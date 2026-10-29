"""
SAM3 Video-based UAP Attack - 与 SAM2 uap_attack 逻辑对齐
使用 SAM Route (点提示 + Tracker, 不经过 Detector/Geometry Encoder)
Video 模式: 三个损失函数项 - loss_t, loss_diff, loss_fea（与 SAM2 一致）

数据划分（与评估一致）:
  - UAP 训练: 训练集 (saco_veval_*_train.json / train_videos.txt)
  - 评估 (train.py val): 测试集 (saco_veval_*_test_split.json)
  因此 UAP 在未见过的测试集上评估，指标不应必然为 0；若出现全 0 可加 --match_sam2_grad 或评估时用 trainer.uap_scale=0.5 减弱扰动。

与 SAM2 uap_attack 的对齐:
  - loss_diff: 已按 SAM2 改为按空间位置算 cosine_similarity(..., dim=1).mean()，不再整特征 flatten。
  - 梯度: SAM2 使用 grad(loss, adv_img, loss)[0] 导致有效步长被 loss 缩放；可选 --match_sam2_grad 复现该行为使攻击更温和。

显存缓解:
  --limit_frames 5/8  每视频少跑帧
  --prompts_num 64/128 少点提示
  --gpu_id 0           指定单卡
  --empty_cache_interval 1  每帧清显存(0=仅每视频结束清)
  --prototype_once_per_video  每视频仅首帧算 prototype
  --offload_features  prototype/target 特征放 CPU，仅算 loss 时上 GPU
  --gradient_checkpoint  backbone 用 checkpoint，反向时重算省显存
  (get_fused_prototype 已改为递推均值不暂存多张特征；prototype 计算包在 no_grad 内)

多卡运行 (数据并行，每卡跑不同视频子集，最后合并 UAP):
  CUDA_VISIBLE_DEVICES=0 python uap_attack_sam3_video.py --video_slice 0,2 ...
  CUDA_VISIBLE_DEVICES=1 python uap_attack_sam3_video.py --video_slice 1,2 ...
  python -c "from uap_attack_sam3_video import merge_multi_gpu_uap; merge_multi_gpu_uap('uap_file/SA-Co_sam3_video', 2)"

前向与显存优化（与 SAM2 uap_attack 等价、少一次 backbone）:
  - 使用 forward_sam_route_with_feature：同一张 adv_img 只跑一次 backbone，同时得到 logits 与 feature。
  - --prototype_once_per_video：每视频仅首帧算 prototype_feature，少 3*(N-1) 次 backbone。
  - 加载后 sam_fwder.to(device)，且 SAM route 内输入统一 .to(self.device)，避免 cuda:N 与 cuda:0 混用。
"""
import os
import random
from argparse import ArgumentParser, Namespace
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.autograd import grad
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
from torch.cuda.amp import autocast
import gc
from attack_setting_sam3 import (
    Sam3VideoForwarder,
    make_multi_prompts,
    seed_everything,
    DarkSamLossModule,
    AdvClipLossModule,
)
from sam3_util import (
    load_sam3_model, infonce_loss, get_fused_prototype,
    choose_dataset_sam3, get_video_to_indices_sam3, get_frame_index_sam3,
    SACO_ROOT, SACO_JSON
)
from dataset_saco import SACoDataset, collate_fn_saco_flat


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="SAM3 Video UAP Attack (SAM Route, 与 SAM2 一致)")

    parser.add_argument('--limit_img', default=-1, type=int, help='ignored when using train split')
    parser.add_argument('--limit_frames', default=15, type=int,
                        help='max frames per video (减小可缓解显存，如 5/8)')
    parser.add_argument('--gpu_id', default=None, type=int,
                        help='指定单卡 id，如 0；不设则用当前 cuda:0')
    parser.add_argument('--empty_cache_interval', default=1, type=int,
                        help='每 N 帧调用 torch.cuda.empty_cache()，0=仅每视频结束清一次')
    parser.add_argument('--video_slice', default=None, type=str,
                        help='多卡分片: rank,total 如 0,2 表示第0片/共2片，只处理 index%%total==rank 的视频')
    parser.add_argument('--train_dataset', default='SA-Co')
    parser.add_argument('--saco_variant', default='saco_yt1b', choices=['saco_yt1b', 'saco_sav', 'saco_sg'],
                        help='SA-Co variant: yt1b/sav/sg, loads from data/splits/*/train_videos.txt')
    parser.add_argument('--root_dir', default='', help='override; default from saco_variant')
    parser.add_argument('--ann_file', default='', help='override; default train JSON from saco_variant')

    parser.add_argument('--checkpoints', default='./sam3/checkpoints/sam3.pt')
    parser.add_argument('--point', default=None, help='point coord (optional); None=grid points')

    parser.add_argument('--seed', default=30, type=int)
    parser.add_argument('--eps', default=4/255, type=float)
    parser.add_argument('--alpha', default=1/255, type=float)
    parser.add_argument('--P_num', default=10, type=int)

    parser.add_argument('--weight_fea', default=0.000001, type=float)
    parser.add_argument('--loss_fea', action='store_true')
    parser.add_argument('--loss_diff', action='store_true')
    parser.add_argument('--loss_t', action='store_true')
    parser.add_argument('--loss_text', action='store_true',
                        help='对 detector+text 的 logits 攻击')
    parser.add_argument('--text_conf_thresh', default=0.25, type=float,
                        help='detector 置信度阈值，与官方 processor 一致')
    parser.add_argument('--weight_text', default=1.0, type=float)
    parser.add_argument('--lambda_darksam', default=0.0, type=float,
                        help='DarkSAM 总项外部权重（单项控制）')
    parser.add_argument('--lambda_advclip', default=0.0, type=float,
                        help='AdvCLIP 总项外部权重（单项控制）')
    parser.add_argument('--darksam_wt', default=1.0, type=float)
    parser.add_argument('--darksam_wf', default=0.1, type=float)
    parser.add_argument('--darksam_wl', default=1.0, type=float)
    parser.add_argument('--darksam_wh', default=0.01, type=float)
    parser.add_argument('--darksam_mask_loss', default='mse', choices=['mse', 'bce'],
                        help='DarkSAM 掩码项：高保真默认 mse，bce 为可选改进')
    parser.add_argument('--darksam_freq_mode', default='haar', choices=['haar', 'avgpool'],
                        help='DarkSAM 频域项：高保真默认 haar，avgpool 为近似')
    parser.add_argument('--darksam_freq_reduction', default='mean', choices=['mean', 'sum'],
                        help='DarkSAM 频域 MSE：sum 与 DarkSAM-main Fre_loss 一致')
    parser.add_argument('--advclip_alpha', default=10.0, type=float)
    parser.add_argument('--advclip_beta', default=5.0, type=float)
    parser.add_argument('--advclip_gamma', default=5.0, type=float)
    parser.add_argument('--advclip_delta', default=1.0, type=float)
    parser.add_argument('--advclip_metric_mode', default='original', choices=['original', 'cosine'],
                        help='AdvCLIP 特征项：original=InfoNCE+UMAP，高保真默认')
    parser.add_argument('--advclip_temperature', default=0.1, type=float,
                        help='AdvCLIP InfoNCE 温度')

    # 简单 MSE 特征攻击分支的独立开关与权重（不影响原有 InfoNCE 分支）
    parser.add_argument('--weight_fea_mse', default=0.0, type=float)
    parser.add_argument('--loss_fea_mse', action='store_true')

    parser.add_argument('--prompts_num', default=256, type=int,
                        help='点提示数量，减小可缓解显存，如 64/128')
    parser.add_argument('--target_image_dir', default='./target_images')
    parser.add_argument('--fea_num', default=30, type=int)
    parser.add_argument('--prototype_once_per_video', action='store_true',
                        help='每视频仅首帧计算 prototype_feature，省显存且少 3*(N-1) 次 backbone')
    parser.add_argument('--offload_features', action='store_true',
                        help='prototype/target 特征放 CPU，仅算 loss 时搬到 GPU，省显存')
    parser.add_argument('--gradient_checkpoint', action='store_true',
                        help='backbone 用 checkpoint，反向时重算以省显存')
    parser.add_argument('--match_sam2_grad', action='store_true',
                        help='与 SAM2 uap_attack 一致：grad 用 grad_outputs=loss，有效步长随 loss 缩放，攻击更温和')
    return parser

def _parse_video_slice(slice_str):
    """Parse 'rank,total' -> (rank, total). None -> (0, 1)."""
    if slice_str is None or slice_str == "":
        return 0, 1
    parts = slice_str.strip().split(",")
    if len(parts) != 2:
        raise ValueError("--video_slice 格式应为 rank,total，如 0,2")
    return int(parts[0]), int(parts[1])


def main():
    parser = get_parser()
    args = parser.parse_args()
    seed_everything(seed=args.seed)

    # 设备：支持单卡指定与多卡分片
    if torch.cuda.is_available():
        device = f"cuda:{args.gpu_id}" if args.gpu_id is not None else "cuda:0"
        torch.cuda.set_device(device)
    else:
        device = "cpu"
    video_rank, video_total = _parse_video_slice(args.video_slice)
    if video_total > 1:
        print(f">> Multi-GPU slice: rank={video_rank}, total={video_total} (only process videos where index % {video_total} == {video_rank})")

    # ========== 1. 加载模型 ==========
    print("=" * 50)
    print("Loading SAM3 Model (Video Mode - SAM Route)")
    print("=" * 50)
    sam_fwder = load_sam3_model(args.checkpoints, device=device)
    if hasattr(sam_fwder, 'forwarder'):
        sam_fwder = sam_fwder.forwarder
    if not isinstance(sam_fwder, Sam3VideoForwarder):
        sam_fwder = Sam3VideoForwarder(sam_fwder.model, device=device)
    # 确保整机在指定 device 上，避免部分参数在 cuda:0、输入在 cuda:N 导致 RuntimeError
    sam_fwder = sam_fwder.to(device)
    sam_fwder.eval()
    if args.gradient_checkpoint:
        setattr(sam_fwder, '_use_backbone_checkpoint', True)
        print(">> Gradient checkpoint enabled for backbone (saves VRAM, more compute)")

    # ========== 2. 加载数据集 ==========
    custom_dataset = choose_dataset_sam3(args)
    video_to_indices = get_video_to_indices_sam3(custom_dataset)
    if video_total > 1:
        video_names = list(video_to_indices.keys())
        video_to_indices = {name: video_to_indices[name] for i, name in enumerate(video_names) if i % video_total == video_rank}
        print(f">> Multi-GPU: this process handles {len(video_to_indices)} videos (slice rank={video_rank}/{video_total})")
    else:
        print(f">> {len(video_to_indices)} videos")

    denorm = lambda x: sam_fwder.denorm_image(x)
    weight_Y = -1

    # ========== 3. 初始化 UAP ==========
    perturbation = torch.empty(1, 3, 1008, 1008).uniform_(-args.eps, args.eps).to(device)
    print(f">> Eps: {args.eps:.4f}, Alpha: {args.alpha:.4f}, Prompts: {args.prompts_num}")
    print(f">> [DEBUG] Initial perturbation: mean={perturbation.mean().item():.6f}, std={perturbation.std().item():.6f}, min={perturbation.min().item():.6f}, max={perturbation.max().item():.6f}")

    # ========== 4. 准备目标特征 ==========
    target_feature = None
    if args.loss_fea:
        target_image_dir = args.target_image_dir
        if os.path.exists(target_image_dir):
            folders = [f for f in os.listdir(target_image_dir) if os.path.isdir(os.path.join(target_image_dir, f))]
            selected_folders = random.sample(folders, min(args.fea_num, len(folders))) if len(folders) >= args.fea_num else folders
            img_list_for_target = []
            for folder in selected_folders[:3]:
                folder_path = os.path.join(target_image_dir, folder)
                image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
                if image_files:
                    img_path = random.choice(image_files)
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize((1008, 1008), Image.Resampling.BICUBIC)
                    img_np = np.array(img)
                    tgt = sam_fwder.transform_image(img_np).to(device)
                    tgt = denorm(tgt)
                    img_list_for_target.append(tgt)
            if img_list_for_target:
                target_feature = get_fused_prototype(img_list_for_target, sam_fwder, device, use_sam_route=True)
                target_feature = target_feature.to(device)
        if target_feature is None:
            print(">> Using zero target feature")
            target_feature = torch.zeros((1, 256, 72, 72), device=device)

    darksam_term = DarkSamLossModule(
        weight_t=args.darksam_wt,
        weight_f=args.darksam_wf,
        weight_l=args.darksam_wl,
        weight_h=args.darksam_wh,
        mask_loss_type=args.darksam_mask_loss,
        freq_mode=args.darksam_freq_mode,
        freq_reduction=getattr(args, 'darksam_freq_reduction', 'mean'),
    ).to(device)
    advclip_term = AdvClipLossModule(
        alpha=args.advclip_alpha,
        beta=args.advclip_beta,
        gamma=args.advclip_gamma,
        delta=args.advclip_delta,
        metric_mode=args.advclip_metric_mode,
        temperature=args.advclip_temperature,
    ).to(device)

    # ========== 5. 训练循环 (与 SAM2 uap_attack 完全一致 - 三损失) ==========
    sample_step_count = 0
    sample_total_g = torch.zeros_like(perturbation, device=device)
    ema_grad = None

    # 保存初始 perturbation 用于对比
    initial_perturbation = perturbation.clone()

    selected_folders = []
    if os.path.exists(args.target_image_dir):
        all_folders = [f for f in os.listdir(args.target_image_dir) if os.path.isdir(os.path.join(args.target_image_dir, f))]
        selected_folders = random.sample(all_folders, min(args.fea_num, len(all_folders))) if len(all_folders) >= args.fea_num else all_folders

    for step in range(args.P_num):
        for video_name, indices in video_to_indices.items():
            print(f"\n{'=' * 40} Processing video: {video_name} {'=' * 40}")

            folder = random.choice(selected_folders) if args.loss_fea and selected_folders else None
            if folder and args.loss_fea:
                folder_path = os.path.join(args.target_image_dir, folder)
                image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
                if image_files:
                    image_path = random.choice(image_files)
                    image = Image.open(image_path).convert("RGB")
                    image = image.resize((1008, 1008), Image.Resampling.BICUBIC)
                    image_np = np.array(image)
                    tgt = sam_fwder.transform_image(image_np).to(device)
                    tgt = denorm(tgt)
                    with torch.no_grad():
                        target_feature = get_fused_prototype([tgt], sam_fwder, device, use_sam_route=True)
                    target_feature = target_feature.to(device)
                    if args.offload_features:
                        target_feature = target_feature.cpu()
                else:
                    target_feature = torch.zeros((1, 256, 72, 72), device=device)
                    if args.offload_features:
                        target_feature = target_feature.cpu()
            elif args.loss_fea and target_feature is None:
                target_feature = torch.zeros((1, 256, 72, 72), device=device)
                if args.offload_features:
                    target_feature = target_feature.cpu()

            video_subset = Subset(custom_dataset, indices)
            video_loader = DataLoader(
                video_subset,
                batch_size=1,
                collate_fn=collate_fn_saco_flat,
                shuffle=False,
                num_workers=0
            )

            prev_adv_feature = None
            prototype_feature_video = None  # 当 --prototype_once_per_video 时每视频只算一次

            for images, text_prompts, gt_masks, sample_ids in tqdm(video_loader, desc=video_name):
                img, mask_gt, sample_id = images[0], gt_masks[0], sample_ids[0]
                tp = text_prompts[0] if text_prompts else ["visual"]
                text_prompts_list = tp if isinstance(tp, (list, tuple)) else ([tp] if tp else ["visual"])
                frame_idx = get_frame_index_sam3(sample_id)

                X = sam_fwder.transform_image(img).to(device)
                benign_img = denorm(X)
                # Y 必须与 forward_sam_route 返回的 mask/logits 尺寸一致（插值到 _orig_hw），否则 loss_t 梯度错误
                H, W = (sam_fwder._orig_hw[0] if (getattr(sam_fwder, "_orig_hw", None) and len(sam_fwder._orig_hw)) else (1008, 1008))
                Y = torch.ones([1, 1, H, W], device=device, dtype=benign_img.dtype) * weight_Y
                Y_bin = Y.bool()

                prompts = make_multi_prompts(args.point, (1008, 1008), args.prompts_num)
                P = sam_fwder.transform_prompts_sam_route(prompts[0], prompts[1])
                P["point_coords"] = P["point_coords"].to(device)
                P["point_labels"] = P["point_labels"].to(device)

                # prototype_feature：每帧算 或 每视频首帧算一次（--prototype_once_per_video 省显存）；no_grad 不存激活
                if args.prototype_once_per_video and prototype_feature_video is not None:
                    prototype_feature = prototype_feature_video
                else:
                    transform1 = transforms.RandomRotation(degrees=15)
                    transform2 = transforms.Lambda(lambda x: x + 0.03 * torch.randn_like(x))
                    transform3 = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.08)
                    aug_imgs = [transform1(benign_img), transform2(benign_img), transform3(benign_img)]
                    with torch.no_grad():
                        prototype_feature = get_fused_prototype(aug_imgs, sam_fwder, device, use_sam_route=True)
                    prototype_feature = prototype_feature.to(device)
                    del aug_imgs
                    if args.prototype_once_per_video:
                        prototype_feature_video = prototype_feature.detach().cpu() if args.offload_features else prototype_feature.detach()
                    if args.offload_features:
                        prototype_feature = prototype_feature.cpu()
                    if device != "cpu" and args.empty_cache_interval > 0:
                        torch.cuda.empty_cache()
                clean_img = benign_img
                adv_img = benign_img + perturbation
                adv_img = torch.clamp(adv_img, 0, 1)
                adv_img.requires_grad = True
                del benign_img, X

                # clean 特征（无梯度，用于简单 MSE 特征攻击分支）
                with torch.no_grad():
                    clean_feature = sam_fwder.get_image_feature_sam_route(clean_img)

                # 一次 backbone 同时得到 logits 与 adv_feature，与 SAM2 逻辑等价但少一次前向、省显存
                # loss_t 使用 Tracker 的 sam_mask_decoder（官方 SAM1/SAM2 点/框→mask 的解码器）
                with autocast(dtype=torch.bfloat16):
                    logits, adv_feature = sam_fwder.forward_sam_route_with_feature(
                        adv_img, P["point_coords"], P["point_labels"]
                    )
                    if logits.dim() == 4 and logits.size(1) > 1:
                        logits = logits[:, 0:1, :, :]
                    mask = logits > sam_fwder.mask_threshold

                    loss_t = torch.tensor(0.0, device=device)
                    loss_ft = torch.tensor(0.0, device=device)
                    feature_diff = torch.tensor(0.0, device=device)
                    loss_fea = torch.tensor(0.0, device=device)

                    weight_loss_t = 0
                    weight_loss_diff = 0

                    if args.loss_t:
                        attacked = mask == Y_bin
                        output = attacked.float() * logits
                        output_f = (~attacked).float() * (1 - logits)
                        loss_t = F.binary_cross_entropy_with_logits(output, Y)
                        loss_ft = -F.binary_cross_entropy_with_logits(output_f, Y)
                        weight_loss_t = 1

                    if args.loss_diff:
                        if prev_adv_feature is not None:
                            # 与 SAM2 uap_attack 一致：按空间位置算 cosine，dim=1 得 (1,H,W) 再 mean
                            feature_diff = -F.cosine_similarity(
                                prev_adv_feature, adv_feature, dim=1
                            ).mean()
                            weight_loss_diff = 1
                        else:
                            feature_diff = torch.tensor(0.0, device=device)

                    # 原有 InfoNCE 特征损失（与 SAM2 保持一致，用于对比 / 迁移）
                    if args.loss_fea and target_feature is not None:
                        pt = prototype_feature.to(device) if prototype_feature.device.type == 'cpu' else prototype_feature
                        tgt_f = target_feature.to(device) if target_feature.device.type == 'cpu' else target_feature
                        loss_fea = infonce_loss(adv_feature, pt, tgt_f)

                    # 新增：最简单的特征攻击分支 —— 拉远 clean 与 adv 特征（负的 MSE），单独一项
                    loss_fea_mse = torch.tensor(0.0, device=device)
                    if args.loss_fea_mse:
                        loss_fea_mse = -F.mse_loss(adv_feature, clean_feature.detach())

                    loss_text = torch.tensor(0.0, device=device)
                    if args.loss_text and text_prompts_list:
                        # loss_text：与 loss_t 同构的 per_slot_losst（符合 SAM3 文本路径：逐候选、不融合）
                        sam_fwder._orig_hw = [(H, W)]
                        thr = getattr(sam_fwder, 'mask_threshold', 0.0)
                        loss_text_sum = torch.tensor(0.0, device=device)
                        for text in text_prompts_list:
                            det_out = sam_fwder.forward_detector_complete(adv_img, text_prompt=text)
                            mask_logits_up = det_out["mask_logits_up"]
                            final_scores = det_out["scores"]
                            if mask_logits_up.dim() == 3:
                                mask_logits_up = mask_logits_up.unsqueeze(0)
                            Q = mask_logits_up.size(1)
                            if Q == 0:
                                continue
                            threshold = getattr(args, 'text_conf_thresh', 0.25)
                            keep_mask = final_scores[0] > threshold
                            if keep_mask.sum() == 0:
                                continue
                            kept_logits = mask_logits_up[0, keep_mask]
                            slot_losses = []
                            for i in range(kept_logits.size(0)):
                                logits_i = kept_logits[i].unsqueeze(0).unsqueeze(0)
                                mask_i = logits_i > thr
                                attacked_i = mask_i == Y_bin
                                out_t = attacked_i.float() * logits_i
                                out_f = (~attacked_i).float() * (1 - logits_i)
                                lt_i = F.binary_cross_entropy_with_logits(out_t, Y)
                                lft_i = -F.binary_cross_entropy_with_logits(out_f, Y)
                                slot_losses.append(lt_i + 0.01 * lft_i)
                            loss_text_sum = loss_text_sum + torch.stack(slot_losses).mean()
                        loss_text = loss_text_sum / max(1, len(text_prompts_list))
                    # 两个特征项拥有各自的权重，不互相干扰
                    loss = (
                        weight_loss_t * loss_t
                        + 0.01 * loss_ft
                        + weight_loss_diff * feature_diff
                        + args.weight_fea * loss_fea
                        + args.weight_fea_mse * loss_fea_mse
                    )
                    if args.loss_text:
                        loss = loss + args.weight_text * loss_text

                    darksam_out = None
                    if args.lambda_darksam != 0.0:
                        darksam_out = darksam_term(
                            logits=logits,
                            clean_img=clean_img,
                            adv_img=adv_img,
                            y_target=weight_Y,
                            mask_threshold=sam_fwder.mask_threshold,
                        )
                        loss = loss + args.lambda_darksam * darksam_out["loss_darksam_total"]

                    advclip_out = None
                    if args.lambda_advclip != 0.0:
                        clean_feat_for_advclip = prototype_feature
                        if clean_feat_for_advclip.device.type == "cpu":
                            clean_feat_for_advclip = clean_feat_for_advclip.to(device)
                        tgt_feat_for_advclip = None
                        if target_feature is not None:
                            tgt_feat_for_advclip = target_feature.to(device) if target_feature.device.type == "cpu" else target_feature
                        advclip_out = advclip_term(
                            adv_feature=adv_feature,
                            clean_feature=clean_feat_for_advclip,
                            adv_img=adv_img,
                            clean_img=clean_img,
                            target_feature=tgt_feat_for_advclip,
                        )
                        loss = loss + args.lambda_advclip * advclip_out["loss_advclip_total"]

                # [DEBUG] 打印损失值和梯度信息
                print(f"  [DEBUG] loss={loss.item():.6f}, loss.requires_grad={loss.requires_grad}, "
                      f"loss_t={loss_t.item():.6f}, loss_diff={feature_diff.item():.6f}, loss_fea={loss_fea.item():.6f}, loss_text={loss_text.item():.6f}")
                if darksam_out is not None:
                    print(f"  [DEBUG] loss_darksam_total={darksam_out['loss_darksam_total'].item():.6f}, "
                          f"d_t={darksam_out['loss_darksam_t'].item():.6f}, d_f={darksam_out['loss_darksam_f'].item():.6f}, "
                          f"d_l={darksam_out['loss_darksam_low'].item():.6f}, d_h={darksam_out['loss_darksam_high'].item():.6f}")
                if advclip_out is not None:
                    print(f"  [DEBUG] loss_advclip_total={advclip_out['loss_advclip_total'].item():.6f}, "
                          f"a_adv={advclip_out['loss_advclip_adv'].item():.6f}, "
                          f"a_umap={advclip_out['loss_advclip_umap'].item():.6f}, "
                          f"a_recon={advclip_out['loss_advclip_recon'].item():.6f}")
                
                if loss.requires_grad:
                    if getattr(args, 'match_sam2_grad', False):
                        g = grad(loss, adv_img, grad_outputs=loss, retain_graph=False)[0]
                    else:
                        g = grad(loss, adv_img, retain_graph=False)[0]
                    print(f"  [DEBUG] gradient: mean={g.mean().item():.8f}, std={g.std().item():.8f}, max_abs={g.abs().max().item():.8f}")
                else:
                    g = torch.zeros_like(adv_img)
                    print(f"  [DEBUG] WARNING: loss has no gradient! Using zero gradient.")

                del loss, Y, Y_bin, P, prototype_feature
                if device != "cpu":
                    torch.cuda.empty_cache()

                beta = 0.95
                if sample_step_count == 0:
                    ema_grad = g.detach()
                else:
                    ema_grad = beta * ema_grad + (1 - beta) * g.detach()
                sample_total_g += ema_grad
                sample_step_count += 1

                if sample_step_count > 0:
                    avg_gradient = sample_total_g / sample_step_count
                    perturbation = (perturbation - avg_gradient.sign() * args.alpha).clamp(-args.eps, args.eps).detach()

                prev_adv_feature = adv_feature.detach()

                del adv_img, logits, mask, adv_feature
                if hasattr(sam_fwder, 'attack_feature_cache'):
                    sam_fwder.attack_feature_cache.clear()
                if hasattr(sam_fwder, 'feature_cache'):
                    sam_fwder.feature_cache.clear()
                if device != "cpu":
                    if args.empty_cache_interval > 0 and sample_step_count % args.empty_cache_interval == 0:
                        torch.cuda.empty_cache()
                        gc.collect()
            # 每个视频结束后清一次显存（empty_cache_interval==0 时仅此处清）
            if device != "cpu" and args.empty_cache_interval == 0:
                torch.cuda.empty_cache()

    # ========== 6. 保存 UAP ==========
    print(f"\n>> [DEBUG] Initial perturbation: mean={initial_perturbation.mean().item():.6f}, std={initial_perturbation.std().item():.6f}")
    print(f">> [DEBUG] Final perturbation:   mean={perturbation.mean().item():.6f}, std={perturbation.std().item():.6f}")
    total_change = (perturbation - initial_perturbation).abs()
    print(f">> [DEBUG] Total change: mean={total_change.mean().item():.6f}, max={total_change.max().item():.6f}")
    print(f">> [DEBUG] Total gradient steps: {sample_step_count}")
    if total_change.max().item() < 1e-6:
        print(">> [WARNING] UAP was NOT updated! Check loss computation and gradient flow.")
    # 输出文件名：数据集 + 攻击边界(eps/alpha) + 损失函数组合
    eps_int = int(round(args.eps * 255))
    alpha_int = int(round(args.alpha * 255))
    loss_tag = (
        f"t{1 if args.loss_t else 0}_diff{1 if args.loss_diff else 0}_fea{1 if args.loss_fea else 0}_text{1 if args.loss_text else 0}"
        f"_darksam{args.lambda_darksam:g}_advclip{args.lambda_advclip:g}"
    )
    if args.loss_text:
        loss_tag += "_per_slot_losst"
    base_name = (
        f"{args.train_dataset}_{args.saco_variant}"
        f"_eps{eps_int}_alpha{alpha_int}"
        f"_{loss_tag}"
    )
    os.makedirs("uap_file3", exist_ok=True)
    if video_total > 1:
        uap_save_path = f"uap_file_10-2/{base_name}_rank{video_rank}_of_{video_total}.pth"
        print(f"\n Multi-GPU: UAP slice saved to {uap_save_path} (merge: load all rank*_of_*.pth and average)")
    else:
        uap_save_path = f"uap_file_new/{base_name}.pth"
    torch.save(perturbation.cpu(), uap_save_path)
    print(f" Global UAP saved to {uap_save_path}")


def merge_multi_gpu_uap(base_path: str, total: int, out_path: str = None):
    """
    合并多卡分片保存的 UAP：加载 base_path_rank0_of_total.pth ... 求平均并保存。
    base_path: 不含 .pth，如 'uap_file/SA-Co_sam3_video'
    total: 分片数
    out_path: 合并后保存路径，默认 base_path + '.pth'
    """
    if out_path is None:
        out_path = base_path + ".pth"
    perts = []
    for rank in range(total):
        p = f"{base_path}_rank{rank}_of_{total}.pth"
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {p}")
        perts.append(torch.load(p, map_location="cpu", weights_only=True))
    merged = sum(perts) / len(perts)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(merged, out_path)
    print(f"Merged {total} UAPs -> {out_path}")




if __name__ == '__main__':
    main()

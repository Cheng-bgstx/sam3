"""
SAM3 Image-based UAP Attack - 与 SAM2 uap_attack 逻辑完全一致
使用 SAM Route (点提示 + Tracker, 不经过 Detector/Geometry Encoder)
Image 模式: 仅 loss_t 和 loss_fea，无时序损失
"""
import os
import random
from argparse import ArgumentParser, Namespace
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.autograd import grad
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.cuda.amp import autocast

from attack_setting_sam3 import (
    Sam3Forwarder,
    Sam3VideoForwarder,
    make_multi_prompts,
    seed_everything,
    DarkSamLossModule,
    AdvClipLossModule,
)
from sam3_util import load_sam3_model, infonce_loss, get_fused_prototype, SACO_ROOT, SACO_JSON
from dataset_saco import SACoDataset


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="SAM3 Image UAP Attack (SAM Route, 与 SAM2 一致)")

    parser.add_argument('--limit_img', default=100, type=int)
    parser.add_argument('--train_dataset', default='SA-Co')
    parser.add_argument('--root_dir', default=SACO_ROOT)
    parser.add_argument('--ann_file', default=SACO_JSON)

    parser.add_argument('--checkpoints', default='./sam3/checkpoints/sam3.pt')
    parser.add_argument('--point', default=None, help='point coord (optional); None=grid points')

    parser.add_argument('--seed', default=30, type=int)
    parser.add_argument('--eps', default=10 / 255, type=float)
    parser.add_argument('--alpha', default=2 / 255, type=float)
    parser.add_argument('--P_num', default=10, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--prompts_num', default=256, type=int)

    parser.add_argument('--weight_fea', default=0.000001, type=float)
    parser.add_argument('--loss_fea', action='store_true')
    # 简单 MSE 特征攻击分支的独立开关与权重（不影响原有 InfoNCE 分支）
    parser.add_argument('--weight_fea_mse', default=0.0, type=float)
    parser.add_argument('--loss_fea_mse', action='store_true')
    parser.add_argument('--loss_t', action='store_true')
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

    parser.add_argument('--target_image_dir', default='./target_images')
    parser.add_argument('--fea_num', default=30, type=int)

    return parser


def collate_fn_image(batch):
    images, text_prompts, gt_masks, sample_ids = [], [], [], []
    for item in batch:
        images.append(item[0])
        text_prompts.append(item[1] if len(item) > 1 else "visual")
        gt_masks.append(item[2] if len(item) > 2 else None)
        sample_ids.append(item[3] if len(item) > 3 else "")
    return images, text_prompts, gt_masks, sample_ids


def main():
    parser = get_parser()
    args = parser.parse_args()
    seed_everything(seed=args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ========== 1. 加载模型 ==========
    print("=" * 50)
    print("Loading SAM3 Model (Image Mode - SAM Route)")
    print("=" * 50)
    sam_fwder = load_sam3_model(args.checkpoints, device=device)
    if hasattr(sam_fwder, 'forwarder'):
        sam_fwder = sam_fwder.forwarder
    sam_fwder.eval()

    # ========== 2. 加载数据集 ==========
    dataset = SACoDataset(
        root_dir=args.root_dir,
        ann_file=args.ann_file,
        target_size=(1008, 1008),
        limit_videos=args.limit_img,
        limit_frames=1
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn_image
    )

    print(f">> {len(dataset)} samples")

    denorm = lambda x: sam_fwder.denorm_image(x)
    weight_Y = -1

    # ========== 3. 初始化 UAP ==========
    perturbation = torch.empty(1, 3, 1008, 1008).uniform_(-args.eps, args.eps).to(device)
    print(f">> Eps: {args.eps:.4f}, Alpha: {args.alpha:.4f}, Prompts: {args.prompts_num}")

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

    # ========== 5. 训练循环 (与 SAM2 uap_attack 完全一致) ==========
    sample_step_count = 0
    sample_total_g = torch.zeros_like(perturbation, device=device)
    ema_grad = None

    for epoch in range(args.P_num):
        print(f"\n{'=' * 40} Epoch {epoch + 1}/{args.P_num} {'=' * 40}")

        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (images, text_prompts, gt_masks, _) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch + 1}")):
            batch_grads = []

            for i, img in enumerate(images):
                mask_gt = gt_masks[i] if i < len(gt_masks) else None

                X = sam_fwder.transform_image(img).to(device)
                benign_img = denorm(X)
                # clean 特征（无梯度，用于简单 MSE 特征攻击分支）
                with torch.no_grad():
                    clean_feature = sam_fwder.get_image_feature_sam_route(benign_img)
                # Y 必须与 forward_sam_route 返回的 mask/logits 尺寸一致（_orig_hw），否则 loss_t 错误
                H, W = (sam_fwder._orig_hw[0] if (getattr(sam_fwder, "_orig_hw", None) and len(sam_fwder._orig_hw)) else (1008, 1008))
                Y = torch.ones([1, 1, H, W], device=device, dtype=benign_img.dtype) * weight_Y
                Y_bin = Y.bool()

                # 生成 grid points，与 SAM2 完全一致
                prompts = make_multi_prompts(args.point, (1008, 1008), args.prompts_num)
                P = sam_fwder.transform_prompts_sam_route(prompts[0], prompts[1])

                # 原型特征 (clean, 无梯度) - 使用 SAM Route
                transform1 = transforms.RandomRotation(degrees=15)
                transform2 = transforms.Lambda(lambda x: x + 0.03 * torch.randn_like(x))
                transform3 = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.08)
                aug_imgs = [transform1(benign_img), transform2(benign_img), transform3(benign_img)]
                prototype_feature = get_fused_prototype(aug_imgs, sam_fwder, device, use_sam_route=True)
                prototype_feature = prototype_feature.to(device)

                # 对抗前向 - SAM Route (点提示，不经过 detector)
                adv_img = benign_img + perturbation
                adv_img = torch.clamp(adv_img, 0, 1)
                adv_img.requires_grad = True

                with autocast(dtype=torch.bfloat16):
                    logits = sam_fwder.forward_sam_route(adv_img, P["point_coords"], P["point_labels"])
                    if logits.dim() == 4 and logits.size(1) > 1:
                        logits = logits[:, 0:1, :, :]
                    mask = logits > sam_fwder.mask_threshold

                    adv_feature = sam_fwder.get_image_feature_sam_route(adv_img)

                    loss_t = torch.tensor(0.0, device=device)
                    loss_ft = torch.tensor(0.0, device=device)
                    weight_loss_t = 0

                    if args.loss_t:
                        # 与 SAM2 uap_attack 完全相同的三个损失项中的 loss_t
                        attacked = mask == Y_bin
                        output = attacked.float() * logits
                        output_f = (~attacked).float() * (1 - logits)
                        loss_t = F.binary_cross_entropy_with_logits(output, Y)
                        loss_ft = -F.binary_cross_entropy_with_logits(output_f, Y)
                        weight_loss_t = 1

                    # 原有 InfoNCE 特征损失（与 SAM2 保持一致，用于对比 / 迁移）
                    loss_fea = torch.tensor(0.0, device=device)
                    if args.loss_fea and target_feature is not None:
                        loss_fea = infonce_loss(adv_feature, prototype_feature, target_feature)

                    # 新增：最简单的特征攻击分支 —— 拉远 clean 与 adv 特征（负的 MSE），单独一项
                    loss_fea_mse = torch.tensor(0.0, device=device)
                    if args.loss_fea_mse:
                        loss_fea_mse = -F.mse_loss(adv_feature, clean_feature.detach())

                    # 两个特征项拥有各自的权重，不互相干扰
                    loss = (
                        weight_loss_t * loss_t
                        + 0.01 * loss_ft
                        + args.weight_fea * loss_fea
                        + args.weight_fea_mse * loss_fea_mse
                    )

                    darksam_out = None
                    if args.lambda_darksam != 0.0:
                        darksam_out = darksam_term(
                            logits=logits,
                            clean_img=benign_img,
                            adv_img=adv_img,
                            y_target=weight_Y,
                            mask_threshold=sam_fwder.mask_threshold,
                        )
                        loss = loss + args.lambda_darksam * darksam_out["loss_darksam_total"]

                    advclip_out = None
                    if args.lambda_advclip != 0.0:
                        tgt_feat_for_advclip = target_feature if target_feature is not None else None
                        advclip_out = advclip_term(
                            adv_feature=adv_feature,
                            clean_feature=prototype_feature,
                            adv_img=adv_img,
                            clean_img=benign_img,
                            target_feature=tgt_feat_for_advclip,
                        )
                        loss = loss + args.lambda_advclip * advclip_out["loss_advclip_total"]

                if loss.requires_grad:
                    g = grad(loss, adv_img, retain_graph=False)[0]
                    batch_grads.append(g.detach())
                    epoch_loss += loss.item()
                    if darksam_out is not None:
                        print(f"[DEBUG] loss_darksam_total={darksam_out['loss_darksam_total'].item():.6f}")
                    if advclip_out is not None:
                        print(f"[DEBUG] loss_advclip_total={advclip_out['loss_advclip_total'].item():.6f}")

                del adv_img, logits, mask, adv_feature

            if batch_grads:
                avg_batch_grad = torch.stack(batch_grads).mean(dim=0)
                beta = 0.95
                if sample_step_count == 0:
                    ema_grad = avg_batch_grad
                else:
                    ema_grad = beta * ema_grad + (1 - beta) * avg_batch_grad
                sample_total_g += ema_grad
                sample_step_count += 1
                num_batches += 1

            torch.cuda.empty_cache()

        if sample_step_count > 0:
            avg_gradient = sample_total_g / sample_step_count
            perturbation = (perturbation - avg_gradient.sign() * args.alpha).clamp(-args.eps, args.eps).detach()

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Avg Loss: {avg_loss:.6f}, UAP L-inf: {perturbation.abs().max().item():.6f}")

    # ========== 6. 保存 UAP ==========
    eps_int = int(round(args.eps * 255))
    alpha_int = int(round(args.alpha * 255))
    loss_tag = (
        f"t{1 if args.loss_t else 0}_fea{1 if args.loss_fea else 0}"
        f"_darksam{args.lambda_darksam:g}_advclip{args.lambda_advclip:g}"
    )
    os.makedirs("uap_file_new", exist_ok=True)
    uap_save_path = f"uap_file_new/{args.train_dataset}_sam3_image_eps{eps_int}_alpha{alpha_int}_{loss_tag}.pth"
    torch.save(perturbation.cpu(), uap_save_path)
    print(f">> Saved to {uap_save_path}")


if __name__ == '__main__':
    main()

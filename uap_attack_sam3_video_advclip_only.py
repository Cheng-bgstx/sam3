"""
SAM3 Video-based UAP Attack (AdvCLIP-only, 对比方案)
--------------------------------------------------

目的：
- 将 AdvCLIP 在 CLIP 特征空间上的“对比 + 结构 + 重建”攻击思想，迁移到 SAM3 的视觉特征空间，
  作为一个**单独的对比基线**，不污染已有 attack 脚本，也不改变 loss_t / loss_diff 等数据流。

特点：
- 只使用 AdvCLIP 损失（InfoNCE + UMAP + 重建），不参与 loss_t / loss_diff / DarkSAM 等其它损失。
- 在一个 batch 内对多帧的 (clean_feature, adv_feature) 做 InfoNCE / UMAP，避免 N=1 退化。
- UAP 为全局扰动 `perturbation`，通过对 adv_img 的梯度做 sign 更新（与 UAP-SAM2 攻击风格一致）。
"""

import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import grad
from torch.utils.data import DataLoader
from tqdm import tqdm

from attack_setting_sam3 import (
    Sam3VideoForwarder,
    seed_everything,
    AdvClipLossModule,
)
from sam3_util import (
    load_sam3_model,
    choose_dataset_sam3,
)
from dataset_saco import SACoDataset, collate_fn_saco_flat


def get_parser() -> ArgumentParser:
    """
    仅 AdvCLIP 的视频 UAP 攻击脚本参数。
    不引入 loss_t / loss_diff / DarkSAM，避免影响现有 attack 脚本。
    """
    parser = ArgumentParser(description="SAM3 Video UAP Attack (AdvCLIP-only baseline)")

    # 数据与模型
    parser.add_argument('--train_dataset', default='SA-Co')
    parser.add_argument('--saco_variant', default='saco_yt1b', choices=['saco_yt1b', 'saco_sav', 'saco_sg'],
                        help='SA-Co variant: yt1b/sav/sg')
    parser.add_argument('--root_dir', default='', help='override; default from saco_variant')
    parser.add_argument('--ann_file', default='', help='override; default train JSON from saco_variant')
    parser.add_argument('--limit_frames', default=15, type=int,
                        help='max frames per video (减小可缓解显存，如 5/8)')
    parser.add_argument('--checkpoints', default='./sam3/checkpoints/sam3.pt')
    parser.add_argument('--gpu_id', default=None, type=int,
                        help='指定单卡 id，如 0；不设则用当前 cuda:0')

    # UAP 超参
    parser.add_argument('--seed', default=30, type=int)
    parser.add_argument('--eps', default=4 / 255, type=float)
    parser.add_argument('--alpha', default=1 / 255, type=float)
    parser.add_argument('--P_num', default=10, type=int,
                        help='攻击轮数（遍历数据集的次数）')

    # AdvCLIP 损失超参（与 AdvCLIP-master 对齐命名）
    parser.add_argument('--lambda_advclip', default=1.0, type=float,
                        help='AdvCLIP 总项外部权重')
    parser.add_argument('--advclip_alpha', default=10.0, type=float)
    parser.add_argument('--advclip_beta', default=5.0, type=float)
    parser.add_argument('--advclip_gamma', default=5.0, type=float)
    parser.add_argument('--advclip_delta', default=1.0, type=float)
    parser.add_argument('--advclip_metric_mode', default='original', choices=['original', 'cosine'],
                        help='AdvCLIP 特征项：original=InfoNCE+UMAP')
    parser.add_argument('--advclip_temperature', default=0.1, type=float,
                        help='AdvCLIP InfoNCE 温度')

    # AdvCLIP 批大小（每次参与 InfoNCE/UMAP 的帧数）
    parser.add_argument('--advclip_batch_frames', default=4, type=int,
                        help='AdvCLIP 批内帧数（>=2 才能避免 N=1 退化）')

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    seed_everything(seed=args.seed)

    # 设备
    if torch.cuda.is_available():
        device = f"cuda:{args.gpu_id}" if args.gpu_id is not None else "cuda:0"
        torch.cuda.set_device(device)
    else:
        device = "cpu"

    # ========== 1. 加载 SAM3 模型 ==========
    print("=" * 50)
    print("Loading SAM3 Model (Video Mode - SAM Route, AdvCLIP-only)")
    print("=" * 50)
    sam_fwder = load_sam3_model(args.checkpoints, device=device)
    if hasattr(sam_fwder, 'forwarder'):
        sam_fwder = sam_fwder.forwarder
    if not isinstance(sam_fwder, Sam3VideoForwarder):
        sam_fwder = Sam3VideoForwarder(sam_fwder.model, device=device)
    sam_fwder = sam_fwder.to(device)
    sam_fwder.eval()

    # ========== 2. 加载数据集（视频被平铺为单帧样本） ==========
    dataset = choose_dataset_sam3(args)
    # 直接按平铺帧加载，不做 video 分片；对 AdvCLIP 来说，只需要 batch 内有多帧即可
    data_loader = DataLoader(
        dataset,
        batch_size=max(2, args.advclip_batch_frames),
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn_saco_flat,
    )
    print(f">> Total frames: {len(dataset)}, AdvCLIP batch_frames={args.advclip_batch_frames}")

    # 反归一化
    denorm = lambda x: sam_fwder.denorm_image(x)

    # ========== 3. 初始化 UAP ==========
    perturbation = torch.empty(1, 3, 1008, 1008).uniform_(-args.eps, args.eps).to(device)
    print(f">> Eps: {args.eps:.6f}, Alpha: {args.alpha:.6f}")

    # ========== 4. AdvCLIP 损失模块 ==========
    advclip_term = AdvClipLossModule(
        alpha=args.advclip_alpha,
        beta=args.advclip_beta,
        gamma=args.advclip_gamma,
        delta=args.advclip_delta,
        metric_mode=args.advclip_metric_mode,
        temperature=args.advclip_temperature,
    ).to(device)

    # ========== 5. 训练循环 (AdvCLIP-only) ==========
    for epoch in range(args.P_num):
        print(f"\n{'=' * 40} AdvCLIP Epoch {epoch + 1}/{args.P_num} {'=' * 40}")

        epoch_loss = 0.0
        num_batches = 0

        for images, text_prompts, gt_masks, sample_ids in tqdm(data_loader, desc=f"Epoch {epoch + 1}"):
            batch_size = len(images)
            if batch_size < 2:
                # N=1 时 InfoNCE/UMAP 退化，直接跳过
                continue

            clean_imgs = []
            adv_imgs_list = []
            clean_feats = []
            adv_feats = []

            # 5.1 为 batch 中每一帧构造 clean / adv 图像与特征
            for i in range(batch_size):
                img_np = images[i]  # numpy array, HWC, [0,255]

                # 归一化 + Resize（内部会设置 sam_fwder._orig_hw）
                X = sam_fwder.transform_image(img_np)  # [1,3,1008,1008], float, 已标准化到 [-1,1]
                X = X.to(device)

                # clean 图像 [0,1]
                benign_img = denorm(X)  # [1,3,H,W]
                clean_imgs.append(benign_img)

                # clean 特征（不需要梯度，作为 InfoNCE/UMAP 正样本）
                with torch.no_grad():
                    clean_feat = sam_fwder.get_image_feature_sam_route(benign_img)
                clean_feats.append(clean_feat.detach())

            # 堆叠 clean 图像与 clean 特征
            clean_imgs_batch = torch.cat(clean_imgs, dim=0)       # [B,3,H,W]
            clean_feats_batch = torch.cat(clean_feats, dim=0)     # [B,C,Hf,Wf]

            # 构造 adv 图像，作为对 UAP 的“观察点”
            adv_imgs_batch = clean_imgs_batch + perturbation      # broadcast [1,3,H,W] -> [B,3,H,W]
            adv_imgs_batch = torch.clamp(adv_imgs_batch, 0.0, 1.0)
            adv_imgs_batch.requires_grad = True

            # 对每个帧单独跑 backbone 取 adv 特征（保留梯度）
            for i in range(batch_size):
                adv_img_i = adv_imgs_batch[i:i + 1]  # [1,3,H,W]
                adv_feat_i = sam_fwder.get_image_feature_sam_route(adv_img_i)
                adv_feats.append(adv_feat_i)

            adv_feats_batch = torch.cat(adv_feats, dim=0)  # [B,C,Hf,Wf]

            # 5.2 AdvCLIP 损失（仅使用 clean_feature vs adv_feature + 重建）
            advclip_out = advclip_term(
                adv_feature=adv_feats_batch,
                clean_feature=clean_feats_batch,
                adv_img=adv_imgs_batch,
                clean_img=clean_imgs_batch,
                text_feature=None,
                target_feature=None,
            )
            loss = args.lambda_advclip * advclip_out["loss_advclip_total"]

            # 5.3 梯度 w.r.t. adv_imgs_batch，用于更新全局 UAP
            if loss.requires_grad:
                g = grad(loss, adv_imgs_batch, retain_graph=False)[0]  # [B,3,H,W]
                g_mean = g.mean(dim=0, keepdim=True)                  # [1,3,H,W]

                # UAP 更新：FGSM-style，按平均梯度方向更新
                perturbation = (perturbation - args.alpha * g_mean.sign()).clamp(-args.eps, args.eps).detach()

                epoch_loss += loss.item()
                num_batches += 1

            # 显存回收
            del clean_imgs, adv_imgs_batch, clean_feats_batch, adv_feats_batch
            torch.cuda.empty_cache()

        avg_loss = epoch_loss / max(1, num_batches)
        print(f">> AdvCLIP Epoch {epoch + 1}: avg_loss={avg_loss:.6f}, UAP L-inf={perturbation.abs().max().item():.6f}")

    # ========== 6. 保存 UAP ==========
    eps_int = int(round(args.eps * 255))
    alpha_int = int(round(args.alpha * 255))
    loss_tag = f"advclip_only_alpha{args.advclip_alpha:g}_beta{args.advclip_beta:g}_gamma{args.advclip_gamma:g}_delta{args.advclip_delta:g}"
    base_name = f"{args.train_dataset}_{args.saco_variant}_eps{eps_int}_alpha{alpha_int}_{loss_tag}"

    os.makedirs("uap_file_10-2", exist_ok=True)
    uap_save_path = f"uap_file_10-2/{base_name}.pth"
    torch.save(perturbation.cpu(), uap_save_path)
    print(f">> AdvCLIP-only UAP saved to {uap_save_path}")


if __name__ == '__main__':
    main()


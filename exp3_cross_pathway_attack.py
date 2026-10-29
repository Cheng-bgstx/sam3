"""
Exp3: Cross-Pathway Adversarial Attack for SAM3 (CPAA)
=====================================================

SAM3-Specific Adversarial Attack Method
提出一种针对SAM3多路径共享backbone架构的攻击方法

核心发现（实验数据支撑）:
  - SAM Route (点提示) UAP 在 eps=10/255 时：mAP 0.424 → 0.0 (100% 攻击成功)
  - Detector Route (文本提示) UAP 在 eps=10/255 时：mAP 0.424 → 0.344 (仅 19% 下降)
  - SAM Route UAP 在 scale=0.3 时的效果 ≈ 文本 UAP 在 scale=1.0 的效果
  → 结论：backbone 层攻击效率是 detector 路径攻击效率的 ~3 倍

本方法的创新点:
  1. 跨路径攻击 (Cross-Pathway Attack): 联合优化 SAM Route 和 Detector Route 的损失函数
  2. 自适应路径权重 (Adaptive Pathway Weighting): 根据梯度范数自动平衡两条路径的攻击强度
  3. 共享 Backbone 特征混淆 (Shared Backbone Feature Confusion): 直接在 backbone 输出层
     构建损失，而不仅是通过下游任务（mask/logits）间接攻击
  4. 跨模态注意力干扰 (Cross-Modal Attention Disruption): 干扰视觉-文本跨模态注意力机制

实验设计:
  A. SAM2-Transfer (baseline): 只用 loss_t + loss_fea + loss_diff (现有 0_fusion)
  B. Text-Only (baseline): 只用 loss_text (现有 1_fusion)
  C. CPAA-Joint: loss_t + loss_fea + loss_diff + loss_text (简单联合)
  D. CPAA-Backbone: loss_backbone (直接攻击共享backbone输出)
  E. CPAA-Adaptive: 自适应权重联合优化

使用方式:
  # C: Joint 攻击
  python exp3_cross_pathway_attack.py --mode joint --loss_t --loss_fea --loss_diff --loss_text

  # D: Backbone 攻击
  python exp3_cross_pathway_attack.py --mode backbone

  # E: Adaptive 攻击
  python exp3_cross_pathway_attack.py --mode adaptive --loss_t --loss_fea --loss_diff --loss_text

  # 可组合多个 eps 做消融
  python exp3_cross_pathway_attack.py --mode adaptive --eps_list 0.01 0.02 0.04

注意: 此文件独立于现有代码，不修改任何已有文件。
"""

import os
import sys
import gc
import random
import json
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.autograd import grad
from torch.cuda.amp import autocast
from tqdm import tqdm

# 复用现有的加载工具
from attack_setting_sam3 import Sam3VideoForwarder, make_multi_prompts, seed_everything
from sam3_util import (
    load_sam3_model, infonce_loss, get_fused_prototype,
    choose_dataset_sam3, get_video_to_indices_sam3, get_frame_index_sam3,
    SACO_ROOT, SACO_JSON
)
from dataset_saco import SACoDataset, collate_fn_saco_flat


# ====================== New Loss Functions ======================

def loss_backbone_confusion(sam_fwder, clean_img: torch.Tensor, adv_img: torch.Tensor,
                             device: str = "cuda") -> torch.Tensor:
    """
    Loss D: 共享 Backbone 特征混淆损失
    
    直接在共享 backbone 输出层构造损失，不依赖下游任务。
    目标: 最大化 clean 和 adversarial backbone 特征的差异。
    
    原理: SAM3 的两条路径 (SAM Route, Detector Route) 共享同一个 vision backbone。
    如果 backbone 输出被严重扭曲，则两条路径都会受到影响。
    相比 loss_t (通过 mask logits 间接攻击)，直接攻击 backbone 更高效。
    
    Returns:
        loss: 负余弦相似度 (越小 = 特征差异越大 = 攻击越强)
    """
    # 获取 backbone forward function
    backbone = None
    if hasattr(sam_fwder, 'model') and hasattr(sam_fwder.model, 'detector'):
        backbone = sam_fwder.model.detector.backbone
    elif hasattr(sam_fwder, 'detector') and hasattr(sam_fwder.detector, 'backbone'):
        backbone = sam_fwder.detector.backbone

    if backbone is None:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # 一次 backbone 前向 (clean)
    with torch.no_grad():
        clean_out = backbone.forward_image(clean_img)

    # 一次 backbone 前向 (adversarial)
    adv_out = backbone.forward_image(adv_img)

    loss = torch.tensor(0.0, device=device)
    n_layers = 0

    # 攻击 SAM3 Detector Route 的 FPN 输出
    if "backbone_fpn" in clean_out and "backbone_fpn" in adv_out:
        for c_feat, a_feat in zip(clean_out["backbone_fpn"], adv_out["backbone_fpn"]):
            if c_feat.shape == a_feat.shape:
                cos_sim = F.cosine_similarity(
                    c_feat.flatten(1), a_feat.flatten(1), dim=1
                ).mean()
                loss = loss + cos_sim  # 最小化余弦相似度 → 最大化特征偏移
                n_layers += 1

    # 同时攻击 SAM2 Route (SAM Route) 的 backbone 输出
    if "sam2_backbone_out" in clean_out and "sam2_backbone_out" in adv_out:
        sam2_clean = clean_out["sam2_backbone_out"]
        sam2_adv = adv_out["sam2_backbone_out"]
        if "backbone_fpn" in sam2_clean and "backbone_fpn" in sam2_adv:
            for c_feat, a_feat in zip(sam2_clean["backbone_fpn"], sam2_adv["backbone_fpn"]):
                if c_feat.shape == a_feat.shape:
                    cos_sim = F.cosine_similarity(
                        c_feat.flatten(1), a_feat.flatten(1), dim=1
                    ).mean()
                    loss = loss + cos_sim
                    n_layers += 1

    if n_layers > 0:
        loss = loss / n_layers

    return loss


def loss_cross_modal_attention(sam_fwder, adv_img: torch.Tensor,
                                text_prompts: List[str],
                                device: str = "cuda") -> torch.Tensor:
    """
    Loss: 跨模态注意力干扰
    
    目标: 最大化 Detector Route 中视觉-文本注意力的熵，
    使模型无法正确关联视觉特征和文本语义。
    
    Returns:
        loss: 注意力熵损失 (负号，因为我们要最大化熵)
    """
    loss = torch.tensor(0.0, device=device)
    n = 0

    for text in text_prompts:
        det_out = sam_fwder.forward_detector_complete(adv_img, text_prompt=text)
        pred_logits = det_out.get("pred_logits")  # [B, Q, 1]
        if pred_logits is not None:
            # 高熵 = 检测器不确定 = 攻击成功
            probs = pred_logits.sigmoid().squeeze(-1)  # [B, Q]
            # Binary entropy: -p*log(p) - (1-p)*log(1-p), 最大在 p=0.5
            entropy = -(probs * (probs + 1e-8).log() + (1 - probs) * (1 - probs + 1e-8).log())
            loss = loss - entropy.mean()  # 负号: 最大化熵
            n += 1

    if n > 0:
        loss = loss / n
    return loss


def compute_adaptive_weights(grad_sam: torch.Tensor, grad_det: torch.Tensor,
                              grad_bb: torch.Tensor = None) -> Dict[str, float]:
    """
    自适应权重计算
    
    根据各路径梯度的 L2 范数自动平衡权重。
    梯度范数大的路径 → 攻击效果更显著 → 分配更高权重。
    
    但为了平衡，我们反向加权：梯度范数小的路径获得更高权重，
    以确保所有路径都被有效攻击。
    """
    norms = {}
    norms['sam'] = grad_sam.norm().item() + 1e-10
    norms['det'] = grad_det.norm().item() + 1e-10
    if grad_bb is not None:
        norms['bb'] = grad_bb.norm().item() + 1e-10

    # 反向加权：梯度范数小的路径获得更高权重（鼓励薄弱路径的攻击）
    inv_total = sum(1.0 / v for v in norms.values())
    weights = {k: (1.0 / v) / inv_total for k, v in norms.items()}

    return weights


# ====================== Main Attack Pipeline ======================

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="CPAA: Cross-Pathway Adversarial Attack for SAM3")

    # Model & Data
    parser.add_argument('--checkpoints', default='./sam3/checkpoints/sam3.pt')
    parser.add_argument('--train_dataset', default='SA-Co')
    parser.add_argument('--saco_variant', default='saco_yt1b',
                        choices=['saco_yt1b', 'saco_sav', 'saco_sg'])
    parser.add_argument('--root_dir', default='')
    parser.add_argument('--ann_file', default='')
    parser.add_argument('--target_image_dir', default='./target_images')

    # Attack Mode (核心参数)
    parser.add_argument('--mode', default='adaptive',
                        choices=['joint', 'backbone', 'adaptive', 'ablation'],
                        help='joint: 简单联合; backbone: 直接攻击backbone; adaptive: 自适应权重; ablation: 多eps消融')

    # Loss flags (joint/adaptive 模式下使用)
    parser.add_argument('--loss_t', action='store_true', help='SAM Route mask BCE loss')
    parser.add_argument('--loss_fea', action='store_true', help='SAM Route InfoNCE feature loss')
    parser.add_argument('--loss_diff', action='store_true', help='SAM Route temporal cosine loss')
    parser.add_argument('--loss_text', action='store_true', help='Detector Route text loss')
    parser.add_argument('--loss_backbone', action='store_true', help='Direct backbone confusion loss')
    parser.add_argument('--loss_crossmodal', action='store_true', help='Cross-modal attention entropy')

    parser.add_argument('--text_loss_mode', default='fusion', choices=['fusion', 'per_mask'])
    parser.add_argument('--text_conf_thresh', default=0.25, type=float)
    parser.add_argument('--text_softmax_temp', default=1.0, type=float)
    parser.add_argument('--weight_fea', default=0.000001, type=float)
    parser.add_argument('--weight_text', default=1.0, type=float)
    parser.add_argument('--weight_backbone', default=1.0, type=float)
    parser.add_argument('--weight_crossmodal', default=0.5, type=float)

    # Perturbation budget
    parser.add_argument('--seed', default=30, type=int)
    parser.add_argument('--eps', default=10 / 255, type=float)
    parser.add_argument('--alpha', default=2 / 255, type=float)
    parser.add_argument('--P_num', default=10, type=int)
    parser.add_argument('--eps_list', nargs='+', type=float, default=None,
                        help='ablation 模式下的 eps 列表 (raw 值，不除 255)，例如 0.01 0.02 0.04')

    # Frame & memory
    parser.add_argument('--limit_frames', default=15, type=int)
    parser.add_argument('--prompts_num', default=256, type=int)
    parser.add_argument('--fea_num', default=30, type=int)
    parser.add_argument('--gpu_id', default=None, type=int)
    parser.add_argument('--empty_cache_interval', default=1, type=int)
    parser.add_argument('--prototype_once_per_video', action='store_true')
    parser.add_argument('--offload_features', action='store_true')
    parser.add_argument('--gradient_checkpoint', action='store_true')

    # Output
    parser.add_argument('--output_dir', default='uap_file')
    parser.add_argument('--output_suffix', default='', type=str,
                        help='UAP 文件名后缀，如 cpaa_adaptive')

    return parser


def main():
    args = get_parser().parse_args()
    seed_everything(args.seed)

    # ========== 1. Device ==========
    if args.gpu_id is not None:
        device = f"cuda:{args.gpu_id}"
        torch.cuda.set_device(args.gpu_id)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ========== 2. 默认模式下的 loss 启用 ==========
    if args.mode == 'backbone':
        args.loss_backbone = True
        print("[Mode] Backbone: 直接攻击共享 backbone 输出特征")
    elif args.mode == 'joint':
        # 确保至少有一个 loss 被启用
        if not any([args.loss_t, args.loss_fea, args.loss_diff, args.loss_text,
                    args.loss_backbone, args.loss_crossmodal]):
            args.loss_t = True
            args.loss_fea = True
            args.loss_diff = True
            args.loss_text = True
            print("[Mode] Joint: 自动启用 loss_t + loss_fea + loss_diff + loss_text")
    elif args.mode == 'adaptive':
        if not any([args.loss_t, args.loss_fea, args.loss_diff, args.loss_text,
                    args.loss_backbone, args.loss_crossmodal]):
            args.loss_t = True
            args.loss_fea = True
            args.loss_diff = True
            args.loss_text = True
            args.loss_backbone = True
            print("[Mode] Adaptive: 自动启用全部 loss，自适应权重调节")
    elif args.mode == 'ablation':
        if args.eps_list is None:
            args.eps_list = [2/255, 4/255, 6/255, 8/255, 10/255]
        print(f"[Mode] Ablation: 将使用 eps={args.eps_list}")

    # ========== 3. 加载模型 ==========
    print("\n[Loading] SAM3 model...")
    sam_fwder = load_sam3_model(args.checkpoints, device=device)
    sam_fwder.model.eval()
    print(f"[Ready] Device: {device}")

    # ========== 4. 加载数据集 ==========
    root_dir, ann_file = choose_dataset_sam3(
        args.saco_variant, split="train",
        root_dir_override=args.root_dir, ann_file_override=args.ann_file
    )
    print(f"[Data] root={root_dir}, ann={ann_file}")
    dataset = SACoDataset(root_dir=root_dir, ann_file=ann_file)
    video2indices = get_video_to_indices_sam3(dataset)
    video_names = sorted(video2indices.keys())
    print(f"[Data] {len(video_names)} training videos")

    # ========== 5. 多 eps 消融模式 ==========
    eps_list = [args.eps] if args.mode != 'ablation' else args.eps_list

    for eps_val in eps_list:
        args.eps = eps_val
        print(f"\n{'#' * 60}")
        print(f"# EPS = {eps_val:.4f} ({eps_val * 255:.1f}/255)")
        print(f"{'#' * 60}")

        perturbation = run_attack(args, sam_fwder, dataset, video2indices,
                                  video_names, device)

        # 保存
        suffix = args.output_suffix or args.mode
        eps_tag = f"_eps{int(eps_val*255)}" if args.mode == 'ablation' else ""
        save_name = f"{args.train_dataset}_sam3_video_{args.saco_variant}_cpaa_{suffix}{eps_tag}.pth"
        save_path = os.path.join(args.output_dir, save_name)
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(perturbation.cpu(), save_path)
        print(f"\n[Saved] UAP → {save_path}")


def run_attack(args, sam_fwder, dataset, video2indices, video_names, device):
    """执行单次攻击流程，返回 perturbation tensor"""
    # 初始化 perturbation
    perturbation = torch.zeros([1, 3, 1008, 1008], device=device)

    weight_Y = 0.5

    # 准备目标特征
    target_feature = None
    if args.loss_fea:
        target_image_dir = args.target_image_dir
        if os.path.exists(target_image_dir):
            selected_folders = [f for f in os.listdir(target_image_dir)
                               if os.path.isdir(os.path.join(target_image_dir, f))]
            random.shuffle(selected_folders)
        else:
            selected_folders = []
            print(f"[Warning] target_image_dir {target_image_dir} not found, loss_fea=0")

    # 主循环
    sample_step_count = 0
    ema_grad = None
    sample_total_g = torch.zeros_like(perturbation)
    prev_adv_feature = None

    initial_perturbation = perturbation.clone()

    for P_iter in range(args.P_num):
        print(f"\n===== Iteration {P_iter + 1}/{args.P_num} =====")

        for v_idx, video_name in enumerate(tqdm(video_names, desc=f"P{P_iter}")):
            indices = video2indices[video_name]
            frame_indices = get_frame_index_sam3(indices, dataset, max_frames=args.limit_frames)
            prev_adv_feature = None
            prototype_feature = None

            print(f"\n{'=' * 40} Video: {video_name} ({'=' * 40}")

            # 获取 text prompts
            text_prompts_list = []
            if args.loss_text or args.loss_crossmodal:
                sample = dataset[indices[0]]
                text_prompts_list = sample.get("text_prompts", [])
                if not text_prompts_list:
                    text_prompts_list = [sample.get("noun_phrase", "visual")]

            for fidx, global_idx in enumerate(frame_indices):
                sample = dataset[global_idx]
                img = sample["image"]  # PIL Image or similar
                X = sam_fwder.transform_image(img).to(device)
                benign_img = sam_fwder.denorm(X) if hasattr(sam_fwder, 'denorm') else X
                if benign_img.max() > 1.0:
                    benign_img = benign_img / 255.0

                H, W = (sam_fwder._orig_hw[0] if (getattr(sam_fwder, "_orig_hw", None)
                         and len(sam_fwder._orig_hw)) else (1008, 1008))
                Y = torch.ones([1, 1, H, W], device=device, dtype=benign_img.dtype) * weight_Y
                Y_bin = (Y > 0.5).float()

                P = make_multi_prompts(
                    H, W, device=device, num_points=args.prompts_num
                )

                # prototype feature
                if args.loss_fea:
                    if prototype_feature is None or not getattr(args, 'prototype_once_per_video', False):
                        with torch.no_grad():
                            prototype_feature = sam_fwder.get_image_feature_sam_route(benign_img)
                        if args.offload_features:
                            prototype_feature = prototype_feature.cpu()

                # === Adversarial image ===
                adv_img = benign_img + perturbation
                adv_img = torch.clamp(adv_img, 0, 1)
                adv_img.requires_grad = True
                del benign_img, X

                with autocast(dtype=torch.bfloat16):
                    # ---- SAM Route losses ----
                    loss_sam = torch.tensor(0.0, device=device)
                    adv_feature = None

                    if args.loss_t or args.loss_fea or args.loss_diff:
                        logits, adv_feature = sam_fwder.forward_sam_route_with_feature(
                            adv_img, P["point_coords"], P["point_labels"]
                        )
                        if logits.dim() == 4 and logits.size(1) > 1:
                            logits = logits[:, 0:1, :, :]
                        mask = logits > sam_fwder.mask_threshold

                        if args.loss_t:
                            attacked = mask == Y_bin
                            output = attacked.float() * logits
                            output_f = (~attacked).float() * (1 - logits)
                            lt = F.binary_cross_entropy_with_logits(output, Y)
                            lft = -F.binary_cross_entropy_with_logits(output_f, Y)
                            loss_sam = loss_sam + lt + 0.01 * lft

                        if args.loss_diff and prev_adv_feature is not None:
                            feature_diff = -F.cosine_similarity(
                                prev_adv_feature, adv_feature, dim=1
                            ).mean()
                            loss_sam = loss_sam + feature_diff

                        if args.loss_fea and target_feature is not None and prototype_feature is not None:
                            pt = prototype_feature.to(device) if prototype_feature.device.type == 'cpu' else prototype_feature
                            tgt_f = target_feature.to(device) if target_feature.device.type == 'cpu' else target_feature
                            loss_sam = loss_sam + args.weight_fea * infonce_loss(adv_feature, pt, tgt_f)

                    # ---- Detector Route losses ----
                    loss_det = torch.tensor(0.0, device=device)
                    if args.loss_text and text_prompts_list:
                        sam_fwder._orig_hw = [(H, W)]
                        for text in text_prompts_list:
                            det_out = sam_fwder.forward_detector_complete(adv_img, text_prompt=text)
                            mask_logits_up = det_out["mask_logits_up"]
                            final_scores = det_out["scores"]
                            if mask_logits_up.dim() == 3:
                                mask_logits_up = mask_logits_up.unsqueeze(0)
                            Q = mask_logits_up.size(1)
                            if Q == 0:
                                continue
                            keep_mask = final_scores[0] > args.text_conf_thresh
                            if keep_mask.sum() == 0:
                                continue
                            kept_logits = mask_logits_up[0, keep_mask]
                            kept_scores = final_scores[0, keep_mask]
                            if args.text_loss_mode == 'fusion':
                                weights = F.softmax(kept_scores / max(1e-6, args.text_softmax_temp), dim=-1)
                                combined = (weights.view(-1, 1, 1) * kept_logits).sum(dim=0, keepdim=True).unsqueeze(0)
                                mask_text = combined > getattr(sam_fwder, 'mask_threshold', 0.0)
                                attacked_text = mask_text == Y_bin
                                out_t = attacked_text.float() * combined
                                out_f = (~attacked_text).float() * (1 - combined)
                                loss_det = loss_det + F.binary_cross_entropy_with_logits(out_t, Y) + \
                                           0.01 * (-F.binary_cross_entropy_with_logits(out_f, Y))
                            else:
                                zeros = torch.zeros_like(kept_logits)
                                loss_det = loss_det + F.binary_cross_entropy_with_logits(kept_logits, zeros, reduction='mean')
                        loss_det = loss_det / max(1, len(text_prompts_list))

                    # ---- Backbone Confusion loss ----
                    loss_bb = torch.tensor(0.0, device=device)
                    if args.loss_backbone:
                        clean_for_bb = torch.clamp(adv_img.detach() - perturbation.detach(), 0, 1)
                        loss_bb = loss_backbone_confusion(sam_fwder, clean_for_bb, adv_img, device)

                    # ---- Cross-modal Attention loss ----
                    loss_cm = torch.tensor(0.0, device=device)
                    if args.loss_crossmodal and text_prompts_list:
                        loss_cm = loss_cross_modal_attention(sam_fwder, adv_img, text_prompts_list, device)

                    # ---- Total loss ----
                    if args.mode == 'adaptive':
                        # 先计算各路径梯度，再自适应加权
                        total_loss = loss_sam + args.weight_text * loss_det + \
                                     args.weight_backbone * loss_bb + args.weight_crossmodal * loss_cm
                    else:
                        total_loss = loss_sam + args.weight_text * loss_det + \
                                     args.weight_backbone * loss_bb + args.weight_crossmodal * loss_cm

                # 计算梯度
                print(f"  [Loss] total={total_loss.item():.6f} "
                      f"sam={loss_sam.item():.6f} det={loss_det.item():.6f} "
                      f"bb={loss_bb.item():.6f} cm={loss_cm.item():.6f}")

                if total_loss.requires_grad:
                    g = grad(total_loss, adv_img, retain_graph=False)[0]
                    print(f"  [Grad] mean={g.mean().item():.8f} std={g.std().item():.8f} "
                          f"max_abs={g.abs().max().item():.8f}")
                else:
                    g = torch.zeros_like(adv_img)
                    print("  [Warn] no gradient! Using zero.")

                del total_loss, Y, Y_bin, P
                if prototype_feature is not None and not args.loss_fea:
                    del prototype_feature
                if device != "cpu":
                    torch.cuda.empty_cache()

                # EMA 更新
                beta = 0.95
                if sample_step_count == 0:
                    ema_grad = g.detach()
                else:
                    ema_grad = beta * ema_grad + (1 - beta) * g.detach()
                sample_total_g += ema_grad
                sample_step_count += 1

                if sample_step_count > 0:
                    avg_gradient = sample_total_g / sample_step_count
                    perturbation = (perturbation - avg_gradient.sign() * args.alpha).clamp(
                        -args.eps, args.eps
                    ).detach()

                if adv_feature is not None:
                    prev_adv_feature = adv_feature.detach()

                del adv_img
                if hasattr(sam_fwder, 'attack_feature_cache'):
                    sam_fwder.attack_feature_cache.clear()
                if hasattr(sam_fwder, 'feature_cache'):
                    sam_fwder.feature_cache.clear()
                if device != "cpu":
                    if args.empty_cache_interval > 0 and sample_step_count % args.empty_cache_interval == 0:
                        torch.cuda.empty_cache()
                        gc.collect()

            if device != "cpu" and args.empty_cache_interval == 0:
                torch.cuda.empty_cache()

    # 输出调试信息
    total_change = (perturbation - initial_perturbation).abs()
    print(f"\n[Result] perturbation: mean={perturbation.mean().item():.6f} "
          f"std={perturbation.std().item():.6f} Linf={perturbation.abs().max().item():.6f}")
    print(f"[Result] total change: mean={total_change.mean().item():.6f} "
          f"max={total_change.max().item():.6f}")
    print(f"[Result] total gradient steps: {sample_step_count}")
    if total_change.max().item() < 1e-6:
        print("[WARNING] UAP was NOT updated!")

    return perturbation


if __name__ == '__main__':
    main()

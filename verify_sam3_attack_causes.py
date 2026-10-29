#!/usr/bin/env python3
"""
验证「SAM3 上攻击效果过强」到底由哪些因素导致。

不修改现有训练/评估核心逻辑，仅通过本脚本跑六种验证实验并汇总结果。

============================================
如何运行所有验证功能（可复制命令）
============================================

准备：UAP 文件 xxx.pth、标注 JSON、图像目录、SAM3 checkpoint（如 sam3.pt）。

1) scale_ablation — 扰动强度消融，看 mAP 是否随 scale 增大而下降
   python verify_sam3_attack_causes.py scale_ablation \\
     --uap_path uap_file/xxx.pth \\
     --ann_file data/annotation/saco_veval_yt1b_test_split.json \\
     --img_folder data/media/saco_yt1b/JPEGImages_6fps \\
     --checkpoint sam3/checkpoints/sam3.pt \\
     --out_dir verify_out/scale_ablation
   （可选：--limit_ids N --batch_size 1）

2) metric_compare — 同一预测文件下「video mAP」与按帧/按实例统计对比
   python verify_sam3_attack_causes.py metric_compare \\
     --pred_clean verify_out/clean.json \\
     --pred_attack verify_out/attack.json \\
     --gt_file data/annotation/saco_veval_yt1b_test_split.json

3) sam_route_only — 仅 SAM 点提示分割 + UAP，看 mask 是否崩溃（双路共享 backbone）
   python verify_sam3_attack_causes.py sam_route_only \\
     --uap_path uap_file/xxx.pth \\
     --checkpoint sam3/checkpoints/sam3.pt \\
     --limit 5

4) probe_logits — 内部评分探针：pred_logits + presence_logits (Clean vs Adv)，呈堂证供
   python verify_sam3_attack_causes.py probe_logits \\
     --uap_path uap_file/xxx.pth \\
     --ann_file data/annotation/saco_veval_yt1b_test_split.json \\
     --img_folder data/media/saco_yt1b/JPEGImages_6fps \\
     --checkpoint sam3/checkpoints/sam3.pt \\
     --limit 10
   （输出 Clean/Adv 的 Presence Mean 与 Sigmoid，用于论文图表证据）

5) force_recall — 强制召回：score_threshold_detection=0，保留所有检测再算 mAP
   python verify_sam3_attack_causes.py force_recall \\
     --uap_path uap_file/xxx.pth \\
     --ann_file data/annotation/saco_veval_yt1b_test_split.json \\
     --img_folder data/media/saco_yt1b/JPEGImages_6fps \\
     --checkpoint sam3/checkpoints/sam3.pt \\
     --out_dir verify_out/force_recall
   （可选：--limit_ids N --gt_annot_file xxx.json）

6) modality_compare — 正常阈值 vs 强制召回 的 mAP 对比
   python verify_sam3_attack_causes.py modality_compare \\
     --uap_path uap_file/xxx.pth \\
     --ann_file data/annotation/saco_veval_yt1b_test_split.json \\
     --img_folder data/media/saco_yt1b/JPEGImages_6fps \\
     --checkpoint sam3/checkpoints/sam3.pt \\
     --out_dir verify_out/modality_compare
   （可选：--limit_ids N --gt_annot_file xxx.json）

7) probe_logits 时保存「Clean vs Attack」Mask 证件照：加 --save_vis [--vis_dir verify_out/probe_vis]

8) visualize_internals — 特征热力图 + 原始 Mask（论文病理切片证据）
   python verify_sam3_attack_causes.py visualize_internals --uap_path ... --ann_file ... --img_folder ... --checkpoint ... [--vis_dir verify_out/visualize_internals]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _save_mask_vis(clean_mask, adv_mask, clean_logit_mean, adv_logit_mean, out_path: str):
    """保存「Clean 分割 vs Attack 全白」对比图，用于论文证件照。"""
    import numpy as np
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[vis] matplotlib not installed, skip saving mask vis.")
        return
    # clean_mask / adv_mask: (H,W) or (1,Q,H,W) -> (H,W)
    def to_hw(t):
        if t is None:
            return None
        x = t.detach().cpu().float().numpy()
        while x.ndim > 2:
            x = x[0]
        return x
    c = to_hw(clean_mask)
    a = to_hw(adv_mask)
    if c is None and a is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    if c is not None:
        axes[0].imshow(c, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Clean")
        axes[0].set_xlabel("Logit mean: %.2f" % (clean_logit_mean if clean_logit_mean is not None else 0))
    else:
        axes[0].set_title("Clean (N/A)")
    if a is not None:
        axes[1].imshow(a, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Attack")
        axes[1].set_xlabel("Logit mean: %.2f" % (adv_logit_mean if adv_logit_mean is not None else 0))
    else:
        axes[1].set_title("Attack (N/A)")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle("Pred Masks (before threshold) — Attack → full white / collapse")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("[probe_logits] Saved mask comparison -> %s" % out_path)


def _save_feature_heatmap(feat_tensor, out_path: str, title: str = "Feature"):
    """将 backbone 特征 (C,H,W) 转为热力图并保存。"""
    import numpy as np
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    x = feat_tensor.detach().cpu().float().numpy()
    if x.ndim == 4:
        x = x[0]
    # (C,H,W) -> (H,W) 用通道均值
    if x.ndim == 3:
        x = x.mean(axis=0)
    x = np.nan_to_num(x)
    plt.figure(figsize=(5, 4))
    plt.imshow(x, cmap="hot")
    plt.colorbar()
    plt.title(title)
    plt.axis("off")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("[vis] Saved feature heatmap -> %s" % out_path)


# ---------- 1) scale_ablation ----------
def run_scale_ablation(args):
    """不同 uap_scale 下跑推理+评估，看 mAP 是否随 scale 增大而下降。"""
    import torch

    uap_path = args.uap_path
    if not uap_path or not os.path.isfile(uap_path):
        print("[scale_ablation] ERROR: --uap_path must point to an existing .pth file.")
        return

    uap = torch.load(uap_path, map_location="cpu", weights_only=True)
    if isinstance(uap, dict):
        uap = uap.get("perturbation", uap.get("uap", next(iter(uap.values()), None)))
    if uap is None:
        print("[scale_ablation] ERROR: Could not load perturbation from", uap_path)
        return
    uap = uap.float()
    if uap.dim() == 3:
        uap = uap.unsqueeze(0)

    scales = [0.0, 0.3, 0.5, 0.7, 1.0]
    out_dir = Path(args.out_dir or "verify_out/scale_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)

    script_path = REPO_ROOT / "sam3" / "run_inference_eval.py"
    if not script_path.exists():
        print("[scale_ablation] ERROR: sam3/run_inference_eval.py not found.")
        return

    cmd_base = [
        sys.executable,
        str(script_path),
        "--ann_file", args.ann_file,
        "--img_folder", args.img_folder,
        "--checkpoint", args.checkpoint,
        "--pred_file", "",  # set per scale
        "--eval_res_file", "",  # set per scale
    ]
    if getattr(args, "limit_ids", None) is not None:
        cmd_base.extend(["--limit_ids", str(args.limit_ids)])
    if getattr(args, "batch_size", None) is not None:
        cmd_base.extend(["--batch_size", str(args.batch_size)])

    results = []
    for scale in scales:
        pred_file = str(out_dir / f"pred_scale_{scale}.json")
        eval_file = str(out_dir / f"metrics_scale_{scale}.json")
        cmd = cmd_base + ["--pred_file", pred_file, "--eval_res_file", eval_file]

        if scale == 0.0:
            # 不加 UAP：不传 --uap_path 即可
            pass
        else:
            uap_scaled = uap * scale
            tmp_pth = out_dir / f"uap_scale_{scale}.pth"
            torch.save(uap_scaled.cpu(), tmp_pth)
            cmd = cmd + ["--uap_path", str(tmp_pth)]

        print(f"[scale_ablation] Running scale={scale} -> pred={pred_file}")
        ret = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if ret.returncode != 0:
            print(f"[scale_ablation] WARNING: scale={scale} returned {ret.returncode}")
            results.append((scale, None))
            continue

        if os.path.isfile(eval_file):
            data = _load_json(eval_file)
            mAP = data.get("dataset_results", {}).get("video_mask_mAP_50_95", None)
            results.append((scale, mAP))
        else:
            results.append((scale, None))

    print("\n--- Scale ablation results ---")
    print("scale\tvideo_mask_mAP_50_95")
    for scale, m in results:
        print(f"{scale}\t{m if m is not None else 'N/A'}")
    print("\n若 mAP 随 scale 增大而明显下降，说明「扰动强度」是 mAP 归零的主因之一。")
    return results


# ---------- 2) metric_compare ----------
def _pred_stats(pred_path: str):
    """从预测 JSON 统计：masklet 数、空 bbox 比例、按 video 的分布。"""
    data = _load_json(pred_path)
    if not isinstance(data, list):
        data = data.get("predictions", data) if isinstance(data, dict) else []
    total = len(data)
    videos = set()
    empty_bbox_count = 0
    for p in data:
        vid = p.get("video_id") or p.get("image_id")
        if vid is not None:
            videos.add(vid)
        bboxes = p.get("bboxes", [])
        for b in bboxes:
            if isinstance(b, (list, tuple)) and len(b) >= 4:
                if b[2] == 0 and b[3] == 0:
                    empty_bbox_count += 1
    total_bbox = sum(len(p.get("bboxes", [])) for p in data)
    return {
        "total_masklets": total,
        "unique_videos": len(videos),
        "total_bboxes": total_bbox,
        "empty_bbox_count": empty_bbox_count,
        "empty_bbox_frac": empty_bbox_count / max(1, total_bbox),
    }


def run_metric_compare(args):
    """对比 clean vs attack 预测的简单统计 + 官方 mAP，看是否「检测大面积失效」导致 mAP 归零。"""
    pred_clean = args.pred_clean
    pred_attack = args.pred_attack
    gt_file = args.gt_file or getattr(args, "ann_file", None)

    if not pred_clean or not os.path.isfile(pred_clean):
        print("[metric_compare] ERROR: --pred_clean must be an existing JSON.")
        return
    if not pred_attack or not os.path.isfile(pred_attack):
        print("[metric_compare] ERROR: --pred_attack must be an existing JSON.")
        return

    s_clean = _pred_stats(pred_clean)
    s_attack = _pred_stats(pred_attack)
    print("\n--- Prediction stats ---")
    print("              clean      attack")
    print(f"masklets      {s_clean['total_masklets']:<10} {s_attack['total_masklets']}")
    print(f"unique_videos {s_clean['unique_videos']:<10} {s_attack['unique_videos']}")
    print(f"total_bboxes  {s_clean['total_bboxes']:<10} {s_attack['total_bboxes']}")
    print(f"empty_bboxes  {s_clean['empty_bbox_count']:<10} {s_attack['empty_bbox_count']}")
    print(f"empty_frac    {s_clean['empty_bbox_frac']:.4f}     {s_attack['empty_bbox_frac']:.4f}")

    if gt_file and os.path.isfile(gt_file):
        try:
            from sam3.eval.saco_veval_eval import VEvalEvaluator
            eval_file_c = "/tmp/verify_metrics_clean.json"
            eval_file_a = "/tmp/verify_metrics_attack.json"
            for name, pred_file, eval_file in [
                ("clean", pred_clean, eval_file_c),
                ("attack", pred_attack, eval_file_a),
            ]:
                ev = VEvalEvaluator(gt_annot_file=gt_file, eval_res_file=eval_file)
                ev.run_eval(pred_file=pred_file)
            mc = _load_json(eval_file_c).get("dataset_results", {})
            ma = _load_json(eval_file_a).get("dataset_results", {})
            mAP_c = mc.get("video_mask_mAP_50_95", "N/A")
            mAP_a = ma.get("video_mask_mAP_50_95", "N/A")
            print("\n--- Official mAP ---")
            print(f"  clean  video_mask_mAP_50_95: {mAP_c}")
            print(f"  attack video_mask_mAP_50_95: {mAP_a}")
        except Exception as e:
            print(f"\n[metric_compare] Could not run VEvalEvaluator: {e}")
    else:
        print("\n[metric_compare] No --gt_file, skip official mAP.")

    print("\n若 attack 下 empty_bbox_frac 很高且 total_bboxes 很少，说明「检测大面积失效」导致 mAP 归零。")
    return {"clean": s_clean, "attack": s_attack}


# ---------- 3) sam_route_only ----------
def run_sam_route_only(args):
    """用同一 UAP 只跑 SAM Route（点提示分割），看 mask 是否也崩溃，验证「双路共享 backbone」是否关键。"""
    try:
        from sam3.model_builder import build_sam3_video_model
        from attack_setting_sam3 import Sam3VideoForwarder, make_multi_prompts
    except ImportError as e:
        print("[sam_route_only] Skip: missing deps:", e)
        return

    import torch
    import torch.nn.functional as F

    device = getattr(args, "device", "cuda") or "cuda"
    limit = getattr(args, "limit", 5)
    uap_path = args.uap_path
    checkpoint = args.checkpoint

    if not uap_path or not os.path.isfile(uap_path):
        print("[sam_route_only] ERROR: --uap_path required.")
        return
    if not checkpoint or not os.path.isfile(checkpoint):
        print("[sam_route_only] ERROR: --checkpoint required.")
        return

    # SAM Route 需要 tracker / inst_interactive_predictor，仅 Video 模型具备；用 Video 模型 + VideoForwarder
    sam3_video = build_sam3_video_model(checkpoint_path=checkpoint, device=device)
    sam_fwder = Sam3VideoForwarder(sam3_video, device=device)
    sam_fwder.eval()
    uap = torch.load(uap_path, map_location=device, weights_only=True)
    if isinstance(uap, dict):
        uap = uap.get("perturbation", uap.get("uap", next(iter(uap.values()), None)))
    if uap.dim() == 3:
        uap = uap.unsqueeze(0)
    uap = uap.to(device)

    prompts = make_multi_prompts(None, (1008, 1008), prompt_num=min(16, 256))
    P = sam_fwder.transform_prompts_sam_route(prompts[0], prompts[1])
    P["point_coords"] = P["point_coords"].to(device)
    P["point_labels"] = P["point_labels"].to(device)

    mask_means = []
    with torch.no_grad():
        for _ in range(limit):
            # 用随机 [0,1] 图像模拟输入（与攻击时一致：在 [0,1] 上加 UAP）
            X = torch.rand(1, 3, 1008, 1008, device=device)
            adv = torch.clamp(X + uap, 0.0, 1.0)
            logits, _ = sam_fwder.forward_sam_route_with_feature(
                adv, P["point_coords"], P["point_labels"]
            )
            if logits.dim() == 4 and logits.size(1) > 1:
                logits = logits[:, 0:1]
            mask = (logits > getattr(sam_fwder, "mask_threshold", 0.0)).float()
            mask_means.append(mask.mean().item())

    avg_mean = sum(mask_means) / len(mask_means)
    print(f"\n[sam_route_only] Over {len(mask_means)} random images, mean mask value (SAM Route + UAP) = {avg_mean:.4f}")
    print("  若接近 0 或 1，说明 SAM Route 在 UAP 下输出几乎全 0/全 1（双路共享 backbone 下两路都会坏）。")
    return mask_means


# ---------- 4) probe_logits：内部评分探针 ----------
def run_probe_logits(args):
    """不经过后处理筛选，Hook 推理过程，统计 Clean vs Adv 的 pred_logits（及可选 presence）分布。"""
    import torch
    sys.path.insert(0, str(REPO_ROOT))
    from sam3.model_builder import build_sam3_video_model
    from sam3.run_inference_eval import (
        build_transforms,
        build_dataset,
        apply_uap_to_batch,
    )
    from sam3.uap_apply_utils import load_uap_tensor
    from sam3.model.utils.misc import copy_data_to_device
    from functools import partial
    from sam3.train.data.collator import collate_fn_api
    from tqdm import tqdm

    device = torch.device(getattr(args, "device", "cuda") if torch.cuda.is_available() else "cpu")
    limit = getattr(args, "limit", 10)
    ann_file = getattr(args, "ann_file", None)
    img_folder = getattr(args, "img_folder", None)
    checkpoint = getattr(args, "checkpoint", None)
    uap_path = getattr(args, "uap_path", None)
    if not checkpoint or not os.path.isfile(checkpoint):
        print("[probe_logits] ERROR: --checkpoint required.")
        return
    if not ann_file or not img_folder:
        print("[probe_logits] ERROR: --ann_file and --img_folder required.")
        return

    model = build_sam3_video_model(
        checkpoint_path=checkpoint,
        device=device,
    )
    model.eval()
    norm_mean = getattr(args, "norm_mean", [0.5, 0.5, 0.5])
    norm_std = getattr(args, "norm_std", [0.5, 0.5, 0.5])
    resolution = getattr(args, "resolution", 1008)
    transforms = build_transforms(resolution, norm_mean, norm_std)
    dataset = build_dataset(ann_file, img_folder, transforms, limit_ids=limit)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(collate_fn_api, dict_key="ytvis_val", with_seg_masks=True, repeats=1),
    )
    uap_tensor = load_uap_tensor(uap_path, device) if uap_path and os.path.isfile(uap_path) else None
    save_vis = getattr(args, "save_vis", False)
    vis_dir = Path(getattr(args, "vis_dir", "verify_out/probe_vis"))

    probe_clean_mask = []
    probe_clean_pres = []
    probe_adv_mask = []
    probe_adv_pres = []
    probe_clean_masks_raw = []  # pred_masks for --save_vis
    probe_adv_masks_raw = []
    probe_phase = ["clean"]  # 用 list 以便在闭包中修改

    orig_multi = model.detector.forward_video_grounding_multigpu
    orig_forward_grounding = model.detector.forward_grounding

    def _capture_clean(*a, **kw):
        o, b = orig_multi(*a, **kw)
        if o is not None:
            if "pred_logits" in o:
                probe_clean_mask.append(o["pred_logits"].detach().cpu().float())
            if save_vis and "pred_masks" in o:
                probe_clean_masks_raw.append(o["pred_masks"].detach().cpu().float())
            if "presence_logit_dec" in o:
                probe_clean_pres.append(o["presence_logit_dec"].detach().cpu().float())
            elif "presence_logits" in o:
                probe_clean_pres.append(o["presence_logits"].detach().cpu().float())
        return o, b

    def _capture_adv(*a, **kw):
        o, b = orig_multi(*a, **kw)
        if o is not None:
            if "pred_logits" in o:
                probe_adv_mask.append(o["pred_logits"].detach().cpu().float())
            if save_vis and "pred_masks" in o:
                probe_adv_masks_raw.append(o["pred_masks"].detach().cpu().float())
            if "presence_logit_dec" in o:
                probe_adv_pres.append(o["presence_logit_dec"].detach().cpu().float())
            elif "presence_logits" in o:
                probe_adv_pres.append(o["presence_logits"].detach().cpu().float())
        return o, b

    def _forward_grounding_capture_presence(*a, **kw):
        out = orig_forward_grounding(*a, **kw)
        if out is not None and probe_phase[0] == "clean":
            if "presence_logit_dec" in out:
                probe_clean_pres.append(out["presence_logit_dec"].detach().cpu().float())
            elif "presence_logits" in out:
                probe_clean_pres.append(out["presence_logits"].detach().cpu().float())
        elif out is not None and probe_phase[0] == "adv":
            if "presence_logit_dec" in out:
                probe_adv_pres.append(out["presence_logit_dec"].detach().cpu().float())
            elif "presence_logits" in out:
                probe_adv_pres.append(out["presence_logits"].detach().cpu().float())
        return out

    with torch.no_grad():
        for batch_dict in tqdm(loader, desc="probe_logits", total=min(limit, len(loader))):
            key, batch = batch_dict.popitem()
            batch = copy_data_to_device(batch, device, non_blocking=True)
            # Clean：未加 UAP，Hook 并跑一遍
            probe_phase[0] = "clean"
            model.detector.forward_grounding = _forward_grounding_capture_presence
            model.detector.forward_video_grounding_multigpu = _capture_clean
            _ = model(batch)
            # Adv：同一 batch 加 UAP 再跑，Hook 捕获
            apply_uap_to_batch(batch, uap_tensor, norm_mean, norm_std, device)
            probe_phase[0] = "adv"
            model.detector.forward_video_grounding_multigpu = _capture_adv
            _ = model(batch)
            if len(probe_clean_mask) >= limit and len(probe_adv_mask) >= limit:
                break
    model.detector.forward_video_grounding_multigpu = orig_multi
    model.detector.forward_grounding = orig_forward_grounding

    def _stats(tensors):
        if not tensors:
            return {}
        cat = torch.cat([t.flatten() for t in tensors])
        return {"mean": cat.mean().item(), "max": cat.max().item(), "min": cat.min().item(), "std": cat.std().item(), "n": len(tensors)}

    s_clean_m = _stats(probe_clean_mask)
    s_adv_m = _stats(probe_adv_mask)
    s_clean_p = _stats(probe_clean_pres)
    s_adv_p = _stats(probe_adv_pres)

    print("\n" + "=" * 60)
    print("Internal Logit Probe (未经过滤)")
    print("=" * 60)
    print("\n--- pred_logits (Mask Quality / IoU 相关) ---")
    print("  Clean: mean=%.4f max=%.4f min=%.4f std=%.4f (n=%d)" % (s_clean_m.get("mean", 0), s_clean_m.get("max", 0), s_clean_m.get("min", 0), s_clean_m.get("std", 0), s_clean_m.get("n", 0)))
    print("  Adv:   mean=%.4f max=%.4f min=%.4f std=%.4f (n=%d)" % (s_adv_m.get("mean", 0), s_adv_m.get("max", 0), s_adv_m.get("min", 0), s_adv_m.get("std", 0), s_adv_m.get("n", 0)))
    if probe_clean_mask and probe_adv_mask:
        p_clean = torch.cat([t.sigmoid().flatten() for t in probe_clean_mask])
        p_adv = torch.cat([t.sigmoid().flatten() for t in probe_adv_mask])
        print("  Clean sigmoid: mean=%.4f   Adv sigmoid: mean=%.4f" % (p_clean.mean().item(), p_adv.mean().item()))

    print("\n--- presence_logits (存在性 / 拒识门控) ---")
    if probe_clean_pres and probe_adv_pres:
        print("  Clean Presence: mean=%.4f max=%.4f min=%.4f (n=%d)" % (s_clean_p.get("mean", 0), s_clean_p.get("max", 0), s_clean_p.get("min", 0), s_clean_p.get("n", 0)))
        print("  Adv Presence:   mean=%.4f max=%.4f min=%.4f (n=%d)" % (s_adv_p.get("mean", 0), s_adv_p.get("max", 0), s_adv_p.get("min", 0), s_adv_p.get("n", 0)))
        pres_clean = torch.cat([t.flatten() for t in probe_clean_pres])
        pres_adv = torch.cat([t.flatten() for t in probe_adv_pres])
        sig_clean = pres_clean.sigmoid().mean().item()
        sig_adv = pres_adv.sigmoid().mean().item()
        print("  Clean Presence Sigmoid ≈ %.4f   Adv Presence Sigmoid ≈ %.4f" % (sig_clean, sig_adv))
        print("  -> 若 Adv Presence Mean 显著低于 Clean（如 +3.5 -> -5.0），Sigmoid 接近 0，则「门控拒识」证据成立。")
    else:
        print("  (presence_logit_dec / presence_logits 未在 multigpu 输出中返回，已从 forward_grounding 尝试捕获；若仍为空请检查 detector 输出 key)")
        if probe_clean_pres or probe_adv_pres:
            print("  Clean pres n=%d  Adv pres n=%d" % (len(probe_clean_pres), len(probe_adv_pres)))
    print("=" * 60)

    # 缺口 A：全屏白 Mask 的「证件照」— Clean vs Attack 对比图
    if save_vis and len(probe_clean_masks_raw) > 0 and len(probe_adv_masks_raw) > 0:
        c_logit = probe_clean_mask[0].mean().item() if probe_clean_mask else None
        a_logit = probe_adv_mask[0].mean().item() if probe_adv_mask else None
        def to_2d(t):
            while t.dim() > 2:
                t = t[0]
            return t
        cm = to_2d(probe_clean_masks_raw[0].clone())
        am = to_2d(probe_adv_masks_raw[0].clone())
        if cm.min() < 0 or cm.max() > 1.1:
            cm = cm.sigmoid()
        if am.min() < 0 or am.max() > 1.1:
            am = am.sigmoid()
        _save_mask_vis(cm, am, c_logit, a_logit, str(vis_dir / "probe_clean_vs_adv_mask.png"))

    return {"clean_mask": s_clean_m, "adv_mask": s_adv_m, "clean_pres": s_clean_p, "adv_pres": s_adv_p}


# ---------- 4.5) visualize_internals：特征图 + 原始 Mask 可视化（论文配图） ----------
def run_visualize_internals(args):
    """Hook Detector 输出：保存 Backbone 特征热力图 + 原始 pred_masks（Clean vs Adv），用于论文「病理切片」证据。"""
    import torch
    sys.path.insert(0, str(REPO_ROOT))
    from sam3.model_builder import build_sam3_video_model
    from sam3.run_inference_eval import (
        build_transforms,
        build_dataset,
        apply_uap_to_batch,
    )
    from sam3.uap_apply_utils import load_uap_tensor
    from sam3.model.utils.misc import copy_data_to_device
    from functools import partial
    from sam3.train.data.collator import collate_fn_api
    from tqdm import tqdm

    device = torch.device(getattr(args, "device", "cuda") if torch.cuda.is_available() else "cpu")
    limit = 1  # 只跑第一个 batch
    ann_file = getattr(args, "ann_file", None)
    img_folder = getattr(args, "img_folder", None)
    checkpoint = getattr(args, "checkpoint", None)
    uap_path = getattr(args, "uap_path", None)
    vis_dir = Path(getattr(args, "vis_dir", "verify_out/visualize_internals"))
    if not checkpoint or not os.path.isfile(checkpoint):
        print("[visualize_internals] ERROR: --checkpoint required.")
        return
    if not ann_file or not img_folder:
        print("[visualize_internals] ERROR: --ann_file and --img_folder required.")
        return

    model = build_sam3_video_model(checkpoint_path=checkpoint, device=device)
    model.eval()
    norm_mean = getattr(args, "norm_mean", [0.5, 0.5, 0.5])
    norm_std = getattr(args, "norm_std", [0.5, 0.5, 0.5])
    resolution = getattr(args, "resolution", 1008)
    transforms = build_transforms(resolution, norm_mean, norm_std)
    dataset = build_dataset(ann_file, img_folder, transforms, limit_ids=limit)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(collate_fn_api, dict_key="ytvis_val", with_seg_masks=True, repeats=1),
    )
    uap_tensor = load_uap_tensor(uap_path, device) if uap_path and os.path.isfile(uap_path) else None

    captured = {"clean": {"feat": None, "mask": None, "logit": None}, "adv": {"feat": None, "mask": None, "logit": None}}
    phase = ["clean"]

    orig_forward_grounding = model.detector.forward_grounding

    def _hook_forward_grounding(backbone_out, find_input, find_target, geometric_prompt):
        out = orig_forward_grounding(backbone_out, find_input, find_target, geometric_prompt)
        if out is None:
            return out
        key = phase[0]
        # Backbone 特征：prev_encoder_out.backbone_out 下可能有 backbone_fpn 或 sam2_backbone_out
        try:
            bo = out.get("prev_encoder_out", {}).get("backbone_out", {})
            if "sam2_backbone_out" in bo:
                fpn = bo["sam2_backbone_out"].get("backbone_fpn")
            else:
                fpn = bo.get("backbone_fpn")
            if fpn is not None and len(fpn) > 0:
                # 取最后一层 (C,H,W)，可能带 batch
                last = fpn[-1]
                if last.dim() == 4:
                    last = last[0]
                captured[key]["feat"] = last.detach().cpu().float()
        except Exception:
            pass
        if "pred_masks" in out:
            m = out["pred_masks"].detach().cpu().float()
            if m.dim() >= 4:
                m = m[0, 0]
            elif m.dim() == 3:
                m = m[0]
            if m.min() < 0 or m.max() > 1.1:
                m = m.sigmoid()
            captured[key]["mask"] = m
        if "pred_logits" in out:
            captured[key]["logit"] = out["pred_logits"].detach().cpu().float().mean().item()
        return out

    model.detector.forward_grounding = _hook_forward_grounding

    with torch.no_grad():
        for batch_dict in tqdm(loader, desc="visualize_internals", total=1):
            key, batch = batch_dict.popitem()
            batch = copy_data_to_device(batch, device, non_blocking=True)
            phase[0] = "clean"
            _ = model(batch)
            apply_uap_to_batch(batch, uap_tensor, norm_mean, norm_std, device)
            phase[0] = "adv"
            _ = model(batch)
            break

    model.detector.forward_grounding = orig_forward_grounding

    vis_dir.mkdir(parents=True, exist_ok=True)
    # 特征热力图 Clean vs Adv
    if captured["clean"]["feat"] is not None:
        _save_feature_heatmap(captured["clean"]["feat"], str(vis_dir / "feature_heatmap_clean.png"), "Backbone FPN (Clean)")
    if captured["adv"]["feat"] is not None:
        _save_feature_heatmap(captured["adv"]["feat"], str(vis_dir / "feature_heatmap_adv.png"), "Backbone FPN (Attack)")
    # Raw mask 对比
    if captured["clean"]["mask"] is not None or captured["adv"]["mask"] is not None:
        _save_mask_vis(
            captured["clean"]["mask"],
            captured["adv"]["mask"],
            captured["clean"].get("logit"),
            captured["adv"].get("logit"),
            str(vis_dir / "raw_mask_clean_vs_adv.png"),
        )
    print("[visualize_internals] Outputs saved under %s" % vis_dir)


# ---------- 5) force_recall：强制召回（阈值绕过） ----------
def run_force_recall(args):
    """推理时强制 score_threshold_detection=0，保留所有检测，保存 JSON 并算 mAP。若 mAP 恢复则 Mask 仍可用。"""
    import torch
    sys.path.insert(0, str(REPO_ROOT))
    from sam3.model_builder import build_sam3_video_model
    from sam3.run_inference_eval import (
        build_transforms,
        build_dataset,
        apply_uap_to_batch,
    )
    from sam3.uap_apply_utils import load_uap_tensor
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.eval.postprocessors import PostProcessNullOp
    from sam3.eval.ytvis_eval import YTVISResultsWriter
    from functools import partial
    from sam3.train.data.collator import collate_fn_api
    from tqdm import tqdm
    from iopath.common.file_io import g_pathmgr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(getattr(args, "out_dir", "verify_out/force_recall"))
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_file = str(out_dir / "pred_force_recall.json")
    eval_file = str(out_dir / "metrics_force_recall.json")
    norm_mean = getattr(args, "norm_mean", [0.5, 0.5, 0.5])
    norm_std = getattr(args, "norm_std", [0.5, 0.5, 0.5])
    resolution = getattr(args, "resolution", 1008)
    limit_ids = getattr(args, "limit_ids", None)

    model = build_sam3_video_model(checkpoint_path=args.checkpoint, device=device)
    model.eval()
    # 关键：绕过内部阈值，强制保留所有检测
    if hasattr(model, "score_threshold_detection"):
        model.score_threshold_detection = 0.0
        print("[force_recall] Patched score_threshold_detection = 0.0")
    else:
        for name, mod in [("model", model)] + list(model.named_children()):
            if hasattr(mod, "score_threshold_detection"):
                mod.score_threshold_detection = 0.0
                print("[force_recall] Patched %s.score_threshold_detection = 0.0" % name)
                break

    transforms = build_transforms(resolution, norm_mean, norm_std)
    dataset = build_dataset(args.ann_file, args.img_folder, transforms, limit_ids)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(collate_fn_api, dict_key="ytvis_val", with_seg_masks=True, repeats=1),
    )
    uap_tensor = load_uap_tensor(args.uap_path, device) if getattr(args, "uap_path", None) and os.path.isfile(args.uap_path) else None
    postprocessor = PostProcessNullOp()
    writer = YTVISResultsWriter(dump_file=pred_file, postprocessor=postprocessor, gather_pred_via_filesys=False, write_eval_metrics_file=False)
    dump = []
    with torch.no_grad():
        for batch_dict in tqdm(loader, desc="force_recall"):
            key, batch = batch_dict.popitem()
            batch = copy_data_to_device(batch, device, non_blocking=True)
            apply_uap_to_batch(batch, uap_tensor, norm_mean, norm_std, device)
            find_stages = model(batch)
            predictions = postprocessor.process_results(find_stages=find_stages, find_metadatas=batch.find_metadatas)
            results = writer.prepare(predictions)
            dump.extend(results)
    os.makedirs(os.path.dirname(pred_file) or ".", exist_ok=True)
    with g_pathmgr.open(pred_file, "w") as f:
        json.dump(dump, f)
    print("[force_recall] Predictions saved to", pred_file, "(%d masklets)" % len(dump))

    gt_file = getattr(args, "gt_annot_file", None) or getattr(args, "ann_file", None)
    if os.path.isfile(gt_file):
        from sam3.eval.saco_veval_eval import VEvalEvaluator
        evaluator = VEvalEvaluator(gt_annot_file=gt_file, eval_res_file=eval_file)
        metrics = evaluator.run_eval(pred_file=pred_file)
        mAP = metrics.get("dataset_results", {}).get("video_mask_mAP_50_95", None)
        print("\n--- Force Recall (threshold=0) mAP ---")
        print("  video_mask_mAP_50_95:", mAP)
        print("  若 mAP 从 0 恢复到 20~30%：Mask 仍可用，主要是评分头/门控被欺骗。若仍接近 0：Backbone 特征已严重破坏。")
    return dump


# ---------- 6) modality_compare：正常阈值 vs 强制召回 mAP 对比 ----------
def run_modality_compare(args):
    """对比「正常阈值 0.5」与「强制召回 threshold=0」下的 mAP，区分门控 vs 特征崩塌。完整 Text vs Point 需 GT 注入。"""
    import torch
    sys.path.insert(0, str(REPO_ROOT))
    from sam3.model_builder import build_sam3_video_model
    from sam3.run_inference_eval import (
        build_transforms,
        build_dataset,
        apply_uap_to_batch,
    )
    from sam3.uap_apply_utils import load_uap_tensor
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.eval.postprocessors import PostProcessNullOp
    from sam3.eval.ytvis_eval import YTVISResultsWriter
    from functools import partial
    from sam3.train.data.collator import collate_fn_api
    from tqdm import tqdm
    from iopath.common.file_io import g_pathmgr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(getattr(args, "out_dir", "verify_out/modality_compare"))
    out_dir.mkdir(parents=True, exist_ok=True)
    norm_mean = getattr(args, "norm_mean", [0.5, 0.5, 0.5])
    norm_std = getattr(args, "norm_std", [0.5, 0.5, 0.5])
    resolution = getattr(args, "resolution", 1008)
    limit_ids = getattr(args, "limit_ids", None)
    gt_file = getattr(args, "gt_annot_file", None) or getattr(args, "ann_file", None)

    def run_inference_with_threshold(score_thresh, pred_path, desc):
        model = build_sam3_video_model(checkpoint_path=args.checkpoint, device=device)
        model.eval()
        if hasattr(model, "score_threshold_detection"):
            model.score_threshold_detection = score_thresh
        for _, mod in model.named_children():
            if hasattr(mod, "score_threshold_detection"):
                mod.score_threshold_detection = score_thresh
                break
        transforms = build_transforms(resolution, norm_mean, norm_std)
        dataset = build_dataset(args.ann_file, args.img_folder, transforms, limit_ids)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=partial(collate_fn_api, dict_key="ytvis_val", with_seg_masks=True, repeats=1),
        )
        uap_tensor = load_uap_tensor(getattr(args, "uap_path", None), device) if getattr(args, "uap_path", None) and os.path.isfile(args.uap_path) else None
        writer = YTVISResultsWriter(dump_file=pred_path, postprocessor=PostProcessNullOp(), gather_pred_via_filesys=False, write_eval_metrics_file=False)
        dump = []
        with torch.no_grad():
            for batch_dict in tqdm(loader, desc=desc):
                key, batch = batch_dict.popitem()
                batch = copy_data_to_device(batch, device, non_blocking=True)
                apply_uap_to_batch(batch, uap_tensor, norm_mean, norm_std, device)
                find_stages = model(batch)
                preds = PostProcessNullOp().process_results(find_stages=find_stages, find_metadatas=batch.find_metadatas)
                dump.extend(writer.prepare(preds))
        os.makedirs(os.path.dirname(pred_path) or ".", exist_ok=True)
        with g_pathmgr.open(pred_path, "w") as f:
            json.dump(dump, f)
        return pred_path, len(dump)

    pred_normal = str(out_dir / "pred_normal_thresh.json")
    pred_force = str(out_dir / "pred_force_recall.json")
    run_inference_with_threshold(0.5, pred_normal, "normal_thresh")
    run_inference_with_threshold(0.0, pred_force, "force_recall")

    if gt_file and os.path.isfile(gt_file):
        from sam3.eval.saco_veval_eval import VEvalEvaluator
        ev = VEvalEvaluator(gt_annot_file=gt_file, eval_res_file=str(out_dir / "metrics_normal.json"))
        m_normal = ev.run_eval(pred_normal).get("dataset_results", {}).get("video_mask_mAP_50_95")
        ev2 = VEvalEvaluator(gt_annot_file=gt_file, eval_res_file=str(out_dir / "metrics_force.json"))
        m_force = ev2.run_eval(pred_force).get("dataset_results", {}).get("video_mask_mAP_50_95")
        print("\n--- Modality / Threshold Compare ---")
        print("  正常阈值 (0.5) mAP:", m_normal)
        print("  强制召回 (0.0) mAP:", m_force)
        print("  若 m_force >> m_normal：门控/评分导致丢弃，Mask 仍可用。若两者都接近 0：特征已崩。")
        print("  完整 Text vs Point 模态对比需在 find_inputs 中注入 GT 中心点，可后续扩展。")
    return None


def main():
    p = argparse.ArgumentParser(description="Verify causes of strong SAM3 attack effect.")
    sub = p.add_subparsers(dest="mode", required=True)

    # scale_ablation
    a1 = sub.add_parser("scale_ablation", help="Run inference at different uap_scale; report mAP.")
    a1.add_argument("--uap_path", required=True, help="UAP .pth")
    a1.add_argument("--ann_file", required=True, help="Test annotation JSON")
    a1.add_argument("--img_folder", required=True, help="Image root")
    a1.add_argument("--checkpoint", required=True, help="SAM3 .pt")
    a1.add_argument("--out_dir", default="verify_out/scale_ablation")
    a1.add_argument("--limit_ids", type=int, default=None)
    a1.add_argument("--batch_size", type=int, default=1)

    # metric_compare
    a2 = sub.add_parser("metric_compare", help="Compare clean vs attack pred stats + mAP.")
    a2.add_argument("--pred_clean", required=True, help="Clean prediction JSON")
    a2.add_argument("--pred_attack", required=True, help="Attack prediction JSON")
    a2.add_argument("--gt_file", default=None, help="GT for official mAP")

    # sam_route_only
    a3 = sub.add_parser("sam_route_only", help="Run SAM Route only with UAP; report mask stats.")
    a3.add_argument("--uap_path", required=True)
    a3.add_argument("--checkpoint", required=True)
    a3.add_argument("--limit", type=int, default=5)
    a3.add_argument("--device", default="cuda")

    # probe_logits
    a4 = sub.add_parser("probe_logits", help="Internal logit probe: pred_logits Clean vs Adv (no threshold).")
    a4.add_argument("--uap_path", required=True)
    a4.add_argument("--ann_file", required=True)
    a4.add_argument("--img_folder", required=True)
    a4.add_argument("--checkpoint", required=True)
    a4.add_argument("--limit", type=int, default=10)
    a4.add_argument("--device", default="cuda")
    a4.add_argument("--save_vis", action="store_true", help="Save Clean vs Adv pred_masks PNG (paper figure).")
    a4.add_argument("--vis_dir", default="verify_out/probe_vis", help="Dir for probe_logits --save_vis output.")

    # force_recall
    a5 = sub.add_parser("force_recall", help="Force recall: score_threshold=0, save JSON and mAP.")
    a5.add_argument("--uap_path", required=True)
    a5.add_argument("--ann_file", required=True)
    a5.add_argument("--img_folder", required=True)
    a5.add_argument("--checkpoint", required=True)
    a5.add_argument("--out_dir", default="verify_out/force_recall")
    a5.add_argument("--limit_ids", type=int, default=None)
    a5.add_argument("--gt_annot_file", default=None)

    # modality_compare
    a6 = sub.add_parser("modality_compare", help="Compare normal thresh vs force_recall mAP.")
    a6.add_argument("--uap_path", required=True)
    a6.add_argument("--ann_file", required=True)
    a6.add_argument("--img_folder", required=True)
    a6.add_argument("--checkpoint", required=True)
    a6.add_argument("--out_dir", default="verify_out/modality_compare")
    a6.add_argument("--limit_ids", type=int, default=None)
    a6.add_argument("--gt_annot_file", default=None)

    # visualize_internals
    a7 = sub.add_parser("visualize_internals", help="Save feature heatmaps + raw mask (Clean vs Adv) for paper figures.")
    a7.add_argument("--uap_path", required=True)
    a7.add_argument("--ann_file", required=True)
    a7.add_argument("--img_folder", required=True)
    a7.add_argument("--checkpoint", required=True)
    a7.add_argument("--vis_dir", default="verify_out/visualize_internals")
    a7.add_argument("--device", default="cuda")

    args = p.parse_args()
    if args.mode == "scale_ablation":
        run_scale_ablation(args)
    elif args.mode == "metric_compare":
        run_metric_compare(args)
    elif args.mode == "sam_route_only":
        run_sam_route_only(args)
    elif args.mode == "probe_logits":
        run_probe_logits(args)
    elif args.mode == "force_recall":
        run_force_recall(args)
    elif args.mode == "modality_compare":
        run_modality_compare(args)
    elif args.mode == "visualize_internals":
        run_visualize_internals(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()

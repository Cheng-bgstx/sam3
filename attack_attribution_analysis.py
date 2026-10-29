#!/usr/bin/env python3
"""
攻击成功率归因分析 + UAP 施加位置/缩放检查

一、UAP 位置与缩放检查（--check_uap）
  - 加载 UAP 文件，打印 shape / min / max / mean
  - 说明：attack 与 trainer 均在「反归一化后的 [0,1] 像素空间」加 UAP，再归一化回模型输入；
  - 无二次放大；评估时 uap_scale 仅线性缩放扰动强度。

二、归因分析（--gt_json + --adv_pred，可选 --clean_pred）
  - 对比「干净预测」与「对抗预测」，或单份对抗预测 vs GT
  - 归因维度：① 覆盖不足（只跑了部分视频） ② score 被阈值过滤 ③ 有效分割帧数过少

用法见下方「可直接复制的运行指令」。
"""

import argparse
import json
import os


def is_empty_bbox(bbox):
    if bbox is None or len(bbox) < 4:
        return True
    return all(x == 0 for x in bbox[:4])


def count_effective_frames(p, use_area=True):
    areas = p.get("areas", [])
    bboxes = p.get("bboxes", [])
    if use_area and areas:
        return sum(1 for a in areas if a and a > 0)
    return sum(1 for b in bboxes if not is_empty_bbox(b))


def check_uap(uap_path):
    """检查 UAP 文件：数值范围、施加位置说明。"""
    import torch
    if not os.path.isfile(uap_path):
        print(f"[ERROR] 文件不存在: {uap_path}")
        return
    uap = torch.load(uap_path, map_location="cpu", weights_only=True)
    if isinstance(uap, dict):
        uap = uap.get("perturbation", uap.get("uap", next(iter(uap.values()), None)))
    if uap is None:
        print("[ERROR] 无法从文件中解析出 perturbation 张量")
        return
    uap = uap.float()
    print("=" * 70)
    print("UAP 施加位置与缩放检查")
    print("=" * 70)
    print(f"  文件: {uap_path}")
    print(f"  shape: {tuple(uap.shape)}")
    print(f"  min:   {uap.min().item():.6f}  max: {uap.max().item():.6f}")
    print(f"  mean:  {uap.mean().item():.6f}  std: {uap.std().item():.6f}")
    eps_10_255 = 10.0 / 255.0
    print(f"  常见 attack eps=10/255 ≈ {eps_10_255:.6f}，UAP 应在 [-{eps_10_255:.4f}, {eps_10_255:.4f}] 附近")
    if uap.abs().max().item() > 0.5:
        print("  [注意] 扰动绝对值较大(>0.5)，若评估时未用 uap_scale<1 可能过强")
    print()
    print("  施加位置（与代码一致）:")
    print("  - 攻击训练(uap_attack_sam3_video): benign_img=denorm(X), adv=benign_img+perturbation, 即 [0,1] 空间")
    print("  - 推理(trainer/run_inference_eval): img_01=img_batch*std+mean, adv=(img_01+uap*uap_scale).clamp(0,1)")
    print("  - 结论: 位置正确，均在 [0,1] 加 UAP，无「归一化空间再加」导致的二次放大。")
    print("  - uap_scale: 仅评估时线性缩放；1.0=与训练时同强度，<1 减弱扰动。")
    print("=" * 70)


def load_pred(path):
    with open(path) as f:
        return json.load(f)


def run_attribution(gt_json, adv_pred_path, clean_pred_path=None, prob_thresh=0.5):
    gt = load_pred(gt_json)
    adv = load_pred(adv_pred_path)
    clean = load_pred(clean_pred_path) if clean_pred_path and os.path.isfile(clean_pred_path) else None

    gt_vids = set(v["id"] for v in gt["videos"])
    gt_pairs = set((a["video_id"], a["category_id"]) for a in gt["annotations"])
    vnp_pairs = set((p["video_id"], p["category_id"]) for p in gt.get("video_np_pairs", []))
    adv_pairs = set((p["video_id"], p["category_id"]) for p in adv)
    adv_scores = [p["score"] for p in adv]
    adv_effective = [count_effective_frames(p) for p in adv]

    print("=" * 70)
    print("攻击成功率归因分析")
    print("=" * 70)
    print(f"  GT: videos={len(gt_vids)}, (vid,cat) 标注对数={len(gt_pairs)}")
    print(f"  对抗 pred: track 数={len(adv)}, (vid,cat) 去重={len(adv_pairs)}")
    if clean is not None:
        clean_pairs = set((p["video_id"], p["category_id"]) for p in clean)
        clean_scores = [p["score"] for p in clean]
        clean_effective = [count_effective_frames(p) for p in clean]
        print(f"  干净 pred: track 数={len(clean)}, (vid,cat) 去重={len(clean_pairs)}")
    print()

    # ① 覆盖不足
    gt_no_adv = gt_pairs - adv_pairs
    print("① 覆盖不足（GT 有标注但对抗 pred 中无）")
    print("  GT 有标注但对抗 pred 无的 (vid,cat) 数:", len(gt_no_adv), "/", len(gt_pairs))
    if len(gt_pairs) > 0:
        print("  占比: %.1f%%" % (100.0 * len(gt_no_adv) / len(gt_pairs)))
    if len(gt_pairs) > 0 and len(gt_no_adv) >= len(gt_pairs) * 0.5:
        print("  -> 归因: 大量 (video, phrase) 无预测，召回极低（可能只跑了部分视频或首帧漏检严重）")
    if clean is not None and len(clean_pairs) > len(adv_pairs):
        print("  -> 与干净对比: 干净有 %d 个 (vid,cat)，对抗仅 %d 个，覆盖下降明显" % (len(clean_pairs), len(adv_pairs)))
    print()

    # ② score 被阈值过滤
    above = sum(1 for s in adv_scores if s >= prob_thresh)
    below = len(adv_scores) - above
    print("② Score 与评估阈值 (prob_thresh=%.2f)" % prob_thresh)
    print("  score >= 阈值 的 track 数:", above, "/", len(adv_scores))
    print("  score <  阈值 的 track 数:", below)
    if len(adv_scores) > 0:
        print("  score 范围: %.4f ~ %.4f" % (min(adv_scores), max(adv_scores)))
    if above == 0 and len(adv_scores) > 0:
        print("  -> 归因: 所有 track 被评估过滤，有效预测数为 0")
    elif below >= len(adv_scores) * 0.5:
        print("  -> 归因: 过半 track score 低于阈值，评估时被丢弃")
    if clean is not None and len(clean_scores) > 0 and len(adv_scores) > 0:
        print("  干净 pred score 范围: %.4f ~ %.4f" % (min(clean_scores), max(clean_scores)))
        clean_above = sum(1 for s in clean_scores if s >= prob_thresh)
        print("  干净 score>=阈值 数:", clean_above, "/", len(clean_scores))
    print()

    # ③ 有效分割帧数
    empty = sum(1 for e in adv_effective if e == 0)
    weak = sum(1 for e in adv_effective if 0 < e <= 5)
    ok = sum(1 for e in adv_effective if e > 5)
    print("③ 有效分割帧数（area>0 的帧数）")
    print("  有效帧=0 的 track 数:", empty)
    print("  有效帧 1~5 的 track 数:", weak)
    print("  有效帧>5 的 track 数:", ok)
    if len(adv_effective) > 0:
        print("  平均有效帧/ track: %.1f" % (sum(adv_effective) / len(adv_effective)))
    if empty >= len(adv) * 0.5:
        print("  -> 归因: 过半 track 几乎没有有效分割（跟踪/分割质量崩溃）")
    if clean is not None and len(clean_effective) > 0:
        print("  干净 pred 平均有效帧/ track: %.1f" % (sum(clean_effective) / len(clean_effective)))
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser(description="UAP 检查 + 攻击成功率归因分析")
    ap.add_argument("--check_uap", type=str, default=None, help="仅检查 UAP 文件位置/缩放，不跑归因")
    ap.add_argument("--gt_json", type=str, default=None, help="GT 标注 JSON")
    ap.add_argument("--adv_pred", type=str, default=None, help="对抗样本上的预测 JSON")
    ap.add_argument("--clean_pred", type=str, default=None, help="干净样本上的预测 JSON（可选，用于对比）")
    ap.add_argument("--prob_thresh", type=float, default=0.5, help="评估使用的 score 阈值")
    args = ap.parse_args()

    if args.check_uap:
        check_uap(args.check_uap)
        return

    if not args.gt_json or not args.adv_pred:
        print("归因分析需要 --gt_json 和 --adv_pred；可选 --clean_pred 对比。")
        print("仅检查 UAP 请用: --check_uap <uap.pth>")
        return
    run_attribution(
        args.gt_json,
        args.adv_pred,
        args.clean_pred,
        args.prob_thresh,
    )


if __name__ == "__main__":
    main()

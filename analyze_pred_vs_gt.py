#!/usr/bin/env python3
"""
预测 JSON vs GT：格式对比 + 攻击成功率归因分析

- 对比 pred 与 GT 的数据结构差异
- 逐条分析 pred：score、有效分割帧数（area>0 的帧数）
- 归因：是「完全没有分割」还是「score 过低被滤掉」还是「只跑了部分视频」等

用法: python analyze_pred_vs_gt.py --pred_json <path> --gt_json <path> [--prob_thresh 0.5]
"""

import argparse
import json


def is_empty_bbox(bbox):
    if bbox is None or len(bbox) < 4:
        return True
    return all(x == 0 for x in bbox[:4])


def count_effective_frames(p, use_area=True):
    """统计该条预测有多少帧有有效分割（area>0 或 bbox 非全零）。"""
    areas = p.get("areas", [])
    bboxes = p.get("bboxes", [])
    n = max(len(areas), len(bboxes))
    if use_area and areas:
        return sum(1 for a in areas if a and a > 0)
    return sum(1 for b in bboxes if not is_empty_bbox(b))


def main():
    ap = argparse.ArgumentParser(description="Pred vs GT format and attack success attribution")
    ap.add_argument("--pred_json", required=True, help="预测 JSON")
    ap.add_argument("--gt_json", required=True, help="GT JSON（如 saco_veval_yt1b_test_split.json）")
    ap.add_argument("--prob_thresh", type=float, default=0.5)
    args = ap.parse_args()

    with open(args.gt_json) as f:
        gt = json.load(f)
    with open(args.pred_json) as f:
        pred = json.load(f)

    thresh = args.prob_thresh

    # ----- 格式说明 -----
    print("=" * 70)
    print("一、格式对比")
    print("=" * 70)
    print("GT:")
    print("  - videos: 列表，每项 id, video_name, file_names, height, width")
    print("  - annotations: 列表，每条 = 一个实例 track；含 id, video_id, category_id, segmentations（按帧，null 表示该帧无标注）")
    print("  - 无 bboxes/areas 顶层字段，评估时由 segmentations 推导")
    print("Pred:")
    print("  - 列表，每条 = 一个 track；含 video_id, category_id, score, bboxes（按帧）, segmentations（按帧）, areas（按帧）")
    print("  - 某帧无检测时通常为 bbox=[0,0,0,0]、area=0、seg 为空 RLE")
    print("对比: 两者都是「每条一条 track、按帧的 segmentations」。可直接按 (video_id, category_id) 对齐评估。")
    print()

    # ----- 基本统计 -----
    print("二、基本统计")
    print("=" * 70)
    gt_ann_pairs = set((a["video_id"], a["category_id"]) for a in gt["annotations"])
    gt_ann_count = len(gt["annotations"])
    vnp_pairs = set((p["video_id"], p["category_id"]) for p in gt.get("video_np_pairs", []))
    pred_pairs = set((p["video_id"], p["category_id"]) for p in pred)

    print("GT:  annotation 条数（实例数）:", gt_ann_count)
    print("GT:  (video_id, category_id) 去重对数:", len(gt_ann_pairs))
    print("GT:  video_np_pairs 条数:", len(vnp_pairs))
    print("Pred: 条数（track 数）:", len(pred))
    print("Pred: (video_id, category_id) 去重:", len(pred_pairs))
    print()

    if not pred:
        print("  -> 预测为空 → 攻击成功归因：模型在对抗样本上完全没有输出 track（或推理未跑全）。")
        return

    # ----- Score 与阈值 -----
    scores = [p["score"] for p in pred]
    above = sum(1 for s in scores if s >= thresh)
    below = len(scores) - above
    print("三、Score 与阈值（评估会过滤 score < prob_thresh）")
    print("=" * 70)
    print("  prob_thresh =", thresh)
    print("  score >= 阈值 的 track 数:", above)
    print("  score <  阈值 的 track 数:", below)
    print("  score 范围: %.4f ~ %.4f" % (min(scores), max(scores)))
    if above == 0:
        print("  -> 所有预测被过滤 → 攻击成功归因：分数过低，评估时有效预测为 0。")
    print()

    # ----- 有效分割帧数（每条 pred） -----
    print("四、每条预测的「有效分割」帧数（area>0 的帧数）")
    print("=" * 70)
    n_total_frames = []
    n_effective = []
    for i, p in enumerate(pred):
        nf = len(p.get("bboxes", []))
        ne = count_effective_frames(p)
        n_total_frames.append(nf)
        n_effective.append(ne)
        if i < 15 or ne == 0 or ne < max(n_effective):
            print("  [%d] vid=%s cat=%s score=%.3f 总帧=%d 有效帧(area>0)=%d" % (
                i, p["video_id"], p["category_id"], p["score"], nf, ne))
    if len(pred) > 15:
        print("  ...")
    print()
    empty_tracks = sum(1 for ne in n_effective if ne == 0)
    weak_tracks = sum(1 for ne in n_effective if 0 < ne <= 5)
    ok_tracks = sum(1 for ne in n_effective if ne > 5)
    print("  汇总: 有效帧=0 的 track 数:", empty_tracks, "（完全没有分割出来）")
    print("        有效帧 1~5 的 track 数:", weak_tracks, "（只少量帧有分割）")
    print("        有效帧>5 的 track 数:", ok_tracks)
    if empty_tracks == len(pred):
        print("  -> 攻击成功归因：每条预测都没有有效分割（全为空框/空 mask）。")
    elif empty_tracks + weak_tracks >= len(pred) * 0.8:
        print("  -> 攻击成功归因：绝大多数 track 几乎没有有效分割。")
    print()

    # ----- 覆盖度：GT 是否有对应 pred -----
    print("五、覆盖度（GT 是否有对应预测）")
    print("=" * 70)
    in_gt_ann = pred_pairs & gt_ann_pairs
    gt_no_pred = gt_ann_pairs - pred_pairs
    pred_no_gt = pred_pairs - vnp_pairs
    print("  预测 (vid,cat) 在 GT annotations 中:", len(in_gt_ann), "/", len(pred_pairs))
    print("  GT 有标注但预测中无的 (vid,cat) 数:", len(gt_no_pred), "/", len(gt_ann_pairs))
    print("  预测 (vid,cat) 不在 GT video_np_pairs 中:", len(pred_no_gt))
    if len(gt_ann_pairs) > 0 and len(pred_pairs) < len(gt_ann_pairs) * 0.2:
        print("  -> 攻击成功归因：只对很少的视频/phrase 做了推理，大量 GT 无预测，召回极低。")
    print("=" * 70)


if __name__ == "__main__":
    main()

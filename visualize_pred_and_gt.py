#!/usr/bin/env python3
"""
完整可视化：预测结果 + GT（框、分割 mask）
==========================================
基于现有 sam3/visualization_utils.py 的接口，从 YT-VIS 格式的 pred/GT JSON 和媒体目录
生成：原图 | GT（框+mask）| 预测（框+mask），并可调预测的 score 阈值以检查「是否全黑」和「调低阈值是否符合攻击设定」。

不修改任何现有代码文件，仅读取 pred/GT JSON 与图像。

用法示例:
  # 可视化某次攻击的预测 vs GT（默认只画 score>=0.25 的预测，与评估一致）
  python visualize_pred_and_gt.py \
    --gt_json data/annotation/saco_veval_yt1b_test_split.json \
    --pred_json attack/SA-Co_sam3_video_saco_yt1b_0_fusion/preds/saco_veval_yt1b_test_split.json \
    --media_dir data/media/saco_yt1b/JPEGImages_6fps \
    --out_dir results/vis_0_fusion \
    --max_videos 3 --max_frames 5

  # 调低预测阈值看「若放宽阈值是否还有东西」
  python visualize_pred_and_gt.py ... --score_threshold 0.01

  # 对比 clean vs attack（两个 pred）
  python visualize_pred_and_gt.py \
    --gt_json data/annotation/saco_veval_yt1b_test_split.json \
    --pred_json attack/SA-Co_sam3_video_saco_yt1b_0_fusion/preds/saco_veval_yt1b_test_split.json \
    --pred_json_b attack/test_yt1b_split_d0.5/preds/saco_veval_yt1b_test_split.json \
    --media_dir data/media/saco_yt1b/JPEGImages_6fps \
    --out_dir results/vis_compare \
    --max_videos 2 --max_frames 4
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
import pycocotools.mask as mask_utils
from PIL import Image
from tqdm import tqdm

# 复用现有可视化颜色
try:
    from sam3.visualization_utils import COLORS, draw_box_on_image
except Exception:
    COLORS = np.random.RandomState(42).rand(128, 3).clip(0, 1)
    draw_box_on_image = None


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def build_video_info(gt):
    """video_id -> {video_name, file_names, height, width}"""
    out = {}
    for v in gt.get("videos", []):
        out[v["id"]] = {
            "video_name": v["video_name"],
            "file_names": v.get("file_names", []),
            "height": v.get("height", 0),
            "width": v.get("width", 0),
        }
    return out


def build_gt_tracks(gt, video_info):
    """(video_id, category_id) -> list of {frame_idx: {bbox_xywh, segmentation_rle}}"""
    vid_to_len = {vid: len(info["file_names"]) for vid, info in video_info.items()}
    ann_by_vc = defaultdict(list)
    for a in gt.get("annotations", []):
        vid = a["video_id"]
        cat = a.get("category_id", 1)
        ann_by_vc[(vid, cat)].append(a)
    tracks = {}
    for (vid, cat), anns in ann_by_vc.items():
        a = anns[0]
        segs = a.get("segmentations", [])
        n = vid_to_len.get(vid, len(segs))
        h, w = video_info.get(vid, {}).get("height", 0), video_info.get(vid, {}).get("width", 0)
        frames = {}
        for t in range(n):
            if t < len(segs) and segs[t] is not None:
                rle = segs[t]
                if isinstance(rle.get("counts"), list) and "size" in rle:
                    rle = mask_utils.frPyObjects(rle, rle["size"][0], rle["size"][1])
                frames[t] = {"segmentation": rle, "height": h, "width": w}
        if frames:
            tracks[(vid, cat)] = frames
    return tracks


def build_pred_tracks(pred_list, score_threshold=0.0):
    """pred_list: list of {video_id, category_id, bboxes, score, segmentations, areas}
    -> (video_id, category_id) -> {score, frames: {frame_idx: {bbox_xywh, segmentation}}}
    """
    tracks = {}
    for p in pred_list:
        if p.get("score", 0) < score_threshold:
            continue
        vid = p["video_id"]
        cat = p["category_id"]
        bboxes = p.get("bboxes", [])
        segs = p.get("segmentations", [])
        score = p.get("score", 0)
        key = (vid, cat)
        frames = {}
        for t, (bbox, seg) in enumerate(zip(bboxes, segs)):
            xywh = bbox if isinstance(bbox, (list, tuple)) else list(bbox)
            if sum(abs(x) for x in xywh) < 1e-6:
                continue
            seg_decoded = None
            if seg is not None and t < len(segs):
                s = segs[t]
                if isinstance(s, dict):
                    try:
                        seg_decoded = mask_utils.decode(s)
                    except Exception:
                        seg_decoded = None
            frames[t] = {"bbox_xywh": xywh, "segmentation": seg_decoded}
        if frames:
            tracks[key] = {"score": score, "frames": frames}
    return tracks


def decode_rle_to_mask(rle, h, w):
    if rle is None:
        return np.zeros((h, w), dtype=np.uint8)
    try:
        m = mask_utils.decode(rle)
        if m.ndim == 3:
            m = m.squeeze()
        m = (m > 0).astype(np.uint8)
        if m.shape[:2] != (h, w) and h and w:
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        return m
    except Exception:
        return np.zeros((h, w), dtype=np.uint8)


def mask_to_bbox(mask):
    """(H,W) -> [x,y,w,h] or None"""
    if mask is None or not np.any(mask):
        return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1)]


def draw_boxes_and_masks(img, frame_tracks, video_h, video_w, title_prefix=""):
    """frame_tracks: list of {bbox_xywh, segmentation (np or rle)}. Draw on img (H,W,3)."""
    out = np.array(img, dtype=np.uint8)
    if out.max() <= 1.0:
        out = (out * 255).astype(np.uint8)
    out = out[..., :3]
    h, w = out.shape[:2]
    for idx, tr in enumerate(frame_tracks):
        color = (np.array(COLORS[idx % len(COLORS)]) * 255).astype(np.uint8).tolist()
        bbox = tr.get("bbox_xywh")
        seg = tr.get("segmentation")
        if bbox and len(bbox) >= 4:
            x, y, bw, bh = [int(round(v)) for v in bbox[:4]]
            x2, y2 = x + bw, y + bh
            cv2.rectangle(out, (max(0, x), max(0, y)), (min(w, x2), min(h, y2)), color, 2)
        if seg is not None:
            if isinstance(seg, dict):
                seg = decode_rle_to_mask(seg, h, w)
            if seg is not None and seg.size > 0:
                if seg.shape[:2] != (h, w):
                    seg = cv2.resize(seg.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                mask_bool = seg > 0
                for c in range(3):
                    out[..., c][mask_bool] = (out[..., c][mask_bool] * 0.5 + color[c] * 0.5).astype(np.uint8)
        elif bbox is None and tr.get("segmentation") is None:
            pass
    return out


def run(
    gt_json,
    pred_json,
    media_dir,
    out_dir,
    pred_json_b=None,
    score_threshold=0.25,
    max_videos=5,
    max_frames=10,
):
    os.makedirs(out_dir, exist_ok=True)
    gt = load_json(gt_json)
    pred_list = load_json(pred_json)
    video_info = build_video_info(gt)
    gt_tracks = build_gt_tracks(gt, video_info)
    pred_tracks = build_pred_tracks(pred_list, score_threshold=score_threshold)
    pred_tracks_b = build_pred_tracks(load_json(pred_json_b), score_threshold=score_threshold) if pred_json_b and os.path.isfile(pred_json_b) else None

    # 确定要画的 video_id 列表（有 GT 且有 file_names）
    video_ids = []
    for vid, info in video_info.items():
        if info["file_names"] and (vid in [k[0] for k in gt_tracks] or vid in [k[0] for k in pred_tracks]):
            video_ids.append(vid)
    video_ids = sorted(video_ids)[:max_videos]

    for video_id in tqdm(video_ids, desc="Videos"):
        info = video_info[video_id]
        base_name = info["video_name"]
        file_names = info["file_names"]
        n_frames = min(len(file_names), max_frames)
        for frame_idx in range(n_frames):
            fpath = os.path.join(media_dir, file_names[frame_idx])
            if not os.path.isfile(fpath):
                continue
            img = np.array(Image.open(fpath).convert("RGB"))
            h, w = img.shape[:2]

            # GT for this (video_id, *) at frame_idx
            gt_frame = []
            for (vid, cat), frames in gt_tracks.items():
                if vid != video_id or frame_idx not in frames:
                    continue
                seg_rle = frames[frame_idx].get("segmentation")
                rh = frames[frame_idx].get("height") or h
                rw = frames[frame_idx].get("width") or w
                seg_np = decode_rle_to_mask(seg_rle, rh, rw) if seg_rle else None
                bbox = mask_to_bbox(seg_np) if seg_np is not None else None
                gt_frame.append({"segmentation": seg_np, "bbox_xywh": bbox})

            # Pred for this (video_id, *) at frame_idx
            pred_frame = []
            for (vid, cat), data in pred_tracks.items():
                if vid != video_id or frame_idx not in data["frames"]:
                    continue
                pred_frame.append(data["frames"][frame_idx])

            pred_frame_b = []
            if pred_tracks_b:
                for (vid, cat), data in pred_tracks_b.items():
                    if vid != video_id or frame_idx not in data["frames"]:
                        continue
                    pred_frame_b.append(data["frames"][frame_idx])

            # 画图：原图 | GT | Pred [| Pred_B]
            panels = [img]
            panels.append(draw_boxes_and_masks(img, gt_frame, h, w))
            panels.append(draw_boxes_and_masks(img, pred_frame, h, w))
            if pred_frame_b is not None:
                panels.append(draw_boxes_and_masks(img, pred_frame_b, h, w))

            # 统一缩放到相同高度便于拼接
            target_h = 400
            scaled = []
            for p in panels:
                r = target_h / p.shape[0]
                tw = int(p.shape[1] * r)
                scaled.append(cv2.resize(p, (tw, target_h), interpolation=cv2.INTER_LINEAR))
            row = np.hstack(scaled)
            titles = ["Image", "GT", "Pred"]
            if pred_frame_b is not None:
                titles.append("Pred_B")
            # 在图上写标题（简单白条+字）
            title_h = 30
            row_with_title = np.ones((target_h + title_h, row.shape[1], 3), dtype=np.uint8) * 255
            row_with_title[title_h:] = row
            step = row.shape[1] // len(titles)
            for i, t in enumerate(titles):
                x0 = i * step
                cv2.putText(row_with_title, t, (x0 + 5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            out_path = os.path.join(out_dir, f"{base_name}_f{frame_idx:03d}.png")
            Image.fromarray(row_with_title).save(out_path)

    print(f"Saved visualizations to {out_dir} (videos={len(video_ids)}, frames per video up to {n_frames})")
    print(f"Pred score threshold used: {score_threshold}")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description="Visualize YT-VIS pred and GT (boxes + masks)")
    ap.add_argument("--gt_json", required=True, help="GT annotation JSON (e.g. saco_veval_yt1b_test_split.json)")
    ap.add_argument("--pred_json", required=True, help="Prediction JSON (e.g. attack/.../preds/...json)")
    ap.add_argument("--pred_json_b", default=None, help="Optional second pred JSON (e.g. clean) for comparison")
    ap.add_argument("--media_dir", required=True, help="Media root (e.g. data/media/saco_yt1b/JPEGImages_6fps)")
    ap.add_argument("--out_dir", default="results/vis_pred_gt", help="Output directory for images")
    ap.add_argument("--score_threshold", type=float, default=0.5, help="Only affects which tracks in JSON to draw; official video eval uses prob_thresh=0.5")
    ap.add_argument("--max_videos", type=int, default=5)
    ap.add_argument("--max_frames", type=int, default=10)
    args = ap.parse_args()
    run(
        args.gt_json,
        args.pred_json,
        args.media_dir,
        args.out_dir,
        pred_json_b=args.pred_json_b,
        score_threshold=args.score_threshold,
        max_videos=args.max_videos,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()

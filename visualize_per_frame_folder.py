#!/usr/bin/env python3
"""
按帧文件夹可视化：每帧一个文件夹，包含干净图、对抗图、GT（框+掩码）、对抗预测（框+掩码）
================================================================================
- 官方视频评估阈值：prob_thresh=0.5（见 sam3/eval/saco_veval_evaluators.py）
- 官方图像评估阈值：detection_threshold=0.3（见 sam3/train/configs/eval_base.yaml）
- 本脚本绘制预测时使用的 score 阈值：默认 0.5，与官方视频 eval 一致；仅影响从 JSON 中筛选要画的 track

输出结构：{out_dir}/{video_name}/frame_{idx:03d}/
  - clean.png          原始干净图像
  - adv.png            对抗样本（clean + UAP，与 trainer 一致：[0,1] 空间加扰动）
  - gt_boxes_masks.png GT 框+掩码
  - pred_boxes_masks.png 有预测时的检测结果（框+掩码）；无预测时为 boxes_masks.png（仅 "No Pred" 标注）

运行示例（在项目根目录执行）:
  python visualize_per_frame_folder.py \
    --gt_json data/annotation/saco_veval_yt1b_test_split.json \
    --pred_json attack/SA-Co_sam3_video_saco_yt1b_0_fusion/preds/saco_veval_yt1b_test_split.json \
    --media_dir data/media/saco_yt1b/JPEGImages_6fps \
    --out_dir results/vis_per_frame \
    --uap_path uap_file/SA-Co_sam3_video_saco_yt1b_0_fusion.pth \
    --max_videos 3 --max_frames 10

不提供 --uap_path 时，adv.png 与 clean.png 相同。
依赖：torch, pycocotools, opencv-python, PIL, tqdm（与项目 requirements 一致）。

问题分析（为何有的帧没有 GT）：
- SA-Co/YT-VIS 标注是稀疏的：每个 track 的 segmentations 列表里，很多帧为 null，只有部分帧有 mask。
- 因此「有的帧根本没有 GT」是数据集设计如此，不是脚本错误。无 GT 的帧上 gt_boxes_masks.png 会打上 "No GT (no annotation)"。
- 评估时：无 GT 且无预测记为 True Negative；无 GT 但有预测可能记为 FP（视评估器而定）。只看有 GT 的帧可加 --frames_with_gt_only。
"""

import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np
import pycocotools.mask as mask_utils
import torch
from PIL import Image
from tqdm import tqdm

try:
    from sam3.visualization_utils import COLORS
except Exception:
    COLORS = np.random.RandomState(42).rand(128, 3).clip(0, 1)


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


def build_cat_id_to_phrase(gt):
    """category_id -> noun_phrase，用于在图上标出每个实例对应的 text。"""
    out = {}
    for c in gt.get("categories", []):
        out[int(c["id"])] = c.get("name", "id:%s" % c["id"])
    for p in gt.get("video_np_pairs", []):
        cid = int(p["category_id"])
        if cid not in out:
            out[cid] = p.get("noun_phrase", "id:%s" % cid)
    return out


def build_gt_tracks(gt, video_info):
    """每条 annotation 一条 track（一个实例）。key=(video_id, category_id, ann_id) -> {frame_idx: {...}}"""
    vid_to_len = {vid: len(info["file_names"]) for vid, info in video_info.items()}
    tracks = {}
    for a in gt.get("annotations", []):
        vid = a["video_id"]
        cat = a.get("category_id", 1)
        ann_id = a.get("id", id(a))
        segs = a.get("segmentations", [])
        n = vid_to_len.get(vid, len(segs))
        h = video_info.get(vid, {}).get("height", 0)
        w = video_info.get(vid, {}).get("width", 0)
        frames = {}
        for t in range(n):
            if t < len(segs) and segs[t] is not None:
                rle = segs[t]
                if isinstance(rle.get("counts"), list) and "size" in rle:
                    rle = mask_utils.frPyObjects(rle, rle["size"][0], rle["size"][1])
                frames[t] = {"segmentation": rle, "height": h, "width": w}
        if frames:
            tracks[(vid, cat, ann_id)] = frames
    return tracks


def build_pred_tracks(pred_list, score_threshold=0.0):
    """(video_id, category_id) -> {score, frames: {frame_idx: {bbox_xywh, segmentation}}}"""
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
                        if seg_decoded.ndim == 3:
                            seg_decoded = seg_decoded.squeeze()
                    except Exception:
                        pass
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
    if mask is None or not np.any(mask):
        return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1)]


def draw_boxes_and_masks(img, frame_tracks, video_h, video_w, overlay_text=None):
    """frame_tracks: list of {bbox_xywh, segmentation, phrase?}. Draw on img (H,W,3).
    overlay_text: if given, draw at top-left (e.g. 'No GT' when frame has no annotation).
    If track has 'phrase', draw it above the box (all text, all instance masks and boxes).
    """
    out = np.array(img, dtype=np.uint8)
    if out.max() <= 1.0:
        out = (out * 255).astype(np.uint8)
    out = out[..., :3].copy()
    h, w = out.shape[:2]
    if overlay_text:
        cv2.putText(
            out, overlay_text, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2
        )
    for idx, tr in enumerate(frame_tracks):
        color = (np.array(COLORS[idx % len(COLORS)]) * 255).astype(np.uint8).tolist()
        bbox = tr.get("bbox_xywh")
        seg = tr.get("segmentation")
        phrase = tr.get("phrase")
        if bbox and len(bbox) >= 4:
            x, y, bw, bh = [int(round(v)) for v in bbox[:4]]
            x2, y2 = x + bw, y + bh
            cv2.rectangle(out, (max(0, x), max(0, y)), (min(w, x2), min(h, y2)), color, 2)
            if phrase:
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.5
                thick = 1
                (tw, th), _ = cv2.getTextSize(phrase, font, scale, thick)
                tx = max(0, min(x, w - tw - 2))
                ty = max(th + 2, y - 2)
                cv2.rectangle(out, (tx, ty - th - 2), (tx + tw + 2, ty + 2), (0, 0, 0), -1)
                cv2.putText(out, phrase, (tx + 1, ty - 1), font, scale, color, thick)
        if seg is not None:
            if isinstance(seg, dict):
                seg = decode_rle_to_mask(seg, h, w)
            if seg is not None and seg.size > 0:
                if seg.shape[:2] != (h, w):
                    seg = cv2.resize(seg.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                mask_bool = seg > 0
                for c in range(3):
                    out[..., c][mask_bool] = (
                        (out[..., c][mask_bool].astype(np.float32) * 0.5 + color[c] * 0.5)
                    ).astype(np.uint8)
    return out


def load_uap(uap_path, uap_scale=1.0):
    """加载 UAP，返回 (H, W, 3) numpy，或 None。与 trainer 一致：.pth 中为 [0,1] 空间扰动。"""
    if not uap_path or not str(uap_path).strip() or not os.path.isfile(uap_path):
        return None
    try:
        with open(uap_path, "rb") as f:
            uap = torch.load(f, map_location="cpu", weights_only=True)
        if isinstance(uap, dict):
            uap = uap.get("perturbation", uap.get("uap", next(iter(uap.values()))))
        uap = uap.numpy()
        if uap.ndim == 3:
            uap = uap[np.newaxis, ...]
        if uap.shape[0] != 1 or uap.shape[1] != 3:
            return None
        uap = uap[0]  # (3, H, W)
        uap = np.transpose(uap, (1, 2, 0))  # (H, W, 3)
        uap = uap.astype(np.float64) * float(uap_scale)
        return uap
    except Exception as e:
        print(f"[WARN] UAP load failed: {e}")
        return None


def apply_uap_to_image(img_uint8, uap_nhwc):
    """
    在 [0,1] 空间施加 UAP（与 trainer 一致），返回 uint8 图像。
    img_uint8: (H, W, 3) 0-255
    uap_nhwc: (Uh, Uw, 3) 已乘过 uap_scale 的扰动，或 None
    """
    if uap_nhwc is None:
        return img_uint8
    uh, uw = uap_nhwc.shape[0], uap_nhwc.shape[1]
    if (uh, uw) != (img_uint8.shape[0], img_uint8.shape[1]):
        uap_resized = cv2.resize(
            uap_nhwc.astype(np.float32),
            (img_uint8.shape[1], img_uint8.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        uap_resized = uap_nhwc.astype(np.float32)
    img_01 = img_uint8.astype(np.float64) / 255.0
    adv_01 = np.clip(img_01 + uap_resized, 0.0, 1.0)
    return (adv_01 * 255.0).astype(np.uint8)


def run(
    gt_json,
    pred_json,
    media_dir,
    out_dir,
    uap_path=None,
    uap_scale=1.0,
    score_threshold=0.5,
    max_videos=5,
    max_frames=20,
    frames_with_gt_only=False,
):
    os.makedirs(out_dir, exist_ok=True)
    gt = load_json(gt_json)
    pred_list = load_json(pred_json)
    video_info = build_video_info(gt)
    gt_tracks = build_gt_tracks(gt, video_info)
    pred_tracks = build_pred_tracks(pred_list, score_threshold=score_threshold)
    cat_id_to_phrase = build_cat_id_to_phrase(gt)

    uap_nhwc = load_uap(uap_path, uap_scale=uap_scale)
    if uap_path and uap_nhwc is None:
        print("[WARN] UAP file not loaded; adv.png will be identical to clean.png.")

    video_ids = []
    for vid, info in video_info.items():
        if not info["file_names"]:
            continue
        if vid in [k[0] for k in gt_tracks] or vid in [k[0] for k in pred_tracks]:
            video_ids.append(vid)
    video_ids = sorted(set(video_ids))
    # 优先选有 pred 的视频，避免前几项全是「无 pred」导致看起来像「啥都没有」
    pred_vids = set(k[0] for k in pred_tracks)
    video_ids = sorted(video_ids, key=lambda v: (v not in pred_vids, v))[:max_videos]

    for video_id in tqdm(video_ids, desc="Videos"):
        info = video_info[video_id]
        video_name = info["video_name"]
        file_names = info["file_names"]
        n_frames = min(len(file_names), max_frames)
        for frame_idx in range(n_frames):
            fpath = os.path.join(media_dir, file_names[frame_idx])
            if not os.path.isfile(fpath):
                continue
            img = np.array(Image.open(fpath).convert("RGB"))
            h, w = img.shape[0], img.shape[1]

            frame_dir = os.path.join(out_dir, video_name, f"frame_{frame_idx:03d}")
            os.makedirs(frame_dir, exist_ok=True)

            Image.fromarray(img).save(os.path.join(frame_dir, "clean.png"))

            adv_img = apply_uap_to_image(img, uap_nhwc)
            Image.fromarray(adv_img).save(os.path.join(frame_dir, "adv.png"))

            gt_frame = []
            for key, frames in gt_tracks.items():
                vid, cat = key[0], key[1]
                if vid != video_id or frame_idx not in frames:
                    continue
                seg_rle = frames[frame_idx].get("segmentation")
                rh = frames[frame_idx].get("height") or h
                rw = frames[frame_idx].get("width") or w
                seg_np = decode_rle_to_mask(seg_rle, rh, rw) if seg_rle else None
                bbox = mask_to_bbox(seg_np) if seg_np is not None else None
                phrase = cat_id_to_phrase.get(cat, "id:%s" % cat)
                gt_frame.append({"segmentation": seg_np, "bbox_xywh": bbox, "phrase": phrase})

            if frames_with_gt_only and len(gt_frame) == 0:
                continue

            pred_frame = []
            for (vid, cat), data in pred_tracks.items():
                if vid != video_id or frame_idx not in data["frames"]:
                    continue
                item = dict(data["frames"][frame_idx])
                item["phrase"] = cat_id_to_phrase.get(cat, "id:%s" % cat)
                pred_frame.append(item)

            gt_label = "No GT (no annotation)" if len(gt_frame) == 0 else None
            pred_label = "No Pred" if len(pred_frame) == 0 else None
            gt_vis = draw_boxes_and_masks(img, gt_frame, h, w, overlay_text=gt_label)
            pred_vis = draw_boxes_and_masks(img, pred_frame, h, w, overlay_text=pred_label)

            Image.fromarray(gt_vis).save(os.path.join(frame_dir, "gt_boxes_masks.png"))
            # 有 pred 时文件名带 pred，无 pred 时用 boxes_masks.png（不含 pred）；删除另一种命名避免残留
            pred_fname = "pred_boxes_masks.png" if len(pred_frame) > 0 else "boxes_masks.png"
            other_fname = "boxes_masks.png" if len(pred_frame) > 0 else "pred_boxes_masks.png"
            other_path = os.path.join(frame_dir, other_fname)
            if os.path.isfile(other_path):
                os.remove(other_path)
            Image.fromarray(pred_vis).save(os.path.join(frame_dir, pred_fname))

    print(f"Saved per-frame folders under {out_dir}")
    print(f"Score threshold (for drawing pred): {score_threshold}; UAP: {uap_path or 'none'}")
    if frames_with_gt_only:
        print("Only frames with at least one GT were written (--frames_with_gt_only).")


def main():
    ap = argparse.ArgumentParser(
        description="Per-frame folder viz: clean, adv, GT boxes+masks, pred boxes+masks"
    )
    ap.add_argument("--gt_json", required=True, help="GT annotation JSON (e.g. saco_veval_yt1b_test_split.json)")
    ap.add_argument("--pred_json", required=True, help="Adversarial prediction JSON")
    ap.add_argument("--media_dir", required=True, help="Image root (e.g. data/media/saco_yt1b/JPEGImages_6fps)")
    ap.add_argument("--out_dir", default="results/vis_per_frame", help="Output root; each frame in {out_dir}/{video_name}/frame_XXX/")
    ap.add_argument("--uap_path", default=None, help="UAP .pth path; if not set, adv.png = clean.png")
    ap.add_argument("--uap_scale", type=float, default=1.0, help="Scale UAP when applying (same as trainer)")
    ap.add_argument("--score_threshold", type=float, default=0.5, help="Pred score threshold for drawing (official video eval: 0.5)")
    ap.add_argument("--max_videos", type=int, default=5)
    ap.add_argument("--max_frames", type=int, default=20)
    ap.add_argument("--frames_with_gt_only", action="store_true", help="Only output folders for frames that have at least one GT (skip frames with no annotation)")
    args = ap.parse_args()
    run(
        args.gt_json,
        args.pred_json,
        args.media_dir,
        args.out_dir,
        uap_path=args.uap_path,
        uap_scale=args.uap_scale,
        score_threshold=args.score_threshold,
        max_videos=args.max_videos,
        max_frames=args.max_frames,
        frames_with_gt_only=args.frames_with_gt_only,
    )


if __name__ == "__main__":
    main()

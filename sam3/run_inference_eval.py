#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""
统一推理 + 评估脚本：不依赖 YAML，纯命令行。
支持：视频/图像、不同数据集（SA-Co SAV/YT1B/SmartGlasses 等）、可选 UAP、视频指标 + 图像指标评估。
逻辑与原有 train/trainer + saco_veval_eval 一致，仅做推理与测试，不训练。
"""
import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path

import torch
from iopath.common.file_io import g_pathmgr
from tqdm import tqdm

# 确保可导入 sam3
if __name__ == "__main__" and "__file__" in dir():
    _root = str(Path(__file__).resolve().parents[1])
    if _root not in sys.path:
        sys.path.insert(0, _root)

from sam3.model.utils.misc import copy_data_to_device
from sam3.model_builder import build_sam3_video_model
from sam3.train.data.coco_json_loaders import SAM3_VEVAL_API_FROM_JSON_NP
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_video_dataset import VideoGroundingDataset
from sam3.train.transforms.basic_for_api import ComposeAPI, NormalizeAPI, RandomResizeAPI, ToTensorAPI
from sam3.train.transforms.segmentation import DecodeRle
from sam3.uap_apply_utils import apply_uap_to_normalized_batch, load_uap_tensor


def get_parser():
    p = argparse.ArgumentParser(
        description="SAM3 推理 + 评估（视频/图像、多数据集、可选 UAP）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 数据
    p.add_argument("--ann_file", type=str, required=True, help="标注 JSON（SA-Co veval 格式，含 video_np_pairs）")
    p.add_argument("--img_folder", type=str, required=True, help="视频帧/图像根目录（与 JSON 中 file_names 相对路径拼接）")
    p.add_argument("--limit_ids", type=int, default=None, help="最多跑多少个视频/样本，默认全部")
    # 模型
    p.add_argument("--checkpoint", type=str, default=None, help="SAM3 权重 .pt，不指定则仅建图不加载")
    p.add_argument("--bpe_path", type=str, default=None, help="BPE 词表路径，不指定则用包内默认")
    p.add_argument("--resolution", type=int, default=1008, help="输入分辨率（正方形）")
    p.add_argument("--norm_mean", type=float, nargs=3, default=[0.5, 0.5, 0.5])
    p.add_argument("--norm_std", type=float, nargs=3, default=[0.5, 0.5, 0.5])
    # UAP
    p.add_argument("--uap_path", type=str, default=None, help="UAP .pth；不指定则不加扰动")
    p.add_argument("--uap_scale", type=float, default=1.0, help="UAP 缩放系数，和 trainer.uap_scale 对齐")
    # 输出
    p.add_argument("--pred_file", type=str, required=True, help="预测结果输出 JSON（YT-VIS 格式）")
    p.add_argument("--eval_res_file", type=str, default=None, help="评估指标输出 JSON；不指定则用 pred_file 同目录下的 _metrics.json")
    p.add_argument("--gt_annot_file", type=str, default=None, help="评估用 GT 标注；不指定则用 --ann_file")
    p.add_argument("--no_eval", action="store_true", help="只推理不跑评估")
    # 运行
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--amp", action="store_true", help="使用 bfloat16 autocast")
    return p


def build_transforms(resolution, norm_mean, norm_std):
    """返回一个 list，满足 dataset 对 _transforms 的迭代约定（for transform in self._transforms）。"""
    compose = ComposeAPI(
        transforms=[
            DecodeRle(),
            RandomResizeAPI(sizes=[resolution], square=True, consistent_transform=True),
            ToTensorAPI(),
            NormalizeAPI(mean=norm_mean, std=norm_std),
        ]
    )
    return [compose]


def build_dataset(ann_file, img_folder, transforms, limit_ids, load_seg=True):
    # Dataset 期望 coco_json_loader 为可调用对象：调用 loader(annFile) 返回 coco-like API
    coco_json_loader = lambda path: SAM3_VEVAL_API_FROM_JSON_NP(path)
    dataset = VideoGroundingDataset(
        img_folder=img_folder,
        ann_file=ann_file,
        transforms=transforms,
        max_ann_per_img=100000,
        multiplier=1,
        training=False,
        load_segmentation=load_seg,
        max_val_queries=100000,
        coco_json_loader=coco_json_loader,
        limit_ids=limit_ids,
    )
    return dataset


def apply_uap_to_batch(batch, uap_tensor, norm_mean, norm_std, device, uap_scale=1.0):
    if uap_tensor is None:
        return
    batch.img_batch = apply_uap_to_normalized_batch(
        img_batch=batch.img_batch,
        uap_tensor=uap_tensor,
        norm_mean=norm_mean,
        norm_std=norm_std,
        uap_scale=uap_scale,
    )


def run_inference(args):
    if not args.checkpoint or not os.path.isfile(args.checkpoint):
        raise FileNotFoundError("For inference --checkpoint is required and must be an existing .pt file.")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    resolution = args.resolution
    norm_mean = args.norm_mean
    norm_std = args.norm_std

    # 模型
    model = build_sam3_video_model(
        checkpoint_path=args.checkpoint,
        bpe_path=args.bpe_path,
        has_presence_token=True,
        geo_encoder_use_img_cross_attn=True,
        apply_temporal_disambiguation=True,
        device=device,
    )
    model.eval()

    # 数据
    transforms = build_transforms(resolution, norm_mean, norm_std)
    dataset = build_dataset(
        args.ann_file,
        args.img_folder,
        transforms,
        args.limit_ids,
    )
    collate_fn = partial(
        collate_fn_api,
        dict_key="ytvis_val",
        with_seg_masks=True,
        repeats=1,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # UAP
    uap_tensor = load_uap_tensor(args.uap_path, device)
    if uap_tensor is not None:
        print(f"[UAP] Loaded from {args.uap_path}, shape {tuple(uap_tensor.shape)}")
    else:
        print("[UAP] No perturbation (uap_path not set or load failed)")

    # 后处理：与 trainer 一致，模型 forward 已返回 {video_id: preds}，PostProcessNullOp 直接透传
    from sam3.eval.postprocessors import PostProcessNullOp
    from sam3.eval.ytvis_eval import YTVISResultsWriter

    postprocessor = PostProcessNullOp()
    writer = YTVISResultsWriter(
        dump_file=args.pred_file,
        postprocessor=postprocessor,
        gather_pred_via_filesys=False,
        write_eval_metrics_file=False,
    )
    dump = []

    with torch.no_grad():
        for batch_dict in tqdm(loader, desc="Inference"):
            key, batch = batch_dict.popitem()
            batch = copy_data_to_device(batch, device, non_blocking=True)
            batch.img_batch = apply_uap_to_normalized_batch(
                img_batch=batch.img_batch,
                uap_tensor=uap_tensor,
                norm_mean=norm_mean,
                norm_std=norm_std,
                uap_scale=args.uap_scale,
            )

            if args.amp and device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    find_stages = model(batch)
            else:
                find_stages = model(batch)

            predictions = postprocessor.process_results(
                find_stages=find_stages,
                find_metadatas=batch.find_metadatas,
            )
            results = writer.prepare(predictions)
            dump.extend(results)

    os.makedirs(os.path.dirname(args.pred_file) or ".", exist_ok=True)
    with g_pathmgr.open(args.pred_file, "w") as f:
        json.dump(dump, f)
    print(f"[Inference] Predictions saved to {args.pred_file} ({len(dump)} masklets)")

    return args.pred_file


def run_evaluation(pred_file, gt_annot_file, eval_res_file):
    if not pred_file or not os.path.isfile(pred_file):
        print("[Eval] No pred_file, skip evaluation.")
        return
    if not gt_annot_file or not os.path.isfile(gt_annot_file):
        print("[Eval] No gt_annot_file, skip evaluation.")
        return

    from sam3.eval.saco_veval_eval import VEvalEvaluator

    os.makedirs(os.path.dirname(eval_res_file) or ".", exist_ok=True)
    evaluator = VEvalEvaluator(gt_annot_file=gt_annot_file, eval_res_file=eval_res_file)
    metrics = evaluator.run_eval(pred_file=pred_file)
    print(f"[Eval] Results saved to {eval_res_file}")
    if metrics.get("dataset_results"):
        for k, v in metrics["dataset_results"].items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    return metrics


def main():
    parser = get_parser()
    args = parser.parse_args()

    pred_file = run_inference(args)

    if args.no_eval:
        return
    gt_file = args.gt_annot_file or args.ann_file
    eval_file = args.eval_res_file
    if eval_file is None:
        base = args.pred_file.rsplit(".", 1)[0]
        eval_file = base + "_metrics.json"
    run_evaluation(pred_file, gt_file, eval_file)


if __name__ == "__main__":
    main()

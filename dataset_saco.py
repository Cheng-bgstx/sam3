import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pycocotools import mask as mask_utils
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from pathlib import Path


class SACoDataset(Dataset):
    """
    SAM3 SA-Co 数据集的平铺式加载器(模仿 Dataset_YOUTUBE 的单帧逻辑)

    特性:
    - 将视频序列拆散为单帧样本
    - 每个样本返回: (image, text_prompt, gt_mask, sample_id)
    - sample_id 格式: "video_name/frame_idx" (例如 "sav_000000/00000")
    """

    def __init__(self, root_dir: str, ann_file: str, target_size=(1008, 1008),
                 limit_videos=-1, limit_frames=-1, transform=None, split_txt: str = None,
                 consecutive_frames: bool = False):
        """
        Args:
            root_dir: 图片根目录 (e.g., path/to/SA-Co-VEval/data)
            ann_file: JSON 标注文件路径
            target_size: 目标分辨率 (height, width)
            limit_videos: 限制视频数量 (-1 表示全部)，当 split_txt 指定时忽略
            limit_frames: 每个视频限制帧数 (-1 表示全部)
            transform: 预处理函数 (通常留空,由 Sam3Forwarder 处理)
            split_txt: 训练/测试划分 txt 路径 (train_videos.txt 或 test_videos.txt)，
                       指定后仅加载该文件中的视频
            consecutive_frames: 若为 True 且 limit_frames>0，则随机选取起始位置取连续帧
        """
        self.root_dir = Path(root_dir)
        self.target_size = target_size
        self.transform = transform
        self.split_txt = split_txt
        self.consecutive_frames = consecutive_frames

        print(f">> Loading SA-Co annotations from {ann_file} ...")
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)

        # 1. 建立索引
        self.video_dict = {v['id']: v for v in self.coco_data['videos']}
        self.video_name_to_id = {v['video_name']: v['id'] for v in self.coco_data['videos']}
        self.ann_dict = defaultdict(list)
        for ann in self.coco_data['annotations']:
            self.ann_dict[ann['video_id']].append(ann)

        # 2. 建立 video_np_pairs 索引 (video_id -> list of noun_phrase)，与推理/评估一致：每视频所有 text 提示
        self.video_to_np = defaultdict(list)
        if 'video_np_pairs' in self.coco_data:
            for pair in self.coco_data['video_np_pairs']:
                vid = pair['video_id']
                np_text = pair['noun_phrase']
                self.video_to_np[vid].append(np_text)
            self.video_to_np = dict(self.video_to_np)

        # 3. 平铺所有帧为独立样本
        self.samples = []
        self.video_frame_count = defaultdict(int)  # 记录每个视频的帧数

        # 3.1 获取 JSON 中所有的 video_id
        all_video_ids = list(self.video_dict.keys())

        # 3.2 过滤：只保留本地磁盘存在的视频
        valid_video_ids = []
        print(">> Checking file existence...")
        all_video_ids.sort()

        for vid in all_video_ids:
            video_info = self.video_dict[vid]
            # 检查该视频的第一帧是否存在，以此判断视频文件夹是否存在
            # 注意：这里假设如果第一帧存在，整个文件夹就是好的。
            # 这种方式比检查每一帧要快得多。
            if len(video_info['file_names']) > 0:
                first_frame_path = self.root_dir / video_info['file_names'][0]
                if first_frame_path.exists():
                    valid_video_ids.append(vid)

        print(f">> Found {len(valid_video_ids)} valid videos on disk out of {len(all_video_ids)} in JSON.")

        # 3.3 应用视频数量限制 (limit_videos)
        # 现在的限制是基于“实际存在的视频”进行的，而不是 JSON 里的视频
        target_video_ids = valid_video_ids
        if split_txt:
            split_path = Path(split_txt)
            if split_path.exists():
                with open(split_path, "r") as f:
                    allowed_names = {line.strip() for line in f if line.strip()}
                target_video_ids = [
                    vid for vid in valid_video_ids
                    if self.video_dict[vid]["video_name"] in allowed_names
                ]
                print(f">> Filtered by split_txt: {len(target_video_ids)} videos (from {len(allowed_names)} in file)")
            else:
                print(f">> [Warning] split_txt not found: {split_txt}, using all valid videos")
        elif limit_videos > 0:
            target_video_ids = valid_video_ids[:limit_videos]

        for video_id in target_video_ids:
            video_info = self.video_dict[video_id]
            video_name = video_info['video_name']
            file_names = video_info['file_names']

            # 限制帧数
            if limit_frames > 0 and len(file_names) > limit_frames:
                if self.consecutive_frames:
                    # 随机选取起始位置，取连续帧（用于 UAP 训练）
                    max_start = len(file_names) - limit_frames
                    start = np.random.randint(0, max(1, max_start + 1))
                    selected_indices = list(range(start, start + limit_frames))
                else:
                    step = max(1, len(file_names) // limit_frames)
                    selected_indices = list(range(0, len(file_names), step))[:limit_frames]
            else:
                selected_indices = list(range(len(file_names)))

            # 获取该视频的所有文本提示（开放词汇：与推理/评估一致）
            text_prompts = self.video_to_np.get(video_id, ["visual"])
            if not text_prompts:
                text_prompts = ["visual"]

            # 获取该视频的所有标注
            annotations = self.ann_dict.get(video_id, [])

            for idx in selected_indices:
                frame_file = file_names[idx]
                sample_id = f"{video_name}/{idx:05d}"

                self.samples.append({
                    'video_id': video_id,
                    'video_name': video_name,
                    'frame_idx': idx,
                    'file_name': frame_file,
                    'text_prompt': text_prompts,
                    'annotations': annotations,
                    'sample_id': sample_id,
                    'height': video_info['height'],
                    'width': video_info['width']
                })

                self.video_frame_count[video_name] += 1

        print(f">> Loaded {len(self.samples)} frame samples from {len(target_video_ids)} videos")
        print(f">> Frame distribution: {dict(self.video_frame_count)}")

    def _load_image(self, file_name):
        """加载图像"""
        path = self.root_dir / file_name
        img = Image.open(path).convert("RGB")
        img = img.resize(self.target_size[::-1], Image.BILINEAR)  # (W, H)
        return np.array(img)

    def _decode_mask(self, segmentation, h, w):
        """解码 RLE 或 Polygon 掩码"""
        if isinstance(segmentation, list):
            # Polygon
            rles = mask_utils.frPyObjects(segmentation, h, w)
            rle = mask_utils.merge(rles)
        elif isinstance(segmentation['counts'], list):
            # Uncompressed RLE
            rle = mask_utils.frPyObjects(segmentation, h, w)
        else:
            # Compressed RLE
            rle = segmentation
        return mask_utils.decode(rle)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 1. 加载图像
        image = self._load_image(sample['file_name'])

        # 2. 加载 GT Mask (融合该帧的所有实例)
        h_orig, w_orig = sample['height'], sample['width']
        h_target, w_target = self.target_size

        mask_accum = np.zeros((h_orig, w_orig), dtype=np.uint8)

        for ann in sample['annotations']:
            # SA-Co 的 segmentations 是列表,每个元素对应一帧
            if 'segmentations' in ann and len(ann['segmentations']) > sample['frame_idx']:
                seg = ann['segmentations'][sample['frame_idx']]
                if seg is not None:  # 可能某些帧没有标注
                    m = self._decode_mask(seg, h_orig, w_orig)
                    mask_accum = np.maximum(mask_accum, m)

        # Resize mask
        mask_pil = Image.fromarray(mask_accum)
        mask_resized = mask_pil.resize(self.target_size[::-1], Image.NEAREST)
        gt_mask = np.array(mask_resized)

        # 3. 文本提示（列表：该视频所有 phrase，与推理/评估一致）
        text_prompt = sample['text_prompt']

        # 4. 样本 ID
        sample_id = sample['sample_id']

        return image, text_prompt, gt_mask, sample_id

    def get_img_id(self, idx):
        """返回样本 ID (用于 get_video_to_indices)"""
        if isinstance(idx, int):
            return self.samples[idx]['sample_id']
        elif isinstance(idx, str):
            return idx
        else:
            raise TypeError(f"Expected idx to be int or str, got {type(idx)}")

    def get_video_name(self, idx):
        """返回视频名称"""
        if isinstance(idx, int):
            return self.samples[idx]['video_name']
        else:
            sample_id = idx
            return sample_id.split('/')[0]

    def get_start_frame_idx(self, video_name):
        """返回视频的起始帧索引 (通常是 0)"""
        for sample in self.samples:
            if sample['video_name'] == video_name:
                return sample['frame_idx']
        return 0


def collate_fn_saco_flat(batch):
    """
    自定义 Collate 函数
    batch: List[(image, text_prompt, gt_mask, sample_id)]
    """
    images, text_prompts, gt_masks, sample_ids = zip(*batch)
    return images, text_prompts, gt_masks, sample_ids
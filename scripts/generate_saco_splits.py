#!/usr/bin/env python3
"""
生成 SA-Co 数据集的训练集/测试集划分

用法:
    python scripts/generate_saco_splits.py

输出:
    data/splits/saco_yt1b/train_videos.txt   # 100 个视频 (用于 UAP 训练)
    data/splits/saco_yt1b/test_videos.txt    # 200-300 个视频 (用于 SAM3 eval 测试)
    data/annotation/saco_veval_yt1b_train.json   # 训练集 JSON (仅 yt1b)
    data/annotation/saco_veval_yt1b_test_split.json  # 测试集 JSON (仅 yt1b)
    data/splits/saco_sav/train_videos.txt
    data/splits/saco_sav/test_videos.txt
    data/splits/saco_sg/train_videos.txt
    data/splits/saco_sg/test_videos.txt

注意: saco_sav 和 saco_sg 若存在对应 annotation 文件，也会生成过滤后的 train/test JSON
"""

import os
import json
import random
from pathlib import Path

# 固定种子，保证可复现
SPLIT_SEED = 42
NUM_TRAIN_VIDEOS = 90
NUM_TEST_VIDEOS = 50

# 路径配置 (支持通过环境变量 DATA_ROOT 覆盖)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(PROJECT_ROOT / "data")))
MEDIA_ROOT = DATA_ROOT / "media"
ANNOTATION_ROOT = DATA_ROOT / "annotation"
SPLITS_ROOT = DATA_ROOT / "splits"

# 数据集配置: (media子路径, annotation文件, 输出 JSON 前缀)
# annotation 用于获取视频列表；若有则同时生成过滤后的 train/test JSON
DATASET_CONFIGS = {
    "saco_yt1b": {
        "media_subdir": "saco_yt1b/JPEGImages_6fps",
        "ann_file": "saco_veval_yt1b_test.json",
        "ann_train_out": "saco_veval_yt1b_train.json",
        "ann_test_out": "saco_veval_yt1b_test_split.json",
        "source": "annotation",
    },
    "saco_sav": {
        "media_subdir": "saco_sav/JPEGImages_24fps",
        "ann_file": "saco_veval_sav_test.json",  # 若存在则生成 JSON
        "ann_train_out": "saco_veval_sav_train.json",
        "ann_test_out": "saco_veval_sav_test_split.json",
        "source": "annotation",
    },
    "saco_sg": {
        "media_subdir": "saco_sg/JPEGImages_6fps",
        "ann_file": "saco_veval_smartglasses_test.json",
        "ann_train_out": "saco_veval_smartglasses_train.json",
        "ann_test_out": "saco_veval_smartglasses_test_split.json",
        "source": "annotation",
    },
}

def count_total_frames(json_data: dict) -> int:
    """统计 JSON 数据中所有视频的总帧数"""
    total_count = 0
    for v in json_data.get("videos", []):
        # 优先读取 file_names 列表长度
        if "file_names" in v and isinstance(v["file_names"], list):
            total_count += len(v["file_names"])
        # 其次读取 length 字段
        elif "length" in v:
            total_count += v["length"]
    return total_count


def get_videos_from_annotation(ann_path: Path, media_root: Path) -> list[str]:
    """从 annotation JSON 获取视频列表，并验证本地存在且有查询标注"""
    with open(ann_path, "r") as f:
        data = json.load(f)

    # 1. 统计所有包含有效查询 (video_np_pairs) 的 video_id
    valid_video_ids = set()
    if "video_np_pairs" in data:
        for pair in data["video_np_pairs"]:
            valid_video_ids.add(pair["video_id"])

    # 如果没有 video_np_pairs (例如纯检测数据集)，则尝试检查 annotations
    if not valid_video_ids and "annotations" in data:
        for ann in data["annotations"]:
            valid_video_ids.add(ann["video_id"])

    print(f"  [Info] JSON中包含有效查询/标注的视频ID数量: {len(valid_video_ids)}")

    video_names = []
    skipped_no_query = 0
    skipped_no_file = 0

    for v in data["videos"]:
        # 2. 检查该视频是否有查询
        if v["id"] not in valid_video_ids:
            skipped_no_query += 1
            continue

        video_name = v["video_name"]
        if len(v.get("file_names", [])) > 0:
            first_frame = v["file_names"][0]
            frame_path = media_root / first_frame
            # 3. 检查文件是否存在
            if frame_path.exists():
                video_names.append(video_name)
            else:
                skipped_no_file += 1

    print(f"  [Info] 跳过无查询的视频: {skipped_no_query}, 跳过文件丢失的视频: {skipped_no_file}")
    return video_names


def get_videos_from_folder(media_dir: Path, require_images: bool = True) -> list[str]:
    """
    从文件夹结构获取视频列表（每个子文件夹为一个视频）
    require_images: 若为 True，仅保留至少含一张 jpg/png 的文件夹
    """
    if not media_dir.exists():
        return []
    candidates = [
        d for d in os.listdir(media_dir)
        if os.path.isdir(media_dir / d) and not d.startswith(".")
    ]
    if not require_images:
        return sorted(candidates)
    video_names = []
    for d in sorted(candidates):
        vdir = media_dir / d
        files = [f for f in os.listdir(vdir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if files:
            video_names.append(d)
    return video_names


def create_minimal_annotation_from_folder(
    media_dir: Path,
    video_names: list[str],
    output_path: Path,
) -> bool:
    """
    当不存在 annotation 时，从文件夹结构创建最小化 JSON，
    便于 SACoDataset 加载 (用于 sav/sg 等无标注数据集)
    """
    from PIL import Image

    videos = []
    annotations = []
    video_np_pairs = []
    categories = [{"id": 1, "name": "object"}]

    for vid, video_name in enumerate(video_names):
        video_dir = media_dir / video_name
        if not video_dir.is_dir():
            continue

        files = sorted([
            f for f in os.listdir(video_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        if not files:
            continue

        file_names = [f"{video_name}/{f}" for f in files]

        # 读取第一帧获取尺寸
        try:
            first_img = Image.open(video_dir / files[0])
            w, h = first_img.size
            first_img.close()
        except Exception:
            h, w = 480, 854

        videos.append({
            "id": vid,
            "video_name": video_name,
            "file_names": file_names,
            "height": h,
            "width": w,
            "length": len(file_names),
        })
        video_np_pairs.append({"video_id": vid, "noun_phrase": "visual", "category_id": 1})
        # 空标注，SACoDataset 会生成全零 mask
        annotations.append({
            "video_id": vid,
            "segmentations": [None] * len(file_names),
        })

    if not videos:
        return False

    data = {
        "info": {"version": "v1", "description": "Minimal annotation from folder"},
        "videos": videos,
        "annotations": annotations,
        "categories": categories,
        "video_np_pairs": video_np_pairs,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return True


def filter_annotation_by_videos(
    ann_path: Path,
    video_names: list[str],
    media_dir: Path,
) -> dict:
    """
    根据视频名列表过滤 annotation JSON，只保留存在的视频
    返回过滤后的完整 JSON 结构
    """
    video_name_set = set(video_names)
    with open(ann_path, "r") as f:
        data = json.load(f)

    kept_videos = []
    kept_video_ids = set()
    for v in data["videos"]:
        video_name = v["video_name"]
        if video_name not in video_name_set:
            continue
        if len(v.get("file_names", [])) > 0:
            first_frame = v["file_names"][0]
            frame_path = media_dir / first_frame
            if frame_path.exists():
                kept_videos.append(v)
                kept_video_ids.add(v["id"])

    # 过滤 annotations (video_id 在保留列表中)
    kept_anns = [a for a in data["annotations"] if a.get("video_id") in kept_video_ids]

    # 过滤 video_np_pairs
    kept_pairs = [p for p in data.get("video_np_pairs", []) if p.get("video_id") in kept_video_ids]

    # 过滤 categories (仅保留被引用的)
    used_cat_ids = set()
    for p in kept_pairs:
        used_cat_ids.add(p.get("category_id"))
    kept_cats = [c for c in data["categories"] if c["id"] in used_cat_ids]

    return {
        "info": data.get("info", {}),
        "videos": kept_videos,
        "annotations": kept_anns,
        "categories": kept_cats,
        "video_np_pairs": kept_pairs,
    }


def generate_split(dataset_name: str, video_names: list[str]) -> tuple[list[str], list[str]]:
    """划分训练集和测试集"""
    required_total = NUM_TRAIN_VIDEOS + NUM_TEST_VIDEOS
    if len(video_names) < required_total:
        print(f"  [Warning] {dataset_name}: 仅 {len(video_names)} 个视频，不足 {required_total}")
        # 优先满足训练集 100，剩下的给测试集
        n_train = min(NUM_TRAIN_VIDEOS, len(video_names))
        n_test = len(video_names) - n_train
    else:
        n_train = NUM_TRAIN_VIDEOS
        n_test = NUM_TEST_VIDEOS

    rng = random.Random(SPLIT_SEED)
    shuffled = video_names.copy()
    rng.shuffle(shuffled)

    train_names = shuffled[:n_train]
    test_names = shuffled[n_train:n_train + n_test]
    return train_names, test_names


def main():
    print("=" * 60)
    print("生成 SA-Co 训练/测试集划分")
    print("=" * 60)

    random.seed(SPLIT_SEED)

    for dataset_name, config in DATASET_CONFIGS.items():
        print(f"\n>> 处理 {dataset_name} ...")

        media_dir = MEDIA_ROOT / config["media_subdir"]
        ann_path = ANNOTATION_ROOT / config.get("ann_file", "")

        # 获取视频列表
        if config["source"] == "annotation" and ann_path.exists():
            video_names = get_videos_from_annotation(ann_path, media_dir)
            source_ann_path = ann_path
        else:
            video_names = get_videos_from_folder(media_dir)
            # sav/sg 无 annotation 时，从文件夹创建最小化 JSON
            if not ann_path.exists() and video_names:
                minimal_ann = ANNOTATION_ROOT / f"{dataset_name}_minimal.json"
                if create_minimal_annotation_from_folder(media_dir, video_names, minimal_ann):
                    source_ann_path = minimal_ann
                    print(f"  已从文件夹创建最小化 annotation: {minimal_ann}")
                else:
                    source_ann_path = None
            else:
                source_ann_path = ann_path if ann_path.exists() else None

        if not video_names:
            print(f"  [Skip] 未找到有效视频")
            continue

        print(f"  有效视频数: {len(video_names)}")
        train_names, test_names = generate_split(dataset_name, video_names)
        print(f"  训练集: {len(train_names)} 个视频")
        print(f"  测试集: {len(test_names)} 个视频")

        # 1. 保存 txt 划分
        out_dir = SPLITS_ROOT / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)

        train_txt = out_dir / "train_videos.txt"
        test_txt = out_dir / "test_videos.txt"

        with open(train_txt, "w") as f:
            f.write("\n".join(train_names) + "\n")
        with open(test_txt, "w") as f:
            f.write("\n".join(test_names) + "\n")

        print(f"  已保存: {train_txt}")
        print(f"  已保存: {test_txt}")

        # 2. 生成过滤后的 train/test JSON (用于 UAP 训练和 SAM3 测试)
        if source_ann_path and config.get("ann_train_out") and config.get("ann_test_out"):
            train_json = filter_annotation_by_videos(source_ann_path, train_names, media_dir)
            test_json = filter_annotation_by_videos(source_ann_path, test_names, media_dir)

            train_out = ANNOTATION_ROOT / config["ann_train_out"]
            test_out = ANNOTATION_ROOT / config["ann_test_out"]

            with open(train_out, "w") as f:
                json.dump(train_json, f, indent=2, ensure_ascii=False)
            with open(test_out, "w") as f:
                json.dump(test_json, f, indent=2, ensure_ascii=False)

            train_frames = count_total_frames(train_json)
            test_frames = count_total_frames(test_json)
            print(f"  已保存: {train_out} ({len(train_json['videos'])} videos) (总帧数: {train_frames})")
            print(f"  已保存: {test_out} ({len(test_json['videos'])} videos) (总帧数: {test_frames})")

    print("\n" + "=" * 60)
    print("完成")
    print("=" * 60)
    print("\n使用说明:")
    print("  1. UAP 训练: uap_attack_sam3_video.py 会从 train_videos.txt 读取训练视频")
    print("     (通过 saco_veval_*_train.json，或 ann_file + train_videos.txt)")
    print("  2. SAM3 测试: 使用 test_split 配置运行 eval，例如:")
    print("     python -m sam3.train.train model=... config=saco_veval_yt1b_test_split")
    print("     或修改 yaml 中 paths.ytvis_json 指向 *_test_split.json")
    print("=" * 60)


if __name__ == "__main__":
    main()

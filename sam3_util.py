import os
import random
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
from typing import Union, Dict, List

from attack_setting_sam3 import Sam3VideoForwarder, Sam3Forwarder, seed_everything
from sam3.model_builder import build_sam3_video_predictor

Data = Union[np.ndarray, torch.Tensor]


def _get_data_root() -> Path:
    """获取 data 目录路径（兼容不同运行环境）"""
    for candidate in [
        Path(__file__).resolve().parent / "data",
        Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parent / "data")),
    ]:
        if candidate.exists():
            return Path(candidate)
    return Path(__file__).resolve().parent / "data"


_DATA_ROOT = _get_data_root()

# 数据集配置: (root_dir, ann_train, ann_test, splits_dir)
# 训练时使用 ann_train + train_videos.txt (或直接 ann_train 已过滤)
# 测试时使用 ann_test (saco_veval_*_test_split.json)
SACO_DATASET_CONFIGS = {
    "saco_yt1b": {
        "root_dir": str(_DATA_ROOT / "media/saco_yt1b/JPEGImages_6fps"),
        "ann_train": str(_DATA_ROOT / "annotation/saco_veval_yt1b_train.json"),
        "ann_test": str(_DATA_ROOT / "annotation/saco_veval_yt1b_test_split.json"),
        "splits_dir": str(_DATA_ROOT / "splits/saco_yt1b"),
    },
    "saco_sav": {
        "root_dir": str(_DATA_ROOT / "media/saco_sav/JPEGImages_24fps"),
        "ann_train": str(_DATA_ROOT / "annotation/saco_veval_sav_train.json"),
        "ann_test": str(_DATA_ROOT / "annotation/saco_veval_sav_test_split.json"),
        "splits_dir": str(_DATA_ROOT / "splits/saco_sav"),
    },
    "saco_sg": {
        "root_dir": str(_DATA_ROOT / "media/saco_sg/JPEGImages_6fps"),
        "ann_train": str(_DATA_ROOT / "annotation/saco_veval_smartglasses_train.json"),
        "ann_test": str(_DATA_ROOT / "annotation/saco_veval_smartglasses_test_split.json"),
        "splits_dir": str(_DATA_ROOT / "splits/saco_sg"),
    },
}

# 向后兼容
SACO_ROOT = SACO_DATASET_CONFIGS["saco_sav"]["root_dir"]
SACO_JSON = SACO_DATASET_CONFIGS["saco_sav"]["ann_train"]


def load_sam3_model(checkpoint_path: str, device: str = "cuda", use_adapter: bool = True):
    """
    加载SAM3模型并封装到Sam3VideoForwarder

    Args:
        checkpoint_path: 模型checkpoint路径
        device: 设备
        use_adapter: 是否使用ForwarderAdapter（推荐用于攻击）
    """
    print(f">> Loading SAM3 model from {checkpoint_path}...")

    if use_adapter:
        # 使用Adapter封装，获得攻击所需的所有接口
        from sam3_forwarder_adapter import build_sam3_video_model_with_forwarder

        sam_fwder = build_sam3_video_model_with_forwarder(
            checkpoint_path=checkpoint_path,
            device=device
        )
    else:
        # 原始方式
        from sam3.model_builder import build_sam3_video_predictor
        predictor = build_sam3_video_predictor(checkpoint_path=checkpoint_path)
        sam3_model = predictor.model if hasattr(predictor, "model") else predictor
        sam3_model.to(device)
        sam3_model.eval()
        sam_fwder = Sam3VideoForwarder(sam3_model, device=device).to(device)

    sam_fwder.eval()
    print(f">> Model loaded successfully on {device}")
    return sam_fwder


def choose_dataset_sam3(args):
    """
    根据 args 选择数据集，按 train_videos.txt 划分加载训练集

    支持: SA-Co，通过 --saco_variant 指定 yt1b / sav / sg
    训练时加载 train_videos.txt 中的视频（100个），测试时使用 test JSON
    """
    from dataset_saco import SACoDataset

    # 解析数据集变体
    variant = getattr(args, "saco_variant", "saco_yt1b")
    if variant not in SACO_DATASET_CONFIGS:
        variant = "saco_yt1b"

    cfg = SACO_DATASET_CONFIGS[variant]
    root_dir = getattr(args, "root_dir", None) or cfg["root_dir"]
    splits_dir = Path(cfg["splits_dir"])
    train_split_txt = splits_dir / "train_videos.txt"

    # 优先使用 train JSON；若不存在则尝试 ann_test 或 full ann + split_txt
    ann_train = Path(cfg["ann_train"])
    ann_test = Path(cfg["ann_test"])
    if ann_train.exists():
        ann_file = getattr(args, "ann_file", None) or str(ann_train)
        split_txt = None  # train JSON 已只含训练视频
    elif ann_test.exists():
        ann_file = getattr(args, "ann_file", None) or str(ann_test)
        split_txt = str(train_split_txt) if train_split_txt.exists() else None
    else:
        ann_file = getattr(args, "ann_file", None) or str(ann_train)
        split_txt = str(train_split_txt) if train_split_txt.exists() else None
    limit_videos = getattr(args, "limit_img", -1)
    limit_frames = getattr(args, "limit_frames", -1)

    # 当使用 split_txt 时，不使用 limit_videos（按划分文件决定）
    if split_txt:
        limit_videos = -1

    # 训练时使用连续帧（随机起始），便于 UAP 时序损失
    consecutive_frames = getattr(args, "consecutive_frames", True)

    dataset = SACoDataset(
        root_dir=root_dir,
        ann_file=ann_file,
        target_size=(1008, 1008),
        limit_videos=limit_videos,
        limit_frames=limit_frames,
        split_txt=split_txt,
        consecutive_frames=consecutive_frames,
    )

    print(f">> Loaded {len(dataset)} samples from {variant} (train split)")
    return dataset


def choose_test_dataset_sam3(variant: str = "saco_yt1b", limit_frames: int = -1):
    """
    加载测试集 (用于 SAM3 eval 或本地验证)
    从 ann_test (saco_veval_*_test_split.json) 加载
    """
    from dataset_saco import SACoDataset

    if variant not in SACO_DATASET_CONFIGS:
        variant = "saco_yt1b"
    cfg = SACO_DATASET_CONFIGS[variant]
    ann_test = Path(cfg["ann_test"])
    if not ann_test.exists():
        raise FileNotFoundError(f"测试集 JSON 不存在: {ann_test}，请先运行 scripts/generate_saco_splits.py")

    dataset = SACoDataset(
        root_dir=cfg["root_dir"],
        ann_file=str(ann_test),
        target_size=(1008, 1008),
        limit_videos=-1,
        limit_frames=limit_frames,
        split_txt=None,
        consecutive_frames=False,
    )
    print(f">> Loaded {len(dataset)} samples from {variant} (test split)")
    return dataset


def get_video_to_indices_sam3(dataset):
    """
    构建视频到样本索引的映射

    返回: {video_name: [idx1, idx2, ...]}
    """
    video_to_indices = defaultdict(list)

    for idx in range(len(dataset)):
        video_name = dataset.get_video_name(idx)
        video_to_indices[video_name].append(idx)

    # 确保每个视频的帧按索引顺序排列
    for video_name in video_to_indices:
        video_to_indices[video_name].sort()

    print(f">> Found {len(video_to_indices)} videos")
    return dict(video_to_indices)


def get_frame_index_sam3(sample_id: str) -> int:
    """
    从sample_id中提取帧索引

    例如: "sav_000000/00042" -> 42
    """
    parts = sample_id.split('/')
    if len(parts) == 2:
        try:
            return int(parts[1])
        except ValueError:
            return 0
    return 0


def infonce_loss(
        adv_feature: torch.Tensor,
        original_feature: torch.Tensor,
        target_feature: torch.Tensor,
        temperature: float = 0.1
) -> torch.Tensor:
    """
    与 SAM2 sam2_util.infonce_loss 完全一致
    对比损失: 让对抗特征接近目标特征,远离原始特征
    """
    adv_feature_flat = adv_feature.reshape(1, -1)
    original_feature_flat = original_feature.reshape(1, -1)
    target_feature_flat = target_feature.reshape(1, -1)
    similarity_adv_original = torch.matmul(adv_feature_flat, original_feature_flat.T) / temperature
    similarity_adv_target = torch.matmul(adv_feature_flat, target_feature_flat.T) / temperature
    loss = -similarity_adv_target + similarity_adv_original
    return loss.squeeze()


def get_fused_prototype(
        img_list: List[torch.Tensor],
        sam_fwder,
        device,
        use_sam_route: bool = False
) -> torch.Tensor:
    """
    提取一组图像的融合特征原型

    Args:
        img_list: 图像列表 (已经是tensor, [0,1]范围)
        sam_fwder: SAM3前向传播器
        device: 设备
        use_sam_route: True=使用SAM Route (get_image_feature_sam_route, 与SAM2一致);
                      False=使用Detector encoder特征
    """
    fused = None
    count = 0

    for i, img in enumerate(img_list):
        if use_sam_route and hasattr(sam_fwder, 'get_image_feature_sam_route'):
            feat = sam_fwder.get_image_feature_sam_route(img, frame_idx=i)
        else:
            if img.min() >= 0 and img.max() <= 1:
                img = sam_fwder.norm_image(img)
            feat = sam_fwder.get_image_feature(img, frame_idx=i)

        if feat is None:
            print(f"[Warning] Failed to extract feature from frame {i}")
            continue
        if fused is None:
            fused = feat.float().clone()
            count = 1
        else:
            # 递推均值，不暂存多张特征，省显存；数学等价于 stack 后 mean
            fused.mul_(count).add_(feat.float()).div_(count + 1)
            count += 1

    if fused is None:
        print("[Error] No features extracted!")
        return None
    if fused.dim() == 3:
        fused = fused.unsqueeze(0)
    return fused


def run_video_inference_via_propagate(
        sam_fwder: Sam3VideoForwarder,
        img_list: List[np.ndarray],
        mask_gt: np.ndarray,
        text_prompt: str = "visual",
        use_gt_init: bool = True
) -> Dict[int, List[torch.Tensor]]:
    """
    通过 pipeline.propagate 运行视频推理（用于无 track_step 的 Sam3VideoForwarder）。
    Returns: {obj_id: [mask_frame0, mask_frame1, ...]}
    """
    if not img_list:
        return {}
    # 构建 [T,3,H,W] 已归一化的帧
    frames_tensors = [sam_fwder.transform_image(im) for im in img_list]
    frames = torch.cat(frames_tensors, dim=0)  # [T,3,H,W]
    h, w = img_list[0].shape[:2]
    sam_fwder.init_video_state(num_frames=len(img_list), frames=frames, orig_height=h, orig_width=w)

    result_masks = defaultdict(list)

    if use_gt_init and len(mask_gt) > 0:
        if isinstance(mask_gt, torch.Tensor):
            mask_gt_np = mask_gt.cpu().numpy().astype(np.uint8)
        else:
            mask_gt_np = np.asarray(mask_gt).astype(np.uint8)
        if mask_gt_np.ndim == 3:
            mask_gt_np = mask_gt_np[0:1]
        if mask_gt_np.size > 0:
            y_indices, x_indices = np.where(mask_gt_np[0] > 0)
            if len(y_indices) > 0:
                center_y = int(np.mean(y_indices))
                center_x = int(np.mean(x_indices))
                point = np.array([[center_x, center_y]], dtype=np.float32)
                label = np.array([1], dtype=np.int32)
                try:
                    sam_fwder.add_new_points(frame_idx=0, points=point, labels=label, obj_id=1)
                except Exception as e:
                    print(f"Warning: add_new_points failed: {e}")

    for frame_idx, obj_ids_list, _low_res, video_res_masks, _scores, _raw in sam_fwder.propagate_in_video_grad(
            inference_state=sam_fwder.inference_state
    ):
        if video_res_masks is None or len(obj_ids_list) == 0:
            continue
        for i, oid in enumerate(obj_ids_list):
            m = video_res_masks[i]
            if m.dim() == 4:
                m = m.squeeze(0)
            result_masks[oid].append(m)

    return dict(result_masks)


def run_video_inference(
        sam_fwder: Sam3VideoForwarder,
        img_list: List[np.ndarray],
        mask_gt: np.ndarray,
        text_prompt: str = "visual",
        use_gt_init: bool = True
) -> Dict[int, List[torch.Tensor]]:
    """
    运行视频推理

    Args:
        sam_fwder: SAM3前向传播器
        img_list: 图像列表
        mask_gt: Ground Truth Mask
        text_prompt: 文本提示
        use_gt_init: 是否使用GT中心点初始化

    Returns:
        {obj_id: [mask_frame0, mask_frame1, ...]}
    """
    if hasattr(sam_fwder, "track_step") and callable(getattr(sam_fwder, "track_step", None)):
        video_len = len(img_list)
        sam_fwder.init_video_state(num_frames=video_len)
        result_masks = defaultdict(list)
        if use_gt_init and len(mask_gt) > 0:
            if isinstance(mask_gt, torch.Tensor):
                mask_gt_np = mask_gt.cpu().numpy().astype(np.uint8)
            else:
                mask_gt_np = mask_gt.astype(np.uint8)
            y_indices, x_indices = np.where(mask_gt_np[0] > 0)
            if len(y_indices) > 0:
                center_y, center_x = int(np.mean(y_indices)), int(np.mean(x_indices))
                point = np.array([[center_x, center_y]], dtype=np.float32)
                label = np.array([1], dtype=np.int32)
                try:
                    sam_fwder.add_new_points(frame_idx=0, points=point, labels=label, obj_id=1)
                except Exception as e:
                    print(f"Warning: Failed to initialize: {e}")
        for frame_idx, img in enumerate(img_list):
            img_tensor = sam_fwder.transform_image(img)
            out_masks = sam_fwder.track_step(frame_idx, img_tensor, text_prompt=text_prompt)
            for obj_id, mask_tensor in out_masks.items():
                m = mask_tensor.squeeze(0) if mask_tensor.dim() == 3 else mask_tensor
                result_masks[obj_id].append(m)
        return dict(result_masks)
    return run_video_inference_via_propagate(
        sam_fwder, img_list, mask_gt, text_prompt=text_prompt, use_gt_init=use_gt_init
    )


def calculate_video_iou(
        pred_masks: Dict[int, List[torch.Tensor]],
        gt_masks: np.ndarray
) -> float:
    """
    计算视频IoU

    Args:
        pred_masks: {obj_id: [mask_frame0, mask_frame1, ...]}
        gt_masks: Ground truth masks (T, H, W)

    Returns:
        平均IoU
    """
    if isinstance(gt_masks, torch.Tensor):
        gt_masks = gt_masks.cpu().numpy().astype(np.uint8)
    else:
        gt_masks = gt_masks.astype(np.uint8)

    iou_sum = 0
    count = 0

    # 假设主对象ID=1
    if 1 not in pred_masks:
        return 0.0

    pred_mask_list = pred_masks[1]

    for frame_idx, pred_mask in enumerate(pred_mask_list):
        if frame_idx >= len(gt_masks):
            break

        gt = gt_masks[frame_idx]

        # Resize预测mask到GT尺寸
        h_gt, w_gt = gt.shape[:2]
        if pred_mask.shape[-2:] != (h_gt, w_gt):
            pred_mask = F.interpolate(
                pred_mask.unsqueeze(0).unsqueeze(0).float(),
                size=(h_gt, w_gt),
                mode='bilinear',
                align_corners=False
            ).squeeze()

        pred_np = (pred_mask > 0.0).cpu().numpy().astype(np.uint8)

        # 计算IoU
        intersection = np.logical_and(pred_np, gt).sum()
        union = np.logical_or(pred_np, gt).sum()
        if union > 0:
            iou = intersection / union
            iou_sum += iou
            count += 1

    return iou_sum / count if count > 0 else 0.0


def save_prediction(
        sam_fwder: Sam3VideoForwarder,
        img_list: List[np.ndarray],
        adv_img_list: List[np.ndarray],
        mask_gt: np.ndarray,
        save_prefix: str,
        output_dir: str,
        text_prompt: str = "visual"
) -> tuple:
    """
    运行推理并保存可视化结果

    对比Clean vs Adversarial的IoU

    Args:
        sam_fwder: SAM3前向传播器
        img_list: 原始图像列表
        adv_img_list: 对抗图像列表
        mask_gt: Ground Truth Mask
        save_prefix: 保存文件前缀
        output_dir: 输出目录
        text_prompt: 文本提示

    Returns:
        (avg_iou_adv, adv_count, avg_iou_clean)
    """
    os.makedirs(output_dir, exist_ok=True)

    # 运行Clean推理
    print(">> Running evaluation on Clean Images...")
    clean_masks = run_video_inference(sam_fwder, img_list, mask_gt, text_prompt)
    clean_iou = calculate_video_iou(clean_masks, mask_gt)

    # 运行Adversarial推理
    print(">> Running evaluation on Adversarial Images...")
    adv_masks = run_video_inference(sam_fwder, adv_img_list, mask_gt, text_prompt)
    adv_iou = calculate_video_iou(adv_masks, mask_gt)

    print(f"Clean Avg IoU: {clean_iou:.4f}")
    print(f"Adv Avg IoU: {adv_iou:.4f}")

    # 可视化保存 (每10帧或首尾帧)
    if isinstance(mask_gt, torch.Tensor):
        mask_gt_np = mask_gt.cpu().numpy().astype(np.uint8)
    else:
        mask_gt_np = mask_gt.astype(np.uint8)

    for frame_idx in range(len(img_list)):
        if frame_idx % 10 == 0 or frame_idx == len(img_list) - 1:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Clean
            axes[0, 0].imshow(img_list[frame_idx])
            axes[0, 0].set_title(f"Clean Frame {frame_idx}")
            axes[0, 0].axis('off')

            axes[0, 1].imshow(img_list[frame_idx])
            if 1 in clean_masks and frame_idx < len(clean_masks[1]):
                mask = clean_masks[1][frame_idx]
                h, w = mask_gt_np[0].shape[:2]
                if mask.shape[-2:] != (h, w):
                    mask = F.interpolate(
                        mask.unsqueeze(0).unsqueeze(0).float(),
                        size=(h, w),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()
                show_mask((mask > 0).cpu().numpy(), axes[0, 1])
            axes[0, 1].set_title(f"Clean Pred")
            axes[0, 1].axis('off')

            # Adversarial
            axes[1, 0].imshow(adv_img_list[frame_idx])
            axes[1, 0].set_title(f"Adv Frame {frame_idx}")
            axes[1, 0].axis('off')

            axes[1, 1].imshow(adv_img_list[frame_idx])
            if 1 in adv_masks and frame_idx < len(adv_masks[1]):
                mask = adv_masks[1][frame_idx]
                h, w = mask_gt_np[0].shape[:2]
                if mask.shape[-2:] != (h, w):
                    mask = F.interpolate(
                        mask.unsqueeze(0).unsqueeze(0).float(),
                        size=(h, w),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()
                show_mask((mask > 0).cpu().numpy(), axes[1, 1])
            axes[1, 1].set_title(f"Adv Pred")
            axes[1, 1].axis('off')

            save_path = os.path.join(
                output_dir,
                f"{save_prefix}_{frame_idx:04d}.jpg"
            )
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', dpi=100)
            plt.close(fig)

    return adv_iou, len(img_list), clean_iou


def show_mask(mask: np.ndarray, ax, random_color: bool = False):
    """可视化辅助函数"""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def collate_fn(batch):
    """DataLoader的collate函数"""
    buffer_list, P_list, sample_id, gt, point = zip(*batch)
    return buffer_list, P_list, sample_id, gt, point
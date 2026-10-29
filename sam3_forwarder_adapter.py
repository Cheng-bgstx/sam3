"""
SAM3 Forwarder 适配器
将 attack_setting_sam3 中的 Forwarder 类适配到原有的 SAM3 训练/推理接口
"""

import os
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
from attack_setting_sam3 import Sam3Forwarder, Sam3VideoForwarder
from sam3.model.data_misc import BatchedDatapoint, FindStage
from sam3.model.model_misc import SAM3Output
import torch.nn.functional as F

class Sam3ForwarderAdapter(nn.Module):
    """
    适配器：让 Sam3Forwarder 兼容原有 SAM3Image 的接口

    使用方式：
        # 替换原有的 build_sam3_image_model
        from sam3_forwarder_adapter import Sam3ForwarderAdapter
        model = Sam3ForwarderAdapter(checkpoint_path, device)
    """

    def __init__(
            self,
            checkpoint_path: str = None,
            device: str = "cuda",
            base_model=None,  # 可选：直接传入已有的 SAM3 模型
            uap_path: str = None,
            **kwargs
    ):
        super().__init__()

        if base_model is not None:
            # 使用已有模型
            sam3_base = base_model
        else:
            # 从 checkpoint 加载
            from sam3.model_builder import build_sam3_image_model
            sam3_base = build_sam3_image_model(
                checkpoint_path=checkpoint_path,
                device=device,
                **kwargs
            )

        # 使用 Forwarder 封装
        self.forwarder = Sam3Forwarder(sam3_base, device=device)
        self.device = device

        # 保留原有模型的关键属性
        self.detector = self.forwarder.detector
        self.model = self.forwarder.model
        self.uap = None
        if uap_path and os.path.exists(uap_path):
            print(f"\n{'=' * 20} UAP ATTACK ENABLED {'=' * 20}")
            print(f">> Loading UAP from: {uap_path}")
            # 加载 UAP 并确保维度正确 [1, 3, H, W]
            uap_tensor = torch.load(uap_path, map_location=device)
            if uap_tensor.dim() == 3:
                uap_tensor = uap_tensor.unsqueeze(0)
            self.uap = uap_tensor.to(device)
            print(f">> UAP Loaded. Shape: {self.uap.shape}, Range: [{self.uap.min():.4f}, {self.uap.max():.4f}]")
        else:
            print(">> Standard inference mode (No UAP loaded).")

    def _inject_uap(self, images):
        """
        [修正版] 注入 UAP
        支持 Tensor 输入 (Batch) 或 List 输入，并保持输出类型一致。
        """
        if self.uap is None:
            return images

        # === 情况 1: 输入是 Tensor (通常是 [B, 3, H, W]) ===
        # 这是导致报错的主要原因，必须直接处理 Tensor 而不是转成 List
        if isinstance(images, torch.Tensor):
            # 1. 反归一化 (forwarder.denorm_image 支持 Batch 广播)
            img_01 = self.forwarder.denorm_image(images)

            # 2. 对齐 UAP 尺寸
            uap_input = self.uap
            if img_01.shape[-2:] != self.uap.shape[-2:]:
                uap_input = F.interpolate(
                    self.uap, size=img_01.shape[-2:], mode='bilinear', align_corners=False
                )

            # 3. 注入扰动 + 截断 (利用广播机制 [B,3,H,W] + [1,3,H,W])
            adv_img = torch.clamp(img_01 + uap_input, 0.0, 1.0)

            # 4. 重新归一化并返回 Tensor
            return self.forwarder.norm_image(adv_img)

        # === 情况 2: 输入是 List (可能是视频帧列表) ===
        elif isinstance(images, list):
            perturbed_images = []
            for img in images:
                # 处理单张图片 (Tensor [3, H, W] 或 [1, 3, H, W])
                if img.dim() == 3:
                    img = img.unsqueeze(0)

                img_01 = self.forwarder.denorm_image(img)

                uap_input = self.uap
                if img_01.shape[-2:] != self.uap.shape[-2:]:
                    uap_input = F.interpolate(
                        self.uap, size=img_01.shape[-2:], mode='bilinear', align_corners=False
                    )

                adv_img = torch.clamp(img_01 + uap_input, 0.0, 1.0)
                adv_img_norm = self.forwarder.norm_image(adv_img).squeeze(0)
                perturbed_images.append(adv_img_norm)

            return perturbed_images

        # 其他情况直接返回
        return images

    def forward(self, batch: BatchedDatapoint) -> List[SAM3Output]:
        """
        保持与原有 SAM3 模型完全一致的 forward 接口

        Args:
            batch: BatchedDatapoint 包含 img_batch, find_inputs, find_targets 等

        Returns:
            List[SAM3Output]: 每个样本的检测结果
        """
        # 直接调用原有模型的 forward
        # 因为 Forwarder 内部保留了完整的 model 实例
        if self.uap is not None:
            # img_batch 是 list[Tensor]
            batch.img_batch = self._inject_uap(batch.img_batch)
        return self.forwarder.model(batch)

    def forward_grounding(self, backbone_out, find_input, find_target=None, geometric_prompt=None):
        """兼容原有的 forward_grounding 接口"""
        return self.detector.forward_grounding(
            backbone_out, find_input, find_target, geometric_prompt
        )

    def get_image_feature(self, image: torch.Tensor, frame_idx: int = 0):
        """特征提取接口（用于攻击）"""
        return self.forwarder.get_image_feature(image, frame_idx)

    def get_detector_features(self, image: torch.Tensor, layer_name: str = "encoder"):
        """多层特征提取接口（用于攻击）"""
        return self.forwarder.get_detector_features(image, layer_name)

    def state_dict(self, *args, **kwargs):
        """保存模型权重"""
        return self.forwarder.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        """加载模型权重"""
        return self.forwarder.model.load_state_dict(state_dict, *args, **kwargs)

    def train(self, mode: bool = True):
        """设置训练模式"""
        super().train(mode)
        self.forwarder.model.train(mode)
        return self

    def eval(self):
        """设置评估模式"""
        super().eval()
        self.forwarder.model.eval()
        return self

    def to(self, *args, **kwargs):
        """移动到设备"""
        super().to(*args, **kwargs)
        self.forwarder = self.forwarder.to(*args, **kwargs)
        return self

    def back_convert(self, x):
        """兼容原有的 back_convert 接口"""
        if hasattr(self.forwarder.model, 'back_convert'):
            return self.forwarder.model.back_convert(x)
        return x


class Sam3VideoForwarderAdapter(nn.Module):
    """
    适配器：让 Sam3VideoForwarder 兼容原有 SAM3Video 的接口

    使用方式：
        from sam3_forwarder_adapter import Sam3VideoForwarderAdapter
        model = Sam3VideoForwarderAdapter(checkpoint_path, device)
    """

    def __init__(
            self,
            checkpoint_path: str = None,
            device: str = "cuda",
            base_model=None,
            uap_path: str = None,
            **kwargs
    ):
        super().__init__()

        if base_model is not None:
            sam3_video_base = base_model
        else:
            from sam3.model_builder import build_sam3_video_predictor
            predictor_kwargs = {k: v for k, v in kwargs.items() if k != 'uap_path'}
            predictor = build_sam3_video_predictor(
                checkpoint_path=checkpoint_path,
                **predictor_kwargs
            )
            sam3_video_base = predictor.model if hasattr(predictor, 'model') else predictor

        # 确保是 Sam3VideoInferenceWithInstanceInteractivity 类型
        from sam3.model.sam3_video_inference import Sam3VideoInferenceWithInstanceInteractivity
        if not isinstance(sam3_video_base, Sam3VideoInferenceWithInstanceInteractivity):
            raise TypeError(
                f"Expected Sam3VideoInferenceWithInstanceInteractivity, got {type(sam3_video_base)}"
            )

        self.forwarder = Sam3VideoForwarder(sam3_video_base, device=device)
        self.device = device

        # 保留原有属性
        self.model = self.forwarder.model
        self.detector = self.forwarder.detector
        self.tracker = self.forwarder.tracker

    def forward(self, batch: BatchedDatapoint) -> List[SAM3Output]:
        """
        保持与原有 SAM3Video 模型完全一致的 forward 接口
        """
        return self.forwarder.model(batch)

    def init_video_state(self, num_frames: int):
        """初始化视频状态"""
        return self.forwarder.init_video_state(num_frames)

    def add_new_points(self, frame_idx: int, points, labels, obj_id: int = 1):
        """添加新的追踪点"""
        return self.forwarder.add_new_points(frame_idx, points, labels, obj_id)

    def track_step(self, frame_idx: int, image: torch.Tensor, **kwargs):
        """单帧追踪"""
        return self.forwarder.track_step(frame_idx, image, **kwargs)

    def get_tracker_features(self, frame_idx: int, image: torch.Tensor,
                             output_dict: Dict, layer_name: str = "memory_attn"):
        """获取追踪器特征（用于攻击）"""
        return self.forwarder.get_tracker_features(frame_idx, image, output_dict, layer_name)

    def state_dict(self, *args, **kwargs):
        return self.forwarder.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        return self.forwarder.model.load_state_dict(state_dict, *args, **kwargs)

    def train(self, mode: bool = True):
        super().train(mode)
        self.forwarder.model.train(mode)
        return self

    def eval(self):
        super().eval()
        self.forwarder.model.eval()
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.forwarder = self.forwarder.to(*args, **kwargs)
        return self

    def back_convert(self, x):
        if hasattr(self.forwarder.model, 'back_convert'):
            return self.forwarder.model.back_convert(x)
        return x


# ============================================================================
# 工厂函数：无缝替换原有的 model_builder 接口
# ============================================================================

def build_sam3_image_model_with_forwarder(
        checkpoint_path: str = None,
        device: str = "cuda",
        uap_path: str = None,
        **kwargs
) -> Sam3ForwarderAdapter:
    """
    替换原有的 build_sam3_image_model

    在配置文件中使用：
        model:
          _target_: sam3_forwarder_adapter.build_sam3_image_model_with_forwarder
          checkpoint_path: ${paths.checkpoint_path}
          device: cuda
    """
    return Sam3ForwarderAdapter(
        checkpoint_path=checkpoint_path,
        device=device,
        uap_path=uap_path,
        **kwargs
    )


def build_sam3_video_model_with_forwarder(
        checkpoint_path: str = None,
        device: str = "cuda",
uap_path: str = None,
        **kwargs
) -> Sam3VideoForwarderAdapter:
    """
    替换原有的 build_sam3_video_predictor

    在配置文件中使用：
        model:
          _target_: sam3_forwarder_adapter.build_sam3_video_model_with_forwarder
          checkpoint_path: ${paths.checkpoint_path}
          device: cuda
    """
    return Sam3VideoForwarderAdapter(
        checkpoint_path=checkpoint_path,
        device=device,
        uap_path=uap_path,
        **kwargs
    )


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    """
    示例1：图像模型
    """
    # 原有方式
    # from sam3.model_builder import build_sam3_image_model
    # model = build_sam3_image_model(checkpoint_path="path/to/ckpt")

    # 新方式（完全兼容）
    model = build_sam3_image_model_with_forwarder(checkpoint_path="path/to/ckpt")

    # 可以正常使用原有接口
    # output = model(batch)

    # 也可以使用新增的攻击接口
    # features = model.get_detector_features(image, layer_name="encoder")

    """
    示例2：视频模型
    """
    video_model = build_sam3_video_model_with_forwarder(checkpoint_path="path/to/ckpt")

    # 初始化视频状态
    # video_model.init_video_state(num_frames=10)

    # 追踪
    # masks = video_model.track_step(frame_idx=0, image=img_tensor)
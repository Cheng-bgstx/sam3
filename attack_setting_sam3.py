import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List, Union, Dict, Callable, Any, Optional
from numpy import ndarray
from numpy.typing import NDArray
import scipy.optimize
from sam3.perflib.masks_ops import mask_iou
from copy import deepcopy
from torch.utils.checkpoint import checkpoint

# SAM3 Imports
from sam3.model.sam3_video_base import Sam3VideoBase, MaskletConfirmationStatus
from sam3.model.sam3_video_inference import Sam3VideoInferenceWithInstanceInteractivity
from sam3.model.data_misc import FindStage, BatchedDatapoint, convert_my_tensors
from sam3.model.geometry_encoders import Prompt
from sam3.model import box_ops
from sam3.model.data_misc import FindStage, interpolate
from torchvision.transforms import v2
npimg_u8 = NDArray[np.uint8]
Size = Tuple[int, int]
Point = Tuple[int, int]
Prompts = Tuple[ndarray, ndarray, None, None]

# SAM3 默认参数
SAM3_MASK_THRESH = 0.0
SAM3_IMAGE_SIZE = 1008


# ==================== 与 SAM2 攻击逻辑完全一致的 Prompt 生成 ====================
def _parse_point(coord: Union[str, None], size: Size, prompt_num: int = 1) -> List[Point]:
    """生成网格点，与 attack_setting.py 完全一致"""
    grid_size = math.ceil(math.sqrt(prompt_num))
    width, height = size
    x_step = width // grid_size
    y_step = height // grid_size

    grid_points = []
    for i in range(grid_size):
        for j in range(grid_size):
            x = np.random.randint(i * x_step, (i + 1) * x_step)
            y = np.random.randint(j * y_step, (j + 1) * y_step)
            grid_points.append((x, y))
    return grid_points


def make_multi_prompts(points: Union[str, np.ndarray], img_size: Size, prompt_num: int = 1) -> Prompts:
    """生成多点 prompt，与 attack_setting.py 完全一致"""
    if isinstance(points, str) or points is None:
        grid_points = _parse_point(points, img_size, prompt_num)
        coords = np.array(grid_points, dtype=np.int32)
        labels = np.ones((coords.shape[0],), dtype=np.int32)
    else:
        coords = np.asarray(points, dtype=np.int32)
        assert coords.ndim == 2 and coords.shape[1] == 2, "Points must have shape (N, 2)"

        if coords.shape[0] < prompt_num:
            grid_points = _parse_point(None, img_size, prompt_num)
            grid_points = np.array(grid_points, dtype=np.int32)
            coords = np.vstack([coords, grid_points[:prompt_num - coords.shape[0]]])
        elif coords.shape[0] > prompt_num:
            coords = coords[:prompt_num, :]

        labels = np.ones((coords.shape[0],), dtype=np.int32)

    return (coords, labels, None, None)


class Sam3Forwarder(nn.Module):
    """
    SAM3 完整前向传播封装，支持白盒攻击。

    功能：
    1. 单帧图像分割 (Image Segmentation)
    2. 细粒度模块拆解 (用于特征攻击)
    3. 特征提取接口 (用于梯度分析)
    """

    def __init__(self, sam3_model: Sam3VideoBase, device='cuda:0'):
        super().__init__()
        self.model = sam3_model
        if hasattr(sam3_model, 'detector'):
            self.detector = sam3_model.detector
        else:
            # 这种情况下 sam3_model 本身就是 Sam3Image 实例
            self.detector = sam3_model
        self.tracker = sam3_model.tracker if hasattr(sam3_model, 'tracker') else None
        self.device = device
        self.mask_threshold = SAM3_MASK_THRESH
        self.image_size = SAM3_IMAGE_SIZE
        self.transform_inference = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(1008, 1008)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.transform_attack = v2.Compose([
            v2.Resize(size=(1008, 1008), antialias=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        # SAM3 归一化参数
        self.pixel_mean = [0.5, 0.5, 0.5]
        self.pixel_std = [0.5, 0.5, 0.5]

        self._orig_hw = None
        self.attack_feature_cache = {}

    # ==================== 图像预处理 ====================

    def norm_image(self, x: Tensor) -> Tensor:
        """标准化图像到 [-1, 1]"""
        pixel_mean = torch.tensor(self.pixel_mean, device=self.device).view(1, 3, 1, 1)
        pixel_std = torch.tensor(self.pixel_std, device=self.device).view(1, 3, 1, 1)
        return (x - pixel_mean) / pixel_std

    def denorm_image(self, x: Tensor) -> Tensor:
        """反标准化图像到 [0, 1]"""
        pixel_mean = torch.tensor(self.pixel_mean, device=self.device).view(1, 3, 1, 1)
        pixel_std = torch.tensor(self.pixel_std, device=self.device).view(1, 3, 1, 1)
        return x * pixel_std + pixel_mean

    def resize_image(self, x: Tensor) -> Tensor:
        """将图像 Resize 到 SAM3 要求的输入尺寸"""
        return F.interpolate(
            x,
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False
        )

    def transform_image(self, im: Union[np.ndarray, Tensor]) -> Tensor:
        if isinstance(im, np.ndarray):
            h, w = im.shape[:2]
            self._orig_hw = [(h, w)]  # 统一存储为列表
            im_tensor = torch.from_numpy(im).permute(2, 0, 1)
            im_tensor = v2.functional.to_image(im_tensor).to(self.device)
            x = self.transform_inference(im_tensor)

        else:
            im_tensor = im.to(self.device)
            x = self.transform_attack(im_tensor)

        if x.dim() == 3:
            x = x.unsqueeze(0)
        return x

    def transform_prompts(self, point_coords=None, point_labels=None, box=None, normalize_coords=True):
        """
        预处理 Prompts:
        将坐标归一化到 [0, 1] 范围(相对于原始图像尺寸)
        """
        img_idx = 0
        unnorm_coords, labels_torch, unnorm_box = None, None, None

        if point_coords is not None:
            point_coords = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.long, device=self.device)

            if point_coords.dim() == 2:
                point_coords = point_coords.unsqueeze(0)
                labels_torch = labels_torch.unsqueeze(0)

            if normalize_coords and self._orig_hw is not None:
                h, w = self._orig_hw[img_idx]
                point_coords = point_coords.clone()
                point_coords[..., 0] /= w
                point_coords[..., 1] /= h
            unnorm_coords = point_coords

        if box is not None:
            box = torch.as_tensor(box, dtype=torch.float, device=self.device)
            if box.dim() == 2:
                box = box.unsqueeze(0)

            if normalize_coords and self._orig_hw is not None:
                h, w = self._orig_hw[img_idx]
                box = box.clone()
                box[..., [0, 2]] /= w
                box[..., [1, 3]] /= h
            unnorm_box = box

        return unnorm_coords, labels_torch, unnorm_box


    # ==================== Detector 模块拆解 (用于白盒攻击) ====================

    def forward_detector_backbone(self, image: Tensor, text_prompt: str = "visual"):
        """
        拆解步骤1: Detector Backbone前向传播
        """
        # 1. 图像backbone
        backbone_out = self.detector.backbone.forward_image(image)

        # 【修正 3】SAM2 特征预处理 (严格对齐官方 set_image 逻辑)
        if "sam2_backbone_out" in backbone_out:
            sam2_backbone_out = backbone_out["sam2_backbone_out"]

            # === 鲁棒的投影层查找策略 ===
            mask_decoder = None

            # 策略 A: 尝试从 Video Predictor 的交互模块获取 (最标准)
            if hasattr(self.model, "inst_interactive_predictor") and self.model.inst_interactive_predictor is not None:
                mask_decoder = self.model.inst_interactive_predictor.model.sam_mask_decoder

            # 策略 B: 尝试从 Tracker 获取 (视频模式常用)
            elif hasattr(self.tracker, "sam_mask_decoder"):
                mask_decoder = self.tracker.sam_mask_decoder

            # 策略 C: 尝试从 Detector 获取 (兜底)
            elif hasattr(self.detector, "sam_mask_decoder"):
                mask_decoder = self.detector.sam_mask_decoder

            # === 执行投影 (如果不投影，Tracker 必挂) ===
            if mask_decoder is not None:
                try:
                    # 原地修改特征，将 64/144 通道投影到 256 通道
                    # 即使已经是 256，经过这个 Conv (1x1) 也是为了特征对齐
                    sam2_backbone_out["backbone_fpn"][0] = mask_decoder.conv_s0(
                        sam2_backbone_out["backbone_fpn"][0]
                    )
                    sam2_backbone_out["backbone_fpn"][1] = mask_decoder.conv_s1(
                        sam2_backbone_out["backbone_fpn"][1]
                    )
                except Exception as e:
                    print(f"[Warning] Backbone projection failed: {e}. Tracker might fail later.")
            else:
                # 如果完全找不到投影层，只能祈祷 Backbone 输出本身就是 256 (对于 Large 模型可能没事，Tiny 必挂)
                pass

        # 2. 文本编码
        text_outputs = self.detector.backbone.forward_text([text_prompt], device=self.device)
        backbone_out.update(text_outputs)

        return backbone_out

    def forward_detector_encoder(self, backbone_out, geometric_prompt):
        """
        [修改] 自动推断 batch_size 并传入 _create_find_input
        """
        # 1. 从 backbone_out 推断当前 Batch Size
        # backbone_out['backbone_fpn'][0] 的形状是 [B, C, H, W]
        current_bs = backbone_out['backbone_fpn'][0].shape[0]

        # 2. 传入 batch_size 创建正确的 Input
        find_input = self._create_find_input(batch_size=current_bs)

        # 3. 继续编码流程
        prompt, prompt_mask, backbone_out = self.detector._encode_prompt(
            backbone_out=backbone_out,
            find_input=find_input,
            geometric_prompt=geometric_prompt
        )

        # Encoder 输出
        backbone_out, encoder_out, _ = self.detector._run_encoder(
            backbone_out, find_input, prompt, prompt_mask
        )

        return encoder_out, prompt, prompt_mask, backbone_out

    def forward_detector_decoder(self, encoder_out, prompt, prompt_mask):
        """
        拆解步骤3: Detector Decoder前向传播
        生成object queries和box预测
        """
        bs = encoder_out["encoder_hidden_states"].shape[1]
        query_embed = self.detector.transformer.decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)

        # 运行decoder
        hs, reference_boxes, dec_presence_out, dec_presence_feats = (
            self.detector.transformer.decoder(
                tgt=tgt,
                memory=encoder_out["encoder_hidden_states"],
                memory_key_padding_mask=encoder_out["padding_mask"],
                pos=encoder_out["pos_embed"],
                reference_boxes=None,
                level_start_index=encoder_out["level_start_index"],
                spatial_shapes=encoder_out["spatial_shapes"],
                valid_ratios=encoder_out["valid_ratios"],
                tgt_mask=None,
                memory_text=prompt,
                text_attention_mask=prompt_mask,
                apply_dac=False,
            )
        )

        hs = hs.transpose(1, 2)  # seq-first to batch-first
        reference_boxes = reference_boxes.transpose(1, 2)

        return hs, reference_boxes, dec_presence_out

    def forward_detector_head(self, hs, backbone_out, encoder_out, prompt, prompt_mask):
        """
        拆解步骤4: Detector Segmentation Head
        生成最终mask预测
        """
        # ==================== 【错误根源在这里】 ====================
        # 原代码：find_input = self._create_find_input()  <-- 默认 batch_size=1

        # 【修正后代码】：必须传入真实的 Batch Size！
        # 我们从 hs (Hidden States) 的第一个维度获取 Batch Size
        current_bs = hs.shape[0]  # 这里应该是 6
        find_input = self._create_find_input(batch_size=current_bs)
        # ==========================================================

        # 运行segmentation head
        seg_out = self.detector.segmentation_head(
            backbone_feats=backbone_out["backbone_fpn"],
            obj_queries=hs,
            image_ids=find_input.img_ids,  # 这里的长度必须是 6，否则只会输出 1 个 mask
            encoder_hidden_states=encoder_out["encoder_hidden_states"],
            prompt=prompt,
            prompt_mask=prompt_mask,
        )

        return seg_out["pred_masks"]

    def forward_detector_scores_and_boxes(self, hs, reference_boxes, prompt, prompt_mask, dec_presence_out):
        """
        拆解步骤5: Detector Scoring & Box Head (补全)
        计算：pred_logits (质量分), pred_boxes (坐标), presence_logits (存在性)
        """
        out = {}
        if dec_presence_out is not None:
             # 这一步在原 Sam3Image._run_decoder 中
            out["presence_logit_dec"] = dec_presence_out

        # 调用 Sam3Image 内部方法，它会填充 out["pred_logits"] 和 out["pred_boxes"]
        # 这保证了与原始推理完全一致的计算图
        self.detector._update_scores_and_boxes(
            out,
            hs=hs,
            reference_boxes=reference_boxes,
            prompt=prompt,
            prompt_mask=prompt_mask,
            dec_presence_out=dec_presence_out
        )
        return out

    def forward_detector_complete(self, image: Tensor, point_coords=None, point_labels=None,
                                  boxes=None, text_prompt="visual") -> Dict[str, Tensor]:
        """
        【Master Function】完整的白盒前向传播
        返回所有中间层的 Logits，用于构建任意攻击 Loss。

        Returns:
            Dict: {
                "pred_masks": [B, Q, H, W],  # Mask Logits
                "pred_logits": [B, Q, 1],    # IoU/Quality Logits
                "pred_boxes": [B, Q, 4],     # Box Coordinates
                "presence_logits": [B, 1, Q] # Object Existence Logits
                "encoder_features": ...      # 用于特征攻击
                "decoder_features": ...      # 用于特征攻击
            }
        """
        # Step 1: Backbone (特征提取)
        backbone_out = self.forward_detector_backbone(image, text_prompt)

        # Step 2: 构造 Geometric Prompt
        geometric_prompt = self._construct_geometric_prompt(point_coords, point_labels, boxes)

        # Step 3: Encoder (Transformer 编码)
        # 获取 encoder_features 用于特征层攻击
        encoder_out, prompt, prompt_mask, backbone_out = self.forward_detector_encoder(
            backbone_out, geometric_prompt
        )

        # Step 4: Decoder (Transformer 解码)
        # 获取 hs (decoder_features) 用于特征层攻击
        hs, reference_boxes, dec_presence_out = self.forward_detector_decoder(
            encoder_out, prompt, prompt_mask
        )

        # Step 5: Scoring & Box Head (关键补充：计算置信度和坐标)
        # 这里包含了 Processor用于过滤的所有分量
        scores_out = self.forward_detector_scores_and_boxes(
            hs, reference_boxes, prompt, prompt_mask, dec_presence_out
        )

        # Step 6: Segmentation Head (Mask 生成)
        pred_masks_logits = self.forward_detector_head(hs, backbone_out, encoder_out, prompt, prompt_mask)

        # Step 7: Resize Mask (为了对齐原始输出，但保持 Logits 属性)
        outputs = {}

        # 7.1 计算最终概率
        pred_logits = scores_out["pred_logits"]
        presence_logits = scores_out.get("presence_logit_dec", None)
        score_probs = pred_logits.sigmoid()

        if presence_logits is not None:
            # 维度适配
            if presence_logits.shape[-1] != score_probs.shape[1] and presence_logits.shape[-2] == score_probs.shape[1]:
                presence_probs = presence_logits.transpose(-1, -2).sigmoid()
            elif presence_logits.dim() == 3 and presence_logits.shape[1] == 1:
                presence_probs = presence_logits.transpose(1, 2).sigmoid()
            else:
                presence_probs = presence_logits.sigmoid()
            final_scores = (score_probs * presence_probs).squeeze(-1)
        else:
            final_scores = score_probs.squeeze(-1)

        # 7.2 Mask 输出 (同时保留低分和高分 Logits)
        outputs["mask_logits"] = pred_masks_logits  # [B, Q, H/4, W/4] (Raw)

        if self._orig_hw is not None:
            h_orig, w_orig = self._orig_hw[0]
            # 插值到原图尺寸 (用于 Mask Loss) - 梯度安全
            mask_logits_up = F.interpolate(
                pred_masks_logits.unsqueeze(1) if pred_masks_logits.dim() == 3 else pred_masks_logits,
                size=(h_orig, w_orig),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
            outputs["mask_logits_up"] = mask_logits_up
        else:
            outputs["mask_logits_up"] = pred_masks_logits

        # 7.3 Box 输出 (保留 Normalized 格式)
        outputs["pred_boxes"] = scores_out["pred_boxes"]
        outputs["scores"] = final_scores

        # 7.4 完整特征包
        outputs.update({
            "pred_masks": pred_masks_logits,
            "pred_logits": pred_logits,
            "presence_logits": scores_out.get("presence_logit_dec"),  # 与 Processor 一致，供 compute_inference_confidence 使用
            "raw_backbone_out": backbone_out,
            "encoder_features": encoder_out["encoder_hidden_states"],
            "decoder_features": hs,
            "backbone_feature": backbone_out["backbone_fpn"][-1]
        })

        return outputs

    def compute_inference_confidence(self, outputs: Dict[str, Tensor]) -> Tensor:
        """
        【辅助函数】完全复刻 Processor 的置信度计算逻辑
        用于在攻击循环中验证目前的攻击效果，或者构建基于置信度的 Loss。

        Args:
            outputs: forward_detector_complete 的返回值
        Returns:
            final_probs: [B, Q] 最终的过滤概率
        """
        # 1. 取出 Logits（与 forward_detector_complete 输出键一致）
        pred_logits = outputs["pred_logits"]  # [B, Q, 1]
        presence_logits = outputs.get("presence_logits")  # [B, 1, Q] 或 None
        if presence_logits is None:
            return pred_logits.sigmoid().squeeze(-1)

        # 2. Sigmoid 归一化与维度对齐（与 Processor 一致）
        score_probs = pred_logits.sigmoid()
        if presence_logits.shape[-1] != score_probs.shape[1] and presence_logits.shape[-2] == score_probs.shape[1]:
            presence_probs = presence_logits.transpose(-1, -2).sigmoid()
        elif presence_logits.dim() == 3 and presence_logits.shape[1] == 1:
            presence_probs = presence_logits.transpose(1, 2).sigmoid()  # [B, Q, 1]
        else:
            presence_probs = presence_logits.sigmoid()
        # 3. 复刻公式: Score = Quality * Presence
        final_probs = score_probs * presence_probs
        return final_probs.squeeze(-1)

    # ==================== 简化接口 (与官方API对齐) ====================

    def forward_with_prompts(self, image: Tensor, point_coords=None, point_labels=None,
                             boxes=None, text_prompt=None) -> Tensor:
        """
        单帧/图像模式的 Mask 生成。
        直接调用 SAM3 Detector (Sam3Image) 的底层逻辑。
        """
        # 1. 前向传播 Backbone
        backbone_out = self.detector.backbone.forward_image(image)

        # 2. 前向传播 Text Encoder
        if text_prompt is None:
            text_prompt = "visual"
        text_outputs = self.detector.backbone.forward_text([text_prompt], device=self.device)
        backbone_out.update(text_outputs)

        # 3. 构造几何 Prompt
        geometric_prompt = self._construct_geometric_prompt(point_coords, point_labels, boxes)

        # 4. 调用 Detector 的 forward_grounding
        find_input = self._create_find_input()

        out = self.detector.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            find_target=None,
            geometric_prompt=geometric_prompt
        )

        # 5. 提取预测 Mask
        pred_masks = out["pred_masks"]
        pred_logits = out["pred_logits"]
        # 6. Resize 回原始尺寸
        if self._orig_hw is not None:
            h, w = self._orig_hw[0]
            if pred_masks.dim() == 4:
                masks_resized = F.interpolate(pred_masks, size=(h, w), mode='bilinear', align_corners=False)
            elif pred_masks.dim() == 3:
                masks_resized = F.interpolate(pred_masks.unsqueeze(0), size=(h, w), mode='bilinear',
                                              align_corners=False).squeeze(0)
            else:
                # 兜底：处理 [B, 1, Slots, H, W] 等情况
                shape = pred_masks.shape
                flattened = pred_masks.view(-1, shape[-2], shape[-1]).unsqueeze(1)
                resized = F.interpolate(flattened, size=(h, w), mode='bilinear', align_corners=False)
                masks_resized = resized.view(*shape[:-2], h, w)
            pred_masks = masks_resized

            # 返回字典，保持与 detector_output 一致的接口
        return {
            "pred_masks": pred_masks,
            "pred_logits": pred_logits,
            "presence_logit_dec": out.get("presence_logit_dec", None)
        }

    def forward(self, image, point_coords=None, point_labels=None, boxes=None, text_prompt=None):
        """
        统一接口:接收[0,1]范围的图像,内部处理norm
        """
        if image.min() >= 0 and image.max() <= 1:
            image = self.norm_image(image)
        return self.forward_with_prompts(image, point_coords, point_labels, boxes, text_prompt)

    # ==================== (用于推理) ====================

    def forward_single_frame_grounding(self, image: Tensor, text_prompt: str = "visual",
                                       point_coords=None, point_labels=None, boxes=None,
                                       confidence_threshold: float = 0.5) -> Dict[str, Union[Tensor, List]]:
        """
        【官方流程复刻】单帧 Grounding 完整流程 (用于评估/推理，非训练)
        完全对齐 sam3_image_processor.py 的 set_text_prompt + _forward_grounding 逻辑

        注意：此函数包含不可导操作（阈值过滤），仅用于评估攻击成功率，不用于反向传播 Loss。
        """
        # Step 1: Backbone 前向
        backbone_out = self.forward_detector_backbone(image, text_prompt)

        # Step 2: 构造 Geometric Prompt
        geometric_prompt = self._construct_geometric_prompt(point_coords, point_labels, boxes)

        # Step 3: 调用官方 forward_grounding (获取原始输出)
        find_input = self._create_find_input()
        outputs = self.detector.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            find_target=None,
            geometric_prompt=geometric_prompt
        )

        # ===== 以下是官方 _forward_grounding 的严格后处理逻辑 =====

        # Step 4: 提取预测 (Normalized)
        out_bbox = outputs["pred_boxes"]  # [B, Q, 4] cxcywh, normalized
        out_logits = outputs["pred_logits"]  # [B, Q, 1]
        out_masks = outputs["pred_masks"]  # [B, Q, H/4, W/4]

        # Step 5: 计算最终概率 (Quality * Presence)
        # 官方逻辑: out_probs = (out_logits.sigmoid() * presence_score).squeeze(-1)
        presence_logit = outputs.get("presence_logit_dec", None)
        if presence_logit is not None:
            presence_score = presence_logit.sigmoid().unsqueeze(1)  # [B, 1, Q]
            # 注意：官方代码这里可能有转置操作，取决于 presence_logit 的形状
            # 通常 presence_logit 是 [B, 1, Q]，out_logits 是 [B, Q, 1]
            if presence_score.shape[-1] == out_logits.shape[1]:
                out_probs = (out_logits.sigmoid().squeeze(-1) * presence_score.squeeze(1))
            else:
                # 维度适配兜底
                out_probs = (out_logits.sigmoid() * presence_score.transpose(-1, -2)).squeeze(-1)
        else:
            out_probs = out_logits.sigmoid().squeeze(-1)

        # Step 6: 置信度过滤 (不可导)
        # 假设 batch_size = 1 进行处理 (与官方 Processor 一致)
        # 如果是 Batch 攻击，这里需要循环处理。这里为了复刻官方逻辑，默认处理 B=0

        # [B, Q] -> [Q]
        out_probs_single = out_probs[0]
        keep = out_probs_single > confidence_threshold

        out_probs_filtered = out_probs_single[keep]
        out_bbox_filtered = out_bbox[0][keep]
        out_masks_filtered = out_masks[0][keep]

        # 如果没有检测到任何物体，返回空结果
        if len(out_probs_filtered) == 0:
            return {
                "masks": torch.zeros((0, *self._orig_hw[0]), device=self.device, dtype=torch.bool),
                "masks_logits": torch.zeros((0, *self._orig_hw[0]), device=self.device),
                "boxes": torch.zeros((0, 4), device=self.device),
                "scores": torch.zeros((0,), device=self.device),
                "raw_outputs": outputs
            }

        # Step 7: Box 格式转换 (cxcywh -> xyxy) 并缩放
        # 官方逻辑: boxes = box_ops.box_cxcywh_to_xyxy(out_bbox[keep])
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(out_bbox_filtered)

        if self._orig_hw is not None:
            img_h, img_w = self._orig_hw[0]
            scale_fct = torch.tensor([img_w, img_h, img_w, img_h],
                                     device=self.device, dtype=torch.float32)
            boxes_xyxy = boxes_xyxy * scale_fct[None, :]

        # Step 8: Mask 插值到原图尺寸并 Sigmoid
        # 官方逻辑: interpolate(..., (img_h, img_w)).sigmoid()
        if self._orig_hw is not None:
            img_h, img_w = self._orig_hw[0]
            out_masks_up = interpolate(
                out_masks_filtered.unsqueeze(1),
                (img_h, img_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1).sigmoid()  # 注意：官方这里做了 sigmoid
        else:
            out_masks_up = out_masks_filtered.sigmoid()

        # Step 9: 二值化
        masks_binary = out_masks_up > 0.5

        # Step 10: 返回结果
        return {
            "masks": masks_binary,  # [N, H, W] (Bool) 最终结果
            "masks_logits": out_masks_up,  # [N, H, W] (Float 0~1) 也就是 post-sigmoid map
            "boxes": boxes_xyxy,  # [N, 4] (Pixels)
            "scores": out_probs_filtered,  # [N]
            "raw_outputs": outputs  # 保留原始 Logits 备用
        }

    # ==================== 特征提取 (用于攻击) ====================

    def get_detector_features(self, image: Tensor, layer_name: str = "encoder") -> Tensor:
        """
        提取detector指定层的特征(用于特征攻击)
        layer_name: 'backbone', 'encoder', 'decoder', 'head'
        """
        backbone_out = self.forward_detector_backbone(image)

        if layer_name == "backbone":
            return backbone_out["backbone_fpn"][-1]

        geometric_prompt = self._get_empty_geometric_prompt()
        encoder_out, prompt, prompt_mask, backbone_out = self.forward_detector_encoder(
            backbone_out, geometric_prompt
        )

        if layer_name == "encoder":
            return encoder_out["encoder_hidden_states"]

        hs, reference_boxes, _ = self.forward_detector_decoder(encoder_out, prompt, prompt_mask)

        if layer_name == "decoder":
            return hs

        # head
        pred_masks = self.forward_detector_head(hs, backbone_out, encoder_out, prompt, prompt_mask)
        return pred_masks

    def get_image_feature(self, image: Tensor, frame_idx: int = 0) -> Tensor:
        """
        提取图像特征(用于特征损失)
        返回: Detector的Encoder Hidden States
        """
        if image.min() >= 0 and image.max() <= 1:
            image = self.norm_image(image)

        backbone_out = self.detector.backbone.forward_image(image)
        text_outputs = self.detector.backbone.forward_text(["visual"], device=self.device)
        backbone_out.update(text_outputs)

        geometric_prompt = self._get_empty_geometric_prompt()

        prompt_emb, prompt_mask, backbone_out = self.detector._encode_prompt(
            backbone_out=backbone_out,
            find_input=self._create_find_input(),
            geometric_prompt=geometric_prompt
        )

        _, encoder_out, _ = self.detector._run_encoder(
            backbone_out,
            self._create_find_input(),
            prompt_emb,
            prompt_mask
        )

        self.attack_feature_cache[frame_idx] = {
            'encoder_hidden_states': encoder_out["encoder_hidden_states"],
            'backbone_out': backbone_out
        }

        return encoder_out["encoder_hidden_states"]

    # ==================== SAM Route (与 SAM2 攻击逻辑一致: 点提示 + Tracker, 不经过 Detector/Geometry Encoder) ====================

    def _get_sam_route_tracker(self):
        """获取用于 SAM 路线的 tracker (点提示 -> sam_prompt_encoder -> sam_mask_decoder，不经过 geometry_encoder)"""
        if self.tracker is not None:
            return self.tracker
        if hasattr(self.model, 'inst_interactive_predictor') and self.model.inst_interactive_predictor is not None:
            return self.model.inst_interactive_predictor.model
        raise RuntimeError("No tracker or inst_interactive_predictor found for SAM route. Use video model.")

    def transform_prompts_sam_route(self, point_coords=None, point_labels=None, normalize_coords=True):
        """
        将 make_multi_prompts 输出转为 SAM Route 所需的 point_inputs 格式。
        与 SAM2 attack_setting.transform_prompts 逻辑一致，坐标需在 [0, image_size] 像素空间。
        """
        if point_coords is None or point_labels is None:
            return None
        point_coords = torch.as_tensor(point_coords, dtype=torch.float32, device=self.device)
        point_labels = torch.as_tensor(point_labels, dtype=torch.int64, device=self.device)
        if point_coords.dim() == 2:
            point_coords = point_coords.unsqueeze(0)  # [1, N, 2]
            point_labels = point_labels.unsqueeze(0)  # [1, N]
        if normalize_coords and self._orig_hw is not None:
            h, w = self._orig_hw[0]
            point_coords = point_coords.clone()
            point_coords[..., 0] = point_coords[..., 0] / w * self.image_size
            point_coords[..., 1] = point_coords[..., 1] / h * self.image_size
        return {"point_coords": point_coords, "point_labels": point_labels}

    def forward_sam_route(self, image: Tensor, point_coords: Tensor = None, point_labels: Tensor = None,
                          boxes: Tensor = None, mask_input: Tensor = None, multi_mask: bool = False) -> Tensor:
        """
        与 SAM2 SamForwarder.forward 完全一致的路线：backbone -> sam_prompt_encoder -> sam_mask_decoder
        不经过 Detector/Geometry Encoder。使用 grid points 作为提示。
        """
        image = image.to(self.device)
        if image.min() >= 0 and image.max() <= 1:
            image = self.norm_image(image)
        if image.dim() == 3:
            image = image.unsqueeze(0)

        tracker = self._get_sam_route_tracker()
        img_resized = F.interpolate(image, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)

        if tracker.backbone is not None:
            backbone_out = tracker.forward_image(img_resized)
        else:
            det_backbone_out = self.model.detector.backbone.forward_image(img_resized)
            sam2_out = det_backbone_out.get("sam2_backbone_out")
            if sam2_out is None:
                raise RuntimeError("forward_sam_route: detector.backbone did not return sam2_backbone_out.")
            backbone_out = {
                "backbone_fpn": [x.clone() for x in sam2_out["backbone_fpn"]],
                "vision_pos_enc": [x.clone() for x in sam2_out["vision_pos_enc"]],
            }
            backbone_out["backbone_fpn"][0] = tracker.sam_mask_decoder.conv_s0(backbone_out["backbone_fpn"][0])
            backbone_out["backbone_fpn"][1] = tracker.sam_mask_decoder.conv_s1(backbone_out["backbone_fpn"][1])
            for i in range(len(backbone_out["backbone_fpn"])):
                backbone_out["backbone_fpn"][i] = backbone_out["backbone_fpn"][i].clone()
                backbone_out["vision_pos_enc"][i] = backbone_out["vision_pos_enc"][i].clone()

        _, vision_feats, vision_pos_embeds, feat_sizes = tracker._prepare_backbone_features(backbone_out)

        B, C = vision_feats[-1].size(1), tracker.hidden_dim
        H, W = feat_sizes[-1]
        pix_feat_with_mem = vision_feats[-1] + tracker.no_mem_embed
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)

        point_inputs = None
        if point_coords is not None and point_labels is not None:
            if point_coords.dim() == 2:
                point_coords = point_coords.unsqueeze(0)
                point_labels = point_labels.unsqueeze(0)
            point_inputs = {"point_coords": point_coords.to(self.device), "point_labels": point_labels.to(self.device)}

        high_res_features = None
        if len(vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
            ]

        low_res_masks, high_res_masks, ious, _, _, _, _ = tracker._forward_sam_heads(
            backbone_features=pix_feat_with_mem,
            point_inputs=point_inputs,
            mask_inputs=mask_input,
            high_res_features=high_res_features,
            multimask_output=multi_mask,
        )

        if self._orig_hw is not None:
            h, w = self._orig_hw[0]
            masks = F.interpolate(low_res_masks.float(), size=(h, w), mode='bilinear', align_corners=False)
        else:
            masks = F.interpolate(low_res_masks.float(), size=(self.image_size, self.image_size),
                                  mode='bilinear', align_corners=False)
        return masks

    def forward_sam_route_with_feature(
        self,
        image: Tensor,
        point_coords: Tensor = None,
        point_labels: Tensor = None,
        boxes: Tensor = None,
        mask_input: Tensor = None,
        multi_mask: bool = False,
    ) -> tuple:
        """
        一次 backbone 前向同时返回 mask logits 与 image feature，与 forward_sam_route + get_image_feature_sam_route 等价但少一次 backbone，省显存。
        返回 (masks, image_feature)，image_feature 形状 [1, C, H, W] 与 get_image_feature_sam_route 一致。
        """
        image = image.to(self.device)
        if image.min() >= 0 and image.max() <= 1:
            image = self.norm_image(image)
        if image.dim() == 3:
            image = image.unsqueeze(0)

        tracker = self._get_sam_route_tracker()
        img_resized = F.interpolate(image, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        def run_backbone(img_input):
            if tracker.backbone is not None:
                backbone_out = tracker.forward_image(img_input)
            else:
                det_backbone_out = self.model.detector.backbone.forward_image(img_input)
                sam2_out = det_backbone_out.get("sam2_backbone_out")
                if sam2_out is None:
                    raise RuntimeError("forward_sam_route_with_feature: detector.backbone did not return sam2_backbone_out.")
                backbone_out = {
                    "backbone_fpn": [x.clone() for x in sam2_out["backbone_fpn"]],
                    "vision_pos_enc": [x.clone() for x in sam2_out["vision_pos_enc"]],
                }
                backbone_out["backbone_fpn"][0] = tracker.sam_mask_decoder.conv_s0(backbone_out["backbone_fpn"][0])
                backbone_out["backbone_fpn"][1] = tracker.sam_mask_decoder.conv_s1(backbone_out["backbone_fpn"][1])
                for i in range(len(backbone_out["backbone_fpn"])):
                    backbone_out["backbone_fpn"][i] = backbone_out["backbone_fpn"][i].clone()
                    backbone_out["vision_pos_enc"][i] = backbone_out["vision_pos_enc"][i].clone()
                return backbone_out
        use_ckpt = getattr(self, "_use_backbone_checkpoint", False)
        if use_ckpt and image.requires_grad:
            backbone_out = checkpoint(run_backbone, img_resized, use_reentrant=False)
        else:
            backbone_out = run_backbone(img_resized)
        _, vision_feats, vision_pos_embeds, feat_sizes = tracker._prepare_backbone_features(backbone_out)
        B, C = vision_feats[-1].size(1), tracker.hidden_dim
        H, W = feat_sizes[-1]
        pix_feat_with_mem = vision_feats[-1] + tracker.no_mem_embed
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)

        # image_feature 与 get_image_feature_sam_route 一致
        image_feature = vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)

        point_inputs = None
        if point_coords is not None and point_labels is not None:
            if point_coords.dim() == 2:
                point_coords = point_coords.unsqueeze(0)
                point_labels = point_labels.unsqueeze(0)
            point_inputs = {"point_coords": point_coords.to(self.device), "point_labels": point_labels.to(self.device)}

        high_res_features = None
        if len(vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
            ]

        low_res_masks, high_res_masks, ious, _, _, _, _ = tracker._forward_sam_heads(
            backbone_features=pix_feat_with_mem,
            point_inputs=point_inputs,
            mask_inputs=mask_input,
            high_res_features=high_res_features,
            multimask_output=multi_mask,
        )

        if self._orig_hw is not None:
            h, w = self._orig_hw[0]
            masks = F.interpolate(low_res_masks.float(), size=(h, w), mode='bilinear', align_corners=False)
        else:
            masks = F.interpolate(low_res_masks.float(), size=(self.image_size, self.image_size),
                                  mode='bilinear', align_corners=False)
        return masks, image_feature

    def get_image_feature_sam_route(self, image: Tensor, frame_idx: int = 0) -> Tensor:
        """
        与 SAM2 get_image_feature 完全一致：返回 backbone 的 image_embed (最后一层特征)。
        用于 loss_fea (InfoNCE) 和 loss_diff (时序一致性)。
        视频模型中 tracker.backbone 为 None，backbone 在 detector 上，需用 detector 跑出 sam2_backbone_out 再经 tracker 的 conv 与 _prepare_backbone_features。
        """
        image = image.to(self.device)
        if image.min() >= 0 and image.max() <= 1:
            image = self.norm_image(image)
        if image.dim() == 3:
            image = image.unsqueeze(0)

        tracker = self._get_sam_route_tracker()
        img_resized = F.interpolate(image, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)

        if tracker.backbone is not None:
            backbone_out = tracker.forward_image(img_resized)
        else:
            # 视频模型：tracker 无 backbone，由 detector.backbone 提供 sam2_backbone_out
            det_backbone_out = self.model.detector.backbone.forward_image(img_resized)
            sam2_out = det_backbone_out.get("sam2_backbone_out")
            if sam2_out is None:
                raise RuntimeError("get_image_feature_sam_route: detector.backbone did not return sam2_backbone_out (video model needs visual with SAM2 features).")
            backbone_out = {
                "backbone_fpn": [x.clone() for x in sam2_out["backbone_fpn"]],
                "vision_pos_enc": [x.clone() for x in sam2_out["vision_pos_enc"]],
            }
            backbone_out["backbone_fpn"][0] = tracker.sam_mask_decoder.conv_s0(backbone_out["backbone_fpn"][0])
            backbone_out["backbone_fpn"][1] = tracker.sam_mask_decoder.conv_s1(backbone_out["backbone_fpn"][1])
            for i in range(len(backbone_out["backbone_fpn"])):
                backbone_out["backbone_fpn"][i] = backbone_out["backbone_fpn"][i].clone()
                backbone_out["vision_pos_enc"][i] = backbone_out["vision_pos_enc"][i].clone()

        _, vision_feats, _, feat_sizes = tracker._prepare_backbone_features(backbone_out)

        B, C, H, W = vision_feats[-1].size(1), tracker.hidden_dim, *feat_sizes[-1]
        feat = vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        return feat

    # ==================== 辅助函数 ====================

    def _create_find_input(self, batch_size=1):
        """
        [修改] 支持动态 Batch Size
        构造 FindStage，所有字段必须是长度为 batch_size 的列表
        """
        # 1. 构造单样本的空数据 (在 GPU 上)
        empty_boxes = torch.zeros(0, 4, device=self.device)
        empty_points = torch.zeros(0, 2, device=self.device)
        empty_mask = torch.zeros(0, device=self.device, dtype=torch.bool)
        empty_label = torch.zeros(0, device=self.device, dtype=torch.long)

        # 2. 构造 Batch 列表 (关键修改：复制 batch_size 份)
        # 这样 convert_my_tensors 才会将其 stack 为 [B, ...] 的形状
        find_stage = FindStage(
            img_ids=[torch.tensor(0, device=self.device, dtype=torch.long) for _ in range(batch_size)],
            text_ids=[torch.tensor(0, device=self.device, dtype=torch.long) for _ in range(batch_size)],

            input_boxes=[empty_boxes for _ in range(batch_size)],
            input_boxes_mask=[empty_mask for _ in range(batch_size)],
            input_boxes_label=[empty_label for _ in range(batch_size)],
            input_points=[empty_points for _ in range(batch_size)],
            input_points_mask=[empty_mask for _ in range(batch_size)],
        )
        return convert_my_tensors(find_stage)

    def _get_empty_geometric_prompt(self) -> Prompt:
        """获取空的geometric prompt"""
        return Prompt(
            box_embeddings=torch.zeros(0, 1, 4, device=self.device),
            box_mask=torch.zeros(1, 0, device=self.device, dtype=torch.bool),
            box_labels=torch.zeros(0, 1, device=self.device, dtype=torch.long),
            point_embeddings=torch.zeros(0, 1, 2, device=self.device),
            point_mask=torch.zeros(1, 0, device=self.device, dtype=torch.bool),
            point_labels=torch.zeros(0, 1, device=self.device, dtype=torch.long),
        )

    def _construct_geometric_prompt(self, point_coords, point_labels, boxes):
        """构造符合官方标准的 geometric prompt"""
        bs = 1  # 假设 batch 为 1

        if point_coords is not None and point_coords.numel() > 0:
            # 官方标准：点坐标 [B, N, 2] -> [N, B, 2]
            point_embeddings = point_coords.transpose(0, 1)
            point_mask = torch.zeros(bs, point_coords.size(1), device=self.device, dtype=torch.bool)

            # 官方标准：点标签 [B, N] -> [N, B]
            # 不要用 unsqueeze(1)，那是导致 [1, 1, 1] 的罪魁祸首
            if point_labels is not None:
                # 确保是 2 维 [B, N]
                if point_labels.dim() == 1:
                    point_labels = point_labels.unsqueeze(0)
                elif point_labels.dim() == 3:
                    point_labels = point_labels.squeeze(-1)

                # 执行转置，得到 [N, B] -> 符合官方 [point_seq_len, bs] 的断言
                point_labels_final = point_labels.transpose(0, 1)
            else:
                point_labels_final = torch.ones(point_coords.size(1), bs, device=self.device, dtype=torch.long)
        else:
            point_embeddings = torch.zeros(0, bs, 2, device=self.device)
            point_mask = torch.zeros(bs, 0, device=self.device, dtype=torch.bool)
            point_labels_final = torch.zeros(0, bs, device=self.device, dtype=torch.long)

        if boxes is not None and boxes.numel() > 0:
            box_embeddings = boxes.transpose(0, 1)
            box_mask = torch.zeros(bs, boxes.size(1), device=self.device, dtype=torch.bool)
            box_labels = torch.ones(boxes.size(1), bs, device=self.device, dtype=torch.long)
        else:
            box_embeddings = torch.zeros(0, bs, 4, device=self.device)
            box_mask = torch.zeros(bs, 0, device=self.device, dtype=torch.bool)
            box_labels = torch.zeros(0, bs, device=self.device, dtype=torch.long)

        if point_labels is not None and point_labels.dim() == 3:
            point_labels = point_labels.squeeze(-1)

        return Prompt(
            box_embeddings=box_embeddings,
            box_mask=box_mask,
            box_labels=box_labels,
            point_embeddings=point_embeddings,
            point_mask=point_mask,
            point_labels=point_labels_final,  # 传入正确转置后的 2 维张量
        )


class Sam3VideoForwarder(Sam3Forwarder):
    def __init__(self, sam3_video_model: Sam3VideoInferenceWithInstanceInteractivity, device='cuda:0'):
        super().__init__(sam3_video_model, device)
        self.pipeline = Sam3VideoOfficialGradPipeline(sam3_video_model, freeze_model=True)
        self.inference_state = None
        self.num_frames = 0

    def init_video_state(self, num_frames: int, frames: torch.Tensor = None, orig_height: int = None, orig_width: int = None):
        if frames is not None and orig_height is not None and orig_width is not None:
            self.num_frames = int(frames.shape[0])
            self.inference_state = self.pipeline.init_state_from_frames(
                frames=frames,
                orig_height=orig_height,
                orig_width=orig_width,
                normalize=False,
                dtype=torch.float32
            )
        else:
            self.num_frames = num_frames
            frames_dummy = torch.zeros(num_frames, 3, self.image_size, self.image_size, device=self.device)
            self.inference_state = self.pipeline.init_state_from_frames(
                frames=frames_dummy,
                orig_height=self.image_size,
                orig_width=self.image_size,
                normalize=False,
                dtype=torch.float32
            )

    def _update_frames_in_state(self, frames: torch.Tensor):
        if frames.dim() == 4:
            T = frames.shape[0]
            if T != self.num_frames:
                raise ValueError(f"frames count {T} != num_frames {self.num_frames}")
            if frames.shape[-2:] != (self.image_size, self.image_size):
                frames = F.interpolate(frames, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
            frames = frames.to(self.device, dtype=torch.float32)
            mean = torch.tensor([0.5, 0.5, 0.5], device=frames.device, dtype=frames.dtype)[:, None, None]
            std = torch.tensor([0.5, 0.5, 0.5], device=frames.device, dtype=frames.dtype)[:, None, None]
            frames = (frames - mean) / std
            self.inference_state["input_batch"].img_batch = frames
    
    def add_new_points(self, frame_idx, points, labels, obj_id=1, clear_old_points=True, rel_coordinates=True):
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        px, py = float(points[0, 0]), float(points[0, 1])
        box_size = 20.0
        H, W = self.image_size, self.image_size
        x1 = max(0.0, px - box_size / 2.0) / W
        y1 = max(0.0, py - box_size / 2.0) / H
        w = box_size / W
        h = box_size / H
        boxes_xywh = [[x1, y1, w, h]]
        box_labels = [1]
        _, post, raw = self.pipeline.add_prompt_grad(
            self.inference_state,
            frame_idx=frame_idx,
            text_str="visual",
            boxes_xywh=boxes_xywh,
            box_labels=box_labels
        )
        if raw and "obj_id_to_mask" in raw:
            masks_dict = raw["obj_id_to_mask"]
            if masks_dict:
                first_mask = list(masks_dict.values())[0]
                if isinstance(first_mask, torch.Tensor):
                    return first_mask.unsqueeze(0).unsqueeze(0)
        return torch.zeros(1, 1, H, W, device=self.device)
    
    def add_new_box(self, frame_idx, box, obj_id=1, use_detector=False):
        if not isinstance(box, torch.Tensor):
            box = torch.as_tensor(box, dtype=torch.float32, device=self.device)
        if box.dim() == 1:
            box = box.unsqueeze(0)
        x1, y1, x2, y2 = box[0].tolist()
        w, h = x2 - x1, y2 - y1
        boxes_xywh = [[x1, y1, w, h]]
        box_labels = [1]
        _, post, raw = self.pipeline.add_prompt_grad(
            self.inference_state,
            frame_idx=frame_idx,
            text_str="visual",
            boxes_xywh=boxes_xywh,
            box_labels=box_labels
        )
        if post and "out_binary_masks" in post and len(post["out_binary_masks"]) > 0:
            m = post["out_binary_masks"][0]
            t = torch.from_numpy(m).float().to(self.device)
            return t.unsqueeze(0).unsqueeze(0)
        if raw and "obj_id_to_mask" in raw and raw["obj_id_to_mask"]:
            first_mask = list(raw["obj_id_to_mask"].values())[0]
            if isinstance(first_mask, torch.Tensor):
                return first_mask.unsqueeze(0).unsqueeze(0)
        return torch.zeros(1, 1, self.image_size, self.image_size, device=self.device)
    
    def propagate_in_video_grad(self, inference_state=None, start_frame_idx=None, max_frame_num_to_track=None, reverse=False, obj_ids=None, run_mem_encoder=True, propagate_preflight=False):
        state = inference_state if inference_state is not None else self.inference_state
        for frame_idx, post, raw in self.pipeline.propagate_in_video_grad_full(
            state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=reverse,
            disable_tqdm=True
        ):
            if post and "out_binary_masks" in post and len(post["out_binary_masks"]) > 0:
                masks_np = post["out_binary_masks"]
                curr_obj_ids = sorted(raw["obj_id_to_mask"].keys()) if raw and "obj_id_to_mask" in raw else list(range(len(masks_np)))
                video_res_masks = torch.stack([torch.from_numpy(m).float().to(self.device) for m in masks_np]).unsqueeze(1)
                obj_ids_list = curr_obj_ids
            elif raw and "obj_id_to_mask" in raw and raw["obj_id_to_mask"]:
                masks_dict = raw["obj_id_to_mask"]
                video_res_masks = torch.stack([v for v in masks_dict.values()]).unsqueeze(1)
                obj_ids_list = list(masks_dict.keys())
            else:
                continue
            obj_scores = raw.get("obj_id_to_score", {}) if raw else {}
            obj_scores_tensor = torch.tensor([obj_scores.get(oid, 0.5) for oid in obj_ids_list], device=self.device)
            low_res_masks = video_res_masks
            yield frame_idx, obj_ids_list, low_res_masks, video_res_masks, obj_scores_tensor, raw


# ==============================================================================
# Grad-enabled official video inference re-wrap
# ==============================================================================

class Sam3VideoOfficialGradPipeline(nn.Module):
    
    def __init__(self, model: Sam3VideoInferenceWithInstanceInteractivity, *, freeze_model: bool = True):
        super().__init__()
        self.model = model
        if freeze_model:
            for p in self.model.parameters():
                p.requires_grad_(False)
    
    def init_state_from_frames(
        self,
        frames: torch.Tensor,
        orig_height: int,
        orig_width: int,
        *,
        is_image_only: bool = False,
        normalize: bool = True,
        img_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        img_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        dtype: torch.dtype = torch.float32,
    ) -> Dict[str, Any]:
        assert isinstance(frames, torch.Tensor) and frames.dim() == 4 and frames.shape[1] == 3
        T = int(frames.shape[0])
        image_size = int(getattr(self.model, "image_size", SAM3_IMAGE_SIZE))
        if frames.shape[-2:] != (image_size, image_size):
            frames = F.interpolate(frames, size=(image_size, image_size), mode="bilinear", align_corners=False)
        frames = frames.to(device=self.model.device, dtype=dtype)
        if normalize:
            mean = torch.tensor(img_mean, device=frames.device, dtype=frames.dtype)[:, None, None]
            std = torch.tensor(img_std, device=frames.device, dtype=frames.dtype)[:, None, None]
            frames = (frames - mean) / std
        inference_state: Dict[str, Any] = {}
        inference_state["image_size"] = image_size
        inference_state["num_frames"] = T
        inference_state["orig_height"] = int(orig_height)
        inference_state["orig_width"] = int(orig_width)
        inference_state["constants"] = {}
        self.model._construct_initial_input_batch(inference_state, frames)
        inference_state["tracker_inference_states"] = []
        inference_state["tracker_metadata"] = {}
        inference_state["feature_cache"] = {}
        inference_state["cached_frame_outputs"] = {}
        inference_state["action_history"] = []
        inference_state["is_image_only"] = bool(is_image_only)
        return inference_state
    
    def _reset_state_grad(self, inference_state: Dict[str, Any]):
        """Inlined reset_state logic without inference_mode, preserves gradient flow."""
        inference_state["input_batch"].find_text_batch[0] = "<text placeholder>"
        inference_state["text_prompt"] = None
        for t in range(inference_state["num_frames"]):
            inference_state["input_batch"].find_inputs[t].text_ids[...] = 0
            inference_state["previous_stages_out"][t] = None
            inference_state["per_frame_raw_point_input"][t] = None
            inference_state["per_frame_raw_box_input"][t] = None
            inference_state["per_frame_visual_prompt"][t] = None
            inference_state["per_frame_geometric_prompt"][t] = None
            inference_state["per_frame_cur_step"][t] = 0
        inference_state["visual_prompt_embed"] = None
        inference_state["visual_prompt_mask"] = None
        inference_state["tracker_inference_states"].clear()
        inference_state["tracker_metadata"].clear()
        inference_state["feature_cache"].clear()
        inference_state["cached_frame_outputs"].clear()
        inference_state["action_history"].clear()

    def add_prompt_grad(
        self,
        inference_state: Dict[str, Any],
        frame_idx: int,
        *,
        text_str: Optional[str] = None,
        boxes_xywh: Optional[Union[List[List[float]], torch.Tensor]] = None,
        box_labels: Optional[Union[List[int], torch.Tensor]] = None,
    ) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
        num_frames = inference_state["num_frames"]
        assert 0 <= frame_idx < num_frames
        assert text_str is not None or boxes_xywh is not None
        assert (boxes_xywh is not None) == (box_labels is not None)
        self._reset_state_grad(inference_state)
        if text_str is not None and text_str != "visual":
            inference_state["text_prompt"] = text_str
            inference_state["input_batch"].find_text_batch[0] = text_str
            text_id = self.model.TEXT_ID_FOR_TEXT
        else:
            inference_state["text_prompt"] = None
            inference_state["input_batch"].find_text_batch[0] = "<text placeholder>"
            text_id = self.model.TEXT_ID_FOR_VISUAL
        for t in range(inference_state["num_frames"]):
            inference_state["input_batch"].find_inputs[t].text_ids[...] = text_id
        if boxes_xywh is not None:
            boxes_xywh_t = boxes_xywh if isinstance(boxes_xywh, torch.Tensor) else torch.as_tensor(boxes_xywh, dtype=torch.float32)
            box_labels_t = box_labels if isinstance(box_labels, torch.Tensor) else torch.as_tensor(box_labels, dtype=torch.long)
            assert boxes_xywh_t.dim() == 2 and boxes_xywh_t.size(-1) == 4
            assert box_labels_t.dim() == 1 and box_labels_t.numel() == boxes_xywh_t.size(0)
            boxes_cxcywh = box_ops.box_xywh_to_cxcywh(boxes_xywh_t)
            new_box_input = boxes_cxcywh, box_labels_t
            inference_state["per_frame_raw_box_input"][frame_idx] = new_box_input
            boxes_cxcywh, box_labels_t, geometric_prompt = self.model._get_visual_prompt(inference_state, frame_idx, boxes_cxcywh, box_labels_t)
            inference_state["per_frame_geometric_prompt"][frame_idx] = geometric_prompt
        raw_out = self._run_single_frame_inference_grad(inference_state, frame_idx, reverse=False)
        if self.model.rank == 0:
            post = self.model._postprocess_output(inference_state, raw_out)
        else:
            post = None
        return frame_idx, post, raw_out
    
    def _run_single_frame_inference_grad(self, inference_state: Dict[str, Any], frame_idx: int, reverse: bool):
        input_batch = inference_state["input_batch"]
        tracker_states_local = inference_state["tracker_inference_states"]
        has_text_prompt = inference_state["text_prompt"] is not None
        has_geometric_prompt = inference_state["per_frame_geometric_prompt"][frame_idx] is not None
        (
            obj_id_to_mask,
            obj_id_to_score,
            tracker_states_local_new,
            tracker_metadata_new,
            frame_stats,
            internals,
        ) = self.model._det_track_one_frame(
            frame_idx=frame_idx,
            num_frames=inference_state["num_frames"],
            reverse=reverse,
            input_batch=input_batch,
            geometric_prompt=(
                inference_state["constants"]["empty_geometric_prompt"]
                if not has_geometric_prompt
                else inference_state["per_frame_geometric_prompt"][frame_idx]
            ),
            tracker_states_local=tracker_states_local,
            tracker_metadata_prev=inference_state["tracker_metadata"],
            feature_cache=inference_state["feature_cache"],
            orig_vid_height=inference_state["orig_height"],
            orig_vid_width=inference_state["orig_width"],
            is_image_only=inference_state["is_image_only"],
            allow_new_detections=has_text_prompt or has_geometric_prompt,
        )
        inference_state["tracker_inference_states"] = tracker_states_local_new
        inference_state["tracker_metadata"] = tracker_metadata_new
        inference_state["previous_stages_out"][frame_idx] = "_THIS_FRAME_HAS_OUTPUTS_"
        if self.model.rank == 0:
            self.model._cache_frame_outputs(inference_state, frame_idx, obj_id_to_mask)
        out = {
            "obj_id_to_mask": obj_id_to_mask,
            "obj_id_to_score": obj_id_to_score,
            "obj_id_to_tracker_score": tracker_metadata_new["obj_id_to_tracker_score_frame_wise"][frame_idx],
            "frame_stats": frame_stats,
            "internals": internals,
        }
        if self.model.rank == 0:
            rank0_metadata = tracker_metadata_new["rank0_metadata"]
            out["removed_obj_ids"] = rank0_metadata["removed_obj_ids"]
            out["suppressed_obj_ids"] = rank0_metadata["suppressed_obj_ids"][frame_idx]
            if getattr(self.model, "masklet_confirmation_enable", False):
                status = rank0_metadata["masklet_confirmation"]["status"]
                is_unconfirmed = status == MaskletConfirmationStatus.UNCONFIRMED.value
                out["unconfirmed_obj_ids"] = tracker_metadata_new["obj_ids_all_gpu"][is_unconfirmed].tolist()
            else:
                out["unconfirmed_obj_ids"] = []
        return out
    
    def propagate_in_video_grad_full(
        self,
        inference_state: Dict[str, Any],
        *,
        start_frame_idx: Optional[int] = None,
        max_frame_num_to_track: Optional[int] = None,
        reverse: bool = False,
        disable_tqdm: bool = True,
        enable_compile: bool = False,
    ):
        if enable_compile:
            self.model._compile_model()
        processing_order, end_frame_idx = self.model._get_processing_order(
            inference_state,
            start_frame_idx,
            max_frame_num_to_track,
            reverse=reverse,
        )
        inference_state.setdefault("feature_cache", {})
        inference_state["feature_cache"]["tracking_bounds"] = {
            "max_frame_num_to_track": max_frame_num_to_track,
            "propagate_in_video_start_frame_idx": start_frame_idx,
        }
        hotstart_buffer = []
        hotstart_removed_obj_ids = set()
        unconfirmed_status_delay = self.model.masklet_confirmation_consecutive_det_thresh - 1
        unconfirmed_obj_ids_per_frame = {}
        iterable = processing_order
        if not disable_tqdm:
            try:
                from tqdm.auto import tqdm
                iterable = tqdm(processing_order, desc="propagate_in_video_grad")
            except Exception:
                iterable = processing_order
        for frame_idx in iterable:
            out = self._run_single_frame_inference_grad(inference_state, frame_idx, reverse)
            if getattr(self.model, "hotstart_delay", 0) > 0:
                hotstart_buffer.append([frame_idx, out])
                if self.model.rank == 0:
                    hotstart_removed_obj_ids.update(out.get("removed_obj_ids", []))
                    unconfirmed_obj_ids = out.get("unconfirmed_obj_ids", None)
                    if unconfirmed_obj_ids is not None:
                        unconfirmed_obj_ids_per_frame[frame_idx] = unconfirmed_obj_ids
                if frame_idx == end_frame_idx:
                    yield_list = hotstart_buffer
                    hotstart_buffer = []
                elif len(hotstart_buffer) >= self.model.hotstart_delay:
                    yield_list = hotstart_buffer[:1]
                    hotstart_buffer = hotstart_buffer[1:]
                else:
                    yield_list = []
            else:
                yield_list = [(frame_idx, out)]
            for yield_frame_idx, yield_out in yield_list:
                if self.model.rank == 0:
                    suppressed_obj_ids = yield_out.get("suppressed_obj_ids", [])
                    unconfirmed_status_frame_idx = (
                        yield_frame_idx + unconfirmed_status_delay
                        if not reverse
                        else yield_frame_idx - unconfirmed_status_delay
                    )
                    num_frames = inference_state["num_frames"]
                    unconfirmed_status_frame_idx = max(0, min(unconfirmed_status_frame_idx, num_frames - 1))
                    unconfirmed_obj_ids = unconfirmed_obj_ids_per_frame.get(unconfirmed_status_frame_idx, None)
                    post = self.model._postprocess_output(
                        inference_state,
                        yield_out,
                        hotstart_removed_obj_ids,
                        suppressed_obj_ids,
                        unconfirmed_obj_ids,
                    )
                    self.model._cache_frame_outputs(
                        inference_state,
                        yield_frame_idx,
                        yield_out["obj_id_to_mask"],
                        suppressed_obj_ids=suppressed_obj_ids,
                        removed_obj_ids=hotstart_removed_obj_ids,
                        unconfirmed_obj_ids=unconfirmed_obj_ids,
                    )
                else:
                    post = None
                yield yield_frame_idx, post, yield_out


FwdPack = Tuple[Union[Sam3VideoForwarder, Sam3Forwarder], Prompts, Callable]


def seed_everything(seed: int):
    print('>> global seed:', seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _expand_like_ref(x: Tensor, ref: Tensor) -> Tensor:
    """Expand a [1, C, H, W] tensor to [B, C, H, W] if needed."""
    if x.shape[0] == ref.shape[0]:
        return x
    if x.shape[0] == 1 and ref.shape[0] > 1:
        return x.expand(ref.shape[0], -1, -1, -1)
    return x


class DarkSamLossModule(nn.Module):
    """
    SAM3 版 DarkSAM 损失，与 DarkSAM-main 损失函数一致。
    - Mask: output_t=attacked*logits, output_f=(~attacked)*logits; loss_t=MSE(output_t,Y), loss_f=MSE(output_f,-Y)。
    - 频域: low=MSE(adv_low,clean_low), high=-MSE(adv_high,clean_high)；freq_reduction='sum' 与原文 Fre_loss 一致。
    """

    def __init__(
        self,
        weight_t: float = 1.0,
        weight_f: float = 0.1,
        weight_l: float = 1.0,
        weight_h: float = 0.01,
        mask_loss_type: str = "mse",
        freq_mode: str = "haar",
        freq_reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.weight_t = float(weight_t)
        self.weight_f = float(weight_f)
        self.weight_l = float(weight_l)
        self.weight_h = float(weight_h)
        self.mask_loss_type = str(mask_loss_type).lower()
        self.freq_mode = str(freq_mode).lower()
        self.freq_reduction = str(freq_reduction).lower()

    def _low_high_decompose(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if self.freq_mode != "haar":
            low = F.avg_pool2d(x, kernel_size=2, stride=2)
            low_up = F.interpolate(low, size=x.shape[-2:], mode="bilinear", align_corners=False)
            high = x - low_up
            return low_up, high
        return self._haar_low_high(x)

    @staticmethod
    def _haar_dwt2(x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        ll = 0.5 * (x00 + x01 + x10 + x11)
        lh = 0.5 * (x00 - x01 + x10 - x11)
        hl = 0.5 * (x00 + x01 - x10 - x11)
        hh = 0.5 * (x00 - x01 - x10 + x11)
        return ll, lh, hl, hh

    @staticmethod
    def _haar_idwt2(ll: Tensor, lh: Tensor, hl: Tensor, hh: Tensor) -> Tensor:
        b, c, h2, w2 = ll.shape
        x00 = 0.5 * (ll + lh + hl + hh)
        x01 = 0.5 * (ll - lh + hl - hh)
        x10 = 0.5 * (ll + lh - hl - hh)
        x11 = 0.5 * (ll - lh - hl + hh)
        out = torch.zeros((b, c, h2 * 2, w2 * 2), device=ll.device, dtype=ll.dtype)
        out[:, :, 0::2, 0::2] = x00
        out[:, :, 0::2, 1::2] = x01
        out[:, :, 1::2, 0::2] = x10
        out[:, :, 1::2, 1::2] = x11
        return out

    def _haar_low_high(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h, w = x.shape[-2:]
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        ll, _, _, hh = self._haar_dwt2(x)
        z = torch.zeros_like(ll)
        low = self._haar_idwt2(ll, z, z, z)
        high = self._haar_idwt2(z, z, z, hh)
        if pad_h or pad_w:
            low = low[:, :, :h, :w]
            high = high[:, :, :h, :w]
        return low, high

    def _mask_losses(self, output_t: Tensor, output_f: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        if self.mask_loss_type == "bce":
            loss_t = F.binary_cross_entropy_with_logits(output_t, y)
            loss_f = -F.binary_cross_entropy_with_logits(output_f, y)
            return loss_t, loss_f
        loss_t = F.mse_loss(output_t, y)
        loss_f = F.mse_loss(output_f, -y)
        return loss_t, loss_f

    def forward(
        self,
        logits: Tensor,
        clean_img: Tensor,
        adv_img: Tensor,
        *,
        y_target: float = -1.0,
        mask_threshold: float = 0.0,
    ) -> Dict[str, Tensor]:
        y = torch.ones_like(logits) * float(y_target)
        y_bin = y.bool()
        mask = logits > mask_threshold
        attacked = mask == y_bin

        # 与 DarkSAM-main darksam_attack.py 一致：output = attacked*logits, output_f = ~attacked*logits
        output_t = attacked.float() * logits
        output_f = (~attacked).float() * logits

        loss_t, loss_f = self._mask_losses(output_t, output_f, y)

        clean_low, clean_high = self._low_high_decompose(clean_img)
        adv_low, adv_high = self._low_high_decompose(adv_img)
        reduction = self.freq_reduction if self.freq_reduction in ("mean", "sum") else "mean"
        loss_l = F.mse_loss(adv_low, clean_low, reduction=reduction)
        loss_h = -F.mse_loss(adv_high, clean_high, reduction=reduction)

        total = (
            self.weight_t * loss_t
            + self.weight_f * loss_f
            + self.weight_l * loss_l
            + self.weight_h * loss_h
        )
        return {
            "loss_darksam_total": total,
            "loss_darksam_t": loss_t,
            "loss_darksam_f": loss_f,
            "loss_darksam_low": loss_l,
            "loss_darksam_high": loss_h,
        }


class AdvClipLossModule(nn.Module):
    """
    SAM3 特征空间的 AdvCLIP 损失，与 AdvCLIP-master advclip.py 损失一致。
    recon + alpha*adv_loss + delta*umap；adv_loss=adv_loss1+beta*adv_loss2；InfoNCE/UMAP/CE 与 utils/nce.py、utils/metrics.py 一致。
    """

    def __init__(
        self,
        alpha: float = 10.0,
        beta: float = 5.0,
        gamma: float = 5.0,
        delta: float = 1.0,
        metric_mode: str = "original",
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.metric_mode = str(metric_mode).lower()
        self.temperature = float(temperature)

    @staticmethod
    def _cosine_term(a: Tensor, b: Tensor) -> Tensor:
        a_flat = a.flatten(1)
        b_flat = b.flatten(1)
        return F.cosine_similarity(a_flat, b_flat, dim=1).mean()

    @staticmethod
    def _flatten_feat(x: Tensor) -> Tensor:
        return x if x.dim() == 2 else x.flatten(1)

    def _info_nce(self, query: Tensor, positive: Tensor) -> Tensor:
        q = F.normalize(self._flatten_feat(query), dim=-1)
        p = F.normalize(self._flatten_feat(positive), dim=-1)
        logits = q @ p.transpose(-2, -1)
        labels = torch.arange(q.shape[0], device=q.device)
        return F.cross_entropy(logits / self.temperature, labels, reduction="mean")

    @staticmethod
    def _ce_prob(p: Tensor, q: Tensor, eps: float = 1e-7) -> Tensor:
        d1 = p * ((p + eps).log() - (q + eps).log())
        d2 = (1 - p) * (((1 - p) + eps).log() - ((1 - q) + eps).log())
        return torch.sum(d1 + d2)

    def _umap(self, output_net: Tensor, target_net: Tensor, eps: float = 1e-7) -> Tensor:
        output_net = self._flatten_feat(output_net)
        target_net = self._flatten_feat(target_net)
        n = output_net.shape[0]
        out = output_net / (torch.norm(output_net, dim=1, keepdim=True) + eps)
        tgt = target_net / (torch.norm(target_net, dim=1, keepdim=True) + eps)
        out = torch.nan_to_num(out, nan=0.0)
        tgt = torch.nan_to_num(tgt, nan=0.0)

        model_similarity = out @ out.transpose(0, 1)
        model_distance = 1.0 - model_similarity
        model_distance[range(n), range(n)] = 3.0
        model_distance = model_distance - torch.min(model_distance, dim=1)[0].view(-1, 1)
        model_distance[range(n), range(n)] = 0.0
        model_similarity = 1.0 - model_distance

        target_similarity = tgt @ tgt.transpose(0, 1)
        target_distance = 1.0 - target_similarity
        target_distance[range(n), range(n)] = 3.0
        target_distance = target_distance - torch.min(target_distance, dim=1)[0].view(-1, 1)
        target_distance[range(n), range(n)] = 0.0
        target_similarity = 1.0 - target_distance

        model_similarity = (model_similarity + 1.0) / 2.0
        target_similarity = (target_similarity + 1.0) / 2.0
        model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
        target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)
        return self._ce_prob(target_similarity, model_similarity, eps=eps)

    def forward(
        self,
        adv_feature: Tensor,
        clean_feature: Tensor,
        adv_img: Tensor,
        clean_img: Tensor,
        *,
        text_feature: Optional[Tensor] = None,
        target_feature: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        clean_feature = _expand_like_ref(clean_feature, adv_feature)
        if text_feature is not None:
            text_feature = _expand_like_ref(text_feature, adv_feature)
        if target_feature is not None:
            target_feature = _expand_like_ref(target_feature, adv_feature)

        if self.metric_mode == "cosine":
            adv_loss_pos1 = self._cosine_term(clean_feature, adv_feature)
            if text_feature is not None:
                adv_loss_pos2 = self._cosine_term(adv_feature, text_feature)
            elif target_feature is not None:
                adv_loss_pos2 = self._cosine_term(adv_feature, target_feature)
            else:
                adv_loss_pos2 = torch.zeros_like(adv_loss_pos1)
            adv_loss1 = -adv_loss_pos1
            adv_loss2 = -adv_loss_pos2
        else:
            adv_loss1 = -self._info_nce(clean_feature, adv_feature)
            if text_feature is not None:
                adv_loss2 = -self._info_nce(adv_feature, text_feature)
            elif target_feature is not None:
                adv_loss2 = -self._info_nce(adv_feature, target_feature)
            else:
                adv_loss2 = torch.zeros_like(adv_loss1)

        adv_loss = adv_loss1 + self.beta * adv_loss2
        reconstruction_loss = F.mse_loss(adv_img, clean_img)

        if self.metric_mode == "cosine":
            umap_loss_pos1 = -self._cosine_term(clean_feature, adv_feature)
            if text_feature is not None:
                umap_loss_pos2 = -self._cosine_term(adv_feature, text_feature)
            elif target_feature is not None:
                umap_loss_pos2 = -self._cosine_term(adv_feature, target_feature)
            else:
                umap_loss_pos2 = torch.zeros_like(umap_loss_pos1)
            umap_loss = umap_loss_pos1 + self.gamma * umap_loss_pos2
        else:
            umap_loss_pos1 = -self._umap(clean_feature, adv_feature)
            if text_feature is not None:
                umap_loss_pos2 = -self._umap(adv_feature, text_feature)
            elif target_feature is not None:
                umap_loss_pos2 = -self._umap(adv_feature, target_feature)
            else:
                umap_loss_pos2 = torch.zeros_like(umap_loss_pos1)
            umap_loss = umap_loss_pos1 + self.gamma * umap_loss_pos2

        total = reconstruction_loss + self.alpha * adv_loss + self.delta * umap_loss
        return {
            "loss_advclip_total": total,
            "loss_advclip_recon": reconstruction_loss,
            "loss_advclip_adv": adv_loss,
            "loss_advclip_umap": umap_loss,
            "loss_advclip_adv1": adv_loss1,
            "loss_advclip_adv2": adv_loss2,
        }

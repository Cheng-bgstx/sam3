import os
from typing import List, Optional

import torch
from iopath.common.file_io import g_pathmgr


def load_uap_tensor(uap_path: Optional[str], device: torch.device) -> Optional[torch.Tensor]:
    if not uap_path or not str(uap_path).strip():
        return None
    try:
        if os.path.isfile(uap_path):
            with open(uap_path, "rb") as f:
                uap = torch.load(f, map_location="cpu", weights_only=True)
        elif g_pathmgr.exists(uap_path):
            with g_pathmgr.open(uap_path, "rb") as f:
                uap = torch.load(f, map_location="cpu", weights_only=True)
        else:
            return None
    except Exception:
        return None

    if isinstance(uap, dict):
        uap = uap.get("perturbation", uap.get("uap", next(iter(uap.values()))))
    uap = uap.to(device)
    if uap.dim() == 3:
        uap = uap.unsqueeze(0)
    if not (uap.dim() == 4 and uap.shape[1] == 3):
        raise AssertionError(f"UAP expected shape (1,3,H,W), got {tuple(uap.shape)}")
    return uap


def apply_uap_to_normalized_batch(
    img_batch: torch.Tensor,
    uap_tensor: Optional[torch.Tensor],
    norm_mean: List[float],
    norm_std: List[float],
    *,
    uap_scale: float = 1.0,
) -> torch.Tensor:
    if uap_tensor is None:
        return img_batch

    uap = uap_tensor
    if img_batch.shape[-2:] != uap.shape[-2:]:
        uap = torch.nn.functional.interpolate(
            uap, size=img_batch.shape[-2:], mode="bilinear", align_corners=False
        )

    mean = torch.tensor(norm_mean, device=img_batch.device, dtype=img_batch.dtype).view(1, 3, 1, 1)
    std = torch.tensor(norm_std, device=img_batch.device, dtype=img_batch.dtype).view(1, 3, 1, 1)

    img_01 = img_batch * std + mean
    img_adv_01 = (img_01 + uap * float(uap_scale)).clamp(0.0, 1.0)
    return (img_adv_01 - mean) / std

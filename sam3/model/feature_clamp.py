"""
Feature Statistics Clamping (FSC): Architecture-native defense.

Core insight: UAP perturbation pushes FPN features outside the clean-data
manifold. By clamping each feature channel to its clean-data range [μ-kσ, μ+kσ],
we can suppress perturbation-induced feature deviations.

Key properties:
- Zero adversarial training (clean-data statistics only)
- Non-differentiable at boundary (BPDA gradient gap)
- Architecture-native (exploits SAM3's FPN feature structure)
- Attack-agnostic
- Dataset-agnostic (calibrate on any clean data)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class FeatureClampDefense(nn.Module):
    """Clamp FPN features to clean-data per-channel statistics.

    Args:
        mins: List of [d_model] tensors, per-scale lower bounds
        maxs: List of [d_model] tensors, per-scale upper bounds
    """
    def __init__(self, mins: List[torch.Tensor], maxs: List[torch.Tensor]):
        super().__init__()
        self.mins = nn.ParameterList([
            nn.Parameter(m, requires_grad=False) for m in mins
        ])
        self.maxs = nn.ParameterList([
            nn.Parameter(m, requires_grad=False) for m in maxs
        ])

    def forward(self, fpn_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Clamp each scale's features to clean-data range."""
        clamped = []
        for i, f in enumerate(fpn_features):
            # mins[i], maxs[i]: [d_model] → reshape to [1, C, 1, 1]
            lo = self.mins[i].view(1, -1, 1, 1).to(f.device)
            hi = self.maxs[i].view(1, -1, 1, 1).to(f.device)
            # Non-differentiable clamp at boundary
            f_clamped = torch.clamp(f, lo, hi)
            clamped.append(f_clamped)
        return clamped


def calibrate_feature_clamp(sf, dataset, n_samples=100, k=3.0, device='cuda:0'):
    """Calibrate per-channel feature ranges from clean data.

    Args:
        sf: Sam3VideoForwarder
        dataset: List of (img_np, anno, gt_np, info) tuples
        n_samples: Number of clean samples for calibration
        k: Number of standard deviations for the clamp range

    Returns:
        mins, maxs: Per-scale [d_model] tensors
    """
    import random
    import numpy as np
    from tqdm import tqdm

    neck = sf.model.detector.backbone.vision_backbone
    trunk = neck.trunk
    trunk.return_interm_layers = True
    d_model = neck.convs[0][-1].out_channels  # Last conv is 3x3 conv with d_model out
    n_scales = len(neck.convs)
    dn = lambda x: sf.denorm_image(x)

    # Collect features
    all_feats = [[] for _ in range(n_scales)]

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    n_done = 0
    for idx in indices:
        if n_done >= n_samples:
            break

        img_np, _, gt_np, _ = dataset[idx]
        if gt_np is None or gt_np.sum() < 100:
            continue

        X = sf.transform_image(img_np).to(device)
        clean = dn(X)
        if clean.dim() == 3:
            clean = clean.unsqueeze(0)

        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                xs = trunk(clean)
                x = xs[-1]
                for i, conv in enumerate(neck.convs):
                    f = conv(x).float()
                    all_feats[i].append(f.cpu())

        n_done += 1
        if n_done % 20 == 0:
            print(f"  [{n_done}/{n_samples}]", flush=True)

    # Compute per-channel statistics
    mins, maxs = [], []
    for i in range(n_scales):
        feats = torch.cat([f.flatten(2) for f in all_feats[i]], dim=2)  # [1, C, total_spatial]
        # Remove batch dim
        feats = feats.squeeze(0)  # [C, total_spatial]
        mu = feats.mean(dim=1)  # [C]
        std = feats.std(dim=1)  # [C]
        lo = mu - k * std
        hi = mu + k * std
        mins.append(lo)
        maxs.append(hi)
        frac_within = ((feats >= lo.unsqueeze(1)) & (feats <= hi.unsqueeze(1))).float().mean().item()
        print(f"  Scale {i}: mu={mu.mean():.3f}, std={std.mean():.3f}, "
              f"frac in [{lo.mean():.3f}, {hi.mean():.3f}] = {frac_within:.3f}", flush=True)

    trunk.return_interm_layers = False
    return mins, maxs


def apply_feature_clamping(sf, mins, maxs):
    """Apply feature clamping to the model's FPN neck.

    This patches the neck's forward to clamp features after FPN convs.
    """
    neck = sf.model.detector.backbone.vision_backbone
    original_forward = neck.forward

    # Create clamp module
    clamp = FeatureClampDefense(mins, maxs)

    def clamped_forward(self, tensor_list):
        xs = self.trunk(tensor_list)
        sam3_out, sam3_pos = [], []
        sam2_out, sam2_pos = None, None
        if self.sam2_convs is not None:
            sam2_out, sam2_pos = [], []

        if self.mlfa_enabled and len(xs) > 1:
            x = sum(xs) / len(xs)
        else:
            x = xs[-1]

        for i in range(len(self.convs)):
            sam3_x_out = self.convs[i](x)
            sam3_pos_out = self.position_encoding(sam3_x_out).to(sam3_x_out.dtype)
            sam3_out.append(sam3_x_out)
            sam3_pos.append(sam3_pos_out)

            if self.sam2_convs is not None:
                sam2_x_out = self.sam2_convs[i](x)
                sam2_pos_out = self.position_encoding(sam2_x_out).to(sam2_x_out.dtype)
                sam2_out.append(sam2_x_out)
                sam2_pos.append(sam2_pos_out)

        # FSC: Clamp features to clean-data range (non-differentiable at boundary)
        sam3_out = clamp(sam3_out)

        # CSFC fusion
        sam3_out = self._csfc_fusion(sam3_out)

        # FBD denoising
        if self.fbd_enabled and self.fbd_denoiser is not None:
            sam3_out = self.fbd_denoiser(sam3_out)

        return sam3_out, sam3_pos, sam2_out, sam2_pos

    import types
    neck.forward = types.MethodType(clamped_forward, neck)
    neck._fsc_clamp = clamp

    return neck

"""
Multi-Path Feature Pyramid (MPFP): Architecture-native defense.

Root vulnerability: SAM3's Sam3DualViTDetNeck creates ALL 4 FPN scales from a
SINGLE ViT output (xs[-1], block 31). This is a single point of failure —
perturbation amplified through 4 global attention blocks (7, 15, 23, 31)
corrupts ALL scales simultaneously.

MPFP: Route DIFFERENT ViT global-attention-block outputs to DIFFERENT FPN scales.
- Block 7 (earliest global attn, least perturbed) → finest scale (stride 4)
- Block 15 → scale stride 2
- Block 23 → scale stride 1
- Block 31 (latest, most semantic) → coarsest scale (stride 0.5)

This eliminates the single point of failure: an attacker must corrupt ALL 4
ViT depths to affect ALL FPN scales. Shallow blocks have fewer perturbation
amplification stages, providing inherent robustness.

Zero learnable parameters. Zero adversarial training. Architecture-native.
"""

import torch
import types
from typing import List


def apply_mpfp(sf, depth_mapping=None):
    """Apply Multi-Path Feature Pyramid defense.

    Enables return_interm_layers on the ViT trunk and patches the neck's
    forward to route different ViT depths to different FPN scales.

    Args:
        sf: Sam3VideoForwarder
        depth_mapping: Optional dict mapping conv_index → xs_index.
                       Default: {0:0, 1:1, 2:2, 3:3} (fine→early, coarse→late)

    Returns:
        neck: The modified neck module
    """
    if depth_mapping is None:
        # Fine scales → early blocks (less perturbed)
        # Coarse scales → late blocks (more semantic)
        depth_mapping = {0: 0, 1: 1, 2: 2, 3: 3}

    neck = sf.model.detector.backbone.vision_backbone
    trunk = neck.trunk

    # Enable intermediate layer outputs (4 global attn blocks)
    trunk.return_interm_layers = True
    print(f"[MPFP] Enabled return_interm_layers=True on ViT trunk")
    print(f"[MPFP] Depth mapping (conv_idx → xs_idx): {depth_mapping}")
    print(f"[MPFP] Full attn block indices: {trunk.full_attn_ids}")

    original_forward = neck.forward

    def mpfp_forward(self, tensor_list: List[torch.Tensor]):
        xs = self.trunk(tensor_list)
        # xs now has 4 elements: [block_7, block_15, block_23, block_31]

        sam3_out, sam3_pos = [], []
        sam2_out, sam2_pos = None, None
        if self.sam2_convs is not None:
            sam2_out, sam2_pos = [], []

        for i in range(len(self.convs)):
            # Each FPN scale gets its own ViT depth input
            src_idx = depth_mapping.get(i, min(i, len(xs) - 1))
            x = xs[src_idx]

            sam3_x_out = self.convs[i](x)
            sam3_pos_out = self.position_encoding(sam3_x_out).to(sam3_x_out.dtype)
            sam3_out.append(sam3_x_out)
            sam3_pos.append(sam3_pos_out)

            if self.sam2_convs is not None:
                sam2_x_out = self.sam2_convs[i](x)
                sam2_pos_out = self.position_encoding(sam2_x_out).to(sam2_x_out.dtype)
                sam2_out.append(sam2_x_out)
                sam2_pos.append(sam2_pos_out)

        # CSFC: Cross-Scale Feature Consensus fusion
        sam3_out = self._csfc_fusion(sam3_out)

        # FBD: Feature Bottleneck Denoising
        if self.fbd_enabled and self.fbd_denoiser is not None:
            sam3_out = self.fbd_denoiser(sam3_out)

        return sam3_out, sam3_pos, sam2_out, sam2_pos

    neck.forward = types.MethodType(mpfp_forward, neck)
    neck._mpfp_enabled = True
    neck._mpfp_depth_mapping = depth_mapping

    return neck


def apply_mpfp_with_rectification(sf, block_mins, block_maxs, k=2.0):
    """Apply MPFP with Cross-Depth Feature Rectification (CDFR).

    In addition to MPFP routing, this rectifies late-block features using
    early-block statistics: if a late-block feature channel deviates beyond
    its clean-data range [μ-kσ, μ+kσ], replace with the corresponding
    early-block feature channel (which is less perturbed).

    This provides a second layer of defense: not only are scales separated,
    but heavily perturbed channels are identified and replaced.

    Args:
        sf: Sam3VideoForwarder
        block_mins: List of [d_model] tensors, per-block lower bounds
        block_maxs: List of [d_model] tensors, per-block upper bounds
        k: Number of std deviations for detection threshold

    Returns:
        neck: The modified neck module
    """
    neck = apply_mpfp(sf)

    trunk = neck.trunk
    full_attn_ids = trunk.full_attn_ids
    n_blocks = len(full_attn_ids)

    # Store rectification params
    neck._mpfp_rectification = {
        'block_mins': block_mins,
        'block_maxs': block_maxs,
        'k': k,
    }

    original_forward = neck.forward

    def mpfp_rect_forward(self, tensor_list):
        xs = self.trunk(tensor_list)

        sam3_out, sam3_pos = [], []
        sam2_out, sam2_pos = None, None
        if self.sam2_convs is not None:
            sam2_out, sam2_pos = [], []

        depth_mapping = self._mpfp_depth_mapping
        rect = self._mpfp_rectification

        # Compute unclamped features first
        raw_features = []
        for i in range(len(self.convs)):
            src_idx = depth_mapping.get(i, min(i, len(xs) - 1))
            x = xs[src_idx]
            f = self.convs[i](x)
            raw_features.append(f)

        # CDFR: For scales that use late blocks (src_idx >= 2), detect
        # anomalous channels and replace with early-block counterpart
        for i in range(len(raw_features)):
            src_idx = depth_mapping.get(i, min(i, len(xs) - 1))
            if src_idx >= 2:
                # This scale uses a late (perturbed) block
                # Check against clean-data statistics
                lo = rect['block_mins'][src_idx].view(1, -1, 1, 1).to(raw_features[i].device)
                hi = rect['block_maxs'][src_idx].view(1, -1, 1, 1).to(raw_features[i].device)
                # Channels within range: keep; outside: flag
                anomalous = ((raw_features[i] < lo) | (raw_features[i] > hi)).float()
                # Use spatial mean of anomalies as channel-wise weight
                anomaly_score = anomalous.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
                # Fall back to early-block counterpart (index 0) for anomalous channels
                # Compute early-block features through the same conv
                early_x = xs[0]
                early_f = self.convs[i](early_x)
                # Blend: weight toward early_f for anomalous channels
                alpha = torch.clamp(anomaly_score * rect['k'], 0.0, 0.9)
                raw_features[i] = (1 - alpha) * raw_features[i] + alpha * early_f

            sam3_x_out = raw_features[i]
            sam3_pos_out = self.position_encoding(sam3_x_out).to(sam3_x_out.dtype)
            sam3_out.append(sam3_x_out)
            sam3_pos.append(sam3_pos_out)

            if self.sam2_convs is not None:
                sam2_x_out = self.sam2_convs[i](xs[depth_mapping.get(i, min(i, len(xs) - 1))])
                sam2_pos_out = self.position_encoding(sam2_x_out).to(sam2_x_out.dtype)
                sam2_out.append(sam2_x_out)
                sam2_pos.append(sam2_pos_out)

        sam3_out = self._csfc_fusion(sam3_out)

        if self.fbd_enabled and self.fbd_denoiser is not None:
            sam3_out = self.fbd_denoiser(sam3_out)

        return sam3_out, sam3_pos, sam2_out, sam2_pos

    neck.forward = types.MethodType(mpfp_rect_forward, neck)
    neck._mpfp_rectification_enabled = True

    return neck


def calibrate_mpfp_rectification(sf, dataset, n_samples=100, k=3.0, device='cuda:0'):
    """Calibrate per-block feature statistics for CDFR.

    Collects features at each global attention block from clean data,
    computes per-channel μ, σ, and stores [μ-kσ, μ+kσ] ranges.

    Args:
        sf: Sam3VideoForwarder
        dataset: List of (img_np, anno, gt_np, info) tuples
        n_samples: Number of clean samples
        k: Number of std deviations

    Returns:
        block_mins, block_maxs: Per-block [d_model] tensors
    """
    import random
    import numpy as np
    from tqdm import tqdm

    neck = sf.model.detector.backbone.vision_backbone
    trunk = neck.trunk
    trunk.return_interm_layers = True

    full_attn_ids = trunk.full_attn_ids
    d_model = neck.convs[0][-1].out_channels
    n_blocks = len(full_attn_ids)

    dn = lambda x: sf.denorm_image(x)
    all_feats = {idx: [] for idx in range(n_blocks)}

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
                for i in range(n_blocks):
                    all_feats[i].append(xs[i].float().cpu())

        n_done += 1
        if n_done % 20 == 0:
            print(f"  [MPFP-CDFR calib] {n_done}/{n_samples}", flush=True)

    block_mins, block_maxs = [], []
    for i in range(n_blocks):
        feats = torch.cat([f.flatten(2) for f in all_feats[i]], dim=2)
        feats = feats.squeeze(0)
        mu = feats.mean(dim=1)
        std = feats.std(dim=1)
        lo = mu - k * std
        hi = mu + k * std
        block_mins.append(lo)
        block_maxs.append(hi)
        frac = ((feats >= lo.unsqueeze(1)) & (feats <= hi.unsqueeze(1))).float().mean().item()
        print(f"  Block {full_attn_ids[i]}: mu={mu.mean():.3f}, "
              f"frac in range = {frac:.3f}", flush=True)

    trunk.return_interm_layers = False
    return block_mins, block_maxs

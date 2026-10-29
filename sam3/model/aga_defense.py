"""
Anomaly-Gated Attention (AGA): Architecture-native defense.

Core insight: In SAM3's global attention blocks, perturbed (anomalous) patches
can corrupt clean patches through the attention mechanism. AGA detects anomalous
patches using clean-data feature statistics and GATES THEM OUT in the attention
computation — they contribute ZERO to the output.

The gate is a HARD threshold (non-differentiable), creating a strong gradient
gap for BPDA attackers. The attacker gets zero gradient through gated-out
patches, forcing perturbations to stay within the clean-data manifold.

Key properties:
- Architecture-native (modifies ViT attention mechanism)
- Zero adversarial training (clean-data statistics only)
- Non-differentiable gate → BPDA gradient gap
- Attack-agnostic
- Dataset-agnostic (calibrate on any clean data)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import types
from typing import List, Optional, Dict

# Import from vitdet for the patched attention forward
from sam3.model.vitdet import concat_rel_pos


def calibrate_aga(sf, dataset, n_samples=100, k=3.0, device='cuda:0'):
    """Calibrate per-block feature statistics for AGA.

    Collects input features to each global attention block from clean data,
    computes per-channel μ, σ for anomaly detection.

    Args:
        sf: Sam3VideoForwarder
        dataset: List of (img_np, anno, gt_np, info) tuples
        n_samples: Number of clean samples
        k: Anomaly threshold (in standard deviations)
        device: GPU device

    Returns:
        block_stats: Dict mapping global_attn_block_index → (mu, sigma)
    """
    import random
    import numpy as np
    from tqdm import tqdm

    neck = sf.model.detector.backbone.vision_backbone
    trunk = neck.trunk
    full_attn_ids = trunk.full_attn_ids

    dn = lambda x: sf.denorm_image(x)

    # Collect features at input to each global attention block
    all_feats = {idx: [] for idx in full_attn_ids}

    def make_hook(idx):
        def hook_fn(module, input_tensor):
            x = input_tensor[0] if isinstance(input_tensor, tuple) else input_tensor
            all_feats[idx].append(x.detach().cpu())
        return hook_fn

    hooks = []
    for i, blk in enumerate(trunk.blocks):
        if i in full_attn_ids:
            hooks.append(blk.register_forward_pre_hook(make_hook(i)))

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
                _ = trunk(clean)

        n_done += 1
        if n_done % 20 == 0:
            print(f"  [AGA calib] {n_done}/{n_samples}", flush=True)

    for h in hooks:
        h.remove()

    # Compute per-channel statistics per block
    block_stats = {}
    for idx in full_attn_ids:
        feats = all_feats[idx]
        if len(feats) == 0:
            print(f"  WARNING: No features collected for block {idx}")
            continue
        cat = torch.cat([f.reshape(-1, f.shape[-1]) for f in feats], dim=0)  # [total_N, C]
        mu = cat.mean(dim=0)  # [C]
        sigma = cat.std(dim=0)  # [C]
        block_stats[idx] = (mu, sigma)
        frac_within = ((cat >= mu - k*sigma) & (cat <= mu + k*sigma)).float().mean().item()
        print(f"  Block {idx}: mu={mu.mean():.3f}, sigma={sigma.mean():.3f}, "
              f"frac in [{mu.mean()-k*sigma.mean():.3f}, {mu.mean()+k*sigma.mean():.3f}] = {frac_within:.3f}", flush=True)

    return block_stats


def apply_aga(sf, block_stats, k=3.0, gate_mode='key_value'):
    """Apply Anomaly-Gated Attention to global attention blocks.

    Patches the Attention.forward of each global attention block to:
    1. Compute per-patch anomaly score from input features
    2. Gate out anomalous patches in the attention weights
    3. The gate is HARD (0/1) and non-differentiable

    Args:
        sf: Sam3VideoForwarder
        block_stats: Dict from calibrate_aga()
        k: Anomaly threshold (in standard deviations)
        gate_mode: 'key_value' (gate K and V), 'query' (gate Q), or 'all'

    Returns:
        vit: The ViT module with patched attention
    """
    vit = sf.model.detector.backbone.vision_backbone.trunk
    full_attn_ids = vit.full_attn_ids

    n_patched = 0
    for i, blk in enumerate(vit.blocks):
        if i not in full_attn_ids or i not in block_stats:
            continue

        mu, sigma = block_stats[i]
        mu = mu.cuda()
        sigma = sigma.cuda()

        original_attn_forward = blk.attn.forward
        blk._aga_block_idx = i
        blk._aga_mu = mu
        blk._aga_sigma = sigma
        blk._aga_k = k
        blk._aga_gate_mode = gate_mode

        def make_aga_forward(orig_fn, blk_ref):
            def aga_forward(self, x):
                # x: [B, H, W, C] (spatial) or [B, N, C] (flat)
                # This mirrors the original Attention.forward exactly,
                # adding AGA gating on K and V.
                mu_dev = blk_ref._aga_mu.to(x.device)
                sigma_dev = blk_ref._aga_sigma.to(x.device)
                k_val = blk_ref._aga_k
                mode = blk_ref._aga_gate_mode
                s = 1 if self.cls_token else 0

                if x.ndim == 4:
                    B, H, W, _ = x.shape
                    assert s == 0
                    L = H * W
                    ndim = 4
                else:
                    assert x.ndim == 3
                    B, L, _ = x.shape
                    ndim = 3
                    H = W = int((L - s) ** 0.5)

                # Compute per-patch anomaly score for gating
                if ndim == 4:
                    x_flat_for_gate = x.reshape(B, L, -1)
                else:
                    x_flat_for_gate = x
                sigma_safe = sigma_dev.clamp(min=1e-6)
                z_score = (x_flat_for_gate.float() - mu_dev) / sigma_safe
                anomaly_score = z_score.abs().max(dim=-1)[0]  # [B, L]
                gate = (anomaly_score <= k_val).float().detach()  # [B, L]
                gate = gate.unsqueeze(1).unsqueeze(-1)  # [B, 1, L, 1]

                # Original qkv processing (exact match to Attention.forward)
                qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, -1)
                q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
                # q, k, v: [B, nHead, L, C_per_head]

                if self.use_rope:
                    q, k = self._apply_rope(q, k)

                if self.use_rel_pos:
                    q, k = concat_rel_pos(
                        q.flatten(0, 1),
                        k.flatten(0, 1),
                        (H, W),
                        x.shape[1:3],
                        self.rel_pos_h,
                        self.rel_pos_w,
                        rescale=True,
                        relative_coords=self.relative_coords,
                    )
                    q = q.reshape(B, self.num_heads, H * W, -1)
                    k = k.reshape(B, self.num_heads, H * W, -1)

                # AGA: Gate anomalous patches in K and V
                if mode in ('key_value', 'all'):
                    k = k * gate
                    v = v * gate
                if mode in ('query', 'all'):
                    q = q * gate

                x_out = F.scaled_dot_product_attention(q, k, v)

                if ndim == 4:
                    x_out = (
                        x_out.view(B, self.num_heads, H, W, -1)
                        .permute(0, 2, 3, 1, 4)
                        .reshape(B, H, W, -1)
                    )
                else:
                    x_out = x_out.view(B, self.num_heads, L, -1).permute(0, 2, 1, 3).reshape(B, L, -1)

                x_out = self.proj(x_out)
                return x_out

            return aga_forward

        blk.attn.forward = types.MethodType(make_aga_forward(original_attn_forward, blk), blk.attn)
        n_patched += 1

    print(f"[AGA] Patched {n_patched} global attention blocks "
          f"(k={k}, mode={gate_mode})")
    return vit


def apply_mpfp_aga(sf, depth_mapping=None, k=3.0):
    """Apply combined MPFP + AGA defense.

    MPFP routes different ViT depths to different FPN scales.
    AGA gates out anomalous patches in global attention.

    These are complementary:
    - MPFP creates multi-path redundancy (harder to corrupt all paths)
    - AGA blocks perturbation propagation through anomalous patches
    - AGA's non-differentiable gate creates BPDA gradient gap
    - MPFP ensures even if some depths are corrupted, others are clean

    Args:
        sf: Sam3VideoForwarder
        depth_mapping: Dict mapping conv_idx → xs_idx
        k: AGA anomaly threshold

    Returns:
        neck: The modified neck module
    """
    from sam3.model.mpfp import apply_mpfp
    import random, numpy as np

    # First apply MPFP
    neck = apply_mpfp(sf, depth_mapping=depth_mapping)

    # Then calibrate AGA using the current model state
    # For now, require pre-calibrated block_stats
    # We'll add calibration in the eval script

    return neck

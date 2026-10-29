"""
High-Pass Residual Attention (HPRA): Frequency-domain architecture-native defense.

Core insight: UAP perturbations are predominantly LOW-FREQUENCY — smooth patterns
that vary slowly across spatial positions. SAM3's 4 global attention blocks
(indices 7, 15, 23, 31) amplify perturbation by allowing EVERY token to attend
to EVERY other token, propagating the low-frequency UAP pattern globally.

HPRA applies a high-pass filter to the input features of global attention blocks:
    x_hp = x - alpha * local_spatial_mean(x)

This attenuates low-frequency (UAP-correlated) components BEFORE attention
computation, preventing perturbation amplification through global attention.

Two non-differentiable mechanisms compound:
1. Randomized alpha — each forward pass uses a different attenuation strength,
   so the BPDA gradient (computed with one alpha) doesn't match the next pass
2. Quantized high-pass — the filtered features are quantized after filtering,
   creating a second gradient gap through round()

Why this is NOVEL:
- No existing defense uses frequency-domain gating INSIDE ViT attention
- Directly targets the FREQUENCY CHARACTERISTIC of UAPs (low-frequency)
- Architecture-native (modifies attention computation, not input preprocessing)
- Zero training (fixed/randomized filter parameters)
- Attack-agnostic (based on perturbation physics, not attack specifics)
- Dataset-agnostic

Properties:
- O(N) overhead (one avg_pool per attention block)
- Complements SPE (SPE protects input→ViT, HPRA protects ViT→FPN)
"""
import types
import random
import math
import torch
import torch.nn.functional as F


def make_hpra_attention_forward(attn_module, alpha_mean=0.3, alpha_std=0.1,
                                  quantize=False, n_bits=8):
    """Create patched forward with high-pass residual filtering.

    Args:
        attn_module: The Attention module to patch
        alpha_mean: Mean attenuation strength (0=off, 1=full high-pass)
        alpha_std: Std of random alpha variation
        quantize: Whether to quantize high-pass features (adds gradient gap)
        n_bits: Quantization bit depth (only if quantize=True)
    """

    def hpra_forward(self, x):
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
            H = W = int(math.sqrt(L - s))

        # === HPRA: High-Pass Residual Filtering ===
        # Reshape to spatial for local mean computation
        if s == 1:
            # Has cls token — handle separately
            x_cls = x[:, :1, :]
            x_spatial = x[:, 1:, :].reshape(B, H, W, -1)
        else:
            x_spatial = x.reshape(B, H, W, -1)

        # Compute local spatial mean (low-frequency component)
        # avg_pool2d operates on [B, C, H, W]
        x_spatial_nchw = x_spatial.permute(0, 3, 1, 2)  # [B, C, H, W]
        local_mean = F.avg_pool2d(x_spatial_nchw, kernel_size=3, padding=1, stride=1)
        local_mean = local_mean.permute(0, 2, 3, 1)  # [B, H, W, C]

        # Randomized alpha (non-differentiable)
        alpha = alpha_mean + random.uniform(-alpha_std, alpha_std)
        alpha = max(0.0, min(1.0, alpha))

        # High-pass: remove low-frequency component
        x_hp_spatial = x_spatial - alpha * local_mean

        # Optional quantization for additional gradient gap
        if quantize:
            n_levels = 2 ** n_bits - 1
            x_hp_spatial = (x_hp_spatial * n_levels).round() / n_levels

        # Restore original format
        if s == 1:
            x_hp = torch.cat([x_cls, x_hp_spatial.reshape(B, L - 1, -1)], dim=1)
        else:
            x_hp = x_hp_spatial.reshape(B, L, -1)

        # Standard attention computation using high-pass features
        qkv = self.qkv(x_hp).reshape(B, L, 3, self.num_heads, -1)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        q, k = self._apply_rope(q, k)
        if self.use_rel_pos:
            from sam3.model.vitdet import concat_rel_pos
            q, k = concat_rel_pos(
                q.flatten(0, 1), k.flatten(0, 1),
                (H, W), x.shape[1:3] if ndim == 4 else None,
                self.rel_pos_h, self.rel_pos_w,
                rescale=True,
                relative_coords=self.relative_coords,
            )
            q = q.reshape(B, self.num_heads, H * W, -1)
            k = k.reshape(B, self.num_heads, H * W, -1)

        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0
        )

        if ndim == 4:
            x = x.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        else:
            x = x.view(B, self.num_heads, L, -1).permute(0, 2, 1, 3).reshape(B, L, -1)

        x = self.proj(x)
        return x

    return hpra_forward


def apply_hpra(sf, alpha_mean=0.3, alpha_std=0.1, quantize=False, n_bits=8,
               global_only=True):
    """Apply High-Pass Residual Attention to ViT model.

    Patches global attention blocks to apply high-pass filtering before
    attention computation. This attenuates low-frequency UAP components
    and prevents perturbation amplification through global attention.

    Args:
        sf: Sam3VideoForwarder
        alpha_mean: Mean high-pass strength (0.3 recommended — moderate filtering)
        alpha_std: Random variation in alpha per forward pass
        quantize: Whether to quantize filtered features
        n_bits: Quantization bit depth
        global_only: If True, only patch global attention blocks (7, 15, 23, 31)

    Returns:
        n_patched: Number of blocks patched
    """
    vit_model = sf.model.detector.backbone.vision_backbone.trunk
    full_attn_ids = set(vit_model.full_attn_ids)

    n_patched = 0
    for i, block in enumerate(vit_model.blocks):
        if global_only and i not in full_attn_ids:
            continue

        attn = block.attn
        hpra_fn = make_hpra_attention_forward(
            attn,
            alpha_mean=alpha_mean,
            alpha_std=alpha_std,
            quantize=quantize,
            n_bits=n_bits,
        )
        attn.forward = types.MethodType(hpra_fn, attn)
        n_patched += 1

    mode = "quantized" if quantize else "random-alpha"
    print(f"  HPRA: patched {n_patched} blocks "
          f"(alpha={alpha_mean}±{alpha_std}, mode={mode})", flush=True)
    return n_patched


def apply_hpra_with_spe(sf, max_shift=3, n_bits=6,
                          alpha_mean=0.3, alpha_std=0.1):
    """Apply HPRA + SPE combined defense.

    SPE protects at input level (random shift + quantization before ViT).
    HPRA protects inside ViT (high-pass filtering in global attention).

    Two non-differentiable mechanisms at DIFFERENT architectural levels:
    1. Input: random spatial shift + pixel quantization (SPE)
    2. ViT interior: randomized high-pass filtering (HPRA)

    The compounding gradient gaps make BPDA optimization extremely difficult.
    """
    from sam3.model.spe_defense import apply_spe

    apply_spe(sf, max_shift=max_shift, n_bits=n_bits)
    n_hpra = apply_hpra(sf, alpha_mean=alpha_mean, alpha_std=alpha_std,
                         quantize=True, n_bits=n_bits)

    print(f"  HPRA+SPE: SPE(shift={max_shift},bits={n_bits}) "
          f"+ HPRA(alpha={alpha_mean}±{alpha_std}, quantized)", flush=True)
    return n_hpra

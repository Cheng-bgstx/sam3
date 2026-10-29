"""
Dual Gradient Gap Defense: SPE + SDS.

Combines two non-differentiable randomizations at different architectural levels:
1. SPE (Stochastic Patch Embedding): Random shift + quantization at ViT input
2. SDS (Stochastic Depth Selection): Random ViT depth selection at FPN neck

Why dual gaps are necessary:
- SPE alone: attacker fights ONE gradient gap (patch grid). 86.5% retention.
- SDS alone: attacker fights ONE gradient gap (depth selection).
- SPE + SDS: attacker fights TWO INDEPENDENT gradient gaps simultaneously.

The attacker must find a perturbation that:
1. Survives random patch grid shifts (SPE)
2. Works at ALL 4 ViT depths (SDS)
3. Each BPDA step computes gradient through ONE random shift + ONE random depth,
   but the perturbation must generalize to ALL shifts and ALL depths.

This directly addresses SAM3's root vulnerability:
- Single FPN source (xs[-1]) → SDS randomizes the source
- 4 global attention blocks amplify perturbations → SPE breaks perturbation alignment
- No hierarchical redundancy → SDS creates artificial redundancy via randomization
"""

import torch
import types
from typing import Optional


def apply_dual_gap(sf, max_shift=3, n_bits=6):
    """Apply Dual Gradient Gap defense (SPE + SDS).

    Args:
        sf: Sam3VideoForwarder
        max_shift: Maximum random shift in pixels (SPE)
        n_bits: Quantization bits (SPE), None = no quantization

    Returns:
        (vit, neck): The modified ViT and neck modules
    """
    # === Step 1: Apply SPE (Stochastic Patch Embedding) ===
    vit = sf.model.detector.backbone.vision_backbone.trunk
    neck = sf.model.detector.backbone.vision_backbone
    original_forward = vit.forward

    vit._spe_max_shift = max_shift
    vit._spe_n_bits = n_bits
    vit._spe_enabled = True

    def spe_forward(self, x: torch.Tensor):
        if self._spe_enabled and self._spe_max_shift > 0:
            import random
            B, C, H, W = x.shape

            # 1. Random spatial shift (match RPD exactly)
            dx = random.randint(-self._spe_max_shift, self._spe_max_shift)
            dy = random.randint(-self._spe_max_shift, self._spe_max_shift)

            if dx != 0 or dy != 0:
                x_shifted = torch.roll(x, shifts=(dy, dx), dims=(-2, -1))
                with torch.no_grad():
                    if dy > 0:
                        x_shifted[:, :, :dy, :] = x[:, :, :1, :]
                    elif dy < 0:
                        sh = -dy
                        x_shifted[:, :, -sh:, :] = x_shifted[:, :, -sh-1:-sh, :]
                    if dx > 0:
                        x_shifted[:, :, :, :dx] = x_shifted[:, :, :, dx:dx+1]
                    elif dx < 0:
                        sh = -dx
                        x_shifted[:, :, :, -sh:] = x_shifted[:, :, :, -sh-1:-sh]
                x = x_shifted

            # 2. Non-differentiable quantization (match RPD exactly)
            if self._spe_n_bits is not None and self._spe_n_bits > 0:
                n_levels = 2 ** self._spe_n_bits - 1
                x = (x * n_levels).round() / n_levels
                x = torch.clamp(x, 0.0, 1.0)

        # Continue with original ViT forward
        x = self.patch_embed(x)
        h, w = x.shape[1], x.shape[2]

        s = 0
        if self.retain_cls_token:
            x = torch.cat([self.class_embedding, x.flatten(1, 2)], dim=1)
            s = 1

        if self.pos_embed is not None:
            x = x + get_abs_pos(
                self.pos_embed,
                self.pretrain_use_cls_token,
                (h, w),
                self.retain_cls_token,
                tiling=self.tile_abs_pos,
            )

        x = self.ln_pre(x)

        outputs = []
        for i, blk in enumerate(self.blocks):
            if self.use_act_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
            if (i == self.full_attn_ids[-1]) or (
                self.return_interm_layers and i in self.full_attn_ids
            ):
                if i == self.full_attn_ids[-1]:
                    x = self.ln_post(x)

                feats = x[:, s:]
                if feats.ndim == 4:
                    feats = feats.permute(0, 3, 1, 2)
                else:
                    assert feats.ndim == 3
                    import math
                    h_f = w_f = int(math.sqrt(feats.shape[1]))
                    feats = feats.reshape(
                        feats.shape[0], h_f, w_f, feats.shape[-1]
                    ).permute(0, 3, 1, 2)

                outputs.append(feats)

        return outputs

    from sam3.model.vitdet import get_abs_pos
    vit.forward = types.MethodType(spe_forward, vit)

    # === Step 2: Apply SDS (Stochastic Depth Selection) ===
    # Enable return_interm_layers so ViT returns all 4 global attention depths
    vit.return_interm_layers = True
    neck.sds_enabled = True

    print(f"[DualGap] Enabled: SPE(max_shift={max_shift}, n_bits={n_bits}) "
          f"+ SDS(depths={vit.full_attn_ids})")
    return vit, neck


def apply_sds_only(sf):
    """Apply SDS (Stochastic Depth Selection) only, without SPE.

    Useful for ablation: isolating the effect of depth randomization.
    """
    vit = sf.model.detector.backbone.vision_backbone.trunk
    neck = sf.model.detector.backbone.vision_backbone

    vit.return_interm_layers = True
    neck.sds_enabled = True

    print(f"[SDS] Enabled: Stochastic Depth Selection (depths={vit.full_attn_ids})")
    return vit, neck

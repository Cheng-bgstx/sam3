"""
Perturbation-Aware Feature Fusion (PAFF): Architecture-native defense.

Core insight from systematic analysis of SAM3's vulnerability:
- SAM3's FPN creates ALL 4 scales from ONE ViT output (xs[-1], block 31)
- This feature has been through ALL 4 global attention blocks (7, 15, 23, 31)
- Each global attention block AMPLIFIES perturbation through O(N^2) attention
- By block 31, perturbation is maximally amplified → all FPN scales corrupted

Other models (SAM2, Mask2Former) use multi-scale features from DIFFERENT
backbone stages, providing inherent redundancy against perturbation.

PAFF creates TWO parallel FPN feature sources:
1. PRIMARY: xs[-1] (block 31) — highest quality, most perturbed
2. ROBUST: MLFA average of all depths — lower quality, perturbation-resistant

On CLEAN data, primary and robust features are SIMILAR (both represent
the same scene). Under PERTURBATION, they DIVERGE because primary has
accumulated more perturbation through all 4 global attention blocks.

PAFF uses this DIVERGENCE as a non-differentiable detection signal:
- Low divergence → features are clean → use high-quality primary
- High divergence → perturbation detected → use robust MLFA fallback

Properties:
- Architecture-native: uses SAM3's own multi-depth ViT features
- Zero adversarial training: thresholds calibrated from clean SA-1B data
- Attack-agnostic: any perturbation that breaks primary↔robust agreement
- Dataset-agnostic: calibrated on diverse SA-1B, not YT1B-specific
- Non-differentiable: hard threshold on divergence creates BPDA gradient gap
- Preserves clean performance: primary pathway unchanged when clean

Novelty:
- No existing defense uses cross-pathway feature divergence for detection
- Exploits SAM3's own architectural property (global attention accumulation)
  as a PERTURBATION DETECTOR rather than a vulnerability
- First defense to use "perturbation amplification through depth" as a signal
"""
import types
import torch
import torch.nn.functional as F


def calibrate_paff(sf, dataset, n_samples=200, percentile=95, device='cuda:0'):
    """Calibrate PAFF divergence thresholds from clean data.

    For each FPN scale, computes the distribution of per-channel differences
    between primary (block 31) and robust (MLFA) features on clean data.

    The threshold is set at the given percentile of clean-data divergence.
    Features with divergence above this threshold are flagged as perturbed.

    Args:
        sf: Sam3VideoForwarder
        dataset: List of (img_np, anno, gt_np, info) tuples
        n_samples: Number of clean samples for calibration
        percentile: Divergence percentile for threshold (95 = flag top 5%)
        device: GPU device

    Returns:
        thresholds: List of [d_model] tensors, per-scale divergence thresholds
    """
    import random
    import numpy as np

    neck = sf.model.detector.backbone.vision_backbone
    trunk = neck.trunk
    trunk.return_interm_layers = True
    d_model = neck.convs[0][-1].out_channels
    n_scales = len(neck.convs)
    dn = lambda x: sf.denorm_image(x)

    # Collect per-channel L2 differences between primary and robust FPN features
    all_divs = [[] for _ in range(n_scales)]

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

                # Primary: block 31 (last)
                x_primary = xs[-1]

                # Robust: MLFA (average of all depths)
                x_robust = sum(xs) / len(xs)

                for i, conv in enumerate(neck.convs):
                    f_primary = conv(x_primary).float()
                    f_robust = conv(x_robust).float()

                    # Per-channel L2 divergence
                    div = ((f_primary - f_robust) ** 2).mean(dim=[2, 3])  # [B, C]
                    div = div.sqrt()  # RMSE
                    all_divs[i].append(div.cpu())

        n_done += 1
        if n_done % 50 == 0:
            print(f"  PAFF calib [{n_done}/{n_samples}]", flush=True)

    trunk.return_interm_layers = False

    # Compute thresholds at given percentile
    thresholds = []
    for i in range(n_scales):
        divs = torch.cat(all_divs[i], dim=0)  # [N_total, C]
        # Flatten batch dim
        divs_flat = divs.flatten()
        thresh = torch.quantile(divs_flat, percentile / 100.0)
        # Per-channel threshold (use global for simplicity, or per-channel)
        per_channel = torch.quantile(divs, percentile / 100.0, dim=0)  # [C]
        thresholds.append(per_channel)
        frac_below = (divs < per_channel.unsqueeze(0)).float().mean().item()
        print(f"  PAFF Scale {i}: thresh mean={per_channel.mean():.4f}, "
              f"frac below thresh={frac_below:.3f}", flush=True)

    return thresholds


class PAFFDefense:
    """Perturbation-Aware Feature Fusion defense.

    Patches the neck's forward to add dual-pathway feature computation
    with non-differentiable divergence-based gating.
    """

    def __init__(self, thresholds, sensitivity=1.0):
        """
        Args:
            thresholds: List of [d_model] per-scale divergence thresholds
            sensitivity: Multiplier for threshold (1.0 = default, >1 = less aggressive)
        """
        self.thresholds = thresholds
        self.sensitivity = sensitivity

    def __call__(self, primary_feats, robust_feats):
        """Apply PAFF gating to FPN features.

        Args:
            primary_feats: List of [B, C, H, W] from block 31
            robust_feats: List of [B, C, H, W] from MLFA

        Returns:
            fused_feats: List of [B, C, H, W] with perturbation gating
        """
        fused = []
        for i, (f_p, f_r) in enumerate(zip(primary_feats, robust_feats)):
            # Per-channel divergence
            div = ((f_p - f_r) ** 2).mean(dim=[2, 3], keepdim=True).sqrt()  # [B, C, 1, 1]

            # Non-differentiable hard threshold
            thresh = self.thresholds[i].to(f_p.device).view(1, -1, 1, 1) * self.sensitivity

            # Binary gate: 1 = clean (use primary), 0 = perturbed (use robust)
            gate = (div < thresh).float()  # [B, C, 1, 1]
            # Non-differentiable: detach gate from computation graph
            gate = gate.detach()

            # Fuse: primary when clean, robust when perturbed
            f_fused = gate * f_p + (1 - gate) * f_r
            fused.append(f_fused)

        return fused


def apply_paff(sf, thresholds, sensitivity=1.0):
    """Apply Perturbation-Aware Feature Fusion to SAM3's FPN neck.

    Patches neck.forward to compute dual-pathway features and fuse
    using divergence-based perturbation detection.

    Args:
        sf: Sam3VideoForwarder
        thresholds: Per-scale divergence thresholds from calibrate_paff()
        sensitivity: Threshold sensitivity multiplier
    """
    neck = sf.model.detector.backbone.vision_backbone
    trunk = neck.trunk
    trunk.return_interm_layers = True

    paff = PAFFDefense(thresholds, sensitivity=sensitivity)

    original_forward = neck.forward

    def paff_forward(self, tensor_list):
        xs = self.trunk(tensor_list)

        # Primary pathway: block 31 only (standard SAM3 behavior)
        x_primary = xs[-1]

        # Robust pathway: MLFA (average of all depths)
        x_robust = sum(xs) / len(xs)

        sam3_out, sam3_pos = [], []
        sam2_out, sam2_pos = None, None
        if self.sam2_convs is not None:
            sam2_out, sam2_pos = [], []

        primary_feats, robust_feats = [], []
        for i in range(len(self.convs)):
            f_p = self.convs[i](x_primary)
            f_r = self.convs[i](x_robust)
            primary_feats.append(f_p)
            robust_feats.append(f_r)

        # PAFF: divergence-based gating
        sam3_out = paff(primary_feats, robust_feats)

        # Position encoding on fused features
        for i, f in enumerate(sam3_out):
            sam3_pos_out = self.position_encoding(f).to(f.dtype)
            sam3_pos.append(sam3_pos_out)

        # SAM2 neck (if present)
        if self.sam2_convs is not None:
            for i in range(len(self.convs)):
                sam2_x_out = self.sam2_convs[i](x_primary)
                sam2_pos_out = self.position_encoding(sam2_x_out).to(sam2_x_out.dtype)
                sam2_out.append(sam2_x_out)
                sam2_pos.append(sam2_pos_out)

        return sam3_out, sam3_pos, sam2_out, sam2_pos

    neck.forward = types.MethodType(paff_forward, neck)
    neck._paff = paff
    neck._paff_enabled = True

    n_scales = len(thresholds)
    print(f"  PAFF: enabled ({n_scales} scales, sensitivity={sensitivity})", flush=True)
    return neck

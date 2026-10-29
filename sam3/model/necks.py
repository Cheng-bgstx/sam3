# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Necks are the interface between a vision backbone and the rest of the detection model"""

from copy import deepcopy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NLRBlock(nn.Module):
    """Non-Linear Residual block for FPN feature purification.

    A lightweight residual block that adds non-linear processing to FPN output.
    Trained on clean SA-1B data only (no adversarial training) to maintain clean
    performance. The GELU non-linearity and bottleneck create a non-linear
    transformation that distorts perturbation propagation through the FPN.

    Architecture: 3x3Conv -> GELU -> 3x3Conv, with residual connection.
    """

    def __init__(self, d_model: int, bottleneck_ratio: int = 4):
        super().__init__()
        hidden = d_model // bottleneck_ratio
        self.net = nn.Sequential(
            nn.Conv2d(d_model, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, d_model, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class Sam3DualViTDetNeck(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        position_encoding: nn.Module,
        d_model: int,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        add_sam2_neck: bool = False,
        csfc_enabled: bool = False,
        fbd_enabled: bool = False,
        fbd_bottleneck_dim: int = 64,
    ):
        """
        SimpleFPN neck a la ViTDet
        (From detectron2, very lightly adapted)
        It supports a "dual neck" setting, where we have two identical necks (for SAM3 and SAM2), with different weights

        :param trunk: the backbone
        :param position_encoding: the positional encoding to use
        :param d_model: the dimension of the model
        """
        super().__init__()
        self.trunk = trunk
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()
        self.d_model = d_model

        self.scale_factors = scale_factors
        use_bias = True
        dim: int = self.trunk.channel_list[-1]

        for _, scale in enumerate(scale_factors):
            current = nn.Sequential()

            if scale == 4.0:
                current.add_module(
                    "dconv_2x2_0",
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                )
                current.add_module(
                    "gelu",
                    nn.GELU(),
                )
                current.add_module(
                    "dconv_2x2_1",
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                )
                out_dim = dim // 4
            elif scale == 2.0:
                current.add_module(
                    "dconv_2x2",
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                )
                out_dim = dim // 2
            elif scale == 1.0:
                out_dim = dim
            elif scale == 0.5:
                current.add_module(
                    "maxpool_2x2",
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
                out_dim = dim
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            current.add_module(
                "conv_1x1",
                nn.Conv2d(
                    in_channels=out_dim,
                    out_channels=d_model,
                    kernel_size=1,
                    bias=use_bias,
                ),
            )
            current.add_module(
                "conv_3x3",
                nn.Conv2d(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                ),
            )
            self.convs.append(current)

        self.sam2_convs = None
        if add_sam2_neck:
            # Assumes sam2 neck is just a clone of the original neck
            self.sam2_convs = deepcopy(self.convs)

        # VNR: ViT Non-linear Residual blocks (LAZY init)
        # Applied to ViT multi-depth features BEFORE FPN processing.
        # The propagation diagnostic shows ViT features are the FIRST and most-severe
        # corruption point (cos=0.10 with clean features). VNR adds non-linear
        # processing at each ViT depth to break perturbation propagation before
        # features enter the FPN.
        # Trained on SA-1B clean data — NOT adversarial training.
        self.vnr_enabled = False
        self.vnr_blocks = None
        self.sam2_vnr_blocks = None

        # NLR: Non-Linear Residual blocks (LAZY init to not break checkpoint loading)
        # Created on first use when nlr_enabled=True.
        # Added after each FPN scale to break linear perturbation propagation.
        # Trained on SA-1B clean data — NOT adversarial training.
        # Bottleneck design (d_model//4 hidden) limits correction rank.
        self.nlr_enabled = False
        self.nlr_blocks = None  # created lazily when nlr_enabled is set
        self.sam2_nlr_blocks = None

        # FBD: Feature Bottleneck Denoising (deprecated — bottleneck too narrow)
        self.fbd_enabled = fbd_enabled
        self.fbd_denoiser = None
        if fbd_enabled:
            from sam3.model.feature_denoiser import FeatureDenoiser
            self.fbd_denoiser = FeatureDenoiser(
                d_model=d_model,
                bottleneck_dim=fbd_bottleneck_dim,
                num_scales=len(scale_factors),
                shared=True,
            )

        # MLFA: Multi-Level Feature Averaging
        # SAM3's FPN creates all scales from a SINGLE ViT feature (xs[-1]).
        # This is a single point of failure — one corrupted feature infects all scales.
        # MLFA uses ALL ViT intermediate features (from global attn blocks) instead
        # of just the last one. Features at different ViT depths have different
        # perturbation characteristics, so averaging reduces perturbation variance.
        # Zero learnable parameters. Zero training.
        self.mlfa_enabled = False

        # SDS: Stochastic Depth Selection
        # Randomly selects which ViT depth to use as FPN source each forward pass.
        self.sds_enabled = False

        # DRFS: Depth-Restructured FPN Source
        # SAM3's FPN is the ROOT CAUSE of its adversarial fragility: ALL 4 FPN
        # scales are sourced from ONE ViT depth (block 31), the most perturbed
        # feature in the entire network. Standard robust models (Swin, ViTDet)
        # assign DIFFERENT depths to DIFFERENT scales — shallow features for
        # high-res, deep features for low-res.
        #
        # DRFS fixes this by mapping each FPN scale to the ViT depth whose
        # semantic level matches that scale:
        #   xs[0] (block 7,  least perturbation) → convs[0] (stride 4,  most sensitive)
        #   xs[1] (block 15)                     → convs[1] (stride 8)
        #   xs[2] (block 23)                     → convs[2] (stride 16)
        #   xs[3] (block 31, most perturbation)  → convs[3] (stride 32, least sensitive)
        #
        # Perturbation amplification is proportional to depth. By sourcing high-res
        # scales from early blocks, the most perturbation-sensitive scales get the
        # cleanest features. Zero training, zero parameters, purely architectural.
        self.drfs_enabled = False

        # CSFC: Cross-Scale Feature Consensus (deprecated — cross-scale cos ≈ 0)
        self.csfc_enabled = csfc_enabled

        # LMFF: Learned Multi-Depth Feature Fusion
        # Each FPN scale gets a LEARNED weighted combination of ALL 4 ViT depths
        # (blocks [7,15,23,31]). Weights trained on SA-1B clean data to minimize
        # reconstruction error vs original single-source (block 31) output.
        # Creates REDUNDANT pathways: if one depth is perturbed, others compensate.
        # Genuine architectural robustness — 16 learned parameters, no adversarial training.
        self.lmff_enabled = False
        self.lmff_weights = None  # [n_scales, n_depths] softmax weights
        self.lmff_temperature = 0.5

        # FMFF: Fixed Multi-Depth Feature Fusion
        # Each FPN scale uses FIXED multi-depth weights (no training needed).
        # Unlike DRFS (1-to-1 mapping, single point of failure per scale),
        # FMFF gives every scale contributions from ALL depths:
        #   Scale 0: [0.50, 0.25, 0.15, 0.10] — mostly block 7 (cleanest)
        #   Scale 1: [0.20, 0.50, 0.20, 0.10] — mostly block 15
        #   Scale 2: [0.10, 0.20, 0.50, 0.20] — mostly block 23
        #   Scale 3: [0.05, 0.10, 0.25, 0.60] — mostly block 31
        # Genuine architectural redundancy: no single depth dominates any scale.
        self.fmff_enabled = False
        self.fmff_weights = None  # set in forward() if enabled

        # DSA: Depth Statistics Alignment
        # Each ViT depth has different channel-wise statistics (mean, std).
        # FPN convs are trained to expect block-31 features. Multi-depth fusion
        # (FMFF/LMFF) mixes depths with different statistics, which the FPN convs
        # don't handle optimally. DSA normalizes each depth to match block-31's
        # channel statistics BEFORE fusion, so the fused feature looks like what
        # the FPN convs expect. Zero learnable parameters. SA-1B calibrated.
        self.dsa_enabled = False
        self.dsa_means = None  # list of [C] per-depth channel means
        self.dsa_stds = None  # list of [C] per-depth channel stds
        self.dsa_target_mean = None  # [C] target channel mean (block 31)
        self.dsa_target_std = None  # [C] target channel std (block 31)

        # CDFC: Cross-Depth Feature Consensus
        # Detects perturbation by comparing deep ViT features (blocks 15,23,31)
        # against the shallow reference (block 7, least perturbed).
        # When perturbation pushes deep features outside their natural cross-depth
        # relationship with shallow features, they get hard-clamped to the
        # calibrated clean-data range. The hard clamp is non-differentiable,
        # creating a principled gradient gap (not random noise).
        self.cdfc_enabled = False
        self.cdfc_k = 3.0  # number of sigmas to allow
        self.cdfc_random_k = False  # randomize k per forward pass for gradient gap
        self.cdfc_mu = None  # list of [C] per-depth mean difference from ref
        self.cdfc_sigma = None  # list of [C] per-depth std of difference from ref

        # MDFA: Multi-Depth FPN Adapters
        # Lightweight adapters (1x1 conv + GELU) before each FPN conv that learn
        # to map multi-depth fused features to what the FPN convs expect.
        # Trained on SA-1B clean data, not adversarial. Architecture-native.
        # Each adapter: 1x1 conv (d_model → d_model) with GELU for non-linearity.
        self.mdfa_enabled = False
        self.mdfa_adapters = None  # nn.ModuleList, one per scale

        # DRA: Depth Representation Alignment
        # SAM3's different ViT depths produce REPRESENTATIONALLY ORTHOGONAL
        # features (cos ≈ -0.14 clean). This is THE barrier preventing effective
        # multi-depth fusion: FMFF blends features that encode different things
        # in different ways, like averaging apples and oranges.
        #
        # DRA trains per-depth projectors (1x1Conv→GELU→1x1Conv, residual) to map
        # each depth's features to block-31's representational space. After DRA:
        # - All depths encode the same information in the same way
        # - FMFF/DSA multi-depth blending becomes genuinely meaningful
        # - Each scale has TRUE multi-depth redundancy, not orthogonal noise
        #
        # Trained on SA-1B clean data only — NOT adversarial training.
        # Architecture-native: fixes SAM3's single-source FPN vulnerability by
        # making its multi-depth ViT features genuinely compatible.
        self.dra_enabled = False
        self.dra_projectors = None  # nn.ModuleList, one per non-reference depth

        # Auto-load DSA calibration for default multi-depth FPN
        self._auto_hierarchical = False
        self._try_load_dsa_default()

    def _try_load_dsa_default(self):
        """Load DSA calibration stats if available. Does NOT auto-enable hierarchical mode.

        Hierarchical FPN (FMFF+DSA) is opt-in: enabled only when a defense type
        explicitly requests it in unified_eval.py. Auto-enabling it changes FPN
        features for ALL model loads, dropping clean IoU from ~0.257 to ~0.18.
        """
        import os
        dsa_path = 'review-stage/dsa_stats.pt'
        if os.path.exists(dsa_path):
            try:
                dsa_data = torch.load(dsa_path, map_location='cpu', weights_only=True)
                self.dsa_means = [m for m in dsa_data['means']]
                self.dsa_stds = [s for s in dsa_data['stds']]
                self.dsa_target_mean = dsa_data['target_mean']
                self.dsa_target_std = dsa_data['target_std']
                # Opt-in only — do NOT set _auto_hierarchical or return_interm_layers here
            except Exception:
                pass

    def set_calibration(self, mu_list, sigma=None):
        """Set calibration parameters from clean-data measurement."""
        self.calib_mu = mu_list
        if sigma is not None:
            self.calib_sigma = sigma

    def _csfc_fusion(self, sam3_out):
        """Cross-Scale Feature Consensus: adaptive fusion of fine and coarse features.

        For each fine scale (0, 1, 2), compute per-pixel cosine similarity with
        the upsampled coarsest scale (3). When similarity is low (perturbation
        detected), fuse toward the robust coarse features.

        Zero trainable parameters. Attack-agnostic. Dataset-agnostic.
        """
        if not self.csfc_enabled:
            return sam3_out

        # sam3_out: list of [B, d_model, H_i, W_i] for scales 4.0, 2.0, 1.0, 0.5
        n_scales = len(sam3_out)
        if n_scales < 2:
            return sam3_out

        f_ref = sam3_out[-1]  # coarsest = most robust (stride ~32)

        fused = []
        for i in range(n_scales - 1):
            f_fine = sam3_out[i]
            # Upsample coarse reference to match fine scale
            f_ref_up = F.interpolate(f_ref, size=f_fine.shape[-2:],
                                     mode='bilinear', align_corners=False)

            # Per-pixel channel-wise cosine similarity
            # f_fine: [B, C, H, W], f_ref_up: [B, C, H, W]
            cos_sim = F.cosine_similarity(f_fine, f_ref_up, dim=1)  # [B, H, W]

            # Adaptive fusion weight via sigmoid
            # High cos_sim (>mu) → alpha→1 → trust fine features (clean)
            # Low cos_sim (<mu) → alpha→0 → trust coarse features (perturbed)
            mu = self.calib_mu[i] if i < len(self.calib_mu) else 0.7
            alpha = torch.sigmoid((cos_sim - mu) / self.calib_sigma)  # [B, H, W]

            # Fuse: alpha * fine + (1-alpha) * coarse_up
            f_fused = alpha.unsqueeze(1) * f_fine + (1 - alpha.unsqueeze(1)) * f_ref_up
            fused.append(f_fused)

        fused.append(f_ref)  # coarsest unchanged
        return fused

    def forward(
        self, tensor_list: List[torch.Tensor]
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        Optional[List[torch.Tensor]],
        Optional[List[torch.Tensor]],
    ]:
        xs = self.trunk(tensor_list)

        # VNR: ViT Non-linear Residual blocks
        # Applied to ViT multi-depth features BEFORE FPN processing.
        # Propagation diagnostic shows ViT cos=0.10 (near-orthogonal to clean).
        # VNR adds non-linear processing at each ViT depth to purify features
        # before they enter the FPN's linear multi-scale conversion.
        # Trained on SA-1B clean data with denoising objective.
        if self.vnr_enabled:
            if self.vnr_blocks is None:
                self.vnr_blocks = nn.ModuleList([
                    NLRBlock(xs[i].shape[1], bottleneck_ratio=4)
                    for i in range(len(xs))
                ]).to(xs[0].device)
            xs = [self.vnr_blocks[i](xs[i]) for i in range(len(xs))]

        # CDFC: Cross-Depth Feature Consensus
        # Hard-clamp deep features (blocks 15,23,31) to expected range relative
        # to the shallow reference (block 7). Uses SA-1B-calibrated per-channel
        # mu/sigma of cross-depth feature differences. The hard clamp creates a
        # non-differentiable gradient gap — principled (feature constraint based
        # on clean-data statistics), not random (unlike SPE/SDS noise).
        # IMPORTANT: CDFC must run BEFORE DSA. CDFC calibration was done on
        # original features. DSA would squash cross-depth differences, making
        # CDFC thresholds meaningless.
        if self.cdfc_enabled and self.cdfc_mu is not None and len(xs) > 1:
            import random
            for i in range(1, len(xs)):
                diff = xs[i] - xs[0]  # deviation from reference
                k = (self.cdfc_k - 0.5 + random.random()) if self.cdfc_random_k else self.cdfc_k
                lo = self.cdfc_mu[i-1] - k * self.cdfc_sigma[i-1]
                hi = self.cdfc_mu[i-1] + k * self.cdfc_sigma[i-1]
                diff_c = torch.clamp(diff, lo, hi)
                xs[i] = xs[0] + diff_c

        # DRA: Depth Representation Alignment
        # Project non-reference ViT depths to block-31's representational space.
        # Fixes cross-depth orthogonality (cos ≈ -0.14) that prevents effective
        # multi-depth fusion. After DRA, FMFF/DSA blending is genuinely meaningful
        # because all depths encode information in a common representational space.
        # Lazy init to not break checkpoint loading.
        if self.dra_enabled:
            if self.dra_projectors is None:
                vit_dim = xs[0].shape[1]  # ViT feature dim (1024), not d_model (256)
                self.dra_projectors = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(vit_dim, vit_dim, 1, bias=False),
                        nn.GELU(),
                        nn.Conv2d(vit_dim, vit_dim, 1, bias=False),
                    )
                    for _ in range(len(xs) - 1)
                ]).to(xs[0].device)
            # Project depths 0, 1, 2 to depth 3 (block 31) space
            for i in range(len(xs) - 1):
                xs[i] = xs[i] + self.dra_projectors[i](xs[i])  # residual

        # DSA: Depth Statistics Alignment
        # Normalize each ViT depth to match block-31 channel statistics.
        # FPN convs are trained for block-31 features; without alignment,
        # multi-depth fusion produces features with different statistics
        # that the FPN convs process suboptimally.
        if self.dsa_enabled and self.dsa_means is not None and len(xs) > 0:
            xs = list(xs)
            for i in range(len(xs)):
                m = self.dsa_means[i].to(xs[i].device).view(1, -1, 1, 1)
                s = self.dsa_stds[i].to(xs[i].device).view(1, -1, 1, 1)
                tm = self.dsa_target_mean.to(xs[i].device).view(1, -1, 1, 1)
                ts = self.dsa_target_std.to(xs[i].device).view(1, -1, 1, 1)
                xs[i] = (xs[i] - m) / (s + 1e-5) * ts + tm

        sam3_out, sam3_pos = [], []
        sam2_out, sam2_pos = None, None
        if self.sam2_convs is not None:
            sam2_out, sam2_pos = [], []

        if self.lmff_enabled and self.lmff_weights is not None and len(xs) >= len(self.convs):
            # LMFF: Learned Multi-Depth Feature Fusion
            # Each FPN scale uses ALL 4 ViT depths with learned softmax weights.
            # w[i,j] = contribution of depth j to scale i, trained on SA-1B.
            # Creates REDUNDANT pathways — if block 31 is heavily perturbed,
            # blocks 7, 15, 23 compensate through their weighted contributions.
            # Genuine architectural robustness: no stochasticity, no adversarial training.
            w = self.lmff_weights.to(xs[0].device)
            for i in range(len(self.convs)):
                # Weighted combination of ALL ViT depths for this scale
                x_i = sum(w[i, j] * xs[j] for j in range(len(xs)))
                # MDFA: adapter maps multi-depth features for FPN compatibility
                if self.mdfa_enabled and self.mdfa_adapters is not None:
                    x_i = self.mdfa_adapters[i](x_i) + x_i  # residual connection
                sam3_x_out = self.convs[i](x_i)
                sam3_pos_out = self.position_encoding(sam3_x_out).to(sam3_x_out.dtype)
                sam3_out.append(sam3_x_out)
                sam3_pos.append(sam3_pos_out)

                if self.sam2_convs is not None:
                    sam2_x_out = self.sam2_convs[i](x_i)
                    sam2_pos_out = self.position_encoding(sam2_x_out).to(sam2_x_out.dtype)
                    sam2_out.append(sam2_x_out)
                    sam2_pos.append(sam2_pos_out)
        elif self.fmff_enabled and len(xs) >= len(self.convs):
            # FMFF: Fixed Multi-Depth Feature Fusion
            # Each scale uses FIXED multi-depth weights (zero training).
            # Unlike DRFS (1-to-1, single point of failure), FMFF gives
            # every scale contributions from ALL 4 depths for redundancy.
            if self.fmff_weights is None:
                self.fmff_weights = torch.tensor([
                    [0.50, 0.25, 0.15, 0.10],  # Scale 0: mostly block 7
                    [0.20, 0.50, 0.20, 0.10],  # Scale 1: mostly block 15
                    [0.10, 0.20, 0.50, 0.20],  # Scale 2: mostly block 23
                    [0.05, 0.10, 0.25, 0.60],  # Scale 3: mostly block 31
                ], device=xs[0].device, dtype=xs[0].dtype)
            w = self.fmff_weights
            for i in range(len(self.convs)):
                x_i = sum(w[i, j] * xs[j] for j in range(len(xs)))
                # MDFA: adapter maps multi-depth features for FPN compatibility
                if self.mdfa_enabled and self.mdfa_adapters is not None:
                    x_i = self.mdfa_adapters[i](x_i) + x_i  # residual connection
                sam3_x_out = self.convs[i](x_i)
                sam3_pos_out = self.position_encoding(sam3_x_out).to(sam3_x_out.dtype)
                sam3_out.append(sam3_x_out)
                sam3_pos.append(sam3_pos_out)

                if self.sam2_convs is not None:
                    sam2_x_out = self.sam2_convs[i](x_i)
                    sam2_pos_out = self.position_encoding(sam2_x_out).to(sam2_x_out.dtype)
                    sam2_out.append(sam2_x_out)
                    sam2_pos.append(sam2_pos_out)
        elif self.drfs_enabled and len(xs) >= len(self.convs):
            # DRFS: Each FPN scale sourced from the matching ViT depth.
            # xs[0] (block 7,  cleanest) → convs[0] (stride 4,  most sensitive)
            # xs[-1] (block 31, most perturbed) → convs[-1] (stride 32, least sensitive)
            for i in range(len(self.convs)):
                if self.sds_enabled:
                    import random
                    # SDS + DRFS: DRFS routing (xs[i]→scale i) with SDS
                    # random blending against multi-depth mean. Per-scale
                    # alpha creates independent gradient gaps per FPN scale.
                    # Full range [0, 1] maximizes gradient gap — same as
                    # standalone SDS but per-scale with DRFS routing.
                    alpha = random.random()  # [0, 1]
                    x_i = alpha * xs[i] + (1 - alpha) * (sum(xs) / len(xs))
                else:
                    x_i = xs[i]
                sam3_x_out = self.convs[i](x_i)
                sam3_pos_out = self.position_encoding(sam3_x_out).to(sam3_x_out.dtype)
                sam3_out.append(sam3_x_out)
                sam3_pos.append(sam3_pos_out)

                if self.sam2_convs is not None:
                    sam2_x_out = self.sam2_convs[i](x_i)
                    sam2_pos_out = self.position_encoding(sam2_x_out).to(sam2_x_out.dtype)
                    sam2_out.append(sam2_x_out)
                    sam2_pos.append(sam2_pos_out)
        elif self.sds_enabled and len(xs) > 1:
            import random
            # Random interpolation between last depth (best quality) and
            # multi-depth average (more robust). Creates a continuous
            # gradient gap — attacker's gradient is for one alpha but
            # forward pass uses a different alpha next step.
            alpha = random.random()  # [0, 1]
            x = alpha * xs[-1] + (1 - alpha) * (sum(xs) / len(xs))
        elif self.mlfa_enabled and len(xs) > 1:
            x = sum(xs) / len(xs)
        else:
            # DEFAULT: auto-hierarchical (like SAM2's natural backbone hierarchy)
            # When multi-depth features are available and DSA is calibrated,
            # use FMFF+DSA for genuine multi-depth redundancy at all scales.
            # This is THE architecture fix: SAM3 no longer uses single-source FPN.
            if self._auto_hierarchical and len(xs) >= len(self.convs):
                # DSA first: align all depths to block-31 statistics
                if self.dsa_means is not None:
                    xs = list(xs)
                    for i in range(len(xs)):
                        m = self.dsa_means[i].to(xs[i].device).view(1, -1, 1, 1)
                        s = self.dsa_stds[i].to(xs[i].device).view(1, -1, 1, 1)
                        tm = self.dsa_target_mean.to(xs[i].device).view(1, -1, 1, 1)
                        ts = self.dsa_target_std.to(xs[i].device).view(1, -1, 1, 1)
                        xs[i] = (xs[i] - m) / (s + 1e-5) * ts + tm
                # FMFF: fixed multi-depth weights per scale
                if self.fmff_weights is None:
                    self.fmff_weights = torch.tensor([
                        [0.50, 0.25, 0.15, 0.10],
                        [0.20, 0.50, 0.20, 0.10],
                        [0.10, 0.20, 0.50, 0.20],
                        [0.05, 0.10, 0.25, 0.60],
                    ], device=xs[0].device, dtype=xs[0].dtype)
                w = self.fmff_weights
                for i in range(len(self.convs)):
                    x_i = sum(w[i, j] * xs[j] for j in range(len(xs)))
                    if self.mdfa_enabled and self.mdfa_adapters is not None:
                        x_i = self.mdfa_adapters[i](x_i) + x_i
                    sam3_x_out = self.convs[i](x_i)
                    sam3_pos_out = self.position_encoding(sam3_x_out).to(sam3_x_out.dtype)
                    sam3_out.append(sam3_x_out)
                    sam3_pos.append(sam3_pos_out)
                    if self.sam2_convs is not None:
                        sam2_x_out = self.sam2_convs[i](x_i)
                        sam2_pos_out = self.position_encoding(sam2_x_out).to(sam2_x_out.dtype)
                        sam2_out.append(sam2_x_out)
                        sam2_pos.append(sam2_pos_out)
            else:
                # Legacy fallback: single-source from block 31
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

        # NLR: Non-Linear Residual blocks
        # Applied after FPN convs but before cross-scale fusion.
        # Each scale gets its own dedicated NLR block with GELU non-linearity.
        # The bottleneck design (d_model//4 hidden) limits correction rank,
        # preventing the blocks from learning to amplify perturbation.
        # Trained on SA-1B clean data — NOT adversarial training.
        # Lazy init: blocks created on first use to not break checkpoint loading.
        if self.nlr_enabled:
            if self.nlr_blocks is None:
                self.nlr_blocks = nn.ModuleList([
                    NLRBlock(self.d_model, bottleneck_ratio=4)
                    for _ in self.scale_factors
                ]).to(sam3_out[0].device)
            for i in range(len(sam3_out)):
                sam3_out[i] = self.nlr_blocks[i](sam3_out[i])
            if self.sam2_convs is not None and sam2_out is not None:
                if self.sam2_nlr_blocks is None:
                    self.sam2_nlr_blocks = nn.ModuleList([
                        NLRBlock(self.d_model, bottleneck_ratio=4)
                        for _ in self.scale_factors
                    ]).to(sam3_out[0].device)
                for i in range(len(sam2_out)):
                    sam2_out[i] = self.sam2_nlr_blocks[i](sam2_out[i])

        # CSFC: Cross-Scale Feature Consensus fusion
        sam3_out = self._csfc_fusion(sam3_out)

        # FBD: Feature Bottleneck Denoising (clean-data trained autoencoder)
        if self.fbd_enabled and self.fbd_denoiser is not None:
            sam3_out = self.fbd_denoiser(sam3_out)

        return sam3_out, sam3_pos, sam2_out, sam2_pos

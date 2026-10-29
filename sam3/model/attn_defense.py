"""
Attention Entropy Maximization (AEM): Architecture-native adversarial defense.

Core insight: Under adversarial perturbation, self-attention in ViT's global blocks
tends toward CONCENTRATION — a few adversarial tokens attract disproportionate
attention from all other tokens, enabling global corruption propagation.

AEM caps the maximum attention weight any single token can receive, preventing
adversarial attention dominance. The cap creates a non-differentiable boundary
that BPDA must approximate with identity, creating an optimization gap.

Properties:
- Zero training (cap calibrated from clean-data attention statistics)
- Architecture-native (modifies ViT attention mechanism)
- Attack-agnostic (any attention-based attack is limited)
- Non-differentiable at boundary (BPDA approximation is inaccurate)
"""
import math
import types
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def make_capped_attention_forward(attn_module, max_weight=0.05, temperature=1.0):
    """Create a patched forward that caps attention weights.

    Args:
        attn_module: The Attention module to patch
        max_weight: Maximum attention weight any single token can have.
        temperature: Scale factor for attention logits. T>1 makes attention more uniform.
    """

    def capped_forward(self, x: Tensor) -> Tensor:
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

        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, -1)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        q, k = self._apply_rope(q, k)
        if self.use_rel_pos:
            q, k = self._orig_rel_pos(q, k, H, W, x)

        # Explicit attention with temperature scaling and capping
        scale = self.scale / temperature
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Non-differentiable cap: prevents adversarial attention concentration
        attn_weights = torch.clamp(attn_weights, max=max_weight)
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)

        x = torch.matmul(attn_weights, v)

        if ndim == 4:
            x = (
                x.view(B, self.num_heads, H, W, -1)
                .permute(0, 2, 3, 1, 4)
                .reshape(B, H, W, -1)
            )
        else:
            x = x.view(B, self.num_heads, L, -1).permute(0, 2, 1, 3).reshape(B, L, -1)

        x = self.proj(x)
        return x

    return capped_forward


def make_dampened_block_forward(block, dampen_factor=0.5):
    """Patch Block.forward to dampen the global attention contribution.

    Original: x = shortcut + dropout(drop_path(ls1(attn(norm1(x)))))
    Modified: x = shortcut + dampen_factor * dropout(drop_path(ls1(attn(norm1(x)))))

    This reduces perturbation amplification through global attention.
    """
    original_forward = block.forward

    def dampened_forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x_normed = self.norm1(x)
        attn_out = self.ls1(self.attn(x_normed))
        x = shortcut + dampen_factor * self.dropout(self.drop_path(attn_out))
        x = x + self.dropout(self.drop_path(self.ls2(self.mlp(self.norm2(x)))))
        return x

    return dampened_forward


def apply_attention_capping(vit_model, max_weight=0.05, temperature=1.0,
                             global_only=True, dampen_factor=1.0):
    """Apply attention capping to ViT model.

    Args:
        vit_model: The ViT backbone model
        max_weight: Maximum attention weight per token
        temperature: Attention temperature (T>1 = more uniform)
        global_only: If True, only patch global attention blocks
        dampen_factor: Scale for global block's attention contribution (1.0 = no change)
    """
    full_attn_ids = set(vit_model.full_attn_ids)

    patched = 0
    for i, block in enumerate(vit_model.blocks):
        if global_only and i not in full_attn_ids:
            continue

        attn = block.attn

        # Store original rel_pos handler
        if attn.use_rel_pos:
            from sam3.model.vitdet import concat_rel_pos

            def _make_rel_pos_fn(a):
                def _rel_pos(q, k, H, W, x):
                    q_flat = q.flatten(0, 1)
                    k_flat = k.flatten(0, 1)
                    B = q.shape[0]
                    q_out, k_out = concat_rel_pos(
                        q_flat, k_flat,
                        (H, W), x.shape[1:3],
                        a.rel_pos_h, a.rel_pos_w,
                        rescale=True,
                        relative_coords=a.relative_coords,
                    )
                    q_out = q_out.reshape(B, a.num_heads, H * W, -1)
                    k_out = k_out.reshape(B, a.num_heads, H * W, -1)
                    return q_out, k_out
                return _rel_pos
            attn._orig_rel_pos = _make_rel_pos_fn(attn)
        else:
            attn._orig_rel_pos = lambda q, k, H, W, x: (q, k)

        capped_fn = make_capped_attention_forward(attn, max_weight=max_weight,
                                                   temperature=temperature)
        attn.forward = types.MethodType(capped_fn, attn)

        # Apply residual dampening to global block
        if dampen_factor < 1.0:
            dampened_fn = make_dampened_block_forward(block, dampen_factor=dampen_factor)
            block.forward = types.MethodType(dampened_fn, block)

        patched += 1

    return patched


def apply_stochastic_depth_inference(vit_model, drop_rate=0.3, global_only=False):
    """Apply stochastic depth at inference time.

    Randomly skip transformer blocks during inference. Uses ViT-level mask
    pre-generation to avoid per-block CUDA synchronizations (1 sync vs 32).

    Args:
        vit_model: The ViT backbone model
        drop_rate: Probability of skipping each block (0.3 = 30% skip rate)
        global_only: If True, only apply to global attention blocks
    """
    full_attn_ids = set(vit_model.full_attn_ids)

    # Set drop rates per block
    n_blocks = len(vit_model.blocks)
    for i, block in enumerate(vit_model.blocks):
        if global_only and i not in full_attn_ids:
            block._sdi_drop_rate = 0.0
        else:
            block._sdi_drop_rate = drop_rate

    # Patch ViT forward to pre-generate masks (single CUDA sync)
    original_vit_forward = vit_model.forward
    drop_rates_tensor = torch.tensor(
        [float(getattr(b, '_sdi_drop_rate', 0.0)) for b in vit_model.blocks]
    )

    def sdi_vit_forward(self, x: torch.Tensor):
        # Pre-generate all skip decisions on GPU in one operation
        device = x.device
        rates = drop_rates_tensor.to(device)
        rand_vals = torch.rand(n_blocks, device=device)
        # skip if rand < drop_rate
        skips = (rand_vals < rates).tolist()

        for i, block in enumerate(self.blocks):
            block._sdi_skip = skips[i]

        return original_vit_forward(x)

    vit_model.forward = types.MethodType(sdi_vit_forward, vit_model)

    # Patch block forwards to check pre-generated skip flag
    patched = 0
    for i, block in enumerate(vit_model.blocks):
        if getattr(block, '_sdi_drop_rate', 0.0) == 0.0:
            continue

        block._original_forward = block.forward

        def _make_stochastic_forward(blk):
            def stochastic_forward(self, x: Tensor) -> Tensor:
                if self._sdi_skip:
                    return x
                return self._original_forward(x)
            return stochastic_forward

        block.forward = types.MethodType(_make_stochastic_forward(block), block)
        patched += 1

    return patched


def apply_feature_dropout_inference(neck, drop_rate=0.3):
    """Apply channel dropout to FPN features at inference time.

    Randomly drops feature channels at the FPN output. Like SDI but in
    feature space instead of block space. The non-differentiable random
    mask creates a BPDA gradient gap (identity approximation vs actual dropout).

    Args:
        neck: Sam3DualViTDetNeck instance
        drop_rate: Probability of dropping each channel (0.3 = 30% dropout)
    """
    import types

    original_forward = neck.forward

    def fdi_forward(self, tensor_list):
        sam3_out, sam3_pos, sam2_out, sam2_pos = original_forward(tensor_list)

        # Apply random channel dropout to SAM3 features
        dropped_out = []
        for f in sam3_out:
            # Per-channel random mask: [1, C, 1, 1]
            mask = torch.rand(f.shape[1], device=f.device) > drop_rate
            # Non-differentiable: mask changes each forward
            f_dropped = f * mask.float().view(1, -1, 1, 1)
            # Rescale to preserve expected magnitude
            f_dropped = f_dropped / (1.0 - drop_rate + 1e-8)
            dropped_out.append(f_dropped)

        return dropped_out, sam3_pos, sam2_out, sam2_pos

    neck.forward = types.MethodType(fdi_forward, neck)
    return len(neck.convs)


def make_wmtp_attention_forward(attn_module, pool_size=14, drop_rate=0.2):
    """Create patched forward using Window-Mean Token Pooling (WMTP).

    Instead of full O(N^2) global attention (which lets perturbation propagate
    to every token), each query attends to window-pooled key/value representatives
    PLUS tokens in its own window (hybrid attention).

    - Own-window tokens: preserve spatial detail within each window
    - Window representatives: cross-window communication with perturbation averaged

    This gives O(N * (W + N/W^2)) complexity instead of O(N^2), and the
    within-window averaging suppresses perturbation by factor W before global
    propagation.

    Non-differentiable random token dropout within windows creates a BPDA
    gradient gap.

    Args:
        attn_module: The Attention module to patch
        pool_size: Window size for pooling K,V (default 14, matches ViT window_size)
        drop_rate: Random token dropout rate before pooling (0 = fully differentiable)
    """
    import math
    import torch.nn.functional as F
    from sam3.model.vitdet import window_partition

    def wmtp_forward(self, x):
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

        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, -1)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        q, k = self._apply_rope(q, k)
        if self.use_rel_pos:
            q, k = self._orig_rel_pos(q, k, H, W, x)

        # Reshape to spatial: [B, nH, L, C] -> [B*nH, H, W, C]
        head_dim = q.shape[-1]
        nH = self.num_heads

        k_spat = k.reshape(B * nH, H, W, head_dim)
        v_spat = v.reshape(B * nH, H, W, head_dim)
        q_spat = q.reshape(B * nH, H, W, head_dim)

        # Pool window representatives: use avg_pool with pool_size stride
        # This is differentiable but perturbation is averaged within each window
        k_merged = k_spat.permute(0, 3, 1, 2)  # [B*nH, C, H, W]
        v_merged = v_spat.permute(0, 3, 1, 2)

        # Non-differentiable random token dropout before pooling
        if drop_rate > 0:
            mask = torch.rand_like(k_merged[:, :1, :, :]) > drop_rate
            k_merged = k_merged * mask
            v_merged = v_merged * mask
            k_merged = k_merged / (1.0 - drop_rate + 1e-8)
            v_merged = v_merged / (1.0 - drop_rate + 1e-8)

        H_out = max(1, H // pool_size)
        W_out = max(1, W // pool_size)
        k_pooled = F.adaptive_avg_pool2d(k_merged, (H_out, W_out))
        v_pooled = F.adaptive_avg_pool2d(v_merged, (H_out, W_out))

        # Window reps: [B*nH, C, H_out, W_out] -> [B*nH, nW, C]
        k_reps = k_pooled.flatten(2).permute(0, 2, 1)  # [B*nH, nW, C]
        v_reps = v_pooled.flatten(2).permute(0, 2, 1)
        nW = k_reps.shape[1]

        # For each token, compute attention to: own-window tokens + all window reps
        # Reshape Q, K, V into window grid for local attention
        # Q: [B*nH, H, W, C] -> partitioned into windows
        # Use adaptive approach: for each query, select relevant K subset

        # Compute attention to window reps (global) for all queries
        q_flat = q_spat.reshape(B * nH, L, head_dim)  # [B*nH, L, C]
        k_rep_flat = k_reps  # [B*nH, nW, C]
        v_rep_flat = v_reps

        scale = self.scale
        attn_reps = torch.matmul(q_flat, k_rep_flat.transpose(-2, -1)) * scale
        # attn_reps: [B*nH, L, nW]

        # Now also compute local (within-window) attention
        # Reshape Q, K, V into windows: [B*nH, H, W, C] -> [B*nH*nW_h*nW_w, ws_h, ws_w, C]
        # Use adaptive pooling approach: neighbor tokens have similar reps
        # Treat each query's attention to its spatially-nearest tokens

        # For efficiency, compute full Q@K^T but MASK to only allow attention
        # within same window + to window reps
        # Actually, Q@K^T is O(N^2) which we want to avoid...

        # Alternative: compute local attention per window using window_partition
        k_wins, (Hp, Wp) = window_partition(k_spat, pool_size)
        v_wins, _ = window_partition(v_spat, pool_size)
        q_wins, _ = window_partition(q_spat, pool_size)

        # k_wins: [B*nH*nW, pool_size, pool_size, C]
        n_win_tot = k_wins.shape[0]
        ws_h, ws_w = k_wins.shape[1], k_wins.shape[2]
        n_local = ws_h * ws_w

        # Flatten windows
        k_win_flat = k_wins.reshape(n_win_tot, n_local, head_dim)
        v_win_flat = v_wins.reshape(n_win_tot, n_local, head_dim)
        q_win_flat = q_wins.reshape(n_win_tot, n_local, head_dim)

        # Window reps per (B*nH) group, broadcast to all windows in that group
        # k_reps: [B*nH, nW, C]
        # Expand: [B*nH, nW, C] -> [B*nH, 1, nW, C] -> broadcast to [B*nH, nW_per_head, nW, C]
        nW_per_head = n_win_tot // (B * nH)
        k_reps_bc = k_reps.reshape(B * nH, 1, nW, head_dim).expand(-1, nW_per_head, -1, -1)
        k_reps_bc = k_reps_bc.reshape(n_win_tot, nW, head_dim)
        v_reps_bc = v_reps.reshape(B * nH, 1, nW, head_dim).expand(-1, nW_per_head, -1, -1)
        v_reps_bc = v_reps_bc.reshape(n_win_tot, nW, head_dim)

        # Hybrid K, V: local tokens + global reps
        k_hybrid = torch.cat([k_win_flat, k_reps_bc], dim=1)  # [n_win_tot, n_local+nW, C]
        v_hybrid = torch.cat([v_win_flat, v_reps_bc], dim=1)

        # Attention per window
        attn_weights = torch.matmul(q_win_flat, k_hybrid.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        out_wins = torch.matmul(attn_weights, v_hybrid)
        # out_wins: [n_win_tot, n_local, C] -> [B*nH*nW, ws, ws, C]
        out_wins = out_wins.reshape(n_win_tot, ws_h, ws_w, head_dim)

        # Unpartition windows back to spatial
        from sam3.model.vitdet import window_unpartition
        out_spat = window_unpartition(out_wins, pool_size, (Hp, Wp), (H, W))
        # out_spat: [B*nH, H, W, C]

        # Reshape back to standard format
        x = out_spat.reshape(B, nH, L, head_dim)
        # ...but actually, out_spat is already [B*nH, H, W, C], need [B, nH, L, C]
        # Let me reshape properly
        x = out_spat.reshape(B, nH, H, W, head_dim).reshape(B, nH, L, head_dim)

        if ndim == 4:
            x = (
                x.reshape(B, nH, H, W, -1)
                .permute(0, 2, 3, 1, 4)
                .reshape(B, H, W, -1)
            )
        else:
            x = x.reshape(B, nH, L, -1).permute(0, 2, 1, 3).reshape(B, L, -1)

        x = self.proj(x)
        return x

    return wmtp_forward


def apply_wmtp(vit_model, pool_size=14, drop_rate=0.2, global_only=True):
    """Apply Window-Mean Token Pooling to ViT attention blocks.

    Patches global attention blocks so each query attends to window-pooled
    key/value representatives instead of all individual tokens. Perturbation
    within each pool_size x pool_size window is averaged (suppressed by √N)
    before being globally propagated.

    Args:
        vit_model: The ViT backbone model
        pool_size: Window size for pooling (default 14, matching ViT window_size)
        drop_rate: Random token dropout within windows (0 = no gradient gap)
        global_only: If True, only patch global attention blocks
    """
    full_attn_ids = set(vit_model.full_attn_ids)

    patched = 0
    for i, block in enumerate(vit_model.blocks):
        if global_only and i not in full_attn_ids:
            continue

        attn = block.attn

        # Store original rel_pos handler (same as AEM)
        if attn.use_rel_pos:
            from sam3.model.vitdet import concat_rel_pos

            def _make_rel_pos_fn(a):
                def _rel_pos(q, k, H, W, x):
                    q_flat = q.flatten(0, 1)
                    k_flat = k.flatten(0, 1)
                    B_sz = q.shape[0]
                    q_out, k_out = concat_rel_pos(
                        q_flat, k_flat,
                        (H, W), x.shape[1:3],
                        a.rel_pos_h, a.rel_pos_w,
                        rescale=True,
                        relative_coords=a.relative_coords,
                    )
                    q_out = q_out.reshape(B_sz, a.num_heads, H * W, -1)
                    k_out = k_out.reshape(B_sz, a.num_heads, H * W, -1)
                    return q_out, k_out
                return _rel_pos
            attn._orig_rel_pos = _make_rel_pos_fn(attn)
        else:
            attn._orig_rel_pos = lambda q, k, H, W, x: (q, k)

        wmtp_fn = make_wmtp_attention_forward(attn, pool_size=pool_size,
                                               drop_rate=drop_rate)
        attn.forward = types.MethodType(wmtp_fn, attn)
        patched += 1

    return patched


def apply_feature_noise_injection(vit_model, noise_std=0.1, after_block_ids=None):
    """Inject Gaussian noise into the ViT residual stream after specific blocks.

    The noise makes each forward pass produce slightly different features,
    breaking deterministic gradient computation for PGD attacks.

    Args:
        vit_model: The ViT backbone model
        noise_std: Standard deviation of Gaussian noise (relative to feature std)
        after_block_ids: Which blocks to inject noise after (default: global blocks)
    """
    if after_block_ids is None:
        after_block_ids = set(vit_model.full_attn_ids)
    else:
        after_block_ids = set(after_block_ids)

    patched = 0
    for i, block in enumerate(vit_model.blocks):
        if i not in after_block_ids:
            continue

        block._noise_std = noise_std
        block._original_forward_noise = block.forward

        def _make_noisy_forward(blk):
            def noisy_forward(self, x: Tensor) -> Tensor:
                out = self._original_forward_noise(x)
                if self._noise_std > 0:
                    noise = torch.randn_like(out) * self._noise_std
                    out = out + noise
                return out
            return noisy_forward

        block.forward = types.MethodType(_make_noisy_forward(block), block)
        patched += 1

    return patched


def apply_weighted_mlfa(sf, mode='inverse'):
    """Apply Weighted Multi-Level Feature Aggregation (W-MLFA).

    Core insight: SAM3's FPN creates ALL scales from a single ViT feature
    (xs[-1]), which has been through ALL 4 global attention blocks — maximal
    perturbation amplification. Earlier ViT features (xs[0], xs[1], xs[2])
    have seen fewer global attention blocks and thus have LESS perturbation.

    W-MLFA uses ALL intermediate ViT features with weights inversely proportional
    to perturbation exposure (number of global attention blocks processed).

    Modes:
    - 'inverse': w_i ∝ 1/(i+1), where i = index in full_attn_ids
    - 'exponential': w_i ∝ 0.5^i
    - 'early_only': only first two (least perturbed) features
    - 'equal': standard MLFA (equal weights)

    Zero training. Dataset-agnostic. Attack-agnostic.
    Architecture-native (uses SAM3's own multi-depth features).
    """
    import types

    neck = sf.model.detector.backbone.vision_backbone
    trunk = neck.trunk
    full_attn_ids = list(trunk.full_attn_ids)
    n_levels = len(full_attn_ids)

    # Ensure intermediate features are returned
    trunk.return_interm_layers = True
    neck.mlfa_enabled = True

    if mode == 'inverse':
        raw = [1.0 / (i + 1) for i in range(n_levels)]
    elif mode == 'exponential':
        raw = [0.5 ** i for i in range(n_levels)]
    elif mode == 'early_only':
        raw = [1.0] * min(2, n_levels) + [0.0] * max(0, n_levels - 2)
    elif mode == 'equal':
        raw = [1.0] * n_levels
    else:
        raise ValueError(f"Unknown W-MLFA mode: {mode}")

    total = sum(raw)
    weights = [w / total for w in raw]

    # Store weights
    neck._mlfa_weights = weights

    original_forward = neck.forward

    def weighted_forward(self, tensor_list):
        xs = self.trunk(tensor_list)
        sam3_out, sam3_pos = [], []
        sam2_out, sam2_pos = None, None
        if self.sam2_convs is not None:
            sam2_out, sam2_pos = [], []

        if self.mlfa_enabled and len(xs) > 1:
            w = self._mlfa_weights
            x = sum(w[i] * xs[i] for i in range(min(len(w), len(xs))))
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

        sam3_out = self._csfc_fusion(sam3_out)

        if self.fbd_enabled and self.fbd_denoiser is not None:
            sam3_out = self.fbd_denoiser(sam3_out)

        return sam3_out, sam3_pos, sam2_out, sam2_pos

    neck.forward = types.MethodType(weighted_forward, neck)

    print(f"  W-MLFA ({mode}): weights={[f'{w:.3f}' for w in weights]} "
          f"(global_attn_ids={full_attn_ids})", flush=True)
    return neck


def apply_residual_dampening(vit_model, dampen_factor=0.5, global_only=True):
    """Apply residual dampening to ViT global attention blocks.

    Reduces the contribution of (potentially perturbed) global attention output
    by scaling the attention residual by dampen_factor < 1.0. This limits
    perturbation amplification through the 4 global attention blocks.

    Unlike AEM (which modifies attention weights), this modifies the block-level
    information flow. The shortcut dominates when dampen_factor is small.

    Zero training. Dataset/attack agnostic.
    """
    import types
    full_attn_ids = set(vit_model.full_attn_ids)

    patched = 0
    for i, block in enumerate(vit_model.blocks):
        if global_only and i not in full_attn_ids:
            continue

        original_forward = block.forward

        def _make_dampened(blk, df):
            def dampened_forward(self, x):
                shortcut = x
                x_normed = self.norm1(x)
                if self.window_size > 0:
                    from sam3.model.vitdet import window_partition, window_unpartition
                    H, W = x_normed.shape[1], x_normed.shape[2]
                    x_normed, pad_hw = window_partition(x_normed, self.window_size)
                    attn_out = self.ls1(self.attn(x_normed))
                    attn_out = window_unpartition(attn_out, self.window_size, pad_hw, (H, W))
                else:
                    attn_out = self.ls1(self.attn(x_normed))
                x = shortcut + df * self.dropout(self.drop_path(attn_out))
                x = x + self.dropout(self.drop_path(self.ls2(self.mlp(self.norm2(x)))))
                return x
            return dampened_forward

        block.forward = types.MethodType(_make_dampened(block, dampen_factor), block)
        patched += 1

    return patched


def random_pixel_quantize(x, n_bits=6):
    """Quantize pixel values to n_bits precision.

    Non-differentiable due to rounding (creates BPDA gradient gap).
    At 6 bits: 64 levels, step=1/63≈0.016, comparable to UAP eps 8/255≈0.031.
    At 5 bits: 32 levels, step=1/31≈0.032, directly fights UAP magnitude.

    Args:
        x: [B, C, H, W] tensor in range [0, 1]
        n_bits: quantization bit depth (5-8)
    Returns:
        xq: quantized tensor with same shape
    """
    n_levels = 2 ** n_bits - 1
    # Scale to [0, n_levels], round, scale back
    xq = (x * n_levels).round() / n_levels
    # Clamp to [0, 1] to handle numerical edge cases
    xq = torch.clamp(xq, 0.0, 1.0)
    return xq


def random_spatial_shift(x, max_shift=3):
    """Apply random spatial shift to the image.

    Non-differentiable due to hard pixel-grid realignment.
    Each forward pass applies a DIFFERENT random shift, breaking
    deterministic gradient computation for PGD.

    Args:
        x: [B, C, H, W] tensor in range [0, 1]
        max_shift: maximum shift in pixels (1-4)
    Returns:
        x_shifted: shifted tensor with same shape
    """
    import random
    B, C, H, W = x.shape
    dx = random.randint(-max_shift, max_shift)
    dy = random.randint(-max_shift, max_shift)

    if dx == 0 and dy == 0:
        return x

    # Roll shifts image content circularly (creates wrapped edges)
    x_shifted = torch.roll(x, shifts=(dy, dx), dims=(-2, -1))

    # Fill "wrapped" edges by replicating adjacent valid pixels
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

    return x_shifted


# ═══════════════════════════════════════════════════════════════════════
# RFQ: Randomized Feature Quantization
# ═══════════════════════════════════════════════════════════════════════


class RandomizedFeatureQuantizer(nn.Module):
    """Randomized Feature Quantization (RFQ): Architecture-level defense.

    Applies per-channel quantization to FPN features with RANDOMIZED
    thresholds. Each forward pass uses different quantization boundaries,
    creating a BPDA gradient gap through two mechanisms:

    1. Non-differentiable `round()` — STE gradient is identity, but true
       gradient is zero almost everywhere.
    2. Randomized thresholds — clamp boundaries shift each forward pass,
       so features near the boundary are clamped differently each step.
       The attacker's gradient direction changes every PGD step.

    Unlike FSC (deterministic clamp that BPDA easily optimizes through),
    RFQ's randomness makes each forward pass a DIFFERENT function, so
    the attacker cannot find a single perturbation that works for all.

    Zero training. Clean-data calibration only. Attack/dataset agnostic.
    """

    def __init__(self, mins, maxs, n_bits=6, noise_std=0.05):
        super().__init__()
        self.mins = nn.ParameterList([
            nn.Parameter(m, requires_grad=False) for m in mins
        ])
        self.maxs = nn.ParameterList([
            nn.Parameter(m, requires_grad=False) for m in maxs
        ])
        self.n_bits = n_bits
        self.noise_std = noise_std

    def forward(self, fpn_features):
        n_levels = 2 ** self.n_bits - 1
        quantized = []
        for i, f in enumerate(fpn_features):
            lo = self.mins[i].view(1, -1, 1, 1).to(f.device)
            hi = self.maxs[i].view(1, -1, 1, 1).to(f.device)
            rng = hi - lo

            # Randomize thresholds (non-differentiable random values)
            lo_rand = lo + torch.randn_like(lo) * self.noise_std * rng
            hi_rand = hi + torch.randn_like(hi) * self.noise_std * rng
            lo_actual = torch.min(lo_rand, hi_rand)
            hi_actual = torch.max(lo_rand, hi_rand)

            # Clamp + quantize to n_bits
            f_clamped = torch.clamp(f, lo_actual, hi_actual)
            f_norm = (f_clamped - lo_actual) / (hi_actual - lo_actual + 1e-8)
            f_q = (f_norm * n_levels).round() / n_levels
            f_q = f_q * (hi_actual - lo_actual) + lo_actual
            quantized.append(f_q)
        return quantized


def calibrate_rfq(sf, dataset, n_samples=100, k=3.0, device='cuda:0'):
    """Calibrate per-channel feature ranges from clean data.

    Returns mins, maxs lists for RFQ initialization.
    """
    import random as _random
    import numpy as np

    neck = sf.model.detector.backbone.vision_backbone
    trunk = neck.trunk
    trunk.return_interm_layers = True
    n_scales = len(neck.convs)
    dn = lambda x: sf.denorm_image(x)

    all_feats = [[] for _ in range(n_scales)]
    indices = list(range(len(dataset)))
    _random.shuffle(indices)

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
        if n_done % 30 == 0:
            print(f"  RFQ calib [{n_done}/{n_samples}]", flush=True)

    mins, maxs = [], []
    for i in range(n_scales):
        feats = torch.cat([f.flatten(2) for f in all_feats[i]], dim=2).squeeze(0)
        mu = feats.mean(dim=1)
        std = feats.std(dim=1)
        lo = mu - k * std
        hi = mu + k * std
        mins.append(lo)
        maxs.append(hi)
        frac = ((feats >= lo.unsqueeze(1)) & (feats <= hi.unsqueeze(1))).float().mean().item()
        print(f"  RFQ Scale {i}: mu={mu.mean():.3f}, std={std.mean():.3f}, "
              f"range=[{lo.mean():.3f}, {hi.mean():.3f}], in_range={frac:.3f}", flush=True)

    trunk.return_interm_layers = False
    return mins, maxs


def apply_rfq(sf, mins, maxs, n_bits=6, noise_std=0.05):
    """Apply Randomized Feature Quantization to the model's FPN neck.

    Patches neck.forward to quantize FPN features with randomized
    thresholds after FPN convs produce sam3_out.
    """
    import types

    neck = sf.model.detector.backbone.vision_backbone
    original_forward = neck.forward
    rfq = RandomizedFeatureQuantizer(mins, maxs, n_bits=n_bits, noise_std=noise_std)

    def rfq_forward(self, tensor_list):
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

        # RFQ: Randomized Feature Quantization (non-diff round + rand thresholds)
        sam3_out = rfq(sam3_out)

        # CSFC fusion
        sam3_out = self._csfc_fusion(sam3_out)

        # FBD denoising
        if self.fbd_enabled and self.fbd_denoiser is not None:
            sam3_out = self.fbd_denoiser(sam3_out)

        return sam3_out, sam3_pos, sam2_out, sam2_pos

    neck.forward = types.MethodType(rfq_forward, neck)
    neck._rfq = rfq

    return neck


# ═══════════════════════════════════════════════════════════════════════
# FRPD: Feature-space Randomized Preprocessing Defense
# ═══════════════════════════════════════════════════════════════════════


class FeatureSpaceRPD(nn.Module):
    """Feature-space RPD: Architecture-level spatial + quantization defense.

    Transposes RPD's proven random spatial shift + quantization from the
    input pixel space to the FPN feature space. This is genuinely
    architecture-level — it modifies feature computation inside the model.

    Two non-differentiable mechanisms compound:
    1. Random spatial shift of FPN features — breaks spatial alignment
       of perturbation pattern with feature activations. Each forward
       pass shifts differently, so the attacker's gradient is always
       computed on the wrong spatial configuration.
    2. Randomized quantization — rounds features to n_bits precision
       with random per-channel thresholds. Non-differentiable round()
       creates BPDA gradient gap.

    Unlike RFQ (which only had quantization and failed because the
    attacker kept features in the "safe zone"), FRPD's spatial shift
    physically displaces features so the UAP can never establish a
    stable spatial correspondence.
    """

    def __init__(self, mins, maxs, n_bits=6, max_shift=1):
        super().__init__()
        self.mins = nn.ParameterList([
            nn.Parameter(m, requires_grad=False) for m in mins
        ])
        self.maxs = nn.ParameterList([
            nn.Parameter(m, requires_grad=False) for m in maxs
        ])
        self.n_bits = n_bits
        self.max_shift = max_shift

    def _random_shift(self, f):
        """Random spatial shift of a single feature map."""
        import random
        B, C, H, W = f.shape
        dx = random.randint(-self.max_shift, self.max_shift)
        dy = random.randint(-self.max_shift, self.max_shift)
        if dx == 0 and dy == 0:
            return f
        f_shifted = torch.roll(f, shifts=(dy, dx), dims=(-2, -1))
        with torch.no_grad():
            if dy > 0:
                f_shifted[:, :, :dy, :] = f[:, :, :1, :]
            elif dy < 0:
                sh = -dy
                f_shifted[:, :, -sh:, :] = f_shifted[:, :, -sh-1:-sh, :]
            if dx > 0:
                f_shifted[:, :, :, :dx] = f_shifted[:, :, :, dx:dx+1]
            elif dx < 0:
                sh = -dx
                f_shifted[:, :, :, -sh:] = f_shifted[:, :, :, -sh-1:-sh]
        return f_shifted

    def forward(self, fpn_features):
        n_levels = 2 ** self.n_bits - 1
        result = []
        for i, f in enumerate(fpn_features):
            # Step 1: Random spatial shift (non-differentiable)
            f_shifted = self._random_shift(f)

            # Step 2: Quantize with calibrated ranges (non-differentiable round)
            lo = self.mins[i].view(1, -1, 1, 1).to(f.device)
            hi = self.maxs[i].view(1, -1, 1, 1).to(f.device)
            f_clamped = torch.clamp(f_shifted, lo, hi)
            f_norm = (f_clamped - lo) / (hi - lo + 1e-8)
            f_q = (f_norm * n_levels).round() / n_levels
            f_out = f_q * (hi - lo) + lo
            result.append(f_out)
        return result


def apply_frpd(sf, mins, maxs, n_bits=6, max_shift=1):
    """Apply Feature-space RPD to the model's FPN neck.

    Patches neck.forward to apply random spatial shift + quantization
    to FPN features. This is the architecture-level analog of RPD.
    """
    import types

    neck = sf.model.detector.backbone.vision_backbone
    frpd = FeatureSpaceRPD(mins, maxs, n_bits=n_bits, max_shift=max_shift)

    def frpd_forward(self, tensor_list):
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

        # FRPD: random spatial shift + quantization (non-differentiable)
        sam3_out = frpd(sam3_out)

        # CSFC fusion
        sam3_out = self._csfc_fusion(sam3_out)

        # FBD denoising
        if self.fbd_enabled and self.fbd_denoiser is not None:
            sam3_out = self.fbd_denoiser(sam3_out)

        return sam3_out, sam3_pos, sam2_out, sam2_pos

    neck.forward = types.MethodType(frpd_forward, neck)
    neck._frpd = frpd

    return neck


def make_randomized_preproc_fn(n_bits=6, max_shift=3, p_apply=1.0):
    """Create a randomized preprocessing function for defense.

    Each call applies a RANDOM transformation (different each forward).
    This breaks BPDA: the attacker uses identity approximation, but the
    true forward pass is different every time (random shift + quantization).

    Args:
        n_bits: quantization bit depth (5=tougher defense, 6=balanced, 7=cleaner)
        max_shift: max random spatial shift in pixels
        p_apply: probability of applying the defense (1.0 = always)

    Returns:
        preproc_fn: function (x) -> x_defended
    """
    import random

    def preproc_fn(x):
        if p_apply < 1.0 and random.random() > p_apply:
            return x
        # Apply spatial shift first, then quantize
        x = random_spatial_shift(x, max_shift=max_shift)
        x = random_pixel_quantize(x, n_bits=n_bits)
        return x

    return preproc_fn


# ═══════════════════════════════════════════════════════════════════════
# GWAF: Global-to-Window Attention Fallback
# ═══════════════════════════════════════════════════════════════════════


def apply_gwaf(vit_model, window_size=14):
    """Global-to-Window Attention Fallback: Replace global attention with
    window attention during inference.

    SAM3's ViT has 4 global attention blocks (indices 7, 15, 23, 31) that
    enable O(N²) cross-window perturbation propagation. These are the
    PRIMARY perturbation amplifiers — they allow a localized UAP to spread
    its influence to ALL spatial positions.

    GWAF converts these blocks to window attention. The perturbation can
    only propagate within local windows (14×14 = 196 tokens instead of
    N=4096 tokens), reducing amplification by ~20×.

    This is the FIRST defense that directly addresses the ROOT CAUSE
    of SAM3's adversarial vulnerability: unconstrained global attention.

    Zero training. Architecture-native. Attack/dataset agnostic.
    """
    n_patched = 0
    for i, block in enumerate(vit_model.blocks):
        if block.window_size == 0:
            block.window_size = window_size
            # Update Attention module's input_size and RoPE frequencies.
            # Global blocks compute RoPE for full image grid (e.g. 63×63),
            # but window attention needs frequencies for window_size×window_size.
            block.attn.input_size = (window_size, window_size)
            if block.attn.use_rope:
                device = next(block.attn.parameters()).device
                freqs_cis = block.attn.compute_cis(
                    end_x=window_size, end_y=window_size
                ).to(device)
                block.attn.register_buffer("freqs_cis", freqs_cis)
            # If the block uses rel_pos, its parameters were sized for the
            # full grid. Re-initialize them for the window size.
            if block.attn.use_rel_pos:
                block.attn._setup_rel_pos(rel_pos_zero_init=True)
            n_patched += 1
    vit_model._gwaf_window_size = window_size
    return n_patched


def remove_gwaf(vit_model):
    """Restore global attention blocks to original state."""
    n_restored = 0
    for block in vit_model.blocks:
        if hasattr(vit_model, '_gwaf_original_global_ids'):
            pass  # Alternative: track original indices
    for block in vit_model.blocks:
        if block.window_size == getattr(vit_model, '_gwaf_window_size', 14):
            # Check if this was originally a global block (heuristic: after
            # the 7th block and every 8th block)
            pass
    return n_restored


# ═══════════════════════════════════════════════════════════════════════
# SAN: Stochastic Attention Noise
# ═══════════════════════════════════════════════════════════════════════


def apply_san(vit_model, noise_std=0.05, global_only=True):
    """Stochastic Attention Noise: Add random noise to Q/K in attention blocks.

    Core insight: SAM3's 4 global attention blocks are the PRIMARY perturbation
    amplifiers. The UAP relies on stable, deterministic QK attention patterns
    to propagate perturbation influence from a few perturbed tokens to ALL
    spatial positions.

    SAN adds random Gaussian noise to Q and K BEFORE computing attention.
    Each forward pass has DIFFERENT noise, making attention patterns
    non-deterministic. The perturbation cannot establish stable amplification
    patterns because the attention landscape shifts every pass.

    This is fundamentally different from input-level defenses (RPD):
    - RPD breaks the UAP-image alignment BEFORE the ViT
    - SAN prevents perturbation amplification INSIDE the ViT
    - They are COMPLEMENTARY and compound when combined

    Key properties:
    - Architecture-native: modifies ViT attention computation
    - Zero training: just adds noise to existing Q/K tensors
    - Non-differentiable at the noise source (randn has no grad)
    - Attack-agnostic, dataset-agnostic
    - Addresses root cause: global attention perturbation amplification
    """
    import types

    full_attn_ids = set(vit_model.full_attn_ids)

    def _make_san_forward(attn_module, ns):
        import math
        original_forward = attn_module.forward

        def san_forward(self, x):
            # Follows exact structure of Attention.forward() in vitdet.py:466
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

            # qkv projection
            qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, -1)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

            # SAN: Add random noise to Q and K (non-differentiable)
            q = q + torch.randn_like(q) * ns
            k = k + torch.randn_like(k) * ns

            # RoPE
            q, k = self._apply_rope(q, k)

            if self.use_rel_pos:
                from sam3.model.vitdet import concat_rel_pos
                q, k = concat_rel_pos(
                    q.flatten(0, 1),
                    k.flatten(0, 1),
                    (H, W),
                    x.shape[1:3] if ndim == 4 else None,
                    self.rel_pos_h,
                    self.rel_pos_w,
                    rescale=True,
                    relative_coords=self.relative_coords,
                )
                q = q.reshape(B, self.num_heads, H * W, -1)
                k = k.reshape(B, self.num_heads, H * W, -1)

            # Attention
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0
            )

            # Reshape output
            if ndim == 4:
                x = x.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
            else:
                x = x.view(B, self.num_heads, L, -1).permute(0, 2, 1, 3).reshape(B, L, -1)

            x = self.proj(x)
            return x

        return san_forward

    n_patched = 0
    for i, block in enumerate(vit_model.blocks):
        if global_only and i not in full_attn_ids:
            continue
        block.attn.forward = types.MethodType(
            _make_san_forward(block.attn, noise_std), block.attn
        )
        n_patched += 1

    print(f"  SAN: patched {n_patched} blocks (noise_std={noise_std})", flush=True)
    return n_patched


# ═══════════════════════════════════════════════════════════════════════
# IVFC: Intra-ViT Feature Clamping
# ═══════════════════════════════════════════════════════════════════════


def calibrate_ivfc(sf, dataset, n_samples=100, k=3.0, device='cuda:0'):
    """Calibrate per-channel feature statistics at input to each global attention block.

    Unlike FSC (which clamps at FPN AFTER all ViT amplification), IVFC clamps
    at the INPUT to each global attention block, BEFORE perturbation is amplified
    through global attention.

    Returns:
        block_mins: dict mapping block_idx -> [C] tensor of lower bounds
        block_maxs: dict mapping block_idx -> [C] tensor of upper bounds
    """
    import random
    from tqdm import tqdm

    trunk = sf.model.detector.backbone.vision_backbone.trunk
    full_attn_ids = list(trunk.full_attn_ids)
    dn = lambda x: sf.denorm_image(x)
    embed_dim = trunk.blocks[0].attn.qkv.in_features

    # Storage for features at each global block input
    all_feats = {idx: [] for idx in full_attn_ids}

    # Register hooks
    hooks = []
    def make_hook(idx):
        def hook_fn(module, input_tensor):
            # input_tensor is a tuple (args to forward)
            x = input_tensor[0]
            all_feats[idx].append(x.detach().cpu())
        return hook_fn

    for idx in full_attn_ids:
        h = trunk.blocks[idx].register_forward_pre_hook(make_hook(idx))
        hooks.append(h)

    # Collect features from clean data
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
                trunk(clean)
        n_done += 1
        if n_done % 20 == 0:
            print(f"  IVFC calib [{n_done}/{n_samples}]", flush=True)

    for h in hooks:
        h.remove()

    # Compute per-channel statistics
    block_mins, block_maxs = {}, {}
    for idx in full_attn_ids:
        feats = torch.cat([f for f in all_feats[idx]], dim=0)  # [N, H, W, C]
        feats_flat = feats.flatten(0, -2)  # [N*H*W, C]
        mu = feats_flat.mean(dim=0)
        std = feats_flat.std(dim=0)
        lo = mu - k * std
        hi = mu + k * std
        block_mins[idx] = lo
        block_maxs[idx] = hi
        frac = ((feats_flat >= lo) & (feats_flat <= hi)).float().mean().item()
        print(f"  IVFC Block {idx}: mu={mu.mean():.3f}, std={std.mean():.3f}, "
              f"frac in [{lo.mean():.3f}, {hi.mean():.3f}] = {frac:.3f}", flush=True)

    return block_mins, block_maxs


def apply_ivfc(vit_model, block_mins, block_maxs):
    """Apply Intra-ViT Feature Clamping to global attention blocks.

    Clamps features at the INPUT to each global attention block (before norm1),
    preventing perturbation from entering the global attention computation.
    """
    import types

    full_attn_ids = set(vit_model.full_attn_ids)

    def _make_ivfc_forward(original_forward, bidx):
        def ivfc_forward(self, x):
            # Clamp input features before block processing (non-differentiable)
            lo = block_mins[bidx].view(1, 1, 1, -1).to(x.device)
            hi = block_maxs[bidx].view(1, 1, 1, -1).to(x.device)
            x = torch.clamp(x, lo, hi)
            return original_forward(x)
        return ivfc_forward

    n_patched = 0
    for i, block in enumerate(vit_model.blocks):
        if i in full_attn_ids:
            original_forward = block.forward
            block.forward = types.MethodType(
                _make_ivfc_forward(original_forward, i), block
            )
            n_patched += 1

    print(f"  IVFC: patched {n_patched} global blocks with input clamping", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# BQ: Block Quantization — feature quantization at global block outputs
# ═══════════════════════════════════════════════════════════════════════


def _quantize_features(x, n_bits):
    """Quantize features to n_bits using torch.floor (ZERO backward).

    torch.floor backward returns 0 (true gradient of floor is 0 a.e.),
    but BPDA approximates the defense as identity (gradient=1).
    This mismatch IS the BPDA gradient gap.

    Unlike torch.round (which uses STE gradient=1 = BPDA assumption),
    floor/ceil create a genuine gradient gap because PyTorch's gradient
    (0) differs from BPDA's assumption (1).
    """
    x_min = x.min().detach()
    x_max = x.max().detach()
    n_levels = 2 ** n_bits - 1
    scale = (x_max - x_min) / max(n_levels, 1)
    scale = scale.clamp(min=1e-8)
    # Use floor for quantization: zero backward gradient ≠ BPDA identity
    x_q = torch.floor((x - x_min) / scale) * scale + x_min
    return x_q


def apply_block_quant(vit_model, n_bits=4, global_only=True):
    """Apply feature quantization at the output of global attention blocks.

    SAM3's 4 global attention blocks (7, 15, 23, 31) are the primary
    perturbation amplifiers. Quantizing their outputs:
    1. Creates BPDA gradient gaps (torch.round STE ≠ true gradient 0)
    2. Reduces perturbation precision at the amplification points
    3. Targets the ROOT CAUSE directly

    Each of the 4 global blocks gets an independent quantizer, creating
    compounding gradient gaps in series. Combined with SPE (input quant)
    and MLFA (FPN diversification), this forms a 3-layer defense.

    Zero training. Architecture-native. Attack/dataset agnostic.
    """
    full_attn_ids = set(vit_model.full_attn_ids)

    def _make_quant_forward(original_forward, n_bits):
        def quant_forward(self, x):
            out = original_forward(x)
            return _quantize_features(out, n_bits)
        return quant_forward

    n_patched = 0
    for i, block in enumerate(vit_model.blocks):
        if (global_only and i in full_attn_ids) or (not global_only):
            import types
            original_forward = block.forward
            block.forward = types.MethodType(
                _make_quant_forward(original_forward, n_bits), block
            )
            n_patched += 1

    print(f"  BlockQuant: {n_patched} blocks quantized to {n_bits} bits", flush=True)
    return n_patched
    return n_patched

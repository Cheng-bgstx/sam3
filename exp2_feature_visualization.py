"""
Exp2: Feature Layer Visualization Analysis
Analyze UAP disruption at different SAM3 feature levels.

Purpose:
  Show that SAM Route UAP (loss_t+loss_fea) causes more backbone corruption
  than Detector Route UAP (loss_text), explaining cross-route transfer.

Analyzed Layers:
  1. Shared Backbone Trunk output (ViT last layer)
  2. SAM3 Neck (Detector Route FPN) per scale
  3. SAM2 Neck (SAM Route FPN) per scale

Metrics:
  - Cosine Similarity (clean vs adversarial): lower = more disruption
  - L2 Distance: feature shift magnitude
  - Feature Magnitude Ratio: norm change

Usage:
  python exp2_feature_visualization.py \
      --uap_path uap_file/SA-Co_sam3_video_saco_yt1b_0_fusion_onlyimage.pth

  python exp2_feature_visualization.py \
      --uap_paths \
          uap_file/SA-Co_sam3_video_saco_yt1b_0_fusion_onlyimage.pth \
          uap_file/SA-Co_sam3_video_saco_yt1b_1_fusion.pth \
      --uap_labels "SAM_Route_loss_t_fea" "Det_Route_loss_text"
"""

import os
import sys
import csv
import json
import argparse
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[Warning] matplotlib not found; plots will be skipped.")


# ========== Model Loading ==========
def load_sam3_model_for_analysis(checkpoint_path, device="cuda"):
    try:
        from sam3_util import load_sam3_model
        sam_fwder = load_sam3_model(checkpoint_path, device=device)
        if hasattr(sam_fwder, 'forwarder'):
            sam_fwder = sam_fwder.forwarder
        return sam_fwder
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        sys.exit(1)


def load_uap(uap_path, device="cuda"):
    uap = torch.load(uap_path, map_location=device, weights_only=True)
    if uap.dim() == 3:
        uap = uap.unsqueeze(0)
    return uap.to(device)


# ========== Feature Hook Manager ==========
class FeatureHookManager:
    def __init__(self):
        self.features = OrderedDict()
        self._hooks = []

    def register_hook(self, module, name):
        def hook_fn(mod, inp, out, layer_name=name):
            if isinstance(out, torch.Tensor):
                self.features[layer_name] = out.detach().clone()
            elif isinstance(out, (tuple, list)):
                for idx, o in enumerate(out):
                    if isinstance(o, torch.Tensor):
                        self.features[f"{layer_name}_out{idx}"] = o.detach().clone()
                    elif isinstance(o, list) and o and isinstance(o[0], torch.Tensor):
                        for j, t in enumerate(o):
                            self.features[f"{layer_name}_out{idx}_s{j}"] = t.detach().clone()
            elif isinstance(out, dict):
                for k, v in out.items():
                    if isinstance(v, torch.Tensor):
                        self.features[f"{layer_name}_{k}"] = v.detach().clone()
                    elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                        for j, t in enumerate(v):
                            self.features[f"{layer_name}_{k}_s{j}"] = t.detach().clone()
        handle = module.register_forward_hook(hook_fn)
        self._hooks.append(handle)

    def clear(self):
        self.features.clear()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def setup_hooks(sam_fwder, hook_mgr):
    """Register hooks on SAM3 backbone layers."""
    backbone = None
    vision_backbone = None

    if hasattr(sam_fwder, 'model') and hasattr(sam_fwder.model, 'backbone'):
        backbone = sam_fwder.model.backbone
    elif hasattr(sam_fwder, 'detector') and hasattr(sam_fwder.detector, 'backbone'):
        backbone = sam_fwder.detector.backbone
    elif hasattr(sam_fwder, 'model') and hasattr(sam_fwder.model, 'detector'):
        if hasattr(sam_fwder.model.detector, 'backbone'):
            backbone = sam_fwder.model.detector.backbone

    if backbone is not None and hasattr(backbone, 'vision_backbone'):
        vision_backbone = backbone.vision_backbone

    hooked = []
    if vision_backbone is not None:
        if hasattr(vision_backbone, 'trunk'):
            hook_mgr.register_hook(vision_backbone.trunk, "shared_trunk")
            hooked.append("shared_trunk")

        if hasattr(vision_backbone, 'convs'):
            for i, conv in enumerate(vision_backbone.convs):
                hook_mgr.register_hook(conv, f"sam3_neck_s{i}")
                hooked.append(f"sam3_neck_s{i}")

        if hasattr(vision_backbone, 'sam2_convs') and vision_backbone.sam2_convs is not None:
            for i, conv in enumerate(vision_backbone.sam2_convs):
                hook_mgr.register_hook(conv, f"sam2_neck_s{i}")
                hooked.append(f"sam2_neck_s{i}")

    print(f"[Hooks] {len(hooked)} hooks: {hooked}")
    return hooked


# ========== Data Loading ==========
def load_sample_images(data_dir, ann_file, num_samples=10, target_size=1008):
    samples = []
    if os.path.exists(ann_file):
        with open(ann_file, 'r') as f:
            ann_data = json.load(f)
        videos = ann_data.get("videos", [])
        for vid in videos[:num_samples * 2]:
            file_names = vid.get("file_names", [])
            if not file_names:
                continue
            frame_path = os.path.join(data_dir, file_names[0])
            if os.path.exists(frame_path):
                img = Image.open(frame_path).convert("RGB")
                img = img.resize((target_size, target_size), Image.Resampling.BICUBIC)
                vid_id = vid.get("id", "")
                samples.append((np.array(img), f"v{vid_id}"))
            if len(samples) >= num_samples:
                break

    if not samples:
        print(f"[Warning] Scanning {data_dir} for images...")
        for root, _, files in os.walk(data_dir):
            for fname in sorted(files):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img = Image.open(os.path.join(root, fname)).convert("RGB")
                    img = img.resize((target_size, target_size), Image.Resampling.BICUBIC)
                    samples.append((np.array(img), fname))
                    if len(samples) >= num_samples:
                        break
            if len(samples) >= num_samples:
                break

    print(f"[Data] {len(samples)} samples loaded")
    return samples


# ========== Feature Metrics ==========
def compute_feature_metrics(clean_feats, adv_feats):
    results = OrderedDict()
    common = set(clean_feats.keys()) & set(adv_feats.keys())

    for key in sorted(common):
        c = clean_feats[key].float()
        a = adv_feats[key].float()
        if c.shape != a.shape:
            continue

        m = {"shape": str(list(c.shape))}

        if c.dim() == 4:
            cos = F.cosine_similarity(c, a, dim=1)
            l2 = (c - a).pow(2).sum(dim=1).sqrt()
        elif c.dim() == 3:
            cos = F.cosine_similarity(c, a, dim=-1)
            l2 = (c - a).pow(2).sum(dim=-1).sqrt()
        elif c.dim() >= 2:
            cos = F.cosine_similarity(c.flatten(1), a.flatten(1), dim=1)
            l2 = (c - a).pow(2).flatten(1).sum(dim=1).sqrt()
        else:
            cos = F.cosine_similarity(c.unsqueeze(0), a.unsqueeze(0))
            l2 = (c - a).pow(2).sum().sqrt().unsqueeze(0)

        m["cosine_sim_mean"] = cos.mean().item()
        m["cosine_sim_std"] = cos.std().item()
        m["cosine_sim_min"] = cos.min().item()
        m["l2_dist_mean"] = l2.mean().item()
        m["l2_dist_max"] = l2.max().item()

        cn = c.norm(p=2)
        an = a.norm(p=2)
        m["magnitude_ratio"] = (an / cn).item() if cn > 0 else float('inf')

        results[key] = m
    return results


def get_spatial_cosine_map(clean_feat, adv_feat):
    if clean_feat.dim() != 4 or adv_feat.dim() != 4:
        return None
    cos = F.cosine_similarity(clean_feat.float(), adv_feat.float(), dim=1)
    return cos[0].cpu().numpy()


# ========== Visualization ==========
def plot_cosine_heatmaps(clean_feats, adv_feats, uap_label, sample_name, output_dir):
    if not HAS_MPL:
        return
    layers_4d = [k for k in sorted(clean_feats.keys())
                 if k in adv_feats and clean_feats[k].dim() == 4 and adv_feats[k].dim() == 4]
    if not layers_4d:
        return

    n = len(layers_4d)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for i, key in enumerate(layers_4d):
        cos_map = get_spatial_cosine_map(clean_feats[key], adv_feats[key])
        if cos_map is not None:
            im = axes[i].imshow(cos_map, cmap='RdYlGn', vmin=-1, vmax=1)
            axes[i].set_title(f"{key}\nmean={cos_map.mean():.4f}", fontsize=8)
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    for i in range(len(layers_4d), len(axes)):
        axes[i].axis('off')

    fig.suptitle(f"Cosine Sim Heatmaps | UAP: {uap_label} | {sample_name}", fontsize=11)
    plt.tight_layout()
    save_dir = os.path.join(output_dir, "heatmaps")
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f"{uap_label}_{sample_name}.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_summary_bar(all_results, output_dir):
    if not HAS_MPL:
        return

    layers = sorted(set(
        l for metrics in all_results.values() for l in metrics
        if "cosine_sim_mean" in metrics.get(l, {})
    ))
    if not layers:
        return

    uap_labels = list(all_results.keys())
    n_uaps = len(uap_labels)
    x = np.arange(len(layers))
    width = 0.8 / max(n_uaps, 1)

    fig, ax = plt.subplots(figsize=(max(14, len(layers) * 1.5), 6))
    for idx, ul in enumerate(uap_labels):
        vals = [all_results[ul].get(l, {}).get("cosine_sim_mean", float('nan')) for l in layers]
        ax.bar(x + idx * width, vals, width, label=ul, alpha=0.85)

    ax.set_ylabel("Cosine Similarity (clean vs adv)")
    ax.set_xlabel("Feature Layer")
    ax.set_title("Per-Layer Feature Disruption by Different UAPs")
    ax.set_xticks(x + width * (n_uaps - 1) / 2)
    ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=7)
    ax.legend(fontsize=8)
    ax.set_ylim(-0.1, 1.1)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0.0, color='red', linestyle='--', alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "summary_cosine_bar.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Saved] {save_path}")


def plot_route_comparison(all_results, output_dir):
    if not HAS_MPL:
        return
    for uap_label, lm in all_results.items():
        sam3_d, sam2_d = {}, {}
        for key, metrics in lm.items():
            if "cosine_sim_mean" not in metrics:
                continue
            if "sam3_neck_s" in key:
                s = key.split("sam3_neck_s")[-1]
                sam3_d[f"s{s}"] = metrics["cosine_sim_mean"]
            elif "sam2_neck_s" in key:
                s = key.split("sam2_neck_s")[-1]
                sam2_d[f"s{s}"] = metrics["cosine_sim_mean"]

        if not sam3_d and not sam2_d:
            continue

        scales = sorted(set(list(sam3_d.keys()) + list(sam2_d.keys())))
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(scales))
        w = 0.35
        ax.bar(x - w / 2, [sam3_d.get(s, float('nan')) for s in scales],
               w, label='SAM3 Neck (Det Route)', color='#e74c3c', alpha=0.8)
        ax.bar(x + w / 2, [sam2_d.get(s, float('nan')) for s in scales],
               w, label='SAM2 Neck (SAM Route)', color='#3498db', alpha=0.8)
        ax.set_ylabel("Cosine Sim (clean vs adv)")
        ax.set_title(f"Route-Specific Disruption | UAP: {uap_label}")
        ax.set_xticks(x)
        ax.set_xticklabels(scales)
        ax.legend()
        ax.set_ylim(-0.1, 1.1)
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"route_compare_{uap_label}.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[Saved] {save_path}")


# ========== Analysis Pipeline ==========
def run_analysis(sam_fwder, uap_tensor, samples, uap_label, output_dir,
                 device="cuda", norm_mean=(0.5, 0.5, 0.5), norm_std=(0.5, 0.5, 0.5)):
    hook_mgr = FeatureHookManager()
    setup_hooks(sam_fwder, hook_mgr)

    mean = torch.tensor(norm_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(norm_std, device=device).view(1, 3, 1, 1)
    uap = uap_tensor.to(device)
    if uap.shape[-1] != 1008 or uap.shape[-2] != 1008:
        uap = F.interpolate(uap, size=(1008, 1008), mode='bilinear', align_corners=False)

    all_metrics = []
    count = 0

    # Find the forward_image function
    def get_backbone_forward(fwder):
        if hasattr(fwder, 'model') and hasattr(fwder.model, 'backbone'):
            return fwder.model.backbone.forward_image
        if hasattr(fwder, 'detector') and hasattr(fwder.detector, 'backbone'):
            return fwder.detector.backbone.forward_image
        if hasattr(fwder, 'model') and hasattr(fwder.model, 'detector'):
            if hasattr(fwder.model.detector, 'backbone'):
                return fwder.model.detector.backbone.forward_image
        return None

    fwd_img = get_backbone_forward(sam_fwder)
    if fwd_img is None:
        print("[Error] Cannot find backbone.forward_image")
        hook_mgr.remove_hooks()
        return {}

    for img_np, sname in tqdm(samples, desc=f"UAP: {uap_label}"):
        try:
            img_t = torch.from_numpy(img_np).permute(2, 0, 1).float().div_(255.0)
            img_t = img_t.unsqueeze(0).to(device)
            img_norm = (img_t - mean) / std

            # Clean forward
            hook_mgr.clear()
            with torch.no_grad():
                fwd_img(img_norm)
            clean_feats = dict(hook_mgr.features)

            # Adversarial forward
            img_adv = torch.clamp(img_t + uap, 0.0, 1.0)
            img_adv_norm = (img_adv - mean) / std
            hook_mgr.clear()
            with torch.no_grad():
                fwd_img(img_adv_norm)
            adv_feats = dict(hook_mgr.features)

            metrics = compute_feature_metrics(clean_feats, adv_feats)
            all_metrics.append(metrics)

            if count < 5:
                plot_cosine_heatmaps(clean_feats, adv_feats, uap_label, sname, output_dir)
            count += 1

        except Exception as e:
            print(f"[Error] {sname}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()

    hook_mgr.remove_hooks()

    if not all_metrics:
        return {}

    # Aggregate
    avg = OrderedDict()
    all_layers = set()
    for m in all_metrics:
        all_layers.update(m.keys())

    metric_keys = ["cosine_sim_mean", "cosine_sim_std", "cosine_sim_min",
                    "l2_dist_mean", "l2_dist_max", "magnitude_ratio"]
    for layer in sorted(all_layers):
        vals = {}
        for mk in metric_keys:
            v_list = [m[layer][mk] for m in all_metrics if layer in m and mk in m[layer]]
            if v_list:
                vals[mk] = float(np.mean(v_list))
        if vals:
            for m in all_metrics:
                if layer in m and "shape" in m[layer]:
                    vals["shape"] = m[layer]["shape"]
                    break
            avg[layer] = vals
    return avg


def save_csv(all_results, output_dir):
    path = os.path.join(output_dir, "feature_metrics.csv")
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for ul, lm in all_results.items():
        for ln, met in lm.items():
            row = {"uap": ul, "layer": ln}
            row.update(met)
            rows.append(row)
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[Saved] {path}")


def print_table(all_results):
    print("\n" + "=" * 95)
    print("FEATURE DISRUPTION SUMMARY")
    print("=" * 95)
    for ul, lm in all_results.items():
        print(f"\n--- UAP: {ul} ---")
        print(f"{'Layer':<40} {'CosSim':>8} {'L2Dist':>10} {'MagRatio':>10} {'Shape'}")
        print("-" * 95)
        for ln, met in lm.items():
            cs = met.get("cosine_sim_mean", float('nan'))
            l2 = met.get("l2_dist_mean", float('nan'))
            mr = met.get("magnitude_ratio", float('nan'))
            sh = met.get("shape", "?")
            print(f"{ln:<40} {cs:>8.4f} {l2:>10.4f} {mr:>10.4f} {sh}")


# ========== CLI ==========
def get_parser():
    p = argparse.ArgumentParser(description="Exp2: Feature Visualization")
    p.add_argument('--checkpoints', default='./sam3/checkpoints/sam3.pt')
    p.add_argument('--uap_path', default=None, type=str)
    p.add_argument('--uap_paths', nargs='+', default=None)
    p.add_argument('--uap_labels', nargs='+', default=None)
    p.add_argument('--data_dir', default='./data/media/saco_yt1b/JPEGImages_6fps/')
    p.add_argument('--ann_file', default='./data/annotation/saco_veval_yt1b_test_split.json')
    p.add_argument('--num_samples', default=10, type=int)
    p.add_argument('--output_dir', default='results/exp2_feature_vis')
    p.add_argument('--norm_mean', nargs=3, type=float, default=[0.5, 0.5, 0.5])
    p.add_argument('--norm_std', nargs=3, type=float, default=[0.5, 0.5, 0.5])
    return p


def main():
    args = get_parser().parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # UAP list
    uap_configs = []
    if args.uap_path:
        label = os.path.splitext(os.path.basename(args.uap_path))[0]
        uap_configs.append((args.uap_path, label))
    elif args.uap_paths:
        labels = args.uap_labels or [
            os.path.splitext(os.path.basename(p))[0] for p in args.uap_paths
        ]
        for p, l in zip(args.uap_paths, labels):
            uap_configs.append((p, l))
    else:
        uap_dir = "uap_file"
        if os.path.exists(uap_dir):
            for fn in sorted(os.listdir(uap_dir)):
                if fn.endswith('.pth'):
                    uap_configs.append((os.path.join(uap_dir, fn),
                                        os.path.splitext(fn)[0]))
        if not uap_configs:
            print("[Error] No UAP files. Use --uap_path or --uap_paths")
            sys.exit(1)

    print(f"[Config] {len(uap_configs)} UAP(s):")
    for p, l in uap_configs:
        print(f"  {l}: {p}")

    # Load model
    print("\n[Model] Loading SAM3...")
    sam_fwder = load_sam3_model_for_analysis(args.checkpoints, device=device)
    sam_fwder.eval()
    print("[Model] OK")

    # Load samples
    samples = load_sample_images(args.data_dir, args.ann_file, args.num_samples)
    if not samples:
        print("[Error] No images")
        sys.exit(1)

    # Run analysis per UAP
    all_results = OrderedDict()
    for uap_path, uap_label in uap_configs:
        print(f"\n{'=' * 60}")
        print(f"Analyzing: {uap_label}")
        print(f"{'=' * 60}")
        uap_tensor = load_uap(uap_path, device)
        print(f"  shape={uap_tensor.shape}, Linf={uap_tensor.abs().max():.6f}")

        avg = run_analysis(sam_fwder, uap_tensor, samples, uap_label,
                           args.output_dir, device,
                           tuple(args.norm_mean), tuple(args.norm_std))
        all_results[uap_label] = avg
        del uap_tensor
        torch.cuda.empty_cache()

    # Output
    print_table(all_results)
    save_csv(all_results, args.output_dir)
    plot_summary_bar(all_results, args.output_dir)
    plot_route_comparison(all_results, args.output_dir)
    print(f"\n[Done] Results in {args.output_dir}/")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Analyze an image and report useful statistics for game asset work."""

import argparse
import os
import sys

import cv2
import numpy as np


def format_size(bytes_val):
    """Format bytes as human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} TB"


def analyze_colors(image):
    """Analyze color statistics of an image."""
    bgr = image[:, :, :3]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    stats = {}
    stats["avg_brightness"] = float(hsv[:, :, 2].mean())
    stats["avg_saturation"] = float(hsv[:, :, 1].mean())
    stats["brightness_range"] = (int(hsv[:, :, 2].min()), int(hsv[:, :, 2].max()))
    stats["saturation_range"] = (int(hsv[:, :, 1].min()), int(hsv[:, :, 1].max()))

    # Dominant color (most common hue bucket).
    hue = hsv[:, :, 0].flatten()
    # Filter out very dark or desaturated pixels.
    mask = (hsv[:, :, 2].flatten() > 30) & (hsv[:, :, 1].flatten() > 30)
    if mask.sum() > 0:
        hue_filtered = hue[mask]
        hist, _ = np.histogram(hue_filtered, bins=18, range=(0, 180))
        dominant_bucket = hist.argmax()
        hue_names = [
            "red", "orange", "yellow", "yellow-green", "green", "green-cyan",
            "cyan", "cyan-blue", "blue", "blue-purple", "purple", "magenta",
            "pink-red", "red", "red-orange", "orange", "gold", "yellow",
        ]
        stats["dominant_hue"] = hue_names[dominant_bucket]
    else:
        stats["dominant_hue"] = "neutral/gray"

    # Unique color count (approximate via downscaled quantization).
    small = cv2.resize(bgr, (100, 100))
    quantized = (small // 16) * 16
    unique = len(set(map(tuple, quantized.reshape(-1, 3))))
    stats["color_richness"] = unique

    return stats


def analyze_depth(image, model_size):
    """Run depth estimation and recommend parallax layers."""
    from PIL import Image as PILImage

    h, w = image.shape[:2]

    # Lazy import to avoid loading torch when --depth is not used.
    from transformers import pipeline as hf_pipeline
    import torch

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model_map = {
        "small": "depth-anything/Depth-Anything-V2-Small-hf",
        "base": "depth-anything/Depth-Anything-V2-Base-hf",
        "large": "depth-anything/Depth-Anything-V2-Large-hf",
    }
    model_name = model_map[model_size]
    print(f"Loading {model_name} on {device}...")
    pipe = hf_pipeline(task="depth-estimation", model=model_name, device=device)

    rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(rgb)
    result = pipe(pil_img)
    depth = np.array(result["depth"], dtype=np.float32)

    # Normalize.
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 0:
        depth = (depth - d_min) / (d_max - d_min)

    # Resize to original dimensions if needed.
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

    stats = {}
    stats["depth_range"] = (float(depth.min()), float(depth.max()))
    stats["depth_mean"] = float(depth.mean())
    stats["depth_std"] = float(depth.std())

    # Recommend number of layers based on depth variance.
    # More depth variation = more useful layers.
    std = stats["depth_std"]
    if std < 0.10:
        rec = 2
        verdict = "low depth variation — 2 layers sufficient"
    elif std < 0.20:
        rec = 3
        verdict = "moderate depth variation — 3 layers recommended"
    elif std < 0.30:
        rec = 4
        verdict = "good depth variation — 4 layers for smooth parallax"
    else:
        rec = 5
        verdict = "high depth variation — 5 layers for rich parallax"

    stats["recommended_layers"] = rec
    stats["verdict"] = verdict

    # Show depth distribution across candidate layer counts.
    stats["layer_table"] = []
    for n in [2, 3, 4, 5]:
        thresholds = [np.percentile(depth.flatten(), 100.0 * i / n) for i in range(n + 1)]
        layer_info = []
        for i in range(n):
            lo, hi = thresholds[i], thresholds[i + 1]
            if i == n - 1:
                mask = (depth >= lo) & (depth <= hi)
            else:
                mask = (depth >= lo) & (depth < hi)
            pct = mask.sum() / depth.size * 100
            layer_info.append((lo, hi, pct))
        stats["layer_table"].append((n, layer_info))

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Analyze an image and report statistics for game asset work."
    )
    parser.add_argument("input", help="Path to the input image")
    parser.add_argument(
        "--depth", action="store_true",
        help="Run Depth Anything V2 for depth analysis and parallax layer recommendations.",
    )
    parser.add_argument(
        "--model", choices=["small", "base", "large"], default="small",
        help="Depth model size when --depth is used (default: small).",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        parser.error(f"Input file not found: {args.input}")

    image = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Error: could not read '{args.input}'.", file=sys.stderr)
        sys.exit(1)

    h, w = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    has_alpha = channels == 4
    file_size = os.path.getsize(args.input)
    fname = os.path.basename(args.input)

    # Basic info.
    print(f"Image: {fname}")
    print(f"  Dimensions: {w}×{h}")
    print(f"  Channels: {channels} ({'BGRA' if has_alpha else 'BGR'})")
    print(f"  File size: {format_size(file_size)}")

    if has_alpha:
        alpha = image[:, :, 3]
        fully_opaque = (alpha == 255).sum() / alpha.size * 100
        fully_transparent = (alpha == 0).sum() / alpha.size * 100
        print(f"  Alpha: {fully_opaque:.1f}% opaque, {fully_transparent:.1f}% transparent")

    # Color stats.
    colors = analyze_colors(image)
    print(f"\nColor:")
    print(f"  Dominant hue: {colors['dominant_hue']}")
    print(f"  Brightness: avg {colors['avg_brightness']:.0f}/255 (range {colors['brightness_range'][0]}–{colors['brightness_range'][1]})")
    print(f"  Saturation: avg {colors['avg_saturation']:.0f}/255 (range {colors['saturation_range'][0]}–{colors['saturation_range'][1]})")
    print(f"  Color richness: ~{colors['color_richness']} unique colors (quantized)")

    # Depth analysis (optional).
    if args.depth:
        print()
        depth = analyze_depth(image, args.model)
        print(f"\nDepth (Depth Anything V2 — {args.model}):")
        print(f"  Mean: {depth['depth_mean']:.3f}")
        print(f"  Std: {depth['depth_std']:.3f}")
        print(f"\nRecommended parallax layers: {depth['recommended_layers']}")
        print(f"  {depth['verdict']}")

        print(f"\n  {'Layers':<8}{'Coverage per layer'}")
        for n, layers in depth["layer_table"]:
            marker = " ✅" if n == depth["recommended_layers"] else ""
            coverages = ", ".join(f"{pct:.0f}%" for _, _, pct in layers)
            print(f"  {n:<8}{coverages}{marker}")

    print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyze an image and report useful statistics for game asset work."""

import argparse
import os
import sys

import cv2
import numpy as np

from tools.lib.console import console, Table, Panel


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
    print(f"Loading {model_name} on {device}...", file=sys.stderr)
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

    # Halo analysis: measure the actual thickness of each thin fragment
    # at layer boundaries and recommend a --clean-edges value.
    rec_n = stats["recommended_layers"]
    thresholds = [np.percentile(depth.flatten(), 100.0 * i / rec_n) for i in range(rec_n + 1)]
    masks = []
    for i in range(rec_n):
        lo, hi = thresholds[i], thresholds[i + 1]
        if i == rec_n - 1:
            mask = (depth >= lo) & (depth <= hi)
        else:
            mask = (depth >= lo) & (depth < hi)
        masks.append(mask)

    from scipy.ndimage import distance_transform_edt

    # For each layer, find fragments that vanish under erosion (halos)
    # and measure their thickness via distance transform.
    max_radius = 12
    kernel_max = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (max_radius * 2 + 1, max_radius * 2 + 1)
    )
    halo_thicknesses = []  # One thickness per halo fragment.
    total_halo_pixels = 0
    for m in masks:
        m_u8 = m.astype(np.uint8)
        eroded = cv2.erode(m_u8, kernel_max, iterations=1)
        # Halo pixels = those in original mask but not surviving max erosion+reconstruction.
        if eroded.any():
            from skimage.morphology import reconstruction
            restored = reconstruction(eroded, m_u8, method="dilation").astype(np.uint8)
            halo_mask = m_u8 - restored
        else:
            halo_mask = m_u8.copy()

        if halo_mask.sum() == 0:
            continue

        total_halo_pixels += int(halo_mask.sum())

        # Label each halo fragment and measure its max thickness.
        n_labels, labels, _, _ = cv2.connectedComponentsWithStats(halo_mask, connectivity=8)
        for label in range(1, n_labels):
            fragment = (labels == label).astype(np.uint8)
            # Distance transform: max distance from edge = half-thickness.
            dist = distance_transform_edt(fragment)
            thickness = int(np.ceil(dist.max() * 2))
            thickness = max(1, thickness)
            halo_thicknesses.append(thickness)

    # Build distribution.
    halo_stats = {}
    halo_stats["total_fragments"] = len(halo_thicknesses)
    halo_stats["total_pixels"] = total_halo_pixels
    halo_stats["pct_of_image"] = total_halo_pixels / depth.size * 100

    if halo_thicknesses:
        arr = np.array(halo_thicknesses)
        halo_stats["min"] = int(arr.min())
        halo_stats["max"] = int(arr.max())
        halo_stats["median"] = int(np.median(arr))
        # Distribution: what % of halos are caught at each radius.
        coverage = []
        for r in sorted(set([1, 2, 3, 4, 5, 6, 8, 10])):
            caught = int((arr <= r).sum())
            pct = caught / len(arr) * 100
            coverage.append((r, caught, pct))
        halo_stats["coverage"] = coverage

        # Recommend radius that catches ≥90% of halos.
        recommended_radius = int(arr.max())  # Fallback: catch all.
        for r, _, pct in coverage:
            if pct >= 90.0:
                recommended_radius = r
                break
    else:
        halo_stats["coverage"] = []
        recommended_radius = 0

    stats["halo_analysis"] = halo_stats
    stats["recommended_clean_edges"] = recommended_radius

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
    info_lines = [
        f"Dimensions: [cyan]{w}×{h}[/cyan]",
        f"Channels: {channels} ({'BGRA' if has_alpha else 'BGR'})",
        f"File size: {format_size(file_size)}",
    ]

    if has_alpha:
        alpha = image[:, :, 3]
        fully_opaque = (alpha == 255).sum() / alpha.size * 100
        fully_transparent = (alpha == 0).sum() / alpha.size * 100
        info_lines.append(f"Alpha: [green]{fully_opaque:.1f}%[/green] opaque, [yellow]{fully_transparent:.1f}%[/yellow] transparent")

    console.print(Panel("\n".join(info_lines), title=f"[bold]{fname}[/bold]", expand=False))

    # Color stats.
    colors = analyze_colors(image)
    color_table = Table(show_header=False, box=None, padding=(0, 2))
    color_table.add_column(style="dim")
    color_table.add_column()
    color_table.add_row("Dominant hue:", colors['dominant_hue'])
    color_table.add_row("Brightness:", f"avg {colors['avg_brightness']:.0f}/255 (range {colors['brightness_range'][0]}–{colors['brightness_range'][1]})")
    color_table.add_row("Saturation:", f"avg {colors['avg_saturation']:.0f}/255 (range {colors['saturation_range'][0]}–{colors['saturation_range'][1]})")
    color_table.add_row("Color richness:", f"~{colors['color_richness']} unique colors (quantized)")
    console.print("\n[bold]Color[/bold]")
    console.print(color_table)

    # Depth analysis (optional).
    if args.depth:
        console.print()
        depth = analyze_depth(image, args.model)
        console.print(f"\n[bold]Depth[/bold] (Depth Anything V2 — {args.model})")
        console.print(f"  Mean: {depth['depth_mean']:.3f}  Std: {depth['depth_std']:.3f}")

        layer_table = Table(title="Layer Coverage", box=None, padding=(0, 2))
        layer_table.add_column("Layers", justify="right")
        layer_table.add_column("Coverage per layer")
        for n, layers in depth["layer_table"]:
            marker = " ✅" if n == depth["recommended_layers"] else ""
            coverages = ", ".join(f"{pct:.0f}%" for _, _, pct in layers)
            layer_table.add_row(str(n), f"{coverages}{marker}")
        console.print(layer_table)

        halo = depth["halo_analysis"]
        console.print(f"\n[bold]Halo analysis[/bold] (for {depth['recommended_layers']} layers)")
        if halo["total_fragments"] > 0:
            console.print(f"  Fragments: {halo['total_fragments']} ({halo['total_pixels']} pixels, {halo['pct_of_image']:.2f}% of image)")
            console.print(f"  Thickness: min {halo['min']}px, median {halo['median']}px, max {halo['max']}px")

            halo_table = Table(box=None, padding=(0, 2))
            halo_table.add_column("Radius", justify="right")
            halo_table.add_column("Halos caught", justify="right")
            halo_table.add_column("Coverage", justify="right")
            for r, caught, pct in halo["coverage"]:
                marker = " ✅" if r == depth["recommended_clean_edges"] else ""
                halo_table.add_row(str(r), str(caught), f"{pct:.0f}%{marker}")
            console.print(halo_table)
        else:
            console.print("  No halos detected.")

        rec_ce = depth["recommended_clean_edges"]
        rec_lines = [
            f"Parallax layers: [bold]{depth['recommended_layers']}[/bold] ({depth['verdict']})",
        ]
        if rec_ce > 0:
            rec_lines.append(f"Clean edges: [bold]{rec_ce}px[/bold]")
        else:
            rec_lines.append("Clean edges: not needed (no significant halos detected)")
        console.print(Panel("\n".join(rec_lines), title="[bold]Recommendations[/bold]", expand=False))

    console.print()


if __name__ == "__main__":
    main()

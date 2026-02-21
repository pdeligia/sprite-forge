#!/usr/bin/env python3
"""Remix animation frames by retiming depth layers independently."""

import argparse
import glob
import os
import shutil
import sys

import cv2
import numpy as np

from tools.lib.console import console, Table, enable_hf_offline
from tools.lib.depth_utils import compute_depth_map, load_depth_pipeline
from tools.image_to_layers import split_layers


def find_static_mask(images, tolerance=2):
    """Find pixels that are identical across all frames.

    Args:
        images: List of BGR/BGRA images (same dimensions).
        tolerance: Max per-channel difference to consider a pixel static.

    Returns:
        Boolean mask (H, W) where True = static pixel.
    """
    stack = np.stack(images, axis=0)  # (N, H, W, C)
    max_diff = np.abs(
        stack.astype(np.int16) - stack[0:1].astype(np.int16)
    ).max(axis=0)  # (H, W, C)
    return np.all(max_diff <= tolerance, axis=2)  # (H, W)


def build_layer_frames(images, all_frame_layers, static_mask, n_layers):
    """Build per-layer frame sequences with static pixels separated.

    For each layer, extracts only the animated pixels from each frame.
    The static pixels form a separate base plate per layer.

    Args:
        images: List of original images.
        all_frame_layers: all_frame_layers[frame][layer] = RGBA image.
        static_mask: Boolean (H, W) mask of static pixels.
        n_layers: Number of depth layers.

    Returns:
        (static_plates, animated_frames) where:
        - static_plates[layer] = RGBA image of static pixels for that layer
        - animated_frames[layer][frame] = RGBA image of animated pixels only
    """
    n_frames = len(images)
    h, w = images[0].shape[:2]

    static_plates = []
    animated_frames = []

    for k in range(n_layers):
        # Static plate: pixels in this layer that are static across all frames.
        ref_layer = all_frame_layers[0][k]
        plate = ref_layer.copy()
        # Keep only static pixels.
        plate[~static_mask, 3] = 0
        static_plates.append(plate)

        # Animated frames: pixels in this layer that change between frames.
        layer_anim = []
        for f in range(n_frames):
            frame_layer = all_frame_layers[f][k].copy()
            # Remove static pixels — keep only animated ones.
            frame_layer[static_mask, 3] = 0
            layer_anim.append(frame_layer)
        animated_frames.append(layer_anim)

    return static_plates, animated_frames


def retime_layer(animated_frames, n_output, speed):
    """Retime a layer's animation frames using nearest-neighbor sampling.

    Args:
        animated_frames: List of RGBA images (original keyframes).
        n_output: Number of output frames to generate.
        speed: Speed multiplier (1.0 = original, 0.5 = half speed, 2.0 = double).

    Returns:
        List of n_output RGBA images.
    """
    n_input = len(animated_frames)
    result = []
    for t in range(n_output):
        # Map output frame t to source frame index.
        src_t = (t * speed) % n_input
        src_idx = int(src_t) % n_input
        result.append(animated_frames[src_idx])
    return result


def composite_frame(static_plates, animated_frame_per_layer):
    """Composite one output frame from static plates + animated layers.

    Layers are composited back-to-front using alpha blending.

    Args:
        static_plates: List of RGBA images (one per layer, back-to-front).
        animated_frame_per_layer: List of RGBA images (one per layer for this frame).

    Returns:
        RGBA image (H, W, 4) — the fully composited frame.
    """
    h, w = static_plates[0].shape[:2]
    result = np.zeros((h, w, 4), dtype=np.uint8)

    for k in range(len(static_plates)):
        # Merge static plate with animated pixels for this layer.
        plate = static_plates[k]
        anim = animated_frame_per_layer[k]

        # Animated pixels override static pixels (they're mutually exclusive).
        layer = plate.copy()
        anim_mask = anim[:, :, 3] > 0
        layer[anim_mask] = anim[anim_mask]

        # Alpha-composite this layer onto the result.
        alpha = layer[:, :, 3].astype(np.float32) / 255.0
        inv_alpha = 1.0 - alpha
        for c in range(3):
            result[:, :, c] = (
                alpha * layer[:, :, c] + inv_alpha * result[:, :, c]
            ).astype(np.uint8)
        result[:, :, 3] = np.clip(
            result[:, :, 3].astype(np.float32) + layer[:, :, 3].astype(np.float32),
            0, 255
        ).astype(np.uint8)

    return result


def dilate_fill_frame(frame):
    """Fill small transparent gaps by dilating neighboring pixels.

    Args:
        frame: RGBA image (H, W, 4) uint8.

    Returns:
        RGBA image with transparent gaps filled, fully opaque.
    """
    alpha = frame[:, :, 3]
    mask = (alpha == 0).astype(np.uint8) * 255
    if mask.max() == 0:
        return frame

    bgr = frame[:, :, :3].copy()
    remaining = mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    for _ in range(20):
        if remaining.max() == 0:
            break
        dilated = cv2.dilate(bgr, kernel, iterations=1)
        fill_mask = remaining > 0
        bgr[fill_mask] = dilated[fill_mask]
        remaining = cv2.erode(remaining, kernel, iterations=1)

    result = frame.copy()
    gap = mask > 0
    result[gap, :3] = bgr[gap]
    result[gap, 3] = 255
    return result


def process(pipe, files, output_dir, n_layers, n_output_frames, speeds,
            suffix, feather, clean_edges, depth_map, tolerance, fill):
    """Main remix pipeline."""
    n_input = len(files)
    console.print()
    console.print("[bold cyan]🎬 Remix[/bold cyan]")
    console.print()

    # Load all frames.
    images = []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Error: could not read '{f}'.", file=sys.stderr)
            return False
        images.append(img)

    h, w = images[0].shape[:2]
    console.print(f"  Input: [cyan]{n_input}[/cyan] frames, [cyan]{w}×{h}[/cyan] px")
    console.print(f"  Output: [cyan]{n_output_frames}[/cyan] frames, [cyan]{n_layers}[/cyan] layers")

    # Find static pixels.
    static_mask = find_static_mask(images, tolerance)
    total = static_mask.size
    n_static = static_mask.sum()
    n_animated = total - n_static
    console.print(
        f"  Static: [green]{n_static / total * 100:.1f}%[/green] "
        f"({n_static:,} px) — "
        f"Animated: [yellow]{n_animated / total * 100:.1f}%[/yellow] "
        f"({n_animated:,} px)"
    )
    console.print()

    # Compute depth and split each frame into layers.
    console.print("[bold cyan]🔍 Depth & Layers[/bold cyan]")
    console.print()
    all_frame_layers = []
    ref_depth = None
    for img in images:
        depth = compute_depth_map(pipe, img)
        if ref_depth is None:
            ref_depth = depth
        layers = split_layers(img, depth, n_layers, feather, clean_edges)
        all_frame_layers.append(layers)

    # Build static plates and per-layer animated frame sequences.
    static_plates, animated_frames = build_layer_frames(
        images, all_frame_layers, static_mask, n_layers
    )

    layer_table = Table(box=None, padding=(0, 2))
    layer_table.add_column("Layer", style="bold")
    layer_table.add_column("Opaque", justify="right")
    layer_table.add_column("Static", justify="right")
    layer_table.add_column("Animated", justify="right")
    layer_table.add_column("Empty", justify="right")
    layer_table.add_column("Speed", justify="right")
    layer_table.add_column("Role", style="dim")

    for k in range(n_layers):
        plate_pct = (static_plates[k][:, :, 3] > 0).sum() / total * 100
        anim_pct = (animated_frames[k][0][:, :, 3] > 0).sum() / total * 100
        opaque_pct = plate_pct + anim_pct
        empty_pct = 100.0 - opaque_pct
        if k == 0:
            role = "back"
        elif k == n_layers - 1:
            role = "front"
        else:
            role = f"mid{k}" if n_layers > 3 else "mid"
        layer_table.add_row(
            f"{k + 1}/{n_layers}",
            f"[bold]{opaque_pct:.1f}%[/bold]",
            f"[green]{plate_pct:.1f}%[/green]",
            f"[yellow]{anim_pct:.1f}%[/yellow]",
            f"[dim]{empty_pct:.1f}%[/dim]",
            f"[cyan]{speeds[k]}×[/cyan]",
            role,
        )

    console.print(layer_table)
    console.print()

    # Retime each layer.
    console.print("[bold cyan]🎞️  Retime & Composite[/bold cyan]")
    console.print()

    retimed = []
    for k in range(n_layers):
        retimed.append(retime_layer(animated_frames[k], n_output_frames, speeds[k]))

    # Composite output frames.
    basename = os.path.basename(files[0])
    name, ext = os.path.splitext(basename)
    if suffix and name.endswith(suffix):
        name = name[: -len(suffix)]
    # Strip trailing frame number (e.g., _01, _02) from name.
    if len(name) >= 3 and name[-3] == "_" and name[-2:].isdigit():
        name = name[:-3]

    pad = max(2, len(str(n_output_frames)))

    frame_table = Table(box=None, padding=(0, 2))
    frame_table.add_column("Frame", style="bold")
    frame_table.add_column("File")
    frame_table.add_column("Opaque", justify="right")
    if fill:
        frame_table.add_column("Filled", justify="right")
    frame_table.add_column("Empty", justify="right")

    for t in range(n_output_frames):
        anim_per_layer = [retimed[k][t] for k in range(n_layers)]
        frame = composite_frame(static_plates, anim_per_layer)

        pre_opaque = (frame[:, :, 3] > 0).sum()
        pre_opaque_pct = pre_opaque / total * 100

        filled_pct = 0.0
        if fill:
            frame = dilate_fill_frame(frame)
            post_opaque = (frame[:, :, 3] > 0).sum()
            filled_pct = (post_opaque - pre_opaque) / total * 100

        final_opaque = (frame[:, :, 3] > 0).sum()
        final_empty_pct = 100.0 - final_opaque / total * 100

        frame_num = str(t + 1).zfill(pad)
        out_name = f"{name}_{frame_num}{suffix}{ext}"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, frame)

        row = [
            f"{t + 1}/{n_output_frames}",
            out_name,
            f"[green]{pre_opaque_pct:.1f}%[/green]",
        ]
        if fill:
            row.append(f"[cyan]+{filled_pct:.1f}%[/cyan]")
        row.append(f"[dim]{final_empty_pct:.1f}%[/dim]")
        frame_table.add_row(*row)

    console.print(frame_table)

    if depth_map and ref_depth is not None:
        preview_path = os.path.join(output_dir, f"{name}_depth_map{suffix}.png")
        depth_vis = (ref_depth * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        cv2.imwrite(preview_path, depth_colored)
        console.print(f"  Depth map: [bold]{preview_path}[/bold]")

    console.print()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Remix animation frames by retiming depth layers independently."
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory of animation frame PNGs (e.g., from mp4-to-frames).",
    )
    parser.add_argument("--prefix", help="Filter input files by prefix")
    parser.add_argument(
        "--layers", type=int, default=3,
        help="Number of depth layers (default: 3)",
    )
    parser.add_argument(
        "--frames", type=int, default=None,
        help="Number of output frames (default: same as input frame count)",
    )
    parser.add_argument(
        "--speed", default=None,
        help="Per-layer speed multipliers, comma-separated back-to-front "
             "(e.g., '0.25,0.5,1.0'). Layers without a value default to 1.0.",
    )
    parser.add_argument(
        "--output-dir", default="./tmp/remix_animation",
        help="Output directory (default: ./tmp/remix_animation)",
    )
    parser.add_argument("--suffix", default="", help="Filename suffix to preserve (e.g., @3x)")
    parser.add_argument(
        "--feather", type=int, default=0,
        help="Edge feather radius in pixels (default: 0)",
    )
    parser.add_argument(
        "--clean-edges", default="off",
        help="Remove outline artifacts: erosion radius in pixels, or 'off' (default).",
    )
    parser.add_argument("--depth-map", action="store_true", help="Output a depth map visualization")
    parser.add_argument(
        "--no-fill", action="store_true",
        help="Disable filling small transparent gaps in output frames.",
    )
    parser.add_argument(
        "--tolerance", type=int, default=2,
        help="Max per-channel difference for static pixel detection (default: 2)",
    )
    parser.add_argument(
        "--model", choices=["small", "base", "large"], default="small",
        help="Depth Anything V2 model size (default: small).",
    )
    parser.add_argument(
        "--no-model", action="store_true",
        help="Skip depth model, use vertical gradient fallback (for testing/CI).",
    )
    parser.add_argument(
        "--fetch-latest-model", action="store_true",
        help="Force re-download HuggingFace models to get the latest version.",
    )
    args = parser.parse_args()

    if not args.fetch_latest_model:
        enable_hf_offline()

    if args.layers < 1:
        parser.error("Number of layers must be at least 1")

    # Collect input files.
    if not os.path.isdir(args.input_dir):
        parser.error(f"Input directory not found: {args.input_dir}")
    pattern = os.path.join(args.input_dir, "*.png")
    files = sorted(glob.glob(pattern))
    if args.prefix:
        files = [f for f in files if os.path.basename(f).startswith(args.prefix)]
    if len(files) < 2:
        parser.error(f"Need at least 2 input frames, found {len(files)}")

    n_input = len(files)
    n_output = args.frames if args.frames else n_input

    # Parse per-layer speeds.
    speeds = [1.0] * args.layers
    if args.speed:
        parts = args.speed.split(",")
        for i, p in enumerate(parts):
            if i >= args.layers:
                break
            try:
                speeds[i] = float(p.strip())
            except ValueError:
                parser.error(f"Invalid speed value: {p}")
            if speeds[i] <= 0:
                parser.error(f"Speed must be positive, got: {speeds[i]}")

    # Clean and recreate output directory.
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    pipe = None if args.no_model else load_depth_pipeline(args.model)

    if process(pipe, files, args.output_dir, args.layers, n_output, speeds,
               args.suffix, args.feather, args.clean_edges, args.depth_map,
               args.tolerance, not args.no_fill):
        console.print(
            f"Done — [bold]{n_output}[/bold] remixed frames → [bold]{args.output_dir}/[/bold]"
        )
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

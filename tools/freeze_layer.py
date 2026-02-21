#!/usr/bin/env python3
"""Freeze depth layers in a video by replacing them with a static reference frame."""

import argparse
import os
import sys

import cv2
import numpy as np
from rich.table import Table

from tools.lib.console import console
from tools.lib.depth_utils import compute_depth_map, load_depth_pipeline


def _make_combined_mask(depth, n_layers, freeze_indices, feather=5):
    """Create a soft mask covering all frozen layers.

    Splits depth [0,1] into n_layers equal bands. Layers are numbered 1=nearest
    to n_layers=farthest. Returns a float32 mask where 1.0 = frozen.
    """
    mask = np.zeros_like(depth, dtype=np.float32)
    for idx in freeze_indices:
        # Layer 1 = nearest (high depth), layer N = farthest (low depth).
        hi = 1.0 - (idx - 1) / n_layers
        lo = 1.0 - idx / n_layers
        mask = np.maximum(mask, ((depth >= lo) & (depth <= hi)).astype(np.float32))

    if feather > 0:
        ksize = feather * 2 + 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    return mask


def main():
    parser = argparse.ArgumentParser(
        description="Freeze depth layers in a video using a static reference frame."
    )
    parser.add_argument("--input", required=True, help="Input video file.")
    parser.add_argument(
        "--output", default="./tmp/freeze_layer/output.mp4",
        help="Output video file path (default: ./tmp/freeze_layer/output.mp4)",
    )
    parser.add_argument(
        "--layers", type=int, required=True,
        help="Number of depth layers to split into.",
    )
    parser.add_argument(
        "--freeze", required=True,
        help="Comma-separated layer indices to freeze (1=nearest, N=farthest). "
        "E.g. '2,4' freezes layers 2 and 4.",
    )
    parser.add_argument(
        "--reference", type=int, default=0,
        help="Frame index to use as static reference (default: 0 = first frame).",
    )
    parser.add_argument(
        "--feather", type=int, default=5,
        help="Edge feathering radius in pixels (default: 5, 0 = hard edges).",
    )
    parser.add_argument(
        "--model-size", choices=["small", "base", "large"], default="small",
        help="Depth model size (default: small).",
    )
    args = parser.parse_args()

    # Parse freeze indices.
    try:
        freeze_indices = [int(x.strip()) for x in args.freeze.split(",")]
    except ValueError:
        parser.error(f"Invalid --freeze '{args.freeze}'. Use comma-separated integers.")

    for idx in freeze_indices:
        if idx < 1 or idx > args.layers:
            parser.error(f"Layer {idx} out of range (1–{args.layers}).")

    animated = [i for i in range(1, args.layers + 1) if i not in freeze_indices]

    # Open video.
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        parser.error(f"Could not open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0

    # Print header.
    console.print()
    console.print("[bold cyan]🧊 Freeze Layer[/bold cyan]")
    console.print()

    info_table = Table(show_header=False, show_edge=False, pad_edge=False, padding=(0, 2))
    info_table.add_column(style="bold")
    info_table.add_column()
    info_table.add_row("Input", f"[cyan]{os.path.abspath(args.input)}[/cyan]")
    info_table.add_row("Resolution", f"[cyan]{vw}×{vh}[/cyan] at [cyan]{fps:.1f}[/cyan] fps")
    info_table.add_row("Frames", f"[cyan]{frame_count}[/cyan] ({duration:.1f}s)")
    info_table.add_row("Layers", f"[cyan]{args.layers}[/cyan] (1=nearest, {args.layers}=farthest)")
    freeze_str = ", ".join(str(i) for i in sorted(freeze_indices))
    anim_str = ", ".join(str(i) for i in animated)
    info_table.add_row("Frozen", f"[cyan]{freeze_str}[/cyan]")
    info_table.add_row("Animated", f"[cyan]{anim_str}[/cyan]")
    info_table.add_row("Reference", f"[cyan]frame {args.reference}[/cyan]")
    info_table.add_row("Feather", f"[cyan]{args.feather}px[/cyan]")
    info_table.add_row("Model", f"[cyan]{args.model_size}[/cyan]")
    console.print(info_table)
    console.print()

    # Read reference frame.
    console.print("  Reading reference frame...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.reference)
    ret, ref_frame = cap.read()
    if not ret:
        print(f"Error: could not read reference frame {args.reference}.", file=sys.stderr)
        sys.exit(1)

    # Load depth model.
    console.print("  Loading depth model...")
    pipe = load_depth_pipeline(args.model_size)

    # Compute reference depth map.
    console.print("  Computing reference depth map...")
    ref_depth = compute_depth_map(pipe, ref_frame)

    # Setup output.
    abs_output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(abs_output) or ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(abs_output, fourcc, fps, (vw, vh))
    if not writer.isOpened():
        print(f"Error: could not create video '{abs_output}'.", file=sys.stderr)
        sys.exit(1)

    # Process every frame.
    console.print(f"  Processing {frame_count} frames...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    total = 0
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Compute per-frame depth to get the mask for this frame.
        depth = compute_depth_map(pipe, frame)
        mask = _make_combined_mask(depth, args.layers, freeze_indices, args.feather)

        # Blend: where mask is active, use reference frame; elsewhere, use current frame.
        mask_3ch = mask[:, :, np.newaxis]
        result = (ref_frame.astype(np.float32) * mask_3ch +
                  frame.astype(np.float32) * (1.0 - mask_3ch)).astype(np.uint8)

        writer.write(result)
        total += 1

        if (i + 1) % 10 == 0 or i == frame_count - 1:
            console.print(f"  [{i + 1}/{frame_count}]", end="\r")

    writer.release()
    cap.release()

    console.print()
    out_duration = total / fps if fps > 0 else 0
    console.print(f"  {total} frames, {out_duration:.1f}s")
    console.print()
    console.print("[bold green]✓ Done[/bold green]")
    console.print()
    console.print(f"file://{abs_output}")
    console.print()


if __name__ == "__main__":
    main()

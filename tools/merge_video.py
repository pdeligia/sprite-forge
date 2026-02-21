#!/usr/bin/env python3
"""Composite a background image behind every frame of a video."""

import argparse
import os
import sys

import cv2
import numpy as np
from rich.table import Table

from tools.lib.console import console


def _flatten_bg(bg):
    """Flatten an RGBA background onto white, returning BGR."""
    if bg.shape[2] == 4:
        alpha = bg[:, :, 3:4].astype(np.float32) / 255.0
        rgb = bg[:, :, :3].astype(np.float32)
        white = np.full_like(rgb, 255.0)
        return (rgb * alpha + white * (1.0 - alpha)).astype(np.uint8)
    return bg[:, :, :3]


def merge_frame(bg_flat, frame, offset):
    """Place a video frame onto the flattened background at the given offset."""
    result = bg_flat.copy()
    ox, oy = offset
    fh, fw = frame.shape[:2]
    result[oy:oy + fh, ox:ox + fw] = frame
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Composite a background image behind every frame of a video."
    )
    parser.add_argument("--input", required=True, help="Input video file.")
    parser.add_argument("--background", required=True, help="Background image (PNG).")
    parser.add_argument(
        "--output", default="./tmp/merge_video/output.mp4",
        help="Output video file path (default: ./tmp/merge_video/output.mp4)",
    )
    parser.add_argument(
        "--offset", nargs=2, type=int, metavar=("X", "Y"), default=[0, 0],
        help="Top-left position to place video frame on background (default: 0 0).",
    )
    args = parser.parse_args()

    # Load background.
    bg = cv2.imread(args.background, cv2.IMREAD_UNCHANGED)
    if bg is None:
        parser.error(f"Could not read background: {args.background}")
    bg_flat = _flatten_bg(bg)
    bh, bw = bg_flat.shape[:2]

    # Open video.
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        parser.error(f"Could not open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0

    ox, oy = args.offset

    # Validate that the frame fits on the background.
    if ox + vw > bw or oy + vh > bh:
        parser.error(
            f"Video frame ({vw}×{vh}) at offset ({ox},{oy}) exceeds "
            f"background ({bw}×{bh})."
        )

    # Print header.
    console.print()
    console.print("[bold cyan]🔀 Merge Video[/bold cyan]")
    console.print()

    info_table = Table(show_header=False, show_edge=False, pad_edge=False, padding=(0, 2))
    info_table.add_column(style="bold")
    info_table.add_column()
    info_table.add_row("Input", f"[cyan]{os.path.abspath(args.input)}[/cyan]")
    info_table.add_row("Background", f"[cyan]{os.path.abspath(args.background)}[/cyan]")
    info_table.add_row("Video size", f"[cyan]{vw}×{vh}[/cyan] at [cyan]{fps:.1f}[/cyan] fps")
    info_table.add_row("BG size", f"[cyan]{bw}×{bh}[/cyan]")
    info_table.add_row("Offset", f"[cyan]{ox},{oy}[/cyan]")
    info_table.add_row("Output size", f"[cyan]{bw}×{bh}[/cyan]")
    info_table.add_row("Frames", f"[cyan]{frame_count}[/cyan] ({duration:.1f}s)")
    console.print(info_table)
    console.print()

    # Write output.
    abs_output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(abs_output) or ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(abs_output, fourcc, fps, (bw, bh))
    if not writer.isOpened():
        print(f"Error: could not create video '{abs_output}'.", file=sys.stderr)
        sys.exit(1)

    console.print("  Processing...")
    total = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        merged = merge_frame(bg_flat, frame, (ox, oy))
        writer.write(merged)
        total += 1

    writer.release()
    cap.release()

    out_duration = total / fps if fps > 0 else 0
    console.print(f"  [cyan]{total}[/cyan] frames, [cyan]{out_duration:.1f}s[/cyan]")
    console.print()
    console.print("[bold green]✓ Done[/bold green]")
    console.print()
    console.print(f"file://{abs_output}")
    console.print()


if __name__ == "__main__":
    main()

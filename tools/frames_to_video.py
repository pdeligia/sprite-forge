#!/usr/bin/env python3
"""Compose an MP4 video from a directory of PNG frames."""

import argparse
import glob
import os
import sys

import cv2
from rich.table import Table

from tools.lib.console import console


def _read_frame(path):
    """Read a PNG frame and convert to BGR for VideoWriter.

    Handles RGBA images by compositing alpha onto a white background,
    avoiding the green-tint artifact from dropping the alpha channel.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        # BGRA → composite onto white background.
        alpha = img[:, :, 3:4].astype(float) / 255.0
        bgr = img[:, :, :3].astype(float)
        white = 255.0 * (1.0 - alpha)
        img = (bgr * alpha + white).clip(0, 255).astype("uint8")
    return img


def compose_video(files, output_path, fps, ping_pong=False):
    """Write PNG frames to an MP4 file.

    Args:
        files: Sorted list of frame file paths.
        output_path: Output MP4 path.
        fps: Frames per second.
        ping_pong: If True, append frames in reverse order (2× length).
    """
    first = _read_frame(files[0])
    if first is None:
        print(f"Error: could not read '{files[0]}'.", file=sys.stderr)
        sys.exit(1)

    h, w = first.shape[:2]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"Error: could not create video '{output_path}'.", file=sys.stderr)
        sys.exit(1)

    sequence = list(files)
    if ping_pong:
        sequence += list(reversed(files[1:-1]))

    total_frames = 0
    for f in sequence:
        frame = _read_frame(f)
        if frame is None:
            print(f"Warning: could not read '{f}', skipping.", file=sys.stderr)
            continue
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_NEAREST)
        writer.write(frame)
        total_frames += 1

    writer.release()
    return total_frames, w, h


def main():
    parser = argparse.ArgumentParser(
        description="Compose an MP4 video from a directory of PNG frames."
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory of PNG frames (sorted alphabetically).",
    )
    parser.add_argument("--prefix", help="Filter input files by prefix")
    parser.add_argument(
        "--output", default="./tmp/frames_to_video/output.mp4",
        help="Output MP4 file path (default: ./tmp/frames_to_video/output.mp4)",
    )
    parser.add_argument(
        "--fps", type=float, default=10,
        help="Frames per second (default: 10). Lower = slower frame switches.",
    )
    parser.add_argument(
        "--ping-pong", action="store_true",
        help="Append frames in reverse to create a seamless back-and-forth loop (2× length).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        parser.error(f"Input directory not found: {args.input_dir}")
    if args.fps <= 0:
        parser.error("FPS must be positive")

    pattern = os.path.join(args.input_dir, "*.png")
    files = sorted(glob.glob(pattern))
    if args.prefix:
        files = [f for f in files if os.path.basename(f).startswith(args.prefix)]
    if not files:
        parser.error(f"No matching PNG files found in {args.input_dir}")

    mode = "ping-pong" if args.ping_pong else "linear"
    n_input = len(files)
    n_total = n_input + (n_input - 2 if args.ping_pong else 0)
    duration = n_total / args.fps

    # Read first frame to get dimensions for the summary.
    first = _read_frame(files[0])
    if first is None:
        print(f"Error: could not read '{files[0]}'.", file=sys.stderr)
        sys.exit(1)
    h, w = first.shape[:2]

    console.print()
    console.print("[bold cyan]🎬 Compose[/bold cyan]")
    console.print()

    table = Table(show_header=False, show_edge=False, pad_edge=False, padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Input", f"{n_input} frames from [cyan]{os.path.abspath(args.input_dir)}[/cyan]")
    table.add_row("Mode", f"[cyan]{mode}[/cyan]")
    table.add_row("Resolution", f"[cyan]{w}×{h}[/cyan]")
    table.add_row("FPS", f"[cyan]{args.fps}[/cyan]")
    table.add_row("Frames", f"[cyan]{n_total}[/cyan]")
    table.add_row("Duration", f"[cyan]{duration:.1f}s[/cyan]")
    console.print(table)

    total_frames, w, h = compose_video(files, args.output, args.fps, args.ping_pong)
    abs_output = os.path.abspath(args.output)

    console.print()
    console.print("[bold green]✓ Done[/bold green]")
    console.print()
    console.print(f"file://{abs_output}")
    console.print()


if __name__ == "__main__":
    main()

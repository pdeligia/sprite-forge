#!/usr/bin/env python3
"""Crop, resize, and trim an MP4 video, outputting every frame."""

import argparse
import os
import sys

import cv2
from rich.table import Table

from tools.lib.console import console


def crop_video(input_path, output_path, region, width, t_start, t_end):
    """Read every frame from input, crop/resize, write to output MP4.

    Args:
        input_path: Source video file.
        output_path: Destination MP4 path.
        region: (x1, y1, x2, y2) crop box or None for full frame.
        width: Target output width (preserves aspect ratio) or None.
        t_start: Start time in seconds (None = beginning).
        t_end: End time in seconds (None = end).

    Returns:
        (total_frames, out_w, out_h, fps)
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: could not open video '{input_path}'.", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Validate region.
    if region:
        x1, y1, x2, y2 = region
        if x2 > vw or y2 > vh:
            print(
                f"Error: region ({x1},{y1})→({x2},{y2}) exceeds video {vw}×{vh}.",
                file=sys.stderr,
            )
            sys.exit(1)

    # Compute output dimensions.
    src_w = (region[2] - region[0]) if region else vw
    src_h = (region[3] - region[1]) if region else vh

    if width:
        scale = width / src_w
        out_w = width
        out_h = round(src_h * scale)
    else:
        out_w = src_w
        out_h = src_h

    start = t_start if t_start is not None else 0.0
    end = t_end if t_end is not None else duration

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        print(f"Error: could not create video '{output_path}'.", file=sys.stderr)
        sys.exit(1)

    interval = 1.0 / fps
    t = start
    total = 0
    while t <= end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            break

        if region:
            x1, y1, x2, y2 = region
            frame = frame[y1:y2, x1:x2]

        if width:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        writer.write(frame)
        total += 1
        t += interval

    writer.release()
    cap.release()
    return total, out_w, out_h, fps


def main():
    parser = argparse.ArgumentParser(
        description="Crop, resize, and trim an MP4 video."
    )
    parser.add_argument("--input", required=True, help="Input MP4 file.")
    parser.add_argument(
        "--output", default="./tmp/crop_video/output.mp4",
        help="Output MP4 file path (default: ./tmp/crop_video/output.mp4)",
    )
    parser.add_argument(
        "--region", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"),
        help="Crop region: top-left (X1, Y1) to bottom-right (X2, Y2) in pixels.",
    )
    parser.add_argument(
        "--width", type=int,
        help="Scale output to this width (preserves aspect ratio).",
    )
    parser.add_argument("--start", type=float, default=None, help="Start time in seconds.")
    parser.add_argument("--end", type=float, default=None, help="End time in seconds.")
    args = parser.parse_args()

    if args.region:
        x1, y1, x2, y2 = args.region
        if x1 >= x2 or y1 >= y2:
            parser.error(f"Invalid region: ({x1},{y1})→({x2},{y2})")

    # Print header.
    console.print()
    console.print("[bold cyan]✂️  Crop[/bold cyan]")
    console.print()

    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()

    src_w = (args.region[2] - args.region[0]) if args.region else vw
    src_h = (args.region[3] - args.region[1]) if args.region else vh
    if args.width:
        scale = args.width / src_w
        out_w = args.width
        out_h = round(src_h * scale)
    else:
        out_w, out_h = src_w, src_h

    t_start = args.start if args.start is not None else 0.0
    t_end = args.end if args.end is not None else duration

    info_table = Table(show_header=False, show_edge=False, pad_edge=False, padding=(0, 2))
    info_table.add_column(style="bold")
    info_table.add_column()
    info_table.add_row("Input", f"[cyan]{os.path.abspath(args.input)}[/cyan]")
    info_table.add_row("Source", f"[cyan]{vw}×{vh}[/cyan] at [cyan]{fps}[/cyan] fps")
    if args.region:
        info_table.add_row("Region", f"[cyan]{args.region[0]},{args.region[1]} → {args.region[2]},{args.region[3]}[/cyan]")
    info_table.add_row("Output size", f"[cyan]{out_w}×{out_h}[/cyan]")
    info_table.add_row("Range", f"[cyan]{t_start:.1f}s – {t_end:.1f}s[/cyan]")
    console.print(info_table)
    console.print()

    console.print("  Processing...")
    total, out_w, out_h, fps = crop_video(
        args.input, args.output, args.region, args.width, args.start, args.end,
    )

    out_duration = total / fps
    abs_output = os.path.abspath(args.output)

    console.print(f"  [cyan]{total}[/cyan] frames, [cyan]{out_duration:.1f}s[/cyan]")
    console.print()
    console.print("[bold green]✓ Done[/bold green]")
    console.print()
    console.print(f"file://{abs_output}")
    console.print()


if __name__ == "__main__":
    main()

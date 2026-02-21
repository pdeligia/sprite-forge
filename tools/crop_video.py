#!/usr/bin/env python3
"""Crop, resize, and trim an MP4 video, outputting every frame."""

import argparse
import os
import sys

import cv2
import numpy as np
from rich.table import Table

from tools.lib.console import console


def detect_content_bounds(input_path, t_start=None, t_end=None, threshold=10, sample_count=10):
    """Detect the bounding box of non-black content across sampled frames.

    Samples frames evenly across the time range, accumulates a brightness mask,
    and finds bounds using column/row density to ignore small logos in black bars.
    Returns (x1, y1, x2, y2).
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start = t_start if t_start is not None else 0.0
    end = t_end if t_end is not None else duration
    span = end - start

    step = span / sample_count if sample_count > 1 else span / 2

    # Accumulate how many sampled frames have non-black pixels at each location.
    heat = np.zeros((vh, vw), dtype=np.float32)
    sampled = 0

    for i in range(sample_count):
        t = start + step * i + step / 2
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        heat += (gray > threshold).astype(np.float32)
        sampled += 1

    cap.release()
    if sampled == 0:
        return None

    # A pixel is "content" if it's non-black in most sampled frames.
    content_mask = heat >= (sampled * 0.5)

    # Project onto columns and rows: fraction of pixels that are content.
    col_density = content_mask.mean(axis=0)  # shape (vw,)
    row_density = content_mask.mean(axis=1)  # shape (vh,)

    # Content columns/rows have high density (>20% of the axis is lit).
    # Black bars with a small logo have very low density.
    density_threshold = 0.2
    content_cols = np.where(col_density > density_threshold)[0]
    content_rows = np.where(row_density > density_threshold)[0]

    if len(content_cols) == 0 or len(content_rows) == 0:
        return None

    x1 = int(content_cols[0])
    x2 = int(content_cols[-1] + 1)
    y1 = int(content_rows[0])
    y2 = int(content_rows[-1] + 1)

    return (x1, y1, x2, y2)


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
    parser.add_argument(
        "--auto-crop", action="store_true",
        help="Auto-detect content bounds by removing black bars.",
    )
    args = parser.parse_args()

    if args.region and args.auto_crop:
        parser.error("Cannot use --region and --auto-crop together")

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

    # Auto-detect content bounds if requested.
    region = args.region
    if args.auto_crop:
        console.print("  Detecting content bounds...")
        bounds = detect_content_bounds(args.input, args.start, args.end)
        if bounds is None:
            print("Error: could not detect content bounds.", file=sys.stderr)
            sys.exit(1)
        region = list(bounds)
        console.print(f"  Detected: [cyan]{region[0]},{region[1]} → {region[2]},{region[3]}[/cyan]")

    src_w = (region[2] - region[0]) if region else vw
    src_h = (region[3] - region[1]) if region else vh
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
    if region:
        info_table.add_row("Region", f"[cyan]{region[0]},{region[1]} → {region[2]},{region[3]}[/cyan]")
    info_table.add_row("Output size", f"[cyan]{out_w}×{out_h}[/cyan]")
    info_table.add_row("Range", f"[cyan]{t_start:.1f}s – {t_end:.1f}s[/cyan]")
    console.print(info_table)
    console.print()

    console.print("  Processing...")
    total, out_w, out_h, fps = crop_video(
        args.input, args.output, region, args.width, args.start, args.end,
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

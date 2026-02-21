#!/usr/bin/env python3
"""Extract frames from an MP4 video as PNGs, with optional cropping, scaling, and loop detection."""

import argparse
import os
import shutil
import sys

import cv2
import numpy as np
from rich.table import Table
from skimage.metrics import structural_similarity as ssim

from tools.lib.console import console
from tools.lib.video_utils import find_loop_segment


def _to_gray_small(frame):
    """Downscale to 160×120 grayscale for fast SSIM comparison."""
    small = cv2.resize(frame, (160, 120))
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def _select_smart_frames(all_frames, n):
    """Greedily pick the N most visually distinct frames.

    Starts with the first frame, then repeatedly picks the frame with the
    maximum minimum SSIM distance from all already-selected frames.
    Returns indices into all_frames, in temporal order.
    """
    if n >= len(all_frames):
        return list(range(len(all_frames)))

    grays = [_to_gray_small(f) for f in all_frames]

    # Start with the first frame.
    selected = [0]
    # Pre-compute SSIM distances lazily via a "min distance to selected" array.
    # Initialize with distance from frame 0 to every other frame.
    min_dist = np.array([
        1.0 - ssim(grays[0], grays[i]) if i != 0 else -1.0
        for i in range(len(grays))
    ])

    for _ in range(n - 1):
        # Pick the frame with the largest min-distance to any selected frame.
        best = np.argmax(min_dist)
        selected.append(int(best))
        min_dist[best] = -1.0  # Mark as selected.

        # Update min distances with the newly selected frame.
        for i in range(len(grays)):
            if min_dist[i] < 0:
                continue
            d = 1.0 - ssim(grays[best], grays[i])
            if d < min_dist[i]:
                min_dist[i] = d

    return sorted(selected)


def extract_frames(input_path, n_frames, output_dir, prefix, suffix, region, width,
                   start, end, loop, smart):
    """Extract frames from the video."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: could not open video '{input_path}'.", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if duration <= 0:
        print("Error: video has zero or negative duration.", file=sys.stderr)
        sys.exit(1)

    # Validate region bounds against actual video dimensions.
    if region:
        x1, y1, x2, y2 = region
        if x2 > vw or y2 > vh:
            print(
                f"Error: region bottom-right ({x2},{y2}) exceeds video dimensions ({vw}x{vh}).",
                file=sys.stderr,
            )
            sys.exit(1)

    # Determine source dimensions (cropped region or full video).
    src_w = (region[2] - region[0]) if region else vw
    src_h = (region[3] - region[1]) if region else vh

    # Determine time range.
    t_start = start if start is not None else 0.0
    t_end = end if end is not None else duration
    span = t_end - t_start
    if span <= 0:
        print(f"Error: time range is empty ({t_start}s\u2013{t_end}s).", file=sys.stderr)
        sys.exit(1)

    # Default N: extract all frames in the time range.
    if n_frames is None:
        n_frames = max(1, int(round(span * fps)))

    # Compute output dimensions if scaling.
    out_w, out_h = None, None
    if width:
        out_w = width
        out_h = round(src_h * (width / src_w))

    # Print header.
    console.print()
    console.print("[bold cyan]🎞️  Extract Frames[/bold cyan]")
    console.print()

    info_table = Table(show_header=False, show_edge=False, pad_edge=False, padding=(0, 2))
    info_table.add_column(style="bold")
    info_table.add_column()
    info_table.add_row("Input", f"[cyan]{os.path.abspath(input_path)}[/cyan]")
    info_table.add_row("Source", f"[cyan]{vw}×{vh}[/cyan] at [cyan]{fps:.1f}[/cyan] fps")
    if region:
        info_table.add_row("Region", f"[cyan]{region[0]},{region[1]} → {region[2]},{region[3]}[/cyan]")
    if out_w and out_h:
        info_table.add_row("Output size", f"[cyan]{out_w}×{out_h}[/cyan]")
    info_table.add_row("Range", f"[cyan]{t_start:.1f}s – {t_end:.1f}s[/cyan]")
    info_table.add_row("Frames", f"[cyan]{n_frames}[/cyan]")
    if loop:
        info_table.add_row("Loop", "[cyan]auto-detect[/cyan]")
    if smart:
        info_table.add_row("Smart", "[cyan]max diversity[/cyan]")
    console.print(info_table)
    console.print()

    # If --loop, find the best loop segment.
    if loop:
        console.print(f"  Scanning for loop in {t_start:.2f}s–{t_end:.2f}s...")
        result = find_loop_segment(cap, t_start, t_end)
        if result is None:
            print("Error: could not find a suitable loop segment.", file=sys.stderr)
            sys.exit(1)
        t_start, t_end, score = result
        span = t_end - t_start
        console.print(f"  Best loop: [cyan]{t_start:.2f}s → {t_end:.2f}s[/cyan] (score: [cyan]{score:.3f}[/cyan])")

    # Clean and recreate output directory.
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    if smart:
        # Read all frames in range, then pick the most diverse N.
        all_timestamps = []
        all_raw_frames = []
        step = 1.0 / fps if fps > 0 else span / n_frames
        t = t_start
        while t < t_end:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if not ret:
                break
            if region:
                rx1, ry1, rx2, ry2 = region
                frame = frame[ry1:ry2, rx1:rx2]
            all_timestamps.append(t)
            all_raw_frames.append(frame)
            t += step

        console.print(f"  Read {len(all_raw_frames)} candidate frames")
        console.print(f"  Selecting {n_frames} most diverse frames...")

        selected_indices = _select_smart_frames(all_raw_frames, n_frames)
        timestamps = [all_timestamps[i] for i in selected_indices]
        frames_to_write = [all_raw_frames[i] for i in selected_indices]
    else:
        # Evenly-spaced timestamps.
        if n_frames == 1:
            timestamps = [t_start + span / 2]
        elif loop:
            step = span / n_frames
            timestamps = [t_start + step * i for i in range(n_frames)]
        else:
            step = span / n_frames
            timestamps = [t_start + step * i + step / 2 for i in range(n_frames)]
        frames_to_write = None

    pad = max(2, len(str(n_frames)))

    console.print(f"  Extracting {n_frames} frames...")

    for i in range(len(timestamps)):
        out_file = os.path.join(output_dir, f"{prefix}_{str(i + 1).zfill(pad)}{suffix}.png")

        if frames_to_write is not None:
            frame = frames_to_write[i]
        else:
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamps[i] * 1000)
            ret, frame = cap.read()
            if not ret:
                print(f"Error: could not read frame at t={timestamps[i]:.2f}s.", file=sys.stderr)
                sys.exit(1)
            if region:
                rx1, ry1, rx2, ry2 = region
                frame = frame[ry1:ry2, rx1:rx2]

        if out_w and out_h:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(out_file, frame)

    cap.release()

    abs_output = os.path.abspath(output_dir)
    console.print(f"  {n_frames} frames saved")
    console.print()
    console.print("[bold green]✓ Done[/bold green]")
    console.print()
    console.print(f"file://{abs_output}")
    console.print()


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from an MP4 video as PNGs."
    )
    parser.add_argument("--input", required=True, help="Path to the input video.")
    parser.add_argument(
        "--frames", type=int, default=None,
        help="Number of frames to extract (default: all frames in range).",
    )
    parser.add_argument("--output-dir", default="./tmp/video_to_frames", help="Output directory (default: ./tmp/video_to_frames)")
    parser.add_argument("--prefix", default="frame", help="Filename prefix (default: frame)")
    parser.add_argument("--suffix", default="", help="Filename suffix before .png (e.g., @3x)")
    parser.add_argument(
        "--region", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"),
        help="Crop region: top-left (X1, Y1) to bottom-right (X2, Y2) in pixels",
    )
    parser.add_argument(
        "--width", type=int, default=0,
        help="Scale output to this width, preserving aspect ratio. Uses nearest-neighbor for pixel art.",
    )
    parser.add_argument(
        "--start", type=float, default=None,
        help="Start time in seconds (default: beginning of video)",
    )
    parser.add_argument(
        "--end", type=float, default=None,
        help="End time in seconds (default: end of video)",
    )
    parser.add_argument(
        "--loop", action="store_true",
        help="Find the best looping segment using SSIM frame comparison.",
    )
    parser.add_argument(
        "--smart", action="store_true",
        help="Pick the N most visually diverse frames instead of evenly-spaced.",
    )

    args = parser.parse_args()

    if args.frames is not None and args.frames < 1:
        parser.error("--frames must be at least 1")
    if args.smart and args.frames is None:
        parser.error("--smart requires --frames to specify how many to pick")
    if not os.path.isfile(args.input):
        parser.error(f"Input file not found: {args.input}")
    if args.region:
        x1, y1, x2, y2 = args.region
        if x1 >= x2 or y1 >= y2:
            parser.error("Region top-left must be above and left of bottom-right")
    if args.width < 0:
        parser.error("Width must be positive")
    if args.start is not None and args.start < 0:
        parser.error("Start time must be non-negative")
    if args.end is not None and args.end < 0:
        parser.error("End time must be non-negative")
    if args.start is not None and args.end is not None and args.start >= args.end:
        parser.error("Start time must be less than end time")

    extract_frames(args.input, args.frames, args.output_dir, args.prefix, args.suffix,
                   args.region, args.width, args.start, args.end, args.loop, args.smart)


if __name__ == "__main__":
    main()

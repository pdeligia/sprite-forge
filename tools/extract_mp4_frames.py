#!/usr/bin/env python3
"""Extract N frames from an MP4 video as PNGs, with optional cropping, scaling, and loop detection."""

import argparse
import os
import shutil
import sys

import cv2
import numpy as np

from tools.lib.video_utils import find_loop_segment


def extract_frames(input_path, n_frames, output_dir, prefix, suffix, region, width, start, end, loop):
    """Extract N evenly-spaced frames from the video."""
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

    # Clean and recreate output directory.
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Determine time range.
    t_start = start if start is not None else 0.0
    t_end = end if end is not None else duration
    span = t_end - t_start
    if span <= 0:
        print(f"Error: time range is empty ({t_start}s\u2013{t_end}s).", file=sys.stderr)
        sys.exit(1)

    # If --loop, find the best loop segment.
    if loop:
        print(f"Scanning for loop in {t_start:.2f}s\u2013{t_end:.2f}s...")
        result = find_loop_segment(cap, t_start, t_end)
        if result is None:
            print("Error: could not find a suitable loop segment.", file=sys.stderr)
            sys.exit(1)
        t_start, t_end, score = result
        span = t_end - t_start
        print(f"Best loop: {t_start:.2f}s \u2192 {t_end:.2f}s (score: {score:.3f})")

    # Calculate evenly-spaced timestamps within the time range.
    # In loop mode, exclude the end frame (it matches the start frame).
    if n_frames == 1:
        timestamps = [t_start + span / 2]
    elif loop:
        step = span / n_frames
        timestamps = [t_start + step * i for i in range(n_frames)]
    else:
        step = span / n_frames
        timestamps = [t_start + step * i + step / 2 for i in range(n_frames)]

    # Zero-pad width based on total frame count.
    pad = max(2, len(str(n_frames)))

    # Compute output dimensions if scaling.
    out_w, out_h = None, None
    if width:
        out_w = width
        out_h = round(src_h * (width / src_w))

    for i, ts in enumerate(timestamps, start=1):
        out_file = os.path.join(output_dir, f"{prefix}_{str(i).zfill(pad)}{suffix}.png")

        # Seek to the target timestamp.
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        if not ret:
            print(f"Error: could not read frame at t={ts:.2f}s.", file=sys.stderr)
            sys.exit(1)

        # Crop if region specified.
        if region:
            x1, y1, x2, y2 = region
            frame = frame[y1:y2, x1:x2]

        # Scale if width specified (nearest-neighbor for pixel art).
        if out_w and out_h:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(out_file, frame)
        print(f"[{i}/{n_frames}] {out_file} (t={ts:.2f}s)")

    cap.release()
    print(f"\nDone \u2014 {n_frames} frame(s) saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Extract N frames from an MP4 video as PNGs."
    )
    parser.add_argument("input", help="Path to the input MP4 video")
    parser.add_argument("n", type=int, help="Number of frames to extract")
    parser.add_argument("--output-dir", default="./tmp/extract_mp4_frames", help="Output directory (default: ./tmp/extract_mp4_frames)")
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
        help="Find the best looping segment using SSIM frame comparison, then extract N frames from it.",
    )

    args = parser.parse_args()

    if args.n < 1:
        parser.error("N must be at least 1")
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

    extract_frames(args.input, args.n, args.output_dir, args.prefix, args.suffix, args.region, args.width, args.start, args.end, args.loop)


if __name__ == "__main__":
    main()

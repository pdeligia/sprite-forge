#!/usr/bin/env python3
"""Analyze an MP4 video and recommend the optimal number of frames for animation extraction."""

import argparse
import math
import os
import sys

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from tools.lib.console import console, Table
from tools.lib.video_utils import find_loop_segment, sample_frames_ssim


# Standard tier N values for the recommendation table.
TIER_N_VALUES = [4, 8, 12, 18, 24]

TIER_LABELS = {
    0: "⚡ Minimal (may look jerky)",
    1: "✅ Good for pixel art / slow motion",
    2: "✅ Smooth for most styles",
    3: "✨ Very smooth",
    4: "💎 Cinematic (diminishing returns)",
}


def compute_delta_ssim(samples, n, span):
    """Compute average Δ-SSIM for N evenly-spaced frames within the sampled segment.

    Δ-SSIM = 1 - avg_ssim_between_consecutive_extracted_frames.
    Higher means more visual change per step.
    """
    if n < 2 or len(samples) < 2:
        return 0.0

    step = span / n
    t_start = samples[0][0]

    # Pick the closest sampled frame for each extraction timestamp.
    extracted = []
    for i in range(n):
        target_t = t_start + step * i + step / 2
        closest = min(samples, key=lambda s: abs(s[0] - target_t))
        extracted.append(closest[1])

    # Compute SSIM between consecutive extracted frames.
    total_ssim = 0.0
    for i in range(len(extracted) - 1):
        total_ssim += ssim(extracted[i], extracted[i + 1])

    avg_ssim = total_ssim / (len(extracted) - 1)
    return round(1.0 - avg_ssim, 4)


def find_optimal_n(samples, span, max_n):
    """Find the optimal N by detecting the diminishing-returns knee.

    Returns the N where adding more frames stops meaningfully reducing Δ-SSIM.
    """
    if max_n < 2:
        return 2

    # Compute Δ-SSIM for each candidate N.
    prev_delta = None
    knee_n = 2
    threshold = 0.005  # Marginal improvement threshold.

    for n in range(2, max_n + 1):
        delta = compute_delta_ssim(samples, n, span)
        if prev_delta is not None:
            improvement = prev_delta - delta
            if improvement < threshold:
                knee_n = n - 1
                break
        prev_delta = delta
        knee_n = n

    return max(2, knee_n)


def estimate_size_mb(width, height, n):
    """Rough estimate of total PNG output size in MB."""
    # RGBA uncompressed = w * h * 4 bytes, typical PNG compression ~3x.
    per_frame = (width * height * 4) / 3
    total = per_frame * n
    return total / (1024 * 1024)


def classify_motion(avg_ssim):
    """Classify motion level from average inter-frame SSIM."""
    if avg_ssim > 0.98:
        return "low"
    elif avg_ssim >= 0.95:
        return "medium"
    else:
        return "high"


def analyze_video(input_path, start, end, loop, region, width):
    """Analyze video and print recommendations."""
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

    # Validate region bounds.
    if region:
        x1, y1, x2, y2 = region
        if x2 > vw or y2 > vh:
            print(
                f"Error: region bottom-right ({x2},{y2}) exceeds video dimensions ({vw}x{vh}).",
                file=sys.stderr,
            )
            sys.exit(1)

    # Determine output dimensions for size estimates.
    src_w = (region[2] - region[0]) if region else vw
    src_h = (region[3] - region[1]) if region else vh
    if width:
        out_w = width
        out_h = round(src_h * (width / src_w))
    else:
        out_w = src_w
        out_h = src_h

    # Determine time range.
    t_start = start if start is not None else 0.0
    t_end = end if end is not None else duration

    # Print header immediately with known metadata.
    fname = os.path.basename(input_path)
    console.print()
    console.print("[bold cyan]🎬 Video[/bold cyan]")
    console.print()
    console.print(f"  File: [cyan]{fname}[/cyan]")
    console.print(f"  Resolution: [cyan]{vw}×{vh}[/cyan]")
    console.print(f"  FPS: [cyan]{fps:.0f}[/cyan]")
    console.print(f"  Duration: [cyan]{duration:.1f}s[/cyan] ({total_frames} frames)")
    if start is not None or end is not None:
        console.print(f"  Segment: [cyan]{t_start:.2f}s – {t_end:.2f}s[/cyan]")
    if region:
        console.print(f"  Region: [cyan]{region[0]},{region[1]} → {region[2]},{region[3]}[/cyan]")
    if width:
        console.print(f"  Output: [cyan]{out_w}×{out_h}[/cyan]")
    console.print()

    # If --loop, find the best loop segment first.
    loop_info = None
    if loop:
        console.print("  Scanning for loop...")
        result = find_loop_segment(cap, t_start, t_end)
        if result is None:
            print("Error: could not find a suitable loop segment.", file=sys.stderr)
            sys.exit(1)
        t_start, t_end, score = result
        loop_info = (t_start, t_end, score)

    span = t_end - t_start
    if span <= 0:
        print(f"Error: time range is empty ({t_start}s–{t_end}s).", file=sys.stderr)
        sys.exit(1)

    native_frames_in_segment = int(span * fps)

    # Sample frames at native FPS for analysis.
    samples = sample_frames_ssim(cap, t_start, t_end)
    cap.release()

    if len(samples) < 2:
        print("Error: not enough frames to analyze.", file=sys.stderr)
        sys.exit(1)

    # Compute average inter-frame SSIM.
    total_ssim = 0.0
    for i in range(len(samples) - 1):
        total_ssim += ssim(samples[i][1], samples[i + 1][1])
    avg_ssim = total_ssim / (len(samples) - 1)

    motion = classify_motion(avg_ssim)

    # Detect jittery frames (SSIM drops below mean - 2σ).
    pair_ssims = []
    for i in range(len(samples) - 1):
        pair_ssims.append(ssim(samples[i][1], samples[i + 1][1]))
    mean_ssim = np.mean(pair_ssims)
    std_ssim = np.std(pair_ssims)
    jitter_threshold = mean_ssim - 2 * std_ssim
    jitter_frames = []
    for i, s in enumerate(pair_ssims):
        if s < jitter_threshold:
            frame_num = i + 1
            t = samples[i][0]
            jitter_frames.append((frame_num, t, s))

    # Find optimal N.
    max_n = min(native_frames_in_segment, 60)  # Cap search at 60.
    optimal_n = find_optimal_n(samples, span, max_n)

    # Build the tier table.
    # Filter tiers that exceed native frame count and insert optimal N.
    tier_ns = [n for n in TIER_N_VALUES if n <= native_frames_in_segment]
    if optimal_n not in tier_ns:
        tier_ns.append(optimal_n)
        tier_ns.sort()

    if start is not None or end is not None:
        console.print(f"  Segment: [cyan]{t_start:.2f}s – {t_end:.2f}s[/cyan] ({span:.1f}s, {native_frames_in_segment} frames)")
        console.print()

    if loop_info:
        ls, le, sc = loop_info
        console.print(f"  Best loop: [cyan]{ls:.2f}s → {le:.2f}s[/cyan] (score: {sc:.3f})")
        console.print()

    # Motion section.
    console.print("[bold cyan]📊 Motion[/bold cyan]")
    console.print()
    console.print(f"  Level: [bold]{motion}[/bold] (avg inter-frame SSIM: {avg_ssim:.3f})")
    console.print()

    # Jitter section.
    console.print("  [bold dim]Jitter Analysis[/bold dim]")
    console.print("  [dim]─────────────────[/dim]")
    if jitter_frames:
        console.print(f"  [bold red]{len(jitter_frames)} jittery frame(s)[/bold red] (threshold: SSIM < {jitter_threshold:.3f})")
        console.print()
        jitter_table = Table(box=None, padding=(0, 2))
        jitter_table.add_column("Frame", style="bold", justify="right")
        jitter_table.add_column("Time", justify="right")
        jitter_table.add_column("SSIM", justify="right")
        for frame_num, t, s in jitter_frames:
            jitter_table.add_row(
                str(frame_num),
                f"[cyan]{t:.2f}s[/cyan]",
                f"[red]{s:.3f}[/red]",
            )
        console.print(jitter_table)
    else:
        console.print("  [green]No jitter detected[/green]")
    console.print()

    # Recommendation section.
    console.print("[bold cyan]🎯 Recommendation[/bold cyan]")
    console.print()
    console.print(f"  Optimal N: [bold]{optimal_n}[/bold] frames")
    console.print()

    # Print table.
    table = Table(box=None, padding=(0, 2))
    table.add_column("N", justify="right")
    table.add_column("fps", justify="right")
    table.add_column("Δ-SSIM", justify="right")
    table.add_column("Size est.", justify="right")
    table.add_column("Rating")
    for idx, n in enumerate(tier_ns):
        eff_fps = n / span
        delta = compute_delta_ssim(samples, n, span)
        size = estimate_size_mb(out_w, out_h, n)

        if n == optimal_n:
            label = "✅ Recommended"
        elif n in TIER_N_VALUES:
            tier_idx = TIER_N_VALUES.index(n)
            label = TIER_LABELS.get(tier_idx, "")
        else:
            label = ""

        size_str = f"~{size:.1f} MB"
        table.add_row(str(n), f"{eff_fps:.1f}", f"{delta:.2f}", size_str, label)
    console.print(table)

    # Playback recommendations (when --loop is used).
    if loop_info:
        _, _, loop_score = loop_info
        time_per_frame = span / optimal_n

        console.print()
        console.print("  [bold dim]Playback[/bold dim]")
        console.print("  [dim]─────────────────[/dim]")
        console.print(f"  timePerFrame: {time_per_frame:.2f}s")
        if loop_score >= 0.98:
            console.print(f"  Strategy: forward (loop score {loop_score:.3f} ≥ 0.98 — seamless)")
            console.print(f"  Cycle: 1→2→…→N→1→… ({optimal_n} steps, {span:.2f}s per cycle)")
        else:
            ping_pong_steps = max(optimal_n * 2 - 2, optimal_n)
            ping_pong_duration = ping_pong_steps * time_per_frame
            console.print(f"  Strategy: ping-pong (loop score {loop_score:.3f} < 0.98 — visible jump if forward-only)")
            console.print(f"  Cycle: 1→2→…→N→…→2→1 ({ping_pong_steps} steps, {ping_pong_duration:.2f}s per cycle)")

    console.print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze an MP4 video and recommend the optimal number of frames for animation extraction."
    )
    parser.add_argument("input", help="Path to the input MP4 video")
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
        help="Find the best looping segment first, then analyze it.",
    )
    parser.add_argument(
        "--region", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"),
        help="Crop region for accurate size estimates: top-left (X1, Y1) to bottom-right (X2, Y2)",
    )
    parser.add_argument(
        "--width", type=int, default=0,
        help="Output width for accurate size estimates (preserves aspect ratio).",
    )

    args = parser.parse_args()

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

    analyze_video(args.input, args.start, args.end, args.loop, args.region, args.width)


if __name__ == "__main__":
    main()

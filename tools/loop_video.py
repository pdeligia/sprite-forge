#!/usr/bin/env python3
"""Create a seamlessly looping MP4 from a video (or a time range of it)."""

import argparse
import os
import sys

import cv2
import numpy as np
from rich.table import Table
from skimage.metrics import structural_similarity as ssim

from tools.lib.console import console


def _extract_frames(cap, fps, start, end):
    """Extract all frames between start and end times (seconds)."""
    frames = []
    interval = 1.0 / fps
    t = start
    while t <= end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        t += interval
    return frames


def _to_gray_small(frame):
    """Downscale to 160×120 grayscale for fast SSIM comparison."""
    small = cv2.resize(frame, (160, 120))
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def _frame_ssim(a, b):
    """Compute SSIM between two BGR frames (downscaled internally)."""
    return ssim(_to_gray_small(a), _to_gray_small(b))


def _crossfade(frame_a, frame_b, alpha):
    """Blend two frames: result = a*(1-alpha) + b*alpha."""
    return cv2.addWeighted(frame_a, 1.0 - alpha, frame_b, alpha, 0)


def find_best_reverse_point(frames, budget):
    """Find the frame closest to frame[0] by walking backwards from the end.

    Returns the index (from end) of the best match within budget.
    """
    first_gray = _to_gray_small(frames[0])
    best_score = -1
    best_k = 1

    for k in range(1, min(budget + 1, len(frames))):
        idx = len(frames) - k
        score = ssim(first_gray, _to_gray_small(frames[idx]))
        if score > best_score:
            best_score = score
            best_k = k

    return best_k, best_score


def find_most_stable_frame(frames, window=3):
    """Find the frame with highest average SSIM to its neighbors.

    Returns (best_index, best_score).
    """
    grays = [_to_gray_small(f) for f in frames]
    best_idx = 0
    best_score = -1

    for i in range(len(frames)):
        scores = []
        for offset in range(1, window + 1):
            j = (i + offset) % len(frames)
            k = (i - offset) % len(frames)
            scores.append(ssim(grays[i], grays[j]))
            scores.append(ssim(grays[i], grays[k]))
        avg = sum(scores) / len(scores)
        if avg > best_score:
            best_score = avg
            best_idx = i

    return best_idx, best_score


def shift_frames(frames, shift_idx):
    """Shift frames so that shift_idx becomes the first frame."""
    return frames[shift_idx:] + frames[:shift_idx]


def build_loop(frames, budget, mode="auto"):
    """Build a seamlessly looping frame sequence.

    Args:
        frames: List of BGR frames (original range).
        budget: Max extra frames to add for the bridge.
        mode: "auto", "crossfade", or "reverse".

    Returns:
        (looped_frames, info_dict) where info_dict has stats about what was done.
    """
    first_last_ssim = _frame_ssim(frames[0], frames[-1])

    info = {
        "first_last_ssim": first_last_ssim,
        "mode_used": mode,
        "bridge_frames": 0,
        "reverse_frames": 0,
        "crossfade_frames": 0,
    }

    # Scale bridge size based on how different first/last are.
    # Even good matches get a small bridge for smooth transition.
    if first_last_ssim > 0.99:
        scaled_budget = max(1, budget // 8)
    elif first_last_ssim > 0.95:
        scaled_budget = max(2, budget // 4)
    elif first_last_ssim > 0.85:
        scaled_budget = max(3, budget // 2)
    else:
        scaled_budget = budget

    if mode == "auto":
        # Try reverse bridge — if we find a good match, use it.
        best_k, best_score = find_best_reverse_point(frames, scaled_budget)
        if best_score > 0.85:
            mode = "reverse"
        else:
            mode = "crossfade"

    if mode == "reverse":
        best_k, best_score = find_best_reverse_point(frames, scaled_budget)

        # Split budget: reverse frames + crossfade tail.
        crossfade_count = min(max(scaled_budget // 4, 3), best_k)
        reverse_count = best_k

        # Build reverse bridge: walk backwards from end.
        bridge = []
        for i in range(1, reverse_count + 1):
            bridge.append(frames[len(frames) - i])

        # Cross-fade the last crossfade_count bridge frames into frame[0].
        for i in range(crossfade_count):
            idx = len(bridge) - crossfade_count + i
            alpha = (i + 1) / (crossfade_count + 1)
            bridge[idx] = _crossfade(bridge[idx], frames[0], alpha)

        info["mode_used"] = "reverse"
        info["bridge_frames"] = len(bridge)
        info["reverse_frames"] = reverse_count
        info["crossfade_frames"] = crossfade_count
        return list(frames) + bridge, info

    else:  # crossfade
        fade_count = min(scaled_budget, len(frames) // 2)
        bridge = []
        for i in range(fade_count):
            alpha = (i + 1) / (fade_count + 1)
            blended = _crossfade(frames[-1], frames[0], alpha)
            bridge.append(blended)

        info["mode_used"] = "crossfade"
        info["bridge_frames"] = fade_count
        info["crossfade_frames"] = fade_count
        return list(frames) + bridge, info


def write_video(frames, output_path, fps, repeat=1):
    """Write frames to an MP4 file, optionally repeating the sequence."""
    if not frames:
        return 0

    h, w = frames[0].shape[:2]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"Error: could not create video '{output_path}'.", file=sys.stderr)
        sys.exit(1)

    total = 0
    for _ in range(repeat):
        for frame in frames:
            writer.write(frame)
            total += 1
    writer.release()
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Create a seamlessly looping MP4 from a video or a time range of it."
    )
    parser.add_argument("--input", required=True, help="Input MP4 file.")
    parser.add_argument(
        "--output", default="./tmp/loop_video/output.mp4",
        help="Output MP4 file path (default: ./tmp/loop_video/output.mp4)",
    )
    parser.add_argument("--start", type=float, default=None, help="Start time in seconds.")
    parser.add_argument("--end", type=float, default=None, help="End time in seconds.")
    parser.add_argument(
        "--budget", type=int, default=15,
        help="Max extra frames the algorithm can add to close the loop (default: 15).",
    )
    parser.add_argument(
        "--mode", choices=["auto", "crossfade", "reverse"], default="auto",
        help="Loop strategy: auto (default), crossfade, or reverse.",
    )
    parser.add_argument(
        "--no-shift", action="store_true",
        help="Disable shifting the start to the most stable frame.",
    )
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="Repeat the loop N times in the output for easy preview (default: 1).",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        parser.error(f"Could not open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    t_start = args.start if args.start is not None else 0.0
    t_end = args.end if args.end is not None else duration

    if t_start < 0 or t_end > duration or t_start >= t_end:
        parser.error(f"Invalid time range: {t_start}-{t_end} (video is {duration:.1f}s)")

    # Print header.
    console.print()
    console.print("[bold cyan]🔁 Loop[/bold cyan]")
    console.print()

    info_table = Table(show_header=False, show_edge=False, pad_edge=False, padding=(0, 2))
    info_table.add_column(style="bold")
    info_table.add_column()
    info_table.add_row("Input", f"[cyan]{os.path.abspath(args.input)}[/cyan]")
    info_table.add_row("Resolution", f"[cyan]{w}×{h}[/cyan]")
    info_table.add_row("Source FPS", f"[cyan]{fps}[/cyan]")
    info_table.add_row("Range", f"[cyan]{t_start:.1f}s – {t_end:.1f}s[/cyan]")
    info_table.add_row("Budget", f"[cyan]{args.budget}[/cyan] extra frames")
    info_table.add_row("Mode", f"[cyan]{args.mode}[/cyan]")
    console.print(info_table)
    console.print()

    # Extract frames.
    console.print("  Extracting frames...")
    frames = _extract_frames(cap, fps, t_start, t_end)
    cap.release()

    if len(frames) < 2:
        print("Error: not enough frames extracted.", file=sys.stderr)
        sys.exit(1)

    console.print(f"  Extracted [cyan]{len(frames)}[/cyan] frames")

    # Shift start to the most stable frame.
    shift_idx = 0
    if not args.no_shift:
        console.print("  Finding most stable frame...")
        shift_idx, stability = find_most_stable_frame(frames)
        if shift_idx > 0:
            frames = shift_frames(frames, shift_idx)
            console.print(
                f"  Shifted start to frame [cyan]{shift_idx + 1}[/cyan] "
                f"(stability [cyan]{stability:.3f}[/cyan])"
            )

    # Build loop.
    console.print("  Building loop...")
    looped, info = build_loop(frames, args.budget, args.mode)

    # Results table.
    console.print()
    result_table = Table(show_header=False, show_edge=False, pad_edge=False, padding=(0, 2))
    result_table.add_column(style="bold")
    result_table.add_column()
    result_table.add_row("First↔Last SSIM", f"[cyan]{info['first_last_ssim']:.3f}[/cyan]")
    if shift_idx > 0:
        result_table.add_row("Shifted to frame", f"[cyan]{shift_idx + 1}[/cyan] (stability [cyan]{stability:.3f}[/cyan])")
    result_table.add_row("Strategy", f"[cyan]{info['mode_used']}[/cyan]")
    result_table.add_row("Bridge frames", f"[cyan]{info['bridge_frames']}[/cyan]")
    if info["reverse_frames"] > 0:
        result_table.add_row("  Reverse", f"[cyan]{info['reverse_frames']}[/cyan]")
    if info["crossfade_frames"] > 0:
        result_table.add_row("  Crossfade", f"[cyan]{info['crossfade_frames']}[/cyan]")
    result_table.add_row("Output frames", f"[cyan]{len(looped)}[/cyan]")
    out_duration = len(looped) / fps
    result_table.add_row("Output duration", f"[cyan]{out_duration:.1f}s[/cyan]")
    if args.repeat > 1:
        total_frames = len(looped) * args.repeat
        total_duration = total_frames / fps
        result_table.add_row("Repeat", f"[cyan]×{args.repeat}[/cyan] ({total_frames} frames, {total_duration:.1f}s)")

    # Check loop quality: compare new last frame to first frame.
    loop_ssim = _frame_ssim(looped[-1], looped[0])
    quality = "excellent" if loop_ssim > 0.95 else "good" if loop_ssim > 0.85 else "fair"
    color = "green" if loop_ssim > 0.95 else "yellow" if loop_ssim > 0.85 else "red"
    result_table.add_row("Loop SSIM", f"[{color}]{loop_ssim:.3f} ({quality})[/{color}]")

    console.print(result_table)

    # Write output.
    write_video(looped, args.output, fps, args.repeat)
    abs_output = os.path.abspath(args.output)

    console.print()
    console.print("[bold green]✓ Done[/bold green]")
    console.print()
    console.print(f"file://{abs_output}")
    console.print()


if __name__ == "__main__":
    main()

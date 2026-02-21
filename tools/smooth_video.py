#!/usr/bin/env python3
"""Detect and fix jittery frames in a video by blending neighbors."""

import argparse
import os
import sys

import cv2
import numpy as np
from rich.table import Table
from skimage.metrics import structural_similarity as ssim

from tools.lib.console import console


def _to_gray_small(frame):
    """Downscale to 160×120 grayscale for fast SSIM comparison."""
    small = cv2.resize(frame, (160, 120))
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def detect_jitter(frames, sigma=2.0):
    """Detect jittery frames using SSIM drop below mean - sigma*std.

    Returns list of (index, ssim_score) for jittery frames.
    """
    pair_ssims = []
    grays = [_to_gray_small(f) for f in frames]
    for i in range(len(grays) - 1):
        pair_ssims.append(ssim(grays[i], grays[i + 1]))

    mean_s = np.mean(pair_ssims)
    std_s = np.std(pair_ssims)
    threshold = mean_s - sigma * std_s

    jittery = []
    for i, s in enumerate(pair_ssims):
        if s < threshold:
            # The jitter is at frame i+1 (it differs from frame i).
            jittery.append((i + 1, s))

    return jittery, threshold, mean_s, std_s


def _group_runs(jittery_indices):
    """Group consecutive indices into runs."""
    jitter_set = set(jittery_indices)
    runs = []
    current_run = []
    for idx in sorted(jitter_set):
        if current_run and idx != current_run[-1] + 1:
            runs.append(current_run)
            current_run = []
        current_run.append(idx)
    if current_run:
        runs.append(current_run)
    return runs


def _compute_window(run, budget, n):
    """Compute the replacement window around a jitter run."""
    center = (run[0] + run[-1]) // 2
    half = budget // 2
    win_start = max(center - half, 1)
    win_end = min(center + half, n - 2)
    return win_start, win_end


def smooth_frames(frames, jittery_indices, budget=1):
    """Replace frames around each jitter point with a linear blend."""
    result = list(frames)
    n = len(frames)

    for run in _group_runs(jittery_indices):
        win_start, win_end = _compute_window(run, budget, n)
        before = win_start - 1
        after = win_end + 1
        win_len = win_end - win_start + 1
        for j in range(win_len):
            idx = win_start + j
            alpha = (j + 1) / (win_len + 1)
            result[idx] = cv2.addWeighted(
                frames[before], 1.0 - alpha, frames[after], alpha, 0
            )

    return result


def smooth_frames_rife(frames, jittery_indices, budget=1):
    """Replace frames around each jitter point using RIFE AI interpolation."""
    import torch
    from torchvision import transforms

    from ccvfi import AutoModel, ConfigType

    model = AutoModel.from_pretrained(
        pretrained_model_name=ConfigType.RIFE_IFNet_v426_heavy
    )
    model.device = torch.device("cpu")
    model.fp16 = False
    model.model = model.model.float().cpu()

    def _to_tensor(img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return transforms.ToTensor()(img_rgb).unsqueeze(0)

    def _interpolate(img0, img1, t):
        inp = torch.stack([_to_tensor(img0), _to_tensor(img1)], dim=1)
        with torch.inference_mode():
            out = model.inference(inp, timestep=t, scale=1.0)
        img = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = (img * 255).clip(0, 255).astype("uint8")
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    result = list(frames)
    n = len(frames)

    runs = _group_runs(jittery_indices)
    for ri, run in enumerate(runs):
        win_start, win_end = _compute_window(run, budget, n)
        before = win_start - 1
        after = win_end + 1
        win_len = win_end - win_start + 1

        console.print(
            f"  RIFE interpolating jitter {ri + 1}/{len(runs)} "
            f"({win_len} frames)..."
        )

        for j in range(win_len):
            idx = win_start + j
            t = (j + 1) / (win_len + 1)
            result[idx] = _interpolate(frames[before], frames[after], t)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Detect and fix jittery frames in a video."
    )
    parser.add_argument("--input", required=True, help="Input video file.")
    parser.add_argument(
        "--output", default="./tmp/smooth_video/output.mp4",
        help="Output video file path (default: ./tmp/smooth_video/output.mp4)",
    )
    parser.add_argument(
        "--sigma", type=float, default=2.0,
        help="Sensitivity: frames with SSIM below mean - sigma*std are jittery (default: 2.0).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only detect and report jitter, don't produce output.",
    )
    parser.add_argument(
        "--budget", type=int, default=0,
        help="Frames to spend per jitter point for gradual transition. "
        "0 = auto (scale based on SSIM gap). Higher = smoother but more ghosting.",
    )
    parser.add_argument(
        "--mode", choices=["blend", "ai"], default="blend",
        help="Interpolation mode: 'blend' (fast linear blend) or "
        "'ai' (AI frame interpolation, slower but motion-aware). Default: blend.",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        parser.error(f"Could not open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0

    # Print header immediately.
    console.print()
    console.print("[bold cyan]🩹 Smooth[/bold cyan]")
    console.print()

    info_table = Table(show_header=False, show_edge=False, pad_edge=False, padding=(0, 2))
    info_table.add_column(style="bold")
    info_table.add_column()
    info_table.add_row("Input", f"[cyan]{os.path.abspath(args.input)}[/cyan]")
    info_table.add_row("Resolution", f"[cyan]{w}×{h}[/cyan]")
    info_table.add_row("FPS", f"[cyan]{fps:.0f}[/cyan]")
    info_table.add_row("Frames", f"[cyan]{frame_count}[/cyan] ({duration:.1f}s)")
    info_table.add_row("Sigma", f"[cyan]{args.sigma}[/cyan]")
    info_table.add_row("Mode", f"[cyan]{args.mode}[/cyan]")
    console.print(info_table)
    console.print()

    # Extract all frames.
    console.print("  Extracting frames...")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) < 3:
        print("Error: need at least 3 frames.", file=sys.stderr)
        sys.exit(1)

    console.print(f"  Extracted [cyan]{len(frames)}[/cyan] frames")
    console.print("  Detecting jitter...")

    jittery, threshold, mean_s, std_s = detect_jitter(frames, args.sigma)

    # Report.
    console.print()
    if jittery:
        console.print(f"  [bold red]{len(jittery)} jittery frame(s)[/bold red] (threshold: SSIM < {threshold:.3f})")
        console.print()

        jitter_table = Table(box=None, padding=(0, 2))
        jitter_table.add_column("Frame", style="bold", justify="right")
        jitter_table.add_column("Time", justify="right")
        jitter_table.add_column("SSIM", justify="right")
        for idx, s in jittery:
            t = idx / fps
            jitter_table.add_row(
                str(idx + 1),
                f"[cyan]{t:.2f}s[/cyan]",
                f"[red]{s:.3f}[/red]",
            )
        console.print(jitter_table)
    else:
        console.print("  [green]No jitter detected[/green]")

    if args.dry_run or not jittery:
        console.print()
        return

    # Fix jittery frames.
    console.print()
    console.print("  Smoothing...")
    jittery_indices = [idx for idx, _ in jittery]

    # Auto budget: scale based on worst SSIM gap.
    budget = args.budget
    if budget == 0:
        worst_ssim = min(s for _, s in jittery)
        gap = 1.0 - worst_ssim
        # Map gap to budget: 0.01→1, 0.1→5, 0.3→11, 0.5→15
        budget = max(1, min(15, int(gap * 30 + 0.5)))

    smoothed = (
        smooth_frames_rife(frames, jittery_indices, budget=budget)
        if args.mode == "ai"
        else smooth_frames(frames, jittery_indices, budget=budget)
    )

    # Write output.
    abs_output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(abs_output) or ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(abs_output, fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"Error: could not create video '{abs_output}'.", file=sys.stderr)
        sys.exit(1)

    for frame in smoothed:
        writer.write(frame)
    writer.release()

    console.print(f"  Replaced [cyan]{len(jittery)}[/cyan] jitter points ([cyan]{budget}[/cyan] frames each, [cyan]{len(jittery) * budget}[/cyan] total)")
    console.print()
    console.print("[bold green]✓ Done[/bold green]")
    console.print()
    console.print(f"file://{abs_output}")
    console.print()


if __name__ == "__main__":
    main()

"""Shared video analysis utilities for sprite-forge tools."""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def find_loop_segment(cap, t_start, t_end, sample_interval=0.1, min_loop_duration=0.5):
    """Find the best loop segment using SSIM frame comparison.

    Returns (loop_start, loop_end, score) or None if no good loop found.
    """
    # Sample frames at regular intervals.
    timestamps = []
    frames = []
    t = t_start
    while t <= t_end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        # Downscale for faster comparison.
        small = cv2.resize(frame, (160, 120))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        timestamps.append(t)
        frames.append(gray)
        t += sample_interval

    if len(frames) < 2:
        return None

    best_score = -1
    best_start = 0
    best_end = 0

    # Compare each frame against later frames to find best loop pair.
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            if timestamps[j] - timestamps[i] < min_loop_duration:
                continue
            score = ssim(frames[i], frames[j])
            if score > best_score:
                best_score = score
                best_start = timestamps[i]
                best_end = timestamps[j]

    if best_score < 0:
        return None

    return (best_start, best_end, best_score)


def sample_frames_ssim(cap, t_start, t_end):
    """Sample all frames at native FPS and return list of (timestamp, grayscale_small_frame).

    Frames are downscaled to 160×120 grayscale for fast SSIM comparison.
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        return []

    interval = 1.0 / fps
    samples = []
    t = t_start
    while t <= t_end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (160, 120))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        samples.append((t, gray))
        t += interval

    return samples

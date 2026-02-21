#!/usr/bin/env python3
"""Tests for frames_to_video tool."""

import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np


TOOLS_DIR = os.path.join(os.path.dirname(__file__), "..", "tools")
COMPOSE = os.path.join(TOOLS_DIR, "frames_to_video.py")


def run_tool(args):
    """Run a tool script and return the result."""
    return subprocess.run(
        [sys.executable] + args,
        capture_output=True, text=True,
    )


def _make_frames(input_dir, n_frames=4, width=100, height=80):
    """Create solid-color test frames with increasing brightness."""
    os.makedirs(input_dir, exist_ok=True)
    for i in range(n_frames):
        val = int(255 * i / max(n_frames - 1, 1))
        img = np.full((height, width, 3), val, dtype=np.uint8)
        cv2.imwrite(os.path.join(input_dir, f"frame_{i + 1:02d}.png"), img)


def test_basic_compose():
    """Compose 4 frames into an MP4."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "frames")
        output = os.path.join(tmpdir, "out.mp4")
        _make_frames(input_dir, n_frames=4)

        result = run_tool([COMPOSE, "--input-dir", input_dir, "--output", output])
        assert result.returncode == 0, f"Failed: {result.stderr}"
        assert os.path.isfile(output), "Output MP4 not created"

        cap = cv2.VideoCapture(output)
        assert cap.isOpened(), "Cannot open output MP4"
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert frame_count == 4, f"Expected 4 frames, got {frame_count}"


def test_custom_fps():
    """Custom FPS should be set in output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "frames")
        output = os.path.join(tmpdir, "out.mp4")
        _make_frames(input_dir, n_frames=4)

        result = run_tool([COMPOSE, "--input-dir", input_dir, "--output", output,
                           "--fps", "24"])
        assert result.returncode == 0, f"Failed: {result.stderr}"

        cap = cv2.VideoCapture(output)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        assert abs(fps - 24) < 1, f"Expected ~24 fps, got {fps}"


def test_ping_pong():
    """--ping-pong should append reversed frames (excluding endpoints)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "frames")
        output = os.path.join(tmpdir, "out.mp4")
        _make_frames(input_dir, n_frames=4)

        result = run_tool([COMPOSE, "--input-dir", input_dir, "--output", output,
                           "--ping-pong"])
        assert result.returncode == 0, f"Failed: {result.stderr}"

        cap = cv2.VideoCapture(output)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        # 4 forward + 2 reversed (excluding first and last) = 6
        assert frame_count == 6, f"Expected 6 frames (4+2), got {frame_count}"


def test_prefix_filter():
    """--prefix should filter input frames."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "frames")
        output = os.path.join(tmpdir, "out.mp4")
        os.makedirs(input_dir)

        for i in range(3):
            img = np.full((50, 100, 3), 128, dtype=np.uint8)
            cv2.imwrite(os.path.join(input_dir, f"bg_{i + 1:02d}.png"), img)
        img = np.full((50, 100, 3), 200, dtype=np.uint8)
        cv2.imwrite(os.path.join(input_dir, "other.png"), img)

        result = run_tool([COMPOSE, "--input-dir", input_dir, "--output", output,
                           "--prefix", "bg_"])
        assert result.returncode == 0, f"Failed: {result.stderr}"

        cap = cv2.VideoCapture(output)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert frame_count == 3, f"Expected 3 frames, got {frame_count}"


def test_no_frames_error():
    """Should fail with empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "empty")
        os.makedirs(input_dir)

        result = run_tool([COMPOSE, "--input-dir", input_dir,
                           "--output", os.path.join(tmpdir, "out.mp4")])
        assert result.returncode != 0, "Should fail with no frames"


if __name__ == "__main__":
    tests = [
        test_basic_compose,
        test_custom_fps,
        test_ping_pong,
        test_prefix_filter,
        test_no_frames_error,
    ]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  ✓ {test.__name__}")
        except AssertionError as e:
            print(f"  ✗ {test.__name__}: {e}")
            failed += 1

    print(f"\n{len(tests) - failed}/{len(tests)} tests passed.")
    sys.exit(1 if failed else 0)

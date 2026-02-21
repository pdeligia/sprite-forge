#!/usr/bin/env python3
"""Tests for gen_video tool."""

import os
import subprocess
import sys
import tempfile

import cv2


TOOLS_DIR = os.path.join(os.path.dirname(__file__), "..", "tools")
GEN_MP4 = os.path.join(TOOLS_DIR, "gen_video.py")


def run_tool(args):
    """Run a tool script and return the result."""
    return subprocess.run(
        [sys.executable] + args,
        capture_output=True, text=True,
    )


def test_default_output():
    """Generate a video with default settings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "test.mp4")
        result = run_tool([GEN_MP4, out])
        assert result.returncode == 0, f"gen_video failed: {result.stderr}"
        assert os.path.isfile(out), "Output file not created"

        cap = cv2.VideoCapture(out)
        assert cap.isOpened(), "Could not open output video"
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 320
        assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 240
        fps = cap.get(cv2.CAP_PROP_FPS)
        assert abs(fps - 30) < 1, f"Expected ~30 fps, got {fps}"
        cap.release()


def test_custom_dimensions():
    """Generate a video with custom width and height."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "test.mp4")
        result = run_tool([GEN_MP4, out, "--width", "640", "--height", "480"])
        assert result.returncode == 0, f"gen_video failed: {result.stderr}"

        cap = cv2.VideoCapture(out)
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 640
        assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 480
        cap.release()


def test_custom_duration():
    """Generate a video with custom duration and verify frame count."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "test.mp4")
        result = run_tool([GEN_MP4, out, "--duration", "2.0", "--fps", "10"])
        assert result.returncode == 0, f"gen_video failed: {result.stderr}"

        cap = cv2.VideoCapture(out)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert total == 20, f"Expected 20 frames (2s * 10fps), got {total}"
        cap.release()


def test_color_preset():
    """Generate a video with a color preset and verify pixel color."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "test.mp4")
        result = run_tool([GEN_MP4, out, "--color", "blue", "--duration", "0.5"])
        assert result.returncode == 0, f"gen_video failed: {result.stderr}"

        cap = cv2.VideoCapture(out)
        ret, frame = cap.read()
        assert ret, "Could not read frame"
        # Blue in BGR = (255, 0, 0). Allow some codec tolerance.
        pixel = frame[0, 0]
        assert pixel[0] > 200, f"Expected high blue channel, got {pixel}"
        cap.release()


def test_color_rgb():
    """Generate a video with custom RGB color."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "test.mp4")
        result = run_tool([GEN_MP4, out, "--color", "0,255,0", "--duration", "0.5"])
        assert result.returncode == 0, f"gen_video failed: {result.stderr}"

        cap = cv2.VideoCapture(out)
        ret, frame = cap.read()
        assert ret, "Could not read frame"
        pixel = frame[0, 0]
        assert pixel[1] > 200, f"Expected high green channel, got {pixel}"
        cap.release()


def test_default_path():
    """Generate a video with no output arg (default ./tmp/gen_video/test.mp4)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run from a temp dir so default ./tmp/gen_video/test.mp4 goes there.
        result = subprocess.run(
            [sys.executable, os.path.abspath(GEN_MP4)],
            capture_output=True, text=True, cwd=tmpdir,
        )
        assert result.returncode == 0, f"gen_video failed: {result.stderr}"
        assert os.path.isfile(os.path.join(tmpdir, "tmp", "gen_video", "test.mp4")), "Default output not created"


if __name__ == "__main__":
    tests = [
        test_default_output,
        test_custom_dimensions,
        test_custom_duration,
        test_color_preset,
        test_color_rgb,
        test_default_path,
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

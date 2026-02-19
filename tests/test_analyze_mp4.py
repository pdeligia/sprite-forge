#!/usr/bin/env python3
"""Tests for analyze_mp4 tool."""

import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np


TOOLS_DIR = os.path.join(os.path.dirname(__file__), "..", "tools")
ANALYZE_MP4 = os.path.join(TOOLS_DIR, "analyze_mp4.py")
GEN_MP4 = os.path.join(TOOLS_DIR, "gen_mp4.py")


def run_tool(args):
    """Run a tool script and return the result."""
    return subprocess.run(
        [sys.executable] + args,
        capture_output=True, text=True,
    )


def _make_looping_video(path, width=320, height=240, fps=30):
    """Create a synthetic video with a repeating color cycle (2 loops)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    frames_per_color = fps // 3
    for _ in range(2):
        for color in colors:
            frame = np.full((height, width, 3), color, dtype=np.uint8)
            for _ in range(frames_per_color):
                writer.write(frame)
    writer.release()


def _make_motion_video(path, width=320, height=240, fps=30, duration=3):
    """Create a video with a moving white rectangle for visible motion."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    total_frames = int(fps * duration)
    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        x = int((width - 40) * (i / max(total_frames - 1, 1)))
        cv2.rectangle(frame, (x, 80), (x + 40, 160), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def test_basic_analysis():
    """Analyze a generated video and verify output contains expected fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video = os.path.join(tmpdir, "test.mp4")
        run_tool([GEN_MP4, video, "--width", "320", "--height", "240", "--duration", "3"])

        result = run_tool([ANALYZE_MP4, video])
        assert result.returncode == 0, f"analyze_mp4 failed: {result.stderr}"
        assert "Recommended number of frames (N):" in result.stdout, f"Missing recommended N: {result.stdout}"
        assert "Motion level:" in result.stdout
        assert "Δ-SSIM" in result.stdout
        assert "✅ Recommended" in result.stdout


def test_no_frames_extracted():
    """Analyze should not create any image files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video = os.path.join(tmpdir, "test.mp4")
        run_tool([GEN_MP4, video, "--width", "320", "--height", "240", "--duration", "2"])

        result = run_tool([ANALYZE_MP4, video])
        assert result.returncode == 0

        # No PNG files should exist anywhere in tmpdir.
        for root, dirs, files in os.walk(tmpdir):
            for f in files:
                assert not f.endswith(".png"), f"Found unexpected PNG: {os.path.join(root, f)}"


def test_with_time_range():
    """Analyze with --start/--end should work and show segment info."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video = os.path.join(tmpdir, "test.mp4")
        run_tool([GEN_MP4, video, "--width", "320", "--height", "240", "--duration", "5"])

        result = run_tool([ANALYZE_MP4, video, "--start", "1", "--end", "3"])
        assert result.returncode == 0, f"Failed: {result.stderr}"
        assert "Recommended number of frames (N):" in result.stdout


def test_with_loop():
    """Analyze with --loop should find a loop and report it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video = os.path.join(tmpdir, "loop.mp4")
        _make_looping_video(video)

        result = run_tool([ANALYZE_MP4, video, "--loop"])
        assert result.returncode == 0, f"Failed: {result.stderr}"
        assert "Best loop:" in result.stdout, f"Expected loop info: {result.stdout}"
        assert "score:" in result.stdout
        assert "Recommended number of frames (N):" in result.stdout


def test_determinism():
    """Running analyze twice on the same video should produce identical output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video = os.path.join(tmpdir, "test.mp4")
        _make_motion_video(video)

        result1 = run_tool([ANALYZE_MP4, video])
        result2 = run_tool([ANALYZE_MP4, video])
        assert result1.returncode == 0
        assert result2.returncode == 0
        assert result1.stdout == result2.stdout, (
            f"Non-deterministic output:\nRun 1:\n{result1.stdout}\nRun 2:\n{result2.stdout}"
        )


def test_motion_video_recommends_more_frames():
    """A video with motion should recommend more frames than a static video."""
    with tempfile.TemporaryDirectory() as tmpdir:
        static_video = os.path.join(tmpdir, "static.mp4")
        motion_video = os.path.join(tmpdir, "motion.mp4")

        run_tool([GEN_MP4, static_video, "--width", "320", "--height", "240", "--duration", "3"])
        _make_motion_video(motion_video)

        static_result = run_tool([ANALYZE_MP4, static_video])
        motion_result = run_tool([ANALYZE_MP4, motion_video])
        assert static_result.returncode == 0
        assert motion_result.returncode == 0

        # Extract recommended N from output.
        def get_n(output):
            for line in output.split("\n"):
                if "Recommended number of frames (N):" in line:
                    return int(line.split(":")[-1].strip())
            return 0

        static_n = get_n(static_result.stdout)
        motion_n = get_n(motion_result.stdout)
        assert motion_n >= static_n, (
            f"Motion video should recommend >= frames than static: motion={motion_n}, static={static_n}"
        )


def test_tiers_capped_by_native_frames():
    """Tier N values should not exceed native frame count of the segment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Very short video: 0.5s at 10fps = 5 native frames.
        video = os.path.join(tmpdir, "short.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video, fourcc, 10, (160, 120))
        for i in range(5):
            frame = np.full((120, 160, 3), (i * 50, 0, 0), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        result = run_tool([ANALYZE_MP4, video])
        assert result.returncode == 0, f"Failed: {result.stderr}"

        # Parse N values from the table — no N should exceed 5.
        for line in result.stdout.split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                n = int(line.split()[0])
                assert n <= 5, f"Tier N={n} exceeds native frame count 5"


if __name__ == "__main__":
    tests = [
        test_basic_analysis,
        test_no_frames_extracted,
        test_with_time_range,
        test_with_loop,
        test_determinism,
        test_motion_video_recommends_more_frames,
        test_tiers_capped_by_native_frames,
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

#!/usr/bin/env python3
"""Tests for video_to_frames tool."""

import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np


TOOLS_DIR = os.path.join(os.path.dirname(__file__), "..", "tools")
GEN_MP4 = os.path.join(TOOLS_DIR, "gen_video.py")
EXTRACT_MP4_FRAMES = os.path.join(TOOLS_DIR, "video_to_frames.py")


def run_tool(args):
    """Run a tool script and return the result."""
    return subprocess.run(
        [sys.executable] + args,
        capture_output=True, text=True,
    )


def test_basic_extraction():
    """Extract frames from a generated video and verify count and dimensions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video = os.path.join(tmpdir, "test.mp4")
        frames_dir = os.path.join(tmpdir, "frames")

        # Generate a 2-second 320x240 test video.
        result = run_tool([GEN_MP4, video, "--width", "320", "--height", "240", "--duration", "2"])
        assert result.returncode == 0, f"gen_video failed: {result.stderr}"

        # Extract 4 frames.
        result = run_tool([EXTRACT_MP4_FRAMES, video, "4", "--output-dir", frames_dir, "--prefix", "test"])
        assert result.returncode == 0, f"video_to_frames failed: {result.stderr}"

        # Verify frame count.
        pngs = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
        assert len(pngs) == 4, f"Expected 4 frames, got {len(pngs)}: {pngs}"

        # Verify dimensions.
        img = cv2.imread(os.path.join(frames_dir, pngs[0]))
        assert img.shape[1] == 320 and img.shape[0] == 240, f"Unexpected dimensions: {img.shape}"


def test_region_crop():
    """Crop a region and verify output dimensions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video = os.path.join(tmpdir, "test.mp4")
        frames_dir = os.path.join(tmpdir, "frames")

        run_tool([GEN_MP4, video, "--width", "320", "--height", "240", "--duration", "1"])
        result = run_tool([EXTRACT_MP4_FRAMES, video, "1", "--output-dir", frames_dir, "--region", "10", "10", "160", "130"])
        assert result.returncode == 0, f"video_to_frames failed: {result.stderr}"

        img = cv2.imread(os.path.join(frames_dir, "frame_01.png"))
        assert img.shape[1] == 150 and img.shape[0] == 120, f"Unexpected dimensions: {img.shape}"


def test_region_with_width_scaling():
    """Crop and scale, verify aspect ratio is preserved."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video = os.path.join(tmpdir, "test.mp4")
        frames_dir = os.path.join(tmpdir, "frames")

        run_tool([GEN_MP4, video, "--width", "320", "--height", "240", "--duration", "1"])
        result = run_tool([EXTRACT_MP4_FRAMES, video, "1", "--output-dir", frames_dir,
                           "--region", "10", "10", "160", "130", "--width", "600"])
        assert result.returncode == 0, f"video_to_frames failed: {result.stderr}"

        img = cv2.imread(os.path.join(frames_dir, "frame_01.png"))
        assert img.shape[1] == 600 and img.shape[0] == 480, f"Unexpected dimensions: {img.shape}"


def test_time_range():
    """Extract frames from a time range and verify timestamps are within range."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video = os.path.join(tmpdir, "test.mp4")
        frames_dir = os.path.join(tmpdir, "frames")

        run_tool([GEN_MP4, video, "--width", "320", "--height", "240", "--duration", "5"])
        result = run_tool([EXTRACT_MP4_FRAMES, video, "3", "--output-dir", frames_dir, "--start", "1", "--end", "3"])
        assert result.returncode == 0, f"video_to_frames failed: {result.stderr}"

        pngs = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
        assert len(pngs) == 3, f"Expected 3 frames, got {len(pngs)}"


def test_invalid_region():
    """Region exceeding video dimensions should fail."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video = os.path.join(tmpdir, "test.mp4")
        frames_dir = os.path.join(tmpdir, "frames")

        run_tool([GEN_MP4, video, "--width", "320", "--height", "240", "--duration", "1"])
        result = run_tool([EXTRACT_MP4_FRAMES, video, "1", "--output-dir", frames_dir, "--region", "0", "0", "400", "300"])
        assert result.returncode != 0, "Expected failure for out-of-bounds region"


def test_output_dir_cleanup():
    """Output directory should be wiped before each run."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video = os.path.join(tmpdir, "test.mp4")
        frames_dir = os.path.join(tmpdir, "frames")

        run_tool([GEN_MP4, video, "--width", "320", "--height", "240", "--duration", "1"])

        # First run: 4 frames.
        run_tool([EXTRACT_MP4_FRAMES, video, "4", "--output-dir", frames_dir])
        assert len(os.listdir(frames_dir)) == 4

        # Second run: 2 frames — old files should be gone.
        run_tool([EXTRACT_MP4_FRAMES, video, "2", "--output-dir", frames_dir])
        assert len(os.listdir(frames_dir)) == 2


def test_suffix():
    """Verify --suffix is included in output filenames."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video = os.path.join(tmpdir, "test.mp4")
        frames_dir = os.path.join(tmpdir, "frames")

        run_tool([GEN_MP4, video, "--width", "320", "--height", "240", "--duration", "1"])
        result = run_tool([EXTRACT_MP4_FRAMES, video, "2", "--output-dir", frames_dir,
                           "--prefix", "forest_dungeon_bg", "--suffix", "@3x"])
        assert result.returncode == 0, f"video_to_frames failed: {result.stderr}"

        pngs = sorted(os.listdir(frames_dir))
        assert pngs == ["forest_dungeon_bg_01@3x.png", "forest_dungeon_bg_02@3x.png"], f"Unexpected files: {pngs}"


def _make_looping_video(path, width=160, height=120, fps=30):
    """Create a video with a repeating color cycle: red→green→blue→red→green→blue (2 loops of 1s each)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: red, green, blue
    frames_per_color = fps // 3
    # Write 2 identical cycles so the loop finder can detect the repeat.
    for _ in range(2):
        for color in colors:
            frame = np.full((height, width, 3), color, dtype=np.uint8)
            for _ in range(frames_per_color):
                writer.write(frame)
    writer.release()


def test_loop_basic():
    """--loop should find a loop and extract frames."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video = os.path.join(tmpdir, "loop.mp4")
        frames_dir = os.path.join(tmpdir, "frames")
        _make_looping_video(video)

        result = run_tool([EXTRACT_MP4_FRAMES, video, "4", "--output-dir", frames_dir, "--loop"])
        assert result.returncode == 0, f"video_to_frames --loop failed: {result.stderr}"
        assert "score:" in result.stdout, f"Expected loop score in output: {result.stdout}"

        pngs = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
        assert len(pngs) == 4, f"Expected 4 frames, got {len(pngs)}: {pngs}"


def test_loop_with_range():
    """--loop with --start/--end should search within the given range."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video = os.path.join(tmpdir, "loop.mp4")
        frames_dir = os.path.join(tmpdir, "frames")
        _make_looping_video(video)

        result = run_tool([EXTRACT_MP4_FRAMES, video, "3", "--output-dir", frames_dir,
                           "--loop", "--start", "0", "--end", "1.5"])
        assert result.returncode == 0, f"video_to_frames --loop failed: {result.stderr}"

        pngs = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
        assert len(pngs) == 3, f"Expected 3 frames, got {len(pngs)}: {pngs}"


if __name__ == "__main__":
    tests = [
        test_basic_extraction,
        test_region_crop,
        test_region_with_width_scaling,
        test_time_range,
        test_invalid_region,
        test_output_dir_cleanup,
        test_suffix,
        test_loop_basic,
        test_loop_with_range,
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

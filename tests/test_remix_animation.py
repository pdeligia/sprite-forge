#!/usr/bin/env python3
"""Tests for remix_animation tool."""

import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np


TOOLS_DIR = os.path.join(os.path.dirname(__file__), "..", "tools")
REMIX = os.path.join(TOOLS_DIR, "remix_animation.py")


def run_tool(args):
    """Run a tool script and return the result."""
    return subprocess.run(
        [sys.executable] + args,
        capture_output=True, text=True,
    )


def _make_animation_frames(input_dir, n_frames=4, width=200, height=100):
    """Create test animation frames with a moving bright band."""
    os.makedirs(input_dir, exist_ok=True)
    for f in range(n_frames):
        img = np.full((height, width, 3), 40, dtype=np.uint8)
        # Static top half (sky).
        img[:height // 2, :] = (100, 120, 140)
        # Animated band moves across bottom half.
        band_y = height // 2 + (f * 10) % (height // 2)
        band_end = min(band_y + 10, height)
        img[band_y:band_end, :] = (200, 220, 255)
        path = os.path.join(input_dir, f"frame_{f + 1:02d}.png")
        cv2.imwrite(path, img)


def test_basic_remix():
    """Remix 4 frames into 4 output frames with default speeds."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "frames")
        out_dir = os.path.join(tmpdir, "out")
        _make_animation_frames(input_dir, n_frames=4)

        result = run_tool([REMIX, "--no-model", "--input-dir", input_dir,
                           "--layers", "2", "--output-dir", out_dir])
        assert result.returncode == 0, f"Failed: {result.stderr}"

        pngs = sorted(f for f in os.listdir(out_dir) if f.endswith(".png"))
        assert len(pngs) == 4, f"Expected 4 output frames, got {len(pngs)}: {pngs}"

        # All should be RGBA.
        for p in pngs:
            img = cv2.imread(os.path.join(out_dir, p), cv2.IMREAD_UNCHANGED)
            assert img.shape[2] == 4, f"{p} is not RGBA"


def test_retime_more_frames():
    """Remix 4 input frames into 8 output frames."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "frames")
        out_dir = os.path.join(tmpdir, "out")
        _make_animation_frames(input_dir, n_frames=4)

        result = run_tool([REMIX, "--no-model", "--input-dir", input_dir,
                           "--layers", "2", "--frames", "8", "--output-dir", out_dir])
        assert result.returncode == 0, f"Failed: {result.stderr}"

        pngs = sorted(f for f in os.listdir(out_dir) if f.endswith(".png"))
        assert len(pngs) == 8, f"Expected 8 output frames, got {len(pngs)}: {pngs}"


def test_per_layer_speed():
    """Per-layer speeds should be accepted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "frames")
        out_dir = os.path.join(tmpdir, "out")
        _make_animation_frames(input_dir, n_frames=4)

        result = run_tool([REMIX, "--no-model", "--input-dir", input_dir,
                           "--layers", "3", "--frames", "12",
                           "--speed", "0.5,1.0,2.0", "--output-dir", out_dir])
        assert result.returncode == 0, f"Failed: {result.stderr}"

        pngs = sorted(f for f in os.listdir(out_dir) if f.endswith(".png"))
        assert len(pngs) == 12, f"Expected 12 output frames, got {len(pngs)}: {pngs}"


def test_output_dir_nuked():
    """Output directory should be wiped clean before each run."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "frames")
        out_dir = os.path.join(tmpdir, "out")
        os.makedirs(out_dir)
        stale = os.path.join(out_dir, "stale.txt")
        with open(stale, "w") as f:
            f.write("old")

        _make_animation_frames(input_dir, n_frames=4)
        run_tool([REMIX, "--no-model", "--input-dir", input_dir,
                  "--layers", "2", "--output-dir", out_dir])

        assert not os.path.exists(stale), "Stale file should have been removed"


def test_suffix_preserved():
    """Output filenames should preserve --suffix."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "frames")
        out_dir = os.path.join(tmpdir, "out")
        os.makedirs(input_dir)

        for f in range(3):
            img = np.full((50, 100, 3), f * 40 + 40, dtype=np.uint8)
            cv2.imwrite(os.path.join(input_dir, f"scene_{f + 1:02d}@3x.png"), img)

        result = run_tool([REMIX, "--no-model", "--input-dir", input_dir,
                           "--layers", "2", "--suffix", "@3x", "--output-dir", out_dir])
        assert result.returncode == 0, f"Failed: {result.stderr}"

        pngs = sorted(os.listdir(out_dir))
        for p in pngs:
            assert "@3x" in p, f"Suffix missing from {p}"


def test_needs_two_frames():
    """Should fail with fewer than 2 input frames."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "frames")
        out_dir = os.path.join(tmpdir, "out")
        os.makedirs(input_dir)

        img = np.full((50, 100, 3), 128, dtype=np.uint8)
        cv2.imwrite(os.path.join(input_dir, "frame_01.png"), img)

        result = run_tool([REMIX, "--no-model", "--input-dir", input_dir,
                           "--layers", "2", "--output-dir", out_dir])
        assert result.returncode != 0, "Should fail with only 1 frame"


def test_depth_map_output():
    """--depth-map should output a depth map image."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "frames")
        out_dir = os.path.join(tmpdir, "out")
        _make_animation_frames(input_dir, n_frames=3)

        result = run_tool([REMIX, "--no-model", "--input-dir", input_dir,
                           "--layers", "2", "--output-dir", out_dir, "--depth-map"])
        assert result.returncode == 0, f"Failed: {result.stderr}"

        files = os.listdir(out_dir)
        depth_maps = [f for f in files if "depth_map" in f]
        assert len(depth_maps) == 1, f"Expected 1 depth map, got {depth_maps}"


if __name__ == "__main__":
    tests = [
        test_basic_remix,
        test_retime_more_frames,
        test_per_layer_speed,
        test_output_dir_nuked,
        test_suffix_preserved,
        test_needs_two_frames,
        test_depth_map_output,
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

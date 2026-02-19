#!/usr/bin/env python3
"""Tests for merge_image tool."""

import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np


TOOLS_DIR = os.path.join(os.path.dirname(__file__), "..", "tools")
MERGE_IMAGE = os.path.join(TOOLS_DIR, "merge_image.py")


def run_tool(args):
    """Run a tool script and return the result."""
    return subprocess.run(
        [sys.executable] + args,
        capture_output=True, text=True,
    )


def make_image(path, width, height, color, alpha=None):
    """Create a solid-color PNG image."""
    if alpha is not None:
        img = np.full((height, width, 4), (*color, alpha), dtype=np.uint8)
    else:
        img = np.full((height, width, 3), color, dtype=np.uint8)
    cv2.imwrite(path, img)


def test_single_merge():
    """Merge a single foreground onto a background."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bg = os.path.join(tmpdir, "bg.png")
        fg = os.path.join(tmpdir, "fg.png")
        make_image(bg, 200, 200, (255, 0, 0))
        make_image(fg, 50, 50, (0, 255, 0))

        result = run_tool([MERGE_IMAGE, bg, "--input", fg])
        assert result.returncode == 0, f"merge_image failed: {result.stderr}"

        img = cv2.imread(fg)
        assert img.shape == (200, 200, 3), f"Unexpected shape: {img.shape}"
        # Top-left 50x50 should be green (foreground).
        assert np.all(img[0, 0] == [0, 255, 0]), f"Expected green at (0,0), got {img[0, 0]}"
        # Bottom-right should be red (background).
        assert np.all(img[199, 199] == [255, 0, 0]), f"Expected red at (199,199), got {img[199, 199]}"


def test_merge_with_offset():
    """Merge with an offset and verify placement."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bg = os.path.join(tmpdir, "bg.png")
        fg = os.path.join(tmpdir, "fg.png")
        make_image(bg, 200, 200, (255, 0, 0))
        make_image(fg, 50, 50, (0, 255, 0))

        result = run_tool([MERGE_IMAGE, bg, "--input", fg, "--offset", "100", "100"])
        assert result.returncode == 0, f"merge_image failed: {result.stderr}"

        img = cv2.imread(fg)
        # Origin should be red (background, not covered).
        assert np.all(img[0, 0] == [255, 0, 0]), f"Expected red at (0,0), got {img[0, 0]}"
        # Offset region should be green.
        assert np.all(img[100, 100] == [0, 255, 0]), f"Expected green at (100,100), got {img[100, 100]}"


def test_batch_with_prefix():
    """Merge multiple images filtered by prefix."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bg = os.path.join(tmpdir, "bg.png")
        make_image(bg, 200, 200, (255, 0, 0))

        fg_dir = os.path.join(tmpdir, "frames")
        os.makedirs(fg_dir)
        make_image(os.path.join(fg_dir, "forest_01.png"), 50, 50, (0, 255, 0))
        make_image(os.path.join(fg_dir, "forest_02.png"), 50, 50, (0, 255, 0))
        make_image(os.path.join(fg_dir, "other_01.png"), 50, 50, (0, 0, 255))

        result = run_tool([MERGE_IMAGE, bg, "--input-dir", fg_dir, "--prefix", "forest"])
        assert result.returncode == 0, f"merge_image failed: {result.stderr}"

        # Forest files should be merged (200x200).
        img1 = cv2.imread(os.path.join(fg_dir, "forest_01.png"))
        assert img1.shape == (200, 200, 3), f"Unexpected shape: {img1.shape}"

        # Other file should be untouched (50x50).
        img_other = cv2.imread(os.path.join(fg_dir, "other_01.png"))
        assert img_other.shape == (50, 50, 3), f"other_01 should be untouched: {img_other.shape}"


def test_batch_all():
    """Merge all PNGs in a directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bg = os.path.join(tmpdir, "bg.png")
        make_image(bg, 200, 200, (255, 0, 0))

        fg_dir = os.path.join(tmpdir, "frames")
        os.makedirs(fg_dir)
        make_image(os.path.join(fg_dir, "a.png"), 50, 50, (0, 255, 0))
        make_image(os.path.join(fg_dir, "b.png"), 50, 50, (0, 0, 255))

        result = run_tool([MERGE_IMAGE, bg, "--input-dir", fg_dir])
        assert result.returncode == 0, f"merge_image failed: {result.stderr}"

        for name in ("a.png", "b.png"):
            img = cv2.imread(os.path.join(fg_dir, name))
            assert img.shape == (200, 200, 3), f"{name} unexpected shape: {img.shape}"


def test_alpha_compositing():
    """Merge a semi-transparent foreground with alpha blending."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bg = os.path.join(tmpdir, "bg.png")
        fg = os.path.join(tmpdir, "fg.png")
        make_image(bg, 100, 100, (255, 0, 0))
        make_image(fg, 50, 50, (0, 255, 0), alpha=128)

        result = run_tool([MERGE_IMAGE, bg, "--input", fg])
        assert result.returncode == 0, f"merge_image failed: {result.stderr}"

        img = cv2.imread(fg)
        # Pixel at (0,0) should be a blend of green and red.
        pixel = img[0, 0]
        assert pixel[1] > 100, f"Expected green component > 100, got {pixel}"
        assert pixel[0] > 100, f"Expected red component > 100, got {pixel}"


def test_invalid_offset():
    """Foreground exceeding background should fail."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bg = os.path.join(tmpdir, "bg.png")
        fg = os.path.join(tmpdir, "fg.png")
        make_image(bg, 100, 100, (255, 0, 0))
        make_image(fg, 50, 50, (0, 255, 0))

        result = run_tool([MERGE_IMAGE, bg, "--input", fg, "--offset", "80", "80"])
        assert result.returncode != 0, "Expected failure for out-of-bounds offset"


if __name__ == "__main__":
    tests = [
        test_single_merge,
        test_merge_with_offset,
        test_batch_with_prefix,
        test_batch_all,
        test_alpha_compositing,
        test_invalid_offset,
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

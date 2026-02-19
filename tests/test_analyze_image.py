#!/usr/bin/env python3
"""Tests for analyze_image tool."""

import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np


TOOLS_DIR = os.path.join(os.path.dirname(__file__), "..", "tools")
ANALYZE = os.path.join(TOOLS_DIR, "analyze_image.py")


def run_tool(args):
    """Run a tool script and return the result."""
    return subprocess.run(
        [sys.executable] + args,
        capture_output=True, text=True,
    )


def _make_test_image(path, width=200, height=100):
    """Create a simple test image with a color gradient."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        val = int(255 * y / max(height - 1, 1))
        img[y, :] = (val, 128, 255 - val)
    cv2.imwrite(path, img)


def _make_rgba_image(path, width=200, height=100):
    """Create an RGBA test image with some transparency."""
    img = np.zeros((height, width, 4), dtype=np.uint8)
    img[:, :, :3] = 128
    img[:, :, 3] = 255
    img[:height // 2, :, 3] = 0  # Top half transparent.
    cv2.imwrite(path, img)


def test_basic_analysis():
    """Analyze a simple image and verify basic output fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "test.png")
        _make_test_image(img_path)

        result = run_tool([ANALYZE, img_path])
        assert result.returncode == 0, f"Failed: {result.stderr}"
        assert "Dimensions:" in result.stdout
        assert "200×100" in result.stdout
        assert "File size:" in result.stdout
        assert "Dominant hue:" in result.stdout
        assert "Brightness:" in result.stdout
        assert "Color richness:" in result.stdout


def test_rgba_analysis():
    """Analyze an RGBA image and verify alpha stats."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "test.png")
        _make_rgba_image(img_path)

        result = run_tool([ANALYZE, img_path])
        assert result.returncode == 0, f"Failed: {result.stderr}"
        assert "BGRA" in result.stdout
        assert "Alpha:" in result.stdout
        assert "opaque" in result.stdout
        assert "transparent" in result.stdout


def test_no_depth_by_default():
    """Without --depth, no depth analysis should appear."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "test.png")
        _make_test_image(img_path)

        result = run_tool([ANALYZE, img_path])
        assert result.returncode == 0
        assert "Depth" not in result.stdout
        assert "Recommended parallax layers" not in result.stdout


def test_invalid_input():
    """Non-existent file should fail."""
    result = run_tool([ANALYZE, "/nonexistent/file.png"])
    assert result.returncode != 0


def test_deterministic():
    """Running twice should produce identical output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "test.png")
        _make_test_image(img_path)

        r1 = run_tool([ANALYZE, img_path])
        r2 = run_tool([ANALYZE, img_path])
        assert r1.returncode == 0
        assert r1.stdout == r2.stdout


if __name__ == "__main__":
    tests = [
        test_basic_analysis,
        test_rgba_analysis,
        test_no_depth_by_default,
        test_invalid_input,
        test_deterministic,
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

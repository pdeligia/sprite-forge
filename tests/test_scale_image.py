#!/usr/bin/env python3
"""Tests for scale_image tool."""

import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np


TOOLS_DIR = os.path.join(os.path.dirname(__file__), "..", "tools")
SCALE_IMAGE = os.path.join(TOOLS_DIR, "scale_image.py")


def run_tool(args):
    """Run a tool script and return the result."""
    return subprocess.run(
        [sys.executable] + args,
        capture_output=True, text=True,
    )


def make_image(path, width, height, color):
    """Create a solid-color PNG image."""
    img = np.full((height, width, 3), color, dtype=np.uint8)
    cv2.imwrite(path, img)


def test_scale_down():
    """Scale an image down by 0.5."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "src.png")
        dst = os.path.join(tmpdir, "dst.png")
        make_image(src, 200, 100, (255, 0, 0))

        result = run_tool([SCALE_IMAGE, "0.5", "--input", src, "--output", dst])
        assert result.returncode == 0, f"scale_image failed: {result.stderr}"

        img = cv2.imread(dst)
        assert img.shape[1] == 100 and img.shape[0] == 50, f"Unexpected dimensions: {img.shape}"


def test_scale_up():
    """Scale an image up by 3x."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "src.png")
        dst = os.path.join(tmpdir, "dst.png")
        make_image(src, 32, 24, (0, 255, 0))

        result = run_tool([SCALE_IMAGE, "3.0", "--input", src, "--output", dst])
        assert result.returncode == 0, f"scale_image failed: {result.stderr}"

        img = cv2.imread(dst)
        assert img.shape[1] == 96 and img.shape[0] == 72, f"Unexpected dimensions: {img.shape}"


def test_overwrite_in_place():
    """Scale without --output should overwrite the input."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "src.png")
        make_image(src, 200, 100, (255, 0, 0))

        result = run_tool([SCALE_IMAGE, "0.5", "--input", src])
        assert result.returncode == 0, f"scale_image failed: {result.stderr}"

        img = cv2.imread(src)
        assert img.shape[1] == 100 and img.shape[0] == 50, f"Unexpected dimensions: {img.shape}"


def test_fractional_scale():
    """Scale by 1/3 (typical @3x to @1x)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "src.png")
        dst = os.path.join(tmpdir, "dst.png")
        make_image(src, 3072, 2412, (128, 128, 128))

        result = run_tool([SCALE_IMAGE, "0.333333", "--input", src, "--output", dst])
        assert result.returncode == 0, f"scale_image failed: {result.stderr}"

        img = cv2.imread(dst)
        assert img.shape[1] == 1024 and img.shape[0] == 804, f"Unexpected dimensions: {img.shape}"


def test_rgba_preserved():
    """Scale an RGBA image and verify alpha channel is preserved."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "src.png")
        dst = os.path.join(tmpdir, "dst.png")
        img = np.full((100, 200, 4), (255, 0, 0, 128), dtype=np.uint8)
        cv2.imwrite(src, img)

        result = run_tool([SCALE_IMAGE, "0.5", "--input", src, "--output", dst])
        assert result.returncode == 0, f"scale_image failed: {result.stderr}"

        out = cv2.imread(dst, cv2.IMREAD_UNCHANGED)
        assert out.shape == (50, 100, 4), f"Unexpected shape: {out.shape}"


def test_invalid_factor():
    """Zero or negative factor should fail."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "src.png")
        make_image(src, 100, 100, (0, 0, 0))

        result = run_tool([SCALE_IMAGE, "0", "--input", src])
        assert result.returncode != 0, "Expected failure for zero factor"

        result = run_tool([SCALE_IMAGE, "-1.5", "--input", src])
        assert result.returncode != 0, "Expected failure for negative factor"


def test_batch_all():
    """Scale all PNGs in a directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = os.path.join(tmpdir, "imgs")
        os.makedirs(img_dir)
        make_image(os.path.join(img_dir, "a.png"), 200, 100, (255, 0, 0))
        make_image(os.path.join(img_dir, "b.png"), 200, 100, (0, 255, 0))

        result = run_tool([SCALE_IMAGE, "0.5", "--input-dir", img_dir])
        assert result.returncode == 0, f"scale_image failed: {result.stderr}"

        for name in ("a.png", "b.png"):
            img = cv2.imread(os.path.join(img_dir, name))
            assert img.shape[1] == 100 and img.shape[0] == 50, f"{name} unexpected: {img.shape}"


def test_batch_with_prefix():
    """Scale only images matching a prefix."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = os.path.join(tmpdir, "imgs")
        os.makedirs(img_dir)
        make_image(os.path.join(img_dir, "forest_01.png"), 200, 100, (255, 0, 0))
        make_image(os.path.join(img_dir, "forest_02.png"), 200, 100, (255, 0, 0))
        make_image(os.path.join(img_dir, "other_01.png"), 200, 100, (0, 255, 0))

        result = run_tool([SCALE_IMAGE, "0.5", "--input-dir", img_dir, "--prefix", "forest"])
        assert result.returncode == 0, f"scale_image failed: {result.stderr}"

        # Forest files should be scaled.
        for name in ("forest_01.png", "forest_02.png"):
            img = cv2.imread(os.path.join(img_dir, name))
            assert img.shape[1] == 100, f"{name} should be scaled: {img.shape}"

        # Other file should be untouched.
        img = cv2.imread(os.path.join(img_dir, "other_01.png"))
        assert img.shape[1] == 200, f"other_01 should be untouched: {img.shape}"


if __name__ == "__main__":
    tests = [
        test_scale_down,
        test_scale_up,
        test_overwrite_in_place,
        test_fractional_scale,
        test_rgba_preserved,
        test_invalid_factor,
        test_batch_all,
        test_batch_with_prefix,
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

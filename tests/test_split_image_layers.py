#!/usr/bin/env python3
"""Tests for split_image_layers tool."""

import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np


TOOLS_DIR = os.path.join(os.path.dirname(__file__), "..", "tools")
SPLIT = os.path.join(TOOLS_DIR, "split_image_layers.py")


def run_tool(args):
    """Run a tool script and return the result."""
    return subprocess.run(
        [sys.executable] + args,
        capture_output=True, text=True,
    )


def _make_gradient_image(path, width=200, height=100):
    """Create a test image with a vertical brightness gradient (dark top, bright bottom)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        val = int(255 * y / (height - 1))
        img[y, :] = (val, val, val)
    cv2.imwrite(path, img)


def _make_colored_image(path, width=200, height=100):
    """Create a test image with distinct color bands (for better HSV separation)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    third = height // 3
    img[:third, :] = (30, 20, 10)         # Dark, desaturated (back)
    img[third:2*third, :] = (80, 120, 60) # Medium (mid)
    img[2*third:, :] = (200, 220, 240)    # Bright, saturated (front)
    cv2.imwrite(path, img)


def test_basic_split():
    """Split a gradient image into 3 layers, verify count and RGBA."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "test.png")
        out_dir = os.path.join(tmpdir, "out")
        _make_gradient_image(img_path)

        result = run_tool([SPLIT, "--no-model", "--input", img_path, "--layers", "3", "--output-dir", out_dir])
        assert result.returncode == 0, f"Failed: {result.stderr}"

        pngs = sorted(f for f in os.listdir(out_dir) if f.endswith(".png"))
        assert len(pngs) == 3, f"Expected 3 layers, got {len(pngs)}: {pngs}"

        # Verify all are RGBA.
        for p in pngs:
            img = cv2.imread(os.path.join(out_dir, p), cv2.IMREAD_UNCHANGED)
            assert img.shape[2] == 4, f"{p} is not RGBA"


def test_layer_naming_with_suffix():
    """Layers should be named with _layer_NN before suffix."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "scene@3x.png")
        out_dir = os.path.join(tmpdir, "out")
        _make_gradient_image(img_path)

        result = run_tool([SPLIT, "--no-model", "--input", img_path, "--layers", "3",
                           "--output-dir", out_dir, "--suffix", "@3x"])
        assert result.returncode == 0, f"Failed: {result.stderr}"

        expected = ["scene_layer_01@3x.png", "scene_layer_02@3x.png", "scene_layer_03@3x.png"]
        actual = sorted(os.listdir(out_dir))
        assert actual == expected, f"Expected {expected}, got {actual}"


def test_layers_have_transparency():
    """Each layer should have some transparent pixels (not fully opaque)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "test.png")
        out_dir = os.path.join(tmpdir, "out")
        _make_gradient_image(img_path)

        run_tool([SPLIT, "--no-model", "--input", img_path, "--layers", "3", "--output-dir", out_dir])

        for f in os.listdir(out_dir):
            img = cv2.imread(os.path.join(out_dir, f), cv2.IMREAD_UNCHANGED)
            alpha = img[:, :, 3]
            transparent = (alpha == 0).sum()
            assert transparent > 0, f"{f} has no transparent pixels"


def test_feathering():
    """With --feather, alpha values should include partial transparency."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "test.png")
        out_dir = os.path.join(tmpdir, "out")
        _make_gradient_image(img_path)

        run_tool([SPLIT, "--no-model", "--input", img_path, "--layers", "2",
                  "--output-dir", out_dir, "--feather", "5"])

        # At least one layer should have semi-transparent pixels (not just 0 or 255).
        found_semi = False
        for f in os.listdir(out_dir):
            img = cv2.imread(os.path.join(out_dir, f), cv2.IMREAD_UNCHANGED)
            alpha = img[:, :, 3]
            semi = ((alpha > 0) & (alpha < 255)).sum()
            if semi > 0:
                found_semi = True
        assert found_semi, "No semi-transparent pixels found with feathering"


def test_preview_depth_map():
    """--preview should output a depth map image."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "test.png")
        out_dir = os.path.join(tmpdir, "out")
        _make_gradient_image(img_path)

        run_tool([SPLIT, "--no-model", "--input", img_path, "--layers", "2",
                  "--output-dir", out_dir, "--preview"])

        files = os.listdir(out_dir)
        depth_maps = [f for f in files if "depth_map" in f]
        assert len(depth_maps) == 1, f"Expected 1 depth map, got {depth_maps}"


def test_batch_mode():
    """Batch mode with --input-dir and --prefix should process matching files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        out_dir = os.path.join(tmpdir, "out")
        os.makedirs(input_dir)

        # Create 2 matching + 1 non-matching file.
        _make_gradient_image(os.path.join(input_dir, "bg_01.png"))
        _make_gradient_image(os.path.join(input_dir, "bg_02.png"))
        _make_gradient_image(os.path.join(input_dir, "other.png"))

        result = run_tool([SPLIT, "--no-model", "--input-dir", input_dir, "--prefix", "bg_",
                           "--layers", "2", "--output-dir", out_dir])
        assert result.returncode == 0, f"Failed: {result.stderr}"

        # 2 files × 2 layers = 4 output PNGs.
        pngs = [f for f in os.listdir(out_dir) if f.endswith(".png")]
        assert len(pngs) == 4, f"Expected 4 layers, got {len(pngs)}: {pngs}"


def test_output_dir_nuked():
    """Output directory should be wiped clean before each run."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "test.png")
        out_dir = os.path.join(tmpdir, "out")
        os.makedirs(out_dir)
        stale = os.path.join(out_dir, "stale.txt")
        with open(stale, "w") as f:
            f.write("old")

        _make_gradient_image(img_path)
        run_tool([SPLIT, "--no-model", "--input", img_path, "--layers", "2", "--output-dir", out_dir])

        assert not os.path.exists(stale), "Stale file should have been removed"


if __name__ == "__main__":
    tests = [
        test_basic_split,
        test_layer_naming_with_suffix,
        test_layers_have_transparency,
        test_feathering,
        test_preview_depth_map,
        test_batch_mode,
        test_output_dir_nuked,
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

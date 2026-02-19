#!/usr/bin/env python3
"""Split an image into N depth layers using HSV-based depth estimation."""

import argparse
import glob
import os
import shutil
import sys

import cv2
import numpy as np


# Depth score weights for HSV channels.
WEIGHT_VALUE = 0.7       # Brightness (V): brighter = closer
WEIGHT_SATURATION = 0.3  # Saturation (S): more saturated = closer


def compute_depth_map(image):
    """Compute a depth score map from an image using HSV channels.

    Returns a float32 array in [0, 1] where 0 = farthest and 1 = nearest.
    """
    hsv = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2HSV).astype(np.float32)
    v = hsv[:, :, 2] / 255.0  # Value (brightness)
    s = hsv[:, :, 1] / 255.0  # Saturation

    depth = WEIGHT_VALUE * v + WEIGHT_SATURATION * s
    # Normalize to [0, 1].
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 0:
        depth = (depth - d_min) / (d_max - d_min)
    else:
        depth = np.zeros_like(depth)

    return depth


def split_layers(image, n_layers, feather=0):
    """Split an image into N depth layers.

    Returns a list of RGBA images (back-to-front, layer 1 = farthest).
    """
    depth = compute_depth_map(image)
    h, w = image.shape[:2]

    # Ensure we have an alpha channel.
    if image.shape[2] == 4:
        bgra = image.copy()
    else:
        bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    layers = []
    band_size = 1.0 / n_layers

    for i in range(n_layers):
        lo = i * band_size
        hi = (i + 1) * band_size

        # Create mask for this depth band.
        if i == n_layers - 1:
            mask = (depth >= lo) & (depth <= hi)
        else:
            mask = (depth >= lo) & (depth < hi)

        mask_f = mask.astype(np.float32)

        # Apply Gaussian feathering to soften edges.
        if feather > 0:
            ksize = feather * 2 + 1
            mask_f = cv2.GaussianBlur(mask_f, (ksize, ksize), 0)

        # Build RGBA layer.
        layer = bgra.copy()
        layer[:, :, 3] = (mask_f * 255).astype(np.uint8)

        layers.append(layer)

    return layers, depth


def layer_name_suffix(idx, n_layers):
    """Return a human-friendly layer label."""
    if n_layers <= 1:
        return ""
    if idx == 0:
        return "back"
    if idx == n_layers - 1:
        return "front"
    if n_layers == 3 and idx == 1:
        return "mid"
    return f"mid{idx}"


def derive_output_name(input_basename, layer_idx, n_layers, suffix, pad):
    """Derive the output filename for a layer.

    Inserts _layer_NN before the suffix. E.g.:
    forest_dungeon_bg_01@3x.png + suffix=@3x + layer 1 → forest_dungeon_bg_01_layer_01@3x.png
    """
    name, ext = os.path.splitext(input_basename)

    # Strip suffix from name if present (so we can re-insert it after layer number).
    if suffix and name.endswith(suffix):
        name = name[: -len(suffix)]

    layer_num = str(layer_idx + 1).zfill(pad)
    return f"{name}_layer_{layer_num}{suffix}{ext}"


def process_file(input_path, output_dir, n_layers, suffix, feather, preview):
    """Split a single image into layers and save them."""
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Error: could not read '{input_path}'.", file=sys.stderr)
        return False

    layers, depth = split_layers(image, n_layers, feather)
    pad = max(2, len(str(n_layers)))
    basename = os.path.basename(input_path)

    for idx, layer in enumerate(layers):
        out_name = derive_output_name(basename, idx, n_layers, suffix, pad)
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, layer)
        label = layer_name_suffix(idx, n_layers)
        print(f"  {out_path} ({label})")

    if preview:
        name, _ = os.path.splitext(basename)
        if suffix and name.endswith(suffix):
            name = name[: -len(suffix)]
        preview_path = os.path.join(output_dir, f"{name}_depth_map{suffix}.png")
        depth_vis = (depth * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        cv2.imwrite(preview_path, depth_colored)
        print(f"  {preview_path} (depth map)")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Split an image into N depth layers using HSV-based depth estimation."
    )
    # Input modes (same pattern as merge_image / scale_image).
    parser.add_argument("--input", dest="input_file", help="Single input image file")
    parser.add_argument("--input-dir", help="Directory of input images (batch mode)")
    parser.add_argument("--prefix", help="Filter batch files by prefix")

    parser.add_argument("--layers", type=int, default=3, help="Number of layers (default: 3)")
    parser.add_argument(
        "--output-dir", default="./tmp/split_image_layers",
        help="Output directory (default: ./tmp/split_image_layers)",
    )
    parser.add_argument("--suffix", default="", help="Filename suffix to preserve (e.g., @3x)")
    parser.add_argument(
        "--feather", type=int, default=0,
        help="Edge feather radius in pixels (0 = hard edges, default: 0)",
    )
    parser.add_argument("--preview", action="store_true", help="Output a depth map visualization")

    args = parser.parse_args()

    if args.layers < 1:
        parser.error("Number of layers must be at least 1")
    if args.feather < 0:
        parser.error("Feather radius must be non-negative")

    # Collect input files.
    files = []
    if args.input_file:
        if not os.path.isfile(args.input_file):
            parser.error(f"Input file not found: {args.input_file}")
        files = [args.input_file]
    elif args.input_dir:
        if not os.path.isdir(args.input_dir):
            parser.error(f"Input directory not found: {args.input_dir}")
        pattern = os.path.join(args.input_dir, "*.png")
        files = sorted(glob.glob(pattern))
        if args.prefix:
            files = [f for f in files if os.path.basename(f).startswith(args.prefix)]
        if not files:
            parser.error(f"No matching PNG files found in {args.input_dir}")
    else:
        parser.error("Provide --input or --input-dir")

    # Clean and recreate output directory.
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    count = 0
    for f in files:
        print(f"[{count + 1}/{len(files)}] {os.path.basename(f)}:")
        if process_file(f, args.output_dir, args.layers, args.suffix, args.feather, args.preview):
            count += 1

    print(f"\nDone — {count} image(s) split into {args.layers} layers each → {args.output_dir}/")


if __name__ == "__main__":
    main()

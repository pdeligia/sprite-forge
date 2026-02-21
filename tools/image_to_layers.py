#!/usr/bin/env python3
"""Split an image into N depth layers using Depth Anything V2 monocular depth estimation."""

import argparse
import glob
import os
import shutil
import sys

import cv2
import numpy as np

from tools.lib.console import console, enable_hf_offline
from tools.lib.depth_utils import compute_depth_map, load_depth_pipeline


def _remove_thin_features(mask, min_thickness=2):
    """Remove features thinner than min_thickness pixels from a binary mask.

    Uses morphological opening (erode + dilate): thin features vanish under
    erosion and don't come back after dilation. Thick regions erode then
    dilate back to approximately their original shape.
    """
    mask_u8 = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (min_thickness * 2 + 1, min_thickness * 2 + 1)
    )
    opened = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    return opened.astype(bool)


def split_layers(image, depth, n_layers, feather=0, clean_edges="off"):
    """Split an image into N depth layers using quantile-based thresholds.

    Uses quantiles so each layer gets roughly equal pixel coverage.
    Returns a list of RGBA images (back-to-front, layer 1 = farthest).

    clean_edges: "off" or int (erosion px).
    """
    h, w = image.shape[:2]

    # Ensure we have an alpha channel.
    if image.shape[2] == 4:
        bgra = image.copy()
    else:
        bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # Compute quantile thresholds for equal pixel distribution.
    flat = depth.flatten()
    thresholds = [0.0]
    for i in range(1, n_layers):
        thresholds.append(np.percentile(flat, 100.0 * i / n_layers))
    thresholds.append(1.0)

    # Build raw masks for each layer.
    masks = []
    for i in range(n_layers):
        lo = thresholds[i]
        hi = thresholds[i + 1]
        if i == n_layers - 1:
            mask = (depth >= lo) & (depth <= hi)
        else:
            mask = (depth >= lo) & (depth < hi)
        masks.append(mask)

    # Clean edges: remove thin outline artifacts at depth boundaries.
    # Use morphological opening to remove thin features, then reassign
    # orphaned pixels to the nearest real layer.
    if clean_edges == "off":
        thickness = 0
    else:
        thickness = int(clean_edges)

    if thickness > 0:
        from scipy.ndimage import distance_transform_edt
        orphaned = np.zeros((h, w), dtype=bool)
        for i in range(n_layers):
            before = masks[i].copy()
            masks[i] = _remove_thin_features(masks[i], thickness)
            orphaned |= (before & ~masks[i])
        # Reassign orphaned pixels to nearest layer (prefer front layers).
        if orphaned.any():
            best_dist = np.full((h, w), np.inf)
            best_layer = np.full((h, w), -1, dtype=int)
            for i in range(n_layers):
                if not masks[i].any():
                    continue
                dist, _ = distance_transform_edt(~masks[i], return_indices=True)
                bias = 1.0 - i * 0.01
                closer = (dist * bias) < best_dist
                best_dist[closer] = dist[closer] * bias
                best_layer[closer] = i
            for i in range(n_layers):
                masks[i] = masks[i] | (orphaned & (best_layer == i))

    layers = []
    for i in range(n_layers):
        mask_f = masks[i].astype(np.float32)

        # Apply Gaussian feathering to soften edges.
        if feather > 0:
            ksize = feather * 2 + 1
            mask_f = cv2.GaussianBlur(mask_f, (ksize, ksize), 0)

        # Build RGBA layer.
        layer = bgra.copy()
        layer[:, :, 3] = (mask_f * bgra[:, :, 3].astype(np.float32) / 255.0 * 255).astype(np.uint8)

        layers.append(layer)

    return layers


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


def process_file(pipe, input_path, output_dir, n_layers, suffix, feather, clean_edges, depth_map):
    """Split a single image into layers and save them."""
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Error: could not read '{input_path}'.", file=sys.stderr)
        return False

    depth = compute_depth_map(pipe, image)
    layers = split_layers(image, depth, n_layers, feather, clean_edges)

    pad = max(2, len(str(n_layers)))
    basename = os.path.basename(input_path)

    for idx, layer in enumerate(layers):
        out_name = derive_output_name(basename, idx, n_layers, suffix, pad)
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, layer)
        label = layer_name_suffix(idx, n_layers)
        console.print(f"  {out_path} ([dim]{label}[/dim])")

    if depth_map:
        name, _ = os.path.splitext(basename)
        if suffix and name.endswith(suffix):
            name = name[: -len(suffix)]
        preview_path = os.path.join(output_dir, f"{name}_depth_map{suffix}.png")
        depth_vis = (depth * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        cv2.imwrite(preview_path, depth_colored)
        console.print(f"  {preview_path} ([dim]depth map[/dim])")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Split an image into N depth layers using Depth Anything V2 depth estimation."
    )
    # Input modes (same pattern as merge_image / scale_image).
    parser.add_argument("--input", dest="input_file", help="Single input image file")
    parser.add_argument("--input-dir", help="Directory of input images (batch mode)")
    parser.add_argument("--prefix", help="Filter batch files by prefix")

    parser.add_argument("--layers", type=int, default=3, help="Number of layers (default: 3)")
    parser.add_argument(
        "--output-dir", default="./tmp/image_to_layers",
        help="Output directory (default: ./tmp/image_to_layers)",
    )
    parser.add_argument("--suffix", default="", help="Filename suffix to preserve (e.g., @3x)")
    parser.add_argument(
        "--feather", type=int, default=0,
        help="Edge feather radius in pixels (0 = hard edges, default: 0)",
    )
    parser.add_argument(
        "--clean-edges", default="off",
        help="Remove outline artifacts: an erosion radius in pixels, or 'off' (default). Use analyze-image --depth to find the recommended value.",
    )
    parser.add_argument("--depth-map", action="store_true", help="Output a depth map visualization")
    parser.add_argument(
        "--model", choices=["small", "base", "large"], default="small",
        help="Depth Anything V2 model size (default: small). Larger = better quality, slower.",
    )
    parser.add_argument(
        "--no-model", action="store_true",
        help="Skip depth model, use vertical gradient fallback (for testing/CI).",
    )
    parser.add_argument(
        "--fetch-latest-model", action="store_true",
        help="Force re-download HuggingFace models to get the latest version.",
    )
    args = parser.parse_args()

    if not args.fetch_latest_model:
        enable_hf_offline()

    if args.layers < 1:
        parser.error("Number of layers must be at least 1")
    if args.feather < 0:
        parser.error("Feather radius must be non-negative")
    if args.clean_edges != "off":
        try:
            val = int(args.clean_edges)
            if val < 1:
                parser.error("--clean-edges must be a positive integer")
        except ValueError:
            parser.error("--clean-edges must be 'off' or a positive integer")

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

    # Load the depth model once, reuse for all files.
    pipe = None if args.no_model else load_depth_pipeline(args.model)

    count = 0
    for f in files:
        console.print(f"[{count + 1}/{len(files)}] [bold]{os.path.basename(f)}[/bold]:")
        if process_file(pipe, f, args.output_dir, args.layers, args.suffix, args.feather, args.clean_edges, args.depth_map):
            count += 1

    console.print(f"\nDone — {count} image(s) split into {args.layers} layers each → {args.output_dir}/")


if __name__ == "__main__":
    main()

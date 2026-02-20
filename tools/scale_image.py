#!/usr/bin/env python3
"""Scale an image by a given factor."""

import argparse
import glob
import os
import sys

import cv2
import numpy as np


def scale_nearest(img, new_w, new_h, factor):
    """Scale using nearest-neighbor (crisp pixel art)."""
    interp = cv2.INTER_NEAREST if factor > 1 else cv2.INTER_AREA
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def scale_smooth(img, new_w, new_h, factor):
    """Scale using Lanczos (smooth, general-purpose)."""
    interp = cv2.INTER_LANCZOS4 if factor > 1 else cv2.INTER_AREA
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def scale_ai(img, new_w, new_h, factor):
    """Scale using Real-ESRGAN AI super-resolution (py-real-esrgan package)."""
    import torch
    from PIL import Image
    from RealESRGAN import RealESRGAN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Choose model scale: 2x or 4x.
    net_scale = 2 if factor <= 2 else 4

    model = RealESRGAN(device, scale=net_scale)
    model.load_weights(f"weights/RealESRGAN_x{net_scale}.pth", download=True)

    # Handle alpha channel separately — ESRGAN only works on RGB.
    has_alpha = len(img.shape) == 3 and img.shape[2] == 4
    if has_alpha:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
    else:
        bgr = img

    # Convert BGR → RGB → PIL for the model.
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    result = model.predict(pil_img)

    # Convert back to BGR numpy array.
    output = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

    if has_alpha:
        alpha_up = cv2.resize(alpha, (output.shape[1], output.shape[0]),
                              interpolation=cv2.INTER_LANCZOS4)
        output = np.dstack([output, alpha_up])

    # Resize to exact target dimensions if model scale != requested factor.
    if output.shape[1] != new_w or output.shape[0] != new_h:
        output = cv2.resize(output, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    return output


SCALE_MODES = {
    "pixel": scale_nearest,
    "smooth": scale_smooth,
    "ai": scale_ai,
}


def scale_image(input_path, output_path, factor, mode="pixel"):
    """Scale an image by the given factor and save to output path."""
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: could not read image '{input_path}'.", file=sys.stderr)
        sys.exit(1)

    h, w = img.shape[:2]
    new_w = round(w * factor)
    new_h = round(h * factor)

    if new_w <= 0 or new_h <= 0:
        print(f"Error: scaled dimensions ({new_w}x{new_h}) are invalid.", file=sys.stderr)
        sys.exit(1)

    scale_fn = SCALE_MODES[mode]
    scaled = scale_fn(img, new_w, new_h, factor)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, scaled)
    print(f"{input_path} ({w}x{h}) -> {output_path} ({new_w}x{new_h}) [x{factor}, {mode}]")


def collect_files(input_file, input_dir, prefix):
    """Collect the list of PNG files to process."""
    if input_file:
        if not os.path.isfile(input_file):
            print(f"Error: input file not found: {input_file}", file=sys.stderr)
            sys.exit(1)
        return [input_file]

    if not os.path.isdir(input_dir):
        print(f"Error: input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    pattern = f"{prefix}*.png" if prefix else "*.png"
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        print(f"Error: no matching PNGs found in {input_dir} (prefix: '{prefix or '*'}').", file=sys.stderr)
        sys.exit(1)
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Scale an image by a given factor."
    )
    parser.add_argument("factor", type=float, help="Scale factor (e.g., 0.333 for 1/3, 2.0 for 2x)")
    parser.add_argument("--input", dest="input_file", help="Single image to scale")
    parser.add_argument("--input-dir", help="Directory of images to scale")
    parser.add_argument("--prefix", default="", help="Filter images by filename prefix (only with --input-dir)")
    parser.add_argument("--output", help="Output path (single file mode only; default: overwrites input)")
    parser.add_argument(
        "--mode", choices=["pixel", "smooth", "ai"], default="pixel",
        help="Scaling method: pixel (nearest-neighbor, default), smooth (Lanczos), ai (Real-ESRGAN)",
    )

    args = parser.parse_args()

    if args.factor <= 0:
        parser.error("Scale factor must be positive")
    if not args.input_file and not args.input_dir:
        parser.error("Must provide either --input or --input-dir")
    if args.input_file and args.input_dir:
        parser.error("Cannot use both --input and --input-dir")
    if args.output and args.input_dir:
        parser.error("--output cannot be used with --input-dir (files are overwritten in place)")

    files = collect_files(args.input_file, args.input_dir, args.prefix)

    for i, path in enumerate(files, start=1):
        output = args.output or path
        scale_image(path, output, args.factor, mode=args.mode)

    if len(files) > 1:
        print(f"\nDone — {len(files)} image(s) scaled.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Composite and merge images together."""

import argparse
import glob
import os
import sys

import cv2
import numpy as np


def merge(bg_path, fg_path, offset):
    """Merge a foreground image onto a background, saving over the foreground file."""
    bg = cv2.imread(bg_path, cv2.IMREAD_UNCHANGED)
    fg = cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)
    if bg is None:
        print(f"Error: could not read background '{bg_path}'.", file=sys.stderr)
        sys.exit(1)
    if fg is None:
        print(f"Error: could not read foreground '{fg_path}'.", file=sys.stderr)
        sys.exit(1)

    ox, oy = offset
    fh, fw = fg.shape[:2]
    bh, bw = bg.shape[:2]

    if ox + fw > bw or oy + fh > bh:
        print(
            f"Error: foreground ({fw}x{fh}) at offset ({ox},{oy}) exceeds "
            f"background ({bw}x{bh}).",
            file=sys.stderr,
        )
        sys.exit(1)

    # Work on a copy of the background.
    result = bg.copy()

    # Ensure both images have the same number of channels for the ROI.
    bg_channels = result.shape[2] if len(result.shape) == 3 else 1
    fg_channels = fg.shape[2] if len(fg.shape) == 3 else 1

    if fg_channels == 4:
        # Alpha compositing.
        alpha = fg[:, :, 3:4].astype(np.float32) / 255.0
        fg_rgb = fg[:, :, :3].astype(np.float32)
        roi = result[oy:oy + fh, ox:ox + fw]
        if bg_channels == 4:
            roi_rgb = roi[:, :, :3].astype(np.float32)
            blended = (fg_rgb * alpha + roi_rgb * (1 - alpha)).astype(np.uint8)
            result[oy:oy + fh, ox:ox + fw, :3] = blended
        else:
            roi_rgb = roi.astype(np.float32) if bg_channels == 3 else np.stack([roi.astype(np.float32)] * 3, axis=-1)
            blended = (fg_rgb * alpha + roi_rgb * (1 - alpha)).astype(np.uint8)
            result[oy:oy + fh, ox:ox + fw] = blended
    else:
        # No alpha — direct paste.
        if fg_channels == 3 and bg_channels == 4:
            result[oy:oy + fh, ox:ox + fw, :3] = fg
        else:
            result[oy:oy + fh, ox:ox + fw] = fg

    cv2.imwrite(fg_path, result)


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
        description="Merge foreground image(s) onto a background image."
    )
    parser.add_argument("background", help="Path to the background image")
    parser.add_argument("--input", dest="input_file", help="Single foreground image to merge")
    parser.add_argument("--input-dir", help="Directory of foreground images")
    parser.add_argument("--prefix", default="", help="Filter images by filename prefix (only with --input-dir)")
    parser.add_argument(
        "--offset", nargs=2, type=int, default=[0, 0], metavar=("X", "Y"),
        help="Top-left position to place foreground on background (default: 0 0)",
    )

    args = parser.parse_args()

    if not args.input_file and not args.input_dir:
        parser.error("Must provide either --input or --input-dir")
    if args.input_file and args.input_dir:
        parser.error("Cannot use both --input and --input-dir")
    if not os.path.isfile(args.background):
        parser.error(f"Background file not found: {args.background}")

    files = collect_files(args.input_file, args.input_dir, args.prefix)

    for i, fg_path in enumerate(files, start=1):
        merge(args.background, fg_path, args.offset)
        print(f"[{i}/{len(files)}] {fg_path}")

    print(f"\nDone — {len(files)} image(s) merged.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate synthetic MP4 videos for testing and prototyping."""

import argparse
import os
import sys

import cv2
import numpy as np

from tools.lib.console import console


def generate_video(output_path, width, height, duration, fps, color):
    """Generate a solid-color MP4 video."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"Error: could not create video '{output_path}'.", file=sys.stderr)
        sys.exit(1)

    total_frames = int(duration * fps)
    frame = np.full((height, width, 3), color, dtype=np.uint8)

    for _ in range(total_frames):
        writer.write(frame)

    writer.release()
    console.print(f"Generated {output_path} ([cyan]{width}x{height}[/cyan], {duration}s, {fps}fps, {total_frames} frames)")


def parse_color(value):
    """Parse a color string like '0,0,0' or 'black' into a BGR tuple."""
    presets = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
    }
    if value.lower() in presets:
        return presets[value.lower()]
    try:
        r, g, b = (int(c.strip()) for c in value.split(","))
        return (b, g, r)  # BGR for opencv
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError(f"Invalid color: '{value}'. Use a preset (black, white, red, green, blue) or R,G,B values.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic MP4 videos."
    )
    parser.add_argument("output", nargs="?", default="./tmp/gen_video/test.mp4", help="Output MP4 file path (default: ./tmp/gen_video/test.mp4)")
    parser.add_argument("--width", type=int, default=320, help="Video width in pixels (default: 320)")
    parser.add_argument("--height", type=int, default=240, help="Video height in pixels (default: 240)")
    parser.add_argument("--duration", type=float, default=1.0, help="Duration in seconds (default: 1.0)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    parser.add_argument("--color", type=parse_color, default="black", help="Fill color: preset name or R,G,B (default: black)")

    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0:
        parser.error("Width and height must be positive")
    if args.duration <= 0:
        parser.error("Duration must be positive")
    if args.fps <= 0:
        parser.error("FPS must be positive")

    generate_video(args.output, args.width, args.height, args.duration, args.fps, args.color)


if __name__ == "__main__":
    main()

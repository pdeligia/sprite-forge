---
name: mp4-to-frames
description: Extract evenly-spaced PNG frames from an MP4 video with optional cropping, scaling, and time range selection. Use this skill when the user wants to generate sprite frames, background animation frames, or image sequences from video files.
---

# mp4-to-frames

A Python tool that extracts N evenly-spaced frames from an MP4 video as PNG files.

## Location
`tools/mp4_to_frames.py` in the sprite-forge repository.

## How to Run
```bash
uv run python tools/mp4_to_frames.py <input.mp4> <N> [options]
```

## Arguments
| Argument | Required | Description |
|----------|----------|-------------|
| `input` | Yes | Path to the input MP4 video file |
| `n` | Yes | Number of frames to extract |

## Options
| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir DIR` | `./tmp` | Output directory (nuked and recreated each run) |
| `--prefix PREFIX` | `frame` | Filename prefix for output PNGs |
| `--region X1 Y1 X2 Y2` | none | Crop region: top-left (X1, Y1) to bottom-right (X2, Y2) in pixels |
| `--width W` | none | Scale output to width W, preserving aspect ratio. Uses nearest-neighbor interpolation (good for pixel art) |
| `--start S` | 0 | Start time in seconds |
| `--end E` | end of video | End time in seconds |

## Output
- Files are named `{prefix}_{NN}.png` with zero-padded numbering (e.g., `forest_01.png`, `forest_02.png`)
- Output directory is wiped clean before each run

## Important Notes
- When using `--region`, coordinates must not exceed the video dimensions. Run without `--region` first to see if extraction works, then add cropping.
- When using `--width` with `--region`, the output height is computed from the crop region's aspect ratio. To hit an exact target resolution (e.g., 1024×735), calculate the crop region so its aspect ratio matches: `crop_height = crop_width × (target_height / target_width)`.
- The tool uses opencv with nearest-neighbor interpolation for scaling, which preserves sharp pixel edges (ideal for pixel art and game assets).
- Requires `uv sync` to have been run first to install dependencies (opencv-python).

## Examples

### Basic extraction
```bash
uv run python tools/mp4_to_frames.py video.mp4 10 --prefix forest
```

### Crop and scale for game assets
```bash
uv run python tools/mp4_to_frames.py video.mp4 10 --prefix forest \
  --region 140 1 1140 719 --width 1024
```

### Extract from a specific time range
```bash
uv run python tools/mp4_to_frames.py video.mp4 10 --start 4 --end 8 --prefix forest
```

### Custom output directory
```bash
uv run python tools/mp4_to_frames.py video.mp4 5 --output-dir ./output --prefix bg
```

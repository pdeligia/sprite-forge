---
name: extract-mp4-frames
description: Extract PNG frames from an MP4 video with optional cropping, scaling, time range, and loop detection. Use this skill when the user wants to generate sprite frames, seamless looping animations, or image sequences from video files.
---

# extract-mp4-frames

A Python tool that extracts N frames from an MP4 video as PNG files, with optional seamless loop detection.

## How to Run
```bash
uv run extract-mp4-frames <input.mp4> <N> [options]
```

## Arguments
| Argument | Required | Description |
|----------|----------|-------------|
| `input` | Yes | Path to the input MP4 video file |
| `n` | Yes | Number of frames to extract |

## Options
| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir DIR` | `./tmp/extract_mp4_frames` | Output directory (nuked and recreated each run) |
| `--prefix PREFIX` | `frame` | Filename prefix for output PNGs |
| `--suffix SUFFIX` | (empty) | Filename suffix before .png (e.g., `@3x`) |
| `--region X1 Y1 X2 Y2` | none | Crop region: top-left (X1, Y1) to bottom-right (X2, Y2) in pixels |
| `--width W` | none | Scale output to width W, preserving aspect ratio. Uses nearest-neighbor interpolation (good for pixel art) |
| `--start S` | 0 | Start time in seconds (supports decimals for sub-second precision) |
| `--end E` | end of video | End time in seconds (supports decimals for sub-second precision) |
| `--loop` | off | Find the best looping segment using SSIM frame comparison, then extract N frames from it |

## Output
- Files are named `{prefix}_{NN}{suffix}.png` with zero-padded numbering (e.g., `forest_dungeon_bg_01@3x.png`)
- Output directory is wiped clean before each run

## Important Notes
- When using `--region`, coordinates must not exceed the video dimensions. Run without `--region` first to see if extraction works, then add cropping.
- When using `--width` with `--region`, the output height is computed from the crop region's aspect ratio. To hit an exact target resolution (e.g., 1024×735), calculate the crop region so its aspect ratio matches: `crop_height = crop_width × (target_height / target_width)`.
- The tool uses opencv with nearest-neighbor interpolation for scaling, which preserves sharp pixel edges (ideal for pixel art and game assets).
- Requires `uv sync` to have been run first to install dependencies (opencv-python, scikit-image).

## Loop Mode
When `--loop` is specified:
- The tool scans frames within the `--start`/`--end` range (or full video) using SSIM (Structural Similarity Index)
- It finds the best pair of frames where a later frame closely matches an earlier one (seamless loop transition)
- N frames are extracted from within that loop segment, excluding the end frame (since it matches the start)
- A loop quality score (0.0–1.0) is reported. Scores above 0.85 typically look seamless
- Use `--start`/`--end` to narrow the search range if the video is long

## Examples

### Basic extraction
```bash
uv run extract-mp4-frames video.mp4 10 --prefix forest
```

### Crop and scale for game assets
```bash
uv run extract-mp4-frames video.mp4 10 --prefix forest \
  --region 140 1 1140 719 --width 1024
```

### Extract from a specific time range
```bash
uv run extract-mp4-frames video.mp4 10 --start 4 --end 8 --prefix forest
```

### Custom output directory
```bash
uv run extract-mp4-frames video.mp4 5 --output-dir ./output --prefix bg
```

### Find best loop and extract 10 frames
```bash
uv run extract-mp4-frames video.mp4 10 --loop --prefix forest
```

### Find loop within a time range
```bash
uv run extract-mp4-frames video.mp4 10 --loop --start 2 --end 8 --prefix forest
```

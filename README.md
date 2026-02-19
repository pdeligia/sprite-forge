# sprite-forge

An AI-powered toolkit for generating game assets from video and image sources.

## Requirements
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for dependency management

## Setup
```bash
uv sync
```

## Tools

### `mp4_to_frames.py`
Extract N evenly-spaced frames from an MP4 video as PNGs, with optional cropping and scaling.

```bash
uv run python tools/mp4_to_frames.py <input.mp4> <N> [options]
```

**Options:**
| Flag | Description |
|------|-------------|
| `--output-dir DIR` | Output directory (default: `./tmp`) |
| `--prefix PREFIX` | Filename prefix (default: `frame`) |
| `--region X1 Y1 X2 Y2` | Crop region: top-left to bottom-right in pixels |
| `--width W` | Scale output to width W, preserving aspect ratio (nearest-neighbor) |
| `--start S` | Start time in seconds (default: beginning) |
| `--end E` | End time in seconds (default: end of video) |

**Examples:**
```bash
# Extract 10 frames from a video
uv run python tools/mp4_to_frames.py video.mp4 10 --prefix forest

# Crop a region and scale up for @3x assets
uv run python tools/mp4_to_frames.py video.mp4 10 --region 140 0 1164 720 --width 3072

# Extract frames from a specific time range
uv run python tools/mp4_to_frames.py video.mp4 10 --start 2 --end 6
```

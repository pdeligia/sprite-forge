---
name: analyze-mp4
description: Analyze an MP4 video and recommend the optimal number of frames for animation extraction. Use this skill before mp4-to-frames to determine how many frames to extract for smooth, efficient animations.
---

# analyze-mp4

A Python tool that analyzes an MP4 video's motion characteristics and recommends the optimal number of frames (N) for animation extraction.

## How to Run
```bash
uv run analyze-mp4 <input.mp4> [options]
```

## Arguments
| Argument | Required | Description |
|----------|----------|-------------|
| `input`  | Yes      | Path to the input MP4 video file |

## Options
| Flag | Default | Description |
|------|---------|-------------|
| `--start S` | 0 | Start time in seconds (supports decimals for sub-second precision) |
| `--end E` | end of video | End time in seconds (supports decimals for sub-second precision) |
| `--loop` | off | Find the best looping segment first, then analyze it |
| `--region X1 Y1 X2 Y2` | none | Crop region for accurate size estimates |
| `--width W` | none | Output width for accurate size estimates (preserves aspect ratio) |

## Output
The tool prints:
- Video metadata (fps, duration, native frame count)
- Loop info (if `--loop` is used): best loop timestamps and SSIM score
- Motion level (low / medium / high) based on average inter-frame SSIM
- **Recommended N**: the optimal frame count (diminishing-returns sweet spot)
- A tier table showing N values with effective fps, Δ-SSIM, estimated file size, and quality ratings
- **Playback recommendation** (when `--loop` is used): `timePerFrame` value and whether to use forward or ping-pong strategy based on loop score

### Tier table columns
| Column | Description |
|--------|-------------|
| N | Number of frames |
| fps | Effective animation playback fps for the segment duration |
| Δ-SSIM | Average visual change between consecutive extracted frames (higher = more motion per step) |
| Size est. | Rough total PNG output size based on output dimensions |
| Rating | Quality label — the optimal N is marked `✅ Recommended` |

### Playback strategy (with `--loop`)
| Loop Score | Strategy | Description |
|------------|----------|-------------|
| ≥ 0.98 | **Forward** | Last frame matches first — seamless forward loop (1→2→…→N→1→…) |
| < 0.98 | **Ping-pong** | Visible jump if played forward — reverse at the end instead (1→2→…→N→…→2→1→…) |

## Important Notes
- This tool does **not** extract or save any image files — it is a dry-run analysis only.
- The output is **fully deterministic** — same video + same parameters always produces the same result.
- Use `--region` and `--width` to get accurate size estimates for your target output dimensions.
- Motion level is derived from average inter-frame SSIM: >0.98 = low, 0.95–0.98 = medium, <0.95 = high.
- Tier N values that exceed the segment's native frame count are automatically omitted.
- Requires `uv sync` to have been run first to install dependencies (opencv-python, scikit-image).

## Workflow
Use this tool before `mp4-to-frames` to decide how many frames to extract:
1. `uv run analyze-mp4 video.mp4 --loop` → get recommended N
2. `uv run mp4-to-frames video.mp4 <N> --loop --prefix forest` → extract frames

## Examples

### Analyze full video
```bash
uv run analyze-mp4 video.mp4
```

### Analyze a specific time range
```bash
uv run analyze-mp4 video.mp4 --start 2.5 --end 6.0
```

### Analyze with loop detection
```bash
uv run analyze-mp4 video.mp4 --loop
```

### Analyze loop within a time range, with output dimensions
```bash
uv run analyze-mp4 video.mp4 --loop \
  --start 2 --end 8 --region 140 1 1140 719 --width 3072
```

---
name: loop-video
description: Create a seamlessly looping MP4 from a video or a time range of it. Use this skill when the user wants to make a video loop smoothly, even if the original doesn't naturally loop.
---

# loop-video

A Python tool that creates a seamlessly looping MP4 by analyzing frame similarity and intelligently adding bridge frames to close the loop. Supports automatic mode selection, crossfade blending, and reverse-bridge strategies.

## How to Run
```bash
uv run loop-video --input <video.mp4> [options]
```

## Options
| Flag | Default | Description |
|------|---------|-------------|
| `--input FILE` | required | Input MP4 file |
| `--output FILE` | `./tmp/loop_video/output.mp4` | Output MP4 file path |
| `--start N` | `0.0` | Start time in seconds |
| `--end N` | (end of video) | End time in seconds |
| `--budget N` | `15` | Max extra frames the algorithm can add to close the loop |
| `--mode MODE` | `auto` | Loop strategy: `auto`, `crossfade`, or `reverse` |
| `--no-shift` | off | Disable shifting the start to the most stable frame |
| `--repeat N` | `1` | Repeat the loop N times in the output for easy preview |

## Modes

### auto (default)
Analyzes the gap between first and last frame, then picks the best strategy:
- If frames already match (SSIM > 0.95): no extra frames needed
- If a good reverse match exists (SSIM > 0.85): uses reverse bridge
- Otherwise: uses crossfade

### crossfade
Generates bridge frames by alpha-blending the last frame into the first frame. Works well for ambient scenes (water, fog, fire).

### reverse
Walks backwards from the last frame to find the closest match to the first frame, then cross-fades the final steps. Works well for organic motion (swaying trees, flickering flames).

## Examples

### Auto-loop a video
```bash
uv run loop-video --input video.mp4 --budget 15
```

### Loop a specific time range
```bash
uv run loop-video --input video.mp4 --start 1.5 --end 6.0 --budget 20
```

### Force crossfade mode
```bash
uv run loop-video --input video.mp4 --mode crossfade --budget 10
```

## Related Tools
- [crop-video](../crop-video/SKILL.md) — Crop and resize before looping
- [smooth-video](../smooth-video/SKILL.md) — Fix jittery frames before looping
- [frames-to-video](../frames-to-video/SKILL.md) — Compose video from PNG frames
- [video-to-frames](../video-to-frames/SKILL.md) — Extract frames from video

---
name: smooth-video
description: Detect and fix jittery frames in a video by blending neighbors. Use this skill when the user wants to stabilize choppy or inconsistent frames in a video.
---

# smooth-video

A Python tool that detects jittery frames in a video (frames with abnormally low SSIM to their neighbors) and replaces them with smooth blends of surrounding good frames. Same frame count in/out — no frames are removed or added.

## How to Run
```bash
uv run smooth-video --input <video> [options]
```

## Options
| Flag | Default | Description |
|------|---------|-------------|
| `--input FILE` | required | Input video file |
| `--output FILE` | `./tmp/smooth_video/output.mp4` | Output video file path |
| `--sigma N` | `2.0` | Sensitivity: frames with SSIM below mean - σ*std are jittery. Lower = more aggressive |
| `--dry-run` | off | Only detect and report jitter, don't produce output |

## How It Works
1. Computes SSIM between every consecutive frame pair
2. Flags frames where SSIM drops below mean - σ × std (statistical outliers)
3. Replaces each jittery frame with a linear blend of the nearest good neighbors
4. For consecutive bad frames, interpolates linearly across the gap

## Examples

### Fix jittery frames
```bash
uv run smooth-video --input video.mp4
```

### Just detect jitter without fixing
```bash
uv run smooth-video --input video.mp4 --dry-run
```

### More aggressive smoothing (lower sigma)
```bash
uv run smooth-video --input video.mp4 --sigma 1.5
```

## Related Tools
- [analyze-video](../analyze-video/SKILL.md) — Also reports jitter detection in its analysis
- [crop-video](../crop-video/SKILL.md) — Crop before smoothing
- [loop-video](../loop-video/SKILL.md) — Loop after smoothing

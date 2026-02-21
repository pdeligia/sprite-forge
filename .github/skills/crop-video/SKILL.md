---
name: crop-video
description: Crop, resize, and trim an MP4 video. Use this skill when the user wants to extract a region from a video, resize it, or trim to a time range.
---

# crop-video

A Python tool that crops, resizes, and trims an MP4 video. Processes every frame in the range, preserving the original frame rate. Useful for extracting a region of interest, upscaling, or trimming to a time window.

## How to Run
```bash
uv run crop-video --input <video.mp4> [options]
```

## Options
| Flag | Default | Description |
|------|---------|-------------|
| `--input FILE` | required | Input MP4 file |
| `--output FILE` | `./tmp/crop_video/output.mp4` | Output MP4 file path |
| `--region X1 Y1 X2 Y2` | (full frame) | Crop region: top-left to bottom-right in pixels |
| `--width N` | (original) | Scale output to this width (preserves aspect ratio) |
| `--start N` | `0.0` | Start time in seconds |
| `--end N` | (end of video) | End time in seconds |

## Examples

### Crop a region and upscale to 3072px wide
```bash
uv run crop-video --input video.mp4 --region 140 1 1140 719 --width 3072
```

### Trim to a time range
```bash
uv run crop-video --input video.mp4 --start 1.5 --end 6.0
```

### Crop, resize, and trim together
```bash
uv run crop-video --input video.mp4 --region 140 1 1140 719 --width 3072 --start 1.5
```

## Related Tools
- [video-to-frames](../video-to-frames/SKILL.md) — Extract individual PNG frames instead of a video
- [smooth-video](../smooth-video/SKILL.md) — Fix jittery frames after cropping
- [loop-video](../loop-video/SKILL.md) — Make a cropped video loop seamlessly

---
name: merge-video
description: Composite a background image behind every frame of a video. Use this skill when the user wants to place a video on top of a static background — such as compositing animated sprites onto game backgrounds.
---

# merge-video

A Python tool for compositing a background image behind every frame of a video.

## How to Run
```bash
uv run merge-video --input <video> --background <image> [options]
```

## Options
| Flag | Default | Description |
|------|---------|-------------|
| `--input FILE` | (required) | Input video file |
| `--background FILE` | (required) | Background image (PNG) |
| `--output FILE` | `./tmp/merge_video/output.mp4` | Output video file path |
| `--offset X Y` | `0 0` | Top-left position to place video frame on background |

## Behavior
- Each video frame is placed on top of the background at the given offset
- Output video has the background's dimensions
- If the background has an alpha channel, it's flattened onto white first
- Video frames are opaque (BGR from video codec)

## Important Notes
- The video frame + offset must fit within the background dimensions
- Output preserves the input video's FPS
- Background is loaded once and reused for every frame (efficient)

## Examples

### Merge a video onto a background (top-left aligned)
```bash
uv run merge-video --input animation.mp4 --background bg@3x.png
```

### Merge with an offset
```bash
uv run merge-video --input animation.mp4 --background bg@3x.png --offset 100 50
```

## Related Tools
- `merge-image` — Merge images together (image + image → image)
- `crop-video` — Crop and resize a video before merging
- `frames-to-video` — Compose a video from PNG frames

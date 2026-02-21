---
name: frames-to-mp4
description: Compose an MP4 video from a directory of PNG frames. Use this skill when the user wants to preview animations, create video from sprite frames, or convert image sequences to video.
---

# frames-to-mp4

A Python tool that composes an MP4 video from a directory of PNG frames. Useful for previewing animations, converting frame sequences to video, or creating looping video assets.

## How to Run
```bash
uv run frames-to-mp4 --input-dir <frames_dir> [options]
```

## Options
| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir DIR` | required | Directory of PNG frames (sorted alphabetically) |
| `--prefix PREFIX` | (none) | Filter input files by prefix |
| `--output FILE` | `./tmp/frames_to_mp4/output.mp4` | Output MP4 file path |
| `--fps N` | `10` | Frames per second. Lower = slower frame switches |
| `--loop N` | `1` | Repeat the frame sequence N times |

## Examples

### Basic: compose frames into video
```bash
uv run frames-to-mp4 --input-dir ./tmp/remix_animation --output preview.mp4
```

### Slow animation preview at 4 fps
```bash
uv run frames-to-mp4 --input-dir ./tmp/remix_animation --fps 4 --output slow.mp4
```

### Loop 3 times for longer preview
```bash
uv run frames-to-mp4 --input-dir ./tmp/mp4_to_frames --loop 3 --output looped.mp4
```

### Filter frames by prefix
```bash
uv run frames-to-mp4 --input-dir ./tmp/mp4_to_frames --prefix forest_ --output forest.mp4
```

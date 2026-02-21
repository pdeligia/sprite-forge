---
name: frames-to-video
description: Compose an MP4 video from a directory of PNG frames. Use this skill when the user wants to preview animations, create video from sprite frames, or convert image sequences to video.
---

# frames-to-video

A Python tool that composes an MP4 video from a directory of PNG frames. Useful for previewing animations, converting frame sequences to video, or creating looping video assets.

## How to Run
```bash
uv run frames-to-video --input-dir <frames_dir> [options]
```

## Options
| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir DIR` | required | Directory of PNG frames (sorted alphabetically) |
| `--prefix PREFIX` | (none) | Filter input files by prefix |
| `--output FILE` | `./tmp/frames_to_video/output.mp4` | Output MP4 file path |
| `--fps N` | `10` | Frames per second. Lower = slower frame switches |
| `--ping-pong` | off | Append frames in reverse for a seamless back-and-forth loop (2× length) |

## Examples

### Basic: compose frames into video
```bash
uv run frames-to-video --input-dir ./tmp/remix_animation --output preview.mp4
```

### Slow animation preview at 4 fps
```bash
uv run frames-to-video --input-dir ./tmp/remix_animation --fps 4 --output slow.mp4
```

### Ping-pong loop
```bash
uv run frames-to-video --input-dir ./tmp/remix_animation --ping-pong --output looped.mp4
```

### Filter frames by prefix
```bash
uv run frames-to-video --input-dir ./tmp/video_to_frames --prefix forest_ --output forest.mp4
```

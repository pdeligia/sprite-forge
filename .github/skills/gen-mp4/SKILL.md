---
name: gen-mp4
description: Generate synthetic MP4 videos with configurable dimensions, duration, frame rate, and fill color. Use this skill when the user needs to create placeholder videos, test assets, or synthetic video data.
---

# gen-mp4

A Python tool that generates synthetic MP4 videos.

## Location
`tools/gen_mp4.py` in the sprite-forge repository.

## How to Run
```bash
uv run python tools/gen_mp4.py [output] [options]
```

## Arguments
| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `output` | No | `./tmp/test.mp4` | Output MP4 file path |

## Options
| Flag | Default | Description |
|------|---------|-------------|
| `--width W` | `320` | Video width in pixels |
| `--height H` | `240` | Video height in pixels |
| `--duration S` | `1.0` | Duration in seconds |
| `--fps N` | `30` | Frames per second |
| `--color COLOR` | `black` | Fill color: preset name (`black`, `white`, `red`, `green`, `blue`) or `R,G,B` values (e.g., `128,0,255`) |

## Examples

### Generate a default 1-second black video
```bash
uv run python tools/gen_mp4.py
```

### Generate a 5-second 1280x720 blue video
```bash
uv run python tools/gen_mp4.py ./tmp/bg.mp4 --width 1280 --height 720 --duration 5 --color blue
```

### Generate with custom RGB color
```bash
uv run python tools/gen_mp4.py ./tmp/custom.mp4 --color 64,128,255
```

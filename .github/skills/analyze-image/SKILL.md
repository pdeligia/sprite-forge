---
name: analyze-image
description: Analyze an image and report statistics (dimensions, colors, depth, parallax layer recommendations). Use this skill before extract-image-layers to determine optimal layer count.
---

# analyze-image

A Python tool that analyzes an image and reports useful statistics for game asset work. Optionally runs Depth Anything V2 for parallax layer recommendations.

## How to Run
```bash
uv run analyze-image <image.png> [options]
```

## Arguments
| Argument | Required | Description |
|----------|----------|-------------|
| `input`  | Yes      | Path to the input image |

## Options
| Flag | Default | Description |
|------|---------|-------------|
| `--depth` | off | Run Depth Anything V2 for depth analysis and parallax layer recommendations |
| `--model SIZE` | `small` | Depth model size when `--depth` is used: `small`, `base`, `large` |

## Output

### Basic report (always shown)
- Image dimensions, channels, file size
- Alpha transparency stats (if RGBA)
- Color analysis: dominant hue, brightness/saturation ranges, color richness

### Depth report (with `--depth`)
- Depth mean and standard deviation
- Recommended number of parallax layers with explanation
- Coverage table showing pixel distribution across 2–5 layer splits

## Workflow
Use before `extract-image-layers` to decide how many layers:
1. `uv run analyze-image image.png --depth` → get recommended layers
2. `uv run extract-image-layers --input image.png --layers N` → split

## Examples

### Quick stats (no model needed)
```bash
uv run analyze-image scene@3x.png
```

### Full analysis with depth
```bash
uv run analyze-image scene@3x.png --depth
```

### With large model for best accuracy
```bash
uv run analyze-image scene@3x.png --depth --model large
```

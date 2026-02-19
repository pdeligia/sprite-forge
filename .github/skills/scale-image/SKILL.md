---
name: scale-image
description: Scale an image up or down by a given factor. Use this skill when the user needs to resize images, generate @1x/@2x/@3x asset variants, or adjust image dimensions for game assets.
---

# scale-image

A Python tool for scaling images by a given factor.

## Location
`tools/scale_image.py` in the sprite-forge repository.

## How to Run
```bash
uv run python tools/scale_image.py <factor> [options]
```

## Arguments
| Argument | Required | Description |
|----------|----------|-------------|
| `factor` | Yes | Scale factor (e.g., `0.333` for 1/3, `2.0` for 2x) |

## Options
| Flag | Default | Description |
|------|---------|-------------|
| `--input FILE` | none | Single image to scale |
| `--input-dir DIR` | none | Directory of images to scale |
| `--prefix PREFIX` | (empty) | Filter images by filename prefix (only with `--input-dir`) |
| `--output PATH` | (overwrites input) | Output file path (single file mode only) |

Must provide either `--input` or `--input-dir` (not both).

## Behavior
- Downscaling (factor < 1) uses `INTER_AREA` interpolation (best quality for shrinking)
- Upscaling (factor > 1) uses `INTER_NEAREST` interpolation (preserves sharp pixel edges for pixel art)
- Supports RGB and RGBA images
- If no `--output` is given, the input file is overwritten

## Examples

### Scale down @3x to @1x
```bash
uv run python tools/scale_image.py 0.333333 --input bg@3x.png --output bg@1x.png
```

### Scale up 2x for pixel art
```bash
uv run python tools/scale_image.py 2.0 --input sprite.png --output sprite@2x.png
```

### Scale all images with a prefix in a directory
```bash
uv run python tools/scale_image.py 0.5 --input-dir ./tmp --prefix forest
```

### Scale all PNGs in a directory
```bash
uv run python tools/scale_image.py 0.5 --input-dir ./tmp
```

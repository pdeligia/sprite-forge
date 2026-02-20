---
name: scale-image
description: Scale an image up or down by a given factor. Use this skill when the user needs to resize images, generate @1x/@2x/@3x asset variants, or adjust image dimensions for game assets.
---

# scale-image

A Python tool for scaling images by a given factor, with multiple scaling methods.

## How to Run
```bash
uv run scale-image <factor> [options]
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
| `--mode MODE` | `pixel` | Scaling method: `pixel`, `smooth`, or `ai` |

Must provide either `--input` or `--input-dir` (not both).

## Modes

### `pixel` (default)
Nearest-neighbor upscaling / area downscaling. Preserves sharp pixel edges — ideal for pixel art.

### `smooth`
Lanczos interpolation. Produces smooth results — good for photos and illustrated art.

### `ai`
Real-ESRGAN AI super-resolution. Best quality for upscaling painted/illustrated art and SVD output. Model weights (~64MB) auto-download on first use. Supports 2x and 4x native scales.

## Behavior
- Supports RGB and RGBA images (alpha is scaled separately in `ai` mode)
- If no `--output` is given, the input file is overwritten

## Examples

### Scale down @3x to @1x
```bash
uv run scale-image 0.333333 --input bg@3x.png --output bg@1x.png
```

### Scale up 2x for pixel art
```bash
uv run scale-image 2.0 --input sprite.png --output sprite@2x.png
```

### Smooth upscale for photos
```bash
uv run scale-image 2.0 --input photo.png --output photo_2x.png --mode smooth
```

### AI upscale for illustrated art
```bash
uv run scale-image 4.0 --input scene.png --output scene_4x.png --mode ai
```

### Scale all images with a prefix in a directory
```bash
uv run scale-image 0.5 --input-dir ./tmp --prefix forest
```

### Scale all PNGs in a directory
```bash
uv run scale-image 0.5 --input-dir ./tmp
```

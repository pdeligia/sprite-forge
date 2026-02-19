---
name: split-image-layers
description: Split an image into N depth layers using HSV-based depth estimation. Use this skill when the user wants to create parallax layers from a single image for 2D game backgrounds.
---

# split-image-layers

A Python tool that splits an image into N depth layers using HSV-based depth estimation, producing RGBA PNGs with transparency for parallax scrolling in games.

## How to Run
```bash
uv run split-image-layers --input <image.png> [options]
```

## Input Modes
| Mode | Flags | Description |
|------|-------|-------------|
| Single file | `--input FILE` | Process one image |
| Batch (all PNGs) | `--input-dir DIR` | Process all PNGs in a directory |
| Batch (filtered) | `--input-dir DIR --prefix PREFIX` | Process PNGs matching a prefix |

## Options
| Flag | Default | Description |
|------|---------|-------------|
| `--layers N` | 3 | Number of depth layers to split into |
| `--output-dir DIR` | `./tmp/split_image_layers` | Output directory (nuked and recreated each run) |
| `--suffix SUFFIX` | (empty) | Filename suffix to preserve (e.g., `@3x`) — layer number is inserted before it |
| `--feather F` | 0 | Edge feather radius in pixels (Gaussian blur, 0 = hard edges) |
| `--preview` | off | Also output a depth map visualization (INFERNO colormap) |

## Output Naming
Layer number is inserted before the suffix. Given input `forest_dungeon_bg_01@3x.png` with `--suffix "@3x"`:
- `forest_dungeon_bg_01_layer_01@3x.png` (back — farthest)
- `forest_dungeon_bg_01_layer_02@3x.png` (mid)
- `forest_dungeon_bg_01_layer_03@3x.png` (front — nearest)

## How It Works
1. Converts the image to HSV color space
2. Computes a depth score per pixel: `depth = 0.7 × V + 0.3 × S` (brightness + saturation)
3. Normalizes the depth score to [0, 1] where 0 = farthest, 1 = nearest
4. Splits the depth range into N equal bands → each band becomes a layer mask
5. Applies optional Gaussian feathering to smooth layer boundaries
6. Outputs RGBA PNGs where pixels outside the layer's band are transparent

## Important Notes
- Layers are numbered back-to-front: layer 01 is the farthest (darkest/desaturated), layer N is the nearest (brightest/saturated).
- This uses a lightweight HSV heuristic — it works well for scenes with depth-correlated brightness (dark sky → bright foreground). For complex scenes, results may need manual adjustment.
- The output directory is wiped clean before each run.
- Use `--preview` to see the computed depth map and verify the layer boundaries make sense.
- Use `--feather` to soften hard edges between layers (recommended: 3–10 pixels).
- Requires `uv sync` to have been run first to install dependencies (opencv-python).

## Examples

### Split a single image into 3 layers
```bash
uv run split-image-layers --input scene@3x.png --layers 3 --suffix "@3x"
```

### Split with feathered edges and depth preview
```bash
uv run split-image-layers --input scene.png --layers 3 --feather 5 --preview
```

### Batch split all frames from mp4_to_frames
```bash
uv run split-image-layers --input-dir ./tmp/mp4_to_frames \
  --prefix forest_dungeon_bg --layers 3 --suffix "@3x"
```

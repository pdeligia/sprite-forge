---
name: split-image-layers
description: Split an image into N depth layers using Depth Anything V2 AI model. Use this skill when the user wants to create parallax layers from a single image for 2D game backgrounds.
---

# split-image-layers

A Python tool that splits an image into N depth layers using Depth Anything V2 (monocular depth estimation), producing RGBA PNGs with transparency for parallax scrolling in games.

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
| `--model SIZE` | `small` | Depth model size: `small` (~25 MB, fast), `base` (~100 MB), `large` (~350 MB, best) |
| `--no-model` | off | Skip AI model, use vertical gradient fallback (for testing/CI) |

## Output Naming
Layer number is inserted before the suffix. Given input `forest_dungeon_bg_01@3x.png` with `--suffix "@3x"`:
- `forest_dungeon_bg_01_layer_01@3x.png` (back — farthest)
- `forest_dungeon_bg_01_layer_02@3x.png` (mid)
- `forest_dungeon_bg_01_layer_03@3x.png` (front — nearest)

## How It Works
1. Runs Depth Anything V2 to compute a per-pixel depth map (0 = far, 1 = near)
2. Uses quantile-based thresholds to split depth into N bands with roughly equal pixel coverage
3. Creates an RGBA layer for each band: original pixels where in-band, transparent elsewhere
4. Applies optional Gaussian feathering to smooth layer boundaries

## Important Notes
- Layers are numbered back-to-front: layer 01 is farthest, layer N is nearest.
- The `--model` flag selects the Depth Anything V2 variant. `small` is fast and good for most use cases; `large` gives better accuracy for complex scenes.
- Use `--no-model` for testing and CI — it replaces the AI model with a simple vertical gradient.
- The output directory is wiped clean before each run.
- Use `--preview` to see the computed depth map and verify layer boundaries.
- Use `--feather` to soften hard edges between layers (recommended: 3–10 pixels).
- Auto-detects MPS (Metal) on Mac for GPU acceleration.

## Examples

### Split a single image into 3 layers
```bash
uv run split-image-layers --input scene@3x.png --layers 3 --suffix "@3x"
```

### Split with feathered edges and depth preview
```bash
uv run split-image-layers --input scene.png --layers 3 --feather 5 --preview
```

### Use large model for best accuracy
```bash
uv run split-image-layers --input scene.png --layers 4 --model large --preview
```

### Batch split all frames from extract-mp4-frames
```bash
uv run split-image-layers --input-dir ./tmp/extract_mp4_frames \
  --prefix forest_dungeon_bg --layers 3 --suffix "@3x"
```

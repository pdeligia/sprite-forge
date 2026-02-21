---
name: remix-animation
description: Remix animation frames by retiming depth layers independently. Use this skill when the user wants to change animation speed per parallax layer, slow down backgrounds, or create new frame counts from existing animation frames.
---

# remix-animation

A Python tool that splits animation frames into depth layers, then retimes each layer independently to produce new composited output frames. Enables per-layer speed control for parallax animations.

## How to Run
```bash
uv run remix-animation --input-dir <frames_dir> [options]
```

## Options
| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir DIR` | required | Directory of animation frame PNGs |
| `--prefix PREFIX` | (none) | Filter input files by prefix |
| `--layers N` | 3 | Number of depth layers |
| `--frames N` | (same as input) | Number of output frames |
| `--speed S1,S2,...` | `1.0` for all | Per-layer speed multipliers, comma-separated back-to-front |
| `--output-dir DIR` | `./tmp/remix_animation` | Output directory (nuked each run) |
| `--suffix SUFFIX` | (empty) | Filename suffix to preserve (e.g., `@3x`) |
| `--feather F` | 0 | Edge feather radius in pixels |
| `--clean-edges VALUE` | `off` | Remove outline artifacts: erosion radius or `off` |
| `--tolerance N` | 2 | Max per-channel diff for static pixel detection |
| `--depth-map` | off | Output a depth map visualization |
| `--model SIZE` | `small` | Depth model size: `small`, `base`, `large` |
| `--no-model` | off | Skip depth model, use gradient fallback |
| `--fetch-latest-model` | off | Force re-download HuggingFace models |

## How It Works
1. Loads all input animation frames
2. Detects **static pixels** (identical across all frames within `--tolerance`)
3. Computes per-frame depth maps and splits each frame into N depth layers
4. Separates each layer into a **static plate** (unchanging pixels) and **animated frames** (changing pixels)
5. Retimes each layer's animation independently using `--speed` multipliers (nearest-neighbor frame sampling)
6. Composites static plates + retimed animated layers back-to-front into final output frames

## Speed Multipliers
- `1.0` = original speed (default)
- `0.5` = half speed (animation plays slower)
- `0.25` = quarter speed (4× slower)
- `2.0` = double speed (animation plays faster)

Layers are specified back-to-front. Missing values default to 1.0.

## Examples

### Slow down background, keep foreground at original speed
```bash
uv run remix-animation --input-dir ./tmp/video_to_frames \
  --layers 4 --frames 16 --speed 0.25,0.5,1.0,1.0 --suffix "@3x"
```

### Double the frame count with uniform speed
```bash
uv run remix-animation --input-dir ./tmp/video_to_frames \
  --layers 3 --frames 8
```

### Preview depth layer assignments
```bash
uv run remix-animation --input-dir ./tmp/video_to_frames \
  --layers 4 --depth-map
```

---
name: freeze-layer
description: Freeze depth layers in a video by replacing them with a static reference frame. Use this skill when the user wants to remove animation from specific depth layers (e.g. freeze foreground grass, freeze background clouds) while keeping other layers animated.
---

# freeze-layer

A Python tool for freezing depth layers in a video using the Depth Anything V2 model.

## How to Run
```bash
uv run freeze-layer --input <video> --layers <N> --freeze <indices> [options]
```

## Options
| Flag | Default | Description |
|------|---------|-------------|
| `--input FILE` | (required) | Input video file |
| `--layers N` | (required) | Number of depth layers to split into |
| `--freeze INDICES` | (required) | Comma-separated layer indices to freeze (1=nearest, N=farthest) |
| `--output FILE` | `./tmp/freeze_layer/output.mp4` | Output video file path |
| `--reference N` | `0` | Frame index to use as the static reference |
| `--feather N` | `5` | Edge feathering radius in pixels (0 = hard edges) |
| `--model-size SIZE` | `small` | Depth model: `small`, `base`, or `large` |

## Behavior
- Splits the depth range into N equal layers (1=nearest to N=farthest)
- Frozen layers are replaced with pixels from the reference frame
- Animated layers keep their original per-frame content
- Feathering creates soft edges at layer boundaries
- Computes per-frame depth maps using Depth Anything V2

## Examples

### Freeze the back 2 layers in a 4-layer split (keep front 2 animated)
```bash
uv run freeze-layer --input animation.mp4 --layers 4 --freeze 3,4
```

### Freeze everything except the front layer
```bash
uv run freeze-layer --input animation.mp4 --layers 3 --freeze 2,3
```

### Fine-grained: 5 layers, freeze only layer 2 and 4
```bash
uv run freeze-layer --input animation.mp4 --layers 5 --freeze 2,4
```

## Related Tools
- `image-to-layers` — Split a single image into depth layers
- `crop-video` — Crop and resize a video before processing
- `analyze-image` — Analyze depth and recommend layer count

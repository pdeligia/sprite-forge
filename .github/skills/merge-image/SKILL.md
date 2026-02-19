---
name: merge-image
description: Composite and merge images together. Use this skill when the user wants to combine, overlay, or layer images — such as placing sprites onto backgrounds, combining textures, or assembling game assets from separate layers.
---

# merge-image

A Python tool for compositing and merging images together.

## How to Run
```bash
uv run merge-image <background.png> [options]
```

## Arguments
| Argument | Required | Description |
|----------|----------|-------------|
| `background` | Yes | Path to the background image |

## Options
| Flag | Default | Description |
|------|---------|-------------|
| `--input FILE` | none | Single foreground image to merge |
| `--input-dir DIR` | none | Directory of foreground images |
| `--prefix PREFIX` | (empty) | Filter images by filename prefix (only with `--input-dir`) |
| `--offset X Y` | `0 0` | Top-left position to place foreground on background |

Must provide either `--input` or `--input-dir` (not both).

## Behavior
- The foreground is placed on top of the background at the given offset
- The result **overwrites the foreground file** (not the background)
- If the foreground has an alpha channel, alpha compositing is used
- The background file is never modified

## Important Notes
- The foreground + offset must fit within the background dimensions, otherwise the tool errors
- When using `--input-dir` without `--prefix`, all PNGs in the directory are processed
- Files are processed in sorted order

## Examples

### Merge a single image onto a background
```bash
uv run merge-image bg.png --input frame_01.png
```

### Merge all images with a prefix onto a background with offset
```bash
uv run merge-image bg.png --input-dir ./tmp --prefix forest --offset 100 50
```

### Merge all PNGs in a directory
```bash
uv run merge-image bg.png --input-dir ./tmp
```

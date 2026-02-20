# SpriteForge ⚒️ - Forge your pixels from any source

[![SpriteForge Banner](assets/image.png)](https://github.com/pdeligia/sprite-forge)

[![CI](https://github.com/pdeligia/sprite-forge/actions/workflows/ci.yml/badge.svg)](https://github.com/pdeligia/sprite-forge/actions/workflows/ci.yml)

An AI-powered toolkit for generating game assets from video and image sources.

## Requirements
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for dependency management

## Setup
```bash
uv sync
```

## Tools

| Tool | Description | Docs |
|------|-------------|------|
| `analyze-image` | Analyze image stats and recommend parallax layers | [SKILL.md](.github/skills/analyze-image/SKILL.md) |
| `analyze-mp4` | Analyze video motion and recommend optimal frame count | [SKILL.md](.github/skills/analyze-mp4/SKILL.md) |
| `extract-image-layers` | Extract depth layers from an image for parallax | [SKILL.md](.github/skills/extract-image-layers/SKILL.md) |
| `extract-mp4-frames` | Extract PNG frames from MP4 video with optional loop detection | [SKILL.md](.github/skills/extract-mp4-frames/SKILL.md) |
| `gen-mp4` | Generate synthetic MP4 videos | [SKILL.md](.github/skills/gen-mp4/SKILL.md) |
| `merge-image` | Composite and merge images together | [SKILL.md](.github/skills/merge-image/SKILL.md) |
| `scale-image` | Scale an image up or down by a given factor | [SKILL.md](.github/skills/scale-image/SKILL.md) |

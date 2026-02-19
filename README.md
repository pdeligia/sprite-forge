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
| `analyze-mp4` | Analyze video motion and recommend optimal frame count | [SKILL.md](.github/skills/analyze-mp4/SKILL.md) |
| `gen-mp4` | Generate synthetic MP4 videos | [SKILL.md](.github/skills/gen-mp4/SKILL.md) |
| `merge-image` | Composite and merge images together | [SKILL.md](.github/skills/merge-image/SKILL.md) |
| `mp4-to-frames` | Extract PNG frames from MP4 video with optional loop detection | [SKILL.md](.github/skills/mp4-to-frames/SKILL.md) |
| `scale-image` | Scale an image up or down by a given factor | [SKILL.md](.github/skills/scale-image/SKILL.md) |
| `split-image-layers` | Split an image into depth layers for parallax | [SKILL.md](.github/skills/split-image-layers/SKILL.md) |

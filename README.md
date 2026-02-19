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
| `tools/gen_mp4.py` | Generate synthetic MP4 videos | [SKILL.md](.github/skills/gen-mp4/SKILL.md) |
| `tools/mp4_to_frames.py` | Extract evenly-spaced PNG frames from MP4 video | [SKILL.md](.github/skills/mp4-to-frames/SKILL.md) |

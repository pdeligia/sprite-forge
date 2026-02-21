# SpriteForge ⚒️ - Forge your pixels from any source

[![SpriteForge Banner](assets/image.png)](https://github.com/pdeligia/sprite-forge)

[![CI](https://github.com/pdeligia/sprite-forge/actions/workflows/ci.yml/badge.svg)](https://github.com/pdeligia/sprite-forge/actions/workflows/ci.yml)

An agentic toolkit for generating game assets from video and image sources.

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
| `analyze-video` | Analyze video motion and recommend optimal frame count | [SKILL.md](.github/skills/analyze-video/SKILL.md) |
| `crop-video` | Crop, resize, and trim a video | [SKILL.md](.github/skills/crop-video/SKILL.md) |
| `frames-to-video` | Compose a video from PNG frames | [SKILL.md](.github/skills/frames-to-video/SKILL.md) |
| `freeze-layer` | Freeze a depth layer in a video | [SKILL.md](.github/skills/freeze-layer/SKILL.md) |
| `gen-video` | Generate synthetic test videos | [SKILL.md](.github/skills/gen-video/SKILL.md) |
| `image-to-layers` | Split an image into depth layers for parallax | [SKILL.md](.github/skills/image-to-layers/SKILL.md) |
| `loop-video` | Create a seamlessly looping video | [SKILL.md](.github/skills/loop-video/SKILL.md) |
| `merge-image` | Composite and merge images together | [SKILL.md](.github/skills/merge-image/SKILL.md) |
| `merge-video` | Composite a background image behind every frame of a video | [SKILL.md](.github/skills/merge-video/SKILL.md) |
| `remix-animation` | Remix animation frames with per-layer speed control | [SKILL.md](.github/skills/remix-animation/SKILL.md) |
| `scale-image` | Scale an image up or down by a given factor | [SKILL.md](.github/skills/scale-image/SKILL.md) |
| `smooth-video` | Detect and fix jittery frames in a video | [SKILL.md](.github/skills/smooth-video/SKILL.md) |
| `video-to-frames` | Extract PNG frames from video with optional loop detection | [SKILL.md](.github/skills/video-to-frames/SKILL.md) |

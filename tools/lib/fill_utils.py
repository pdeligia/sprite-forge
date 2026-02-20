"""Fill utilities for transparent layer regions."""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Dilate fill — fast, no model, good for parallax with small shifts
# ---------------------------------------------------------------------------

def dilate_fill(image_bgr, mask, iterations=50):
    """Fill masked regions by iteratively dilating edge pixels inward.

    Repeatedly expands the border of known pixels into the transparent
    region, then applies a blur to smooth the result.

    Args:
        image_bgr: Input image (H, W, 3) BGR uint8.
        mask: Binary mask (H, W) uint8 where 255 = region to fill.
        iterations: Number of dilation passes (more = fills larger gaps).

    Returns:
        Filled image (H, W, 3) BGR uint8.
    """
    if mask.max() == 0:
        return image_bgr

    result = image_bgr.copy()
    remaining = mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    for _ in range(iterations):
        if remaining.max() == 0:
            break

        # Dilate the known region by one step.
        dilated = cv2.dilate(result, kernel, iterations=1)

        # Only write into pixels that are still masked.
        fill_mask = remaining > 0
        result[fill_mask] = dilated[fill_mask]

        # Erode the remaining mask (shrink the unknown region).
        remaining = cv2.erode(remaining, kernel, iterations=1)

    # Smooth the filled regions to reduce blockiness.
    blurred = cv2.GaussianBlur(result, (15, 15), 0)
    fill_region = mask > 0
    result[fill_region] = blurred[fill_region]

    return result


# ---------------------------------------------------------------------------
# LaMa fill — AI-based, better quality, needs model download
# ---------------------------------------------------------------------------

def load_lama_model(device="cpu"):
    """Load the big-lama model from HuggingFace.

    Downloads the TorchScript model (~200MB) on first use.
    Returns a callable model that takes (image_tensor, mask_tensor).
    """
    import torch
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(
        repo_id="JosephCatrambone/big-lama-torchscript",
        filename="lama.pt",
    )
    print(f"Loading LaMa inpainting model on {device}...")
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model


def lama_fill(model, image_bgr, mask, device="cpu"):
    """Fill masked regions of an image using LaMa.

    Args:
        model: Loaded LaMa JIT model.
        image_bgr: Input image (H, W, 3) BGR uint8.
        mask: Binary mask (H, W) uint8 where 255 = region to fill.
        device: Torch device string.

    Returns:
        Filled image (H, W, 3) BGR uint8.
    """
    import torch

    if mask.max() == 0:
        return image_bgr

    h, w = image_bgr.shape[:2]

    # LaMa expects dimensions divisible by 8.
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8

    # Convert BGR→RGB, normalize to [0, 1].
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mask_f = (mask.astype(np.float32) / 255.0)

    # Pad if needed.
    if pad_h > 0 or pad_w > 0:
        rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        mask_f = np.pad(mask_f, ((0, pad_h), (0, pad_w)), mode="reflect")

    # To tensors: (1, C, H, W).
    img_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask_f).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        result = model(img_t, mask_t)

    # Back to numpy BGR.
    out = result.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    out = (out * 255).astype(np.uint8)

    # Remove padding.
    if pad_h > 0 or pad_w > 0:
        out = out[:h, :w]

    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Stable Diffusion inpainting — scene-aware, only touches masked pixels
# ---------------------------------------------------------------------------

SD_INPAINT_MODEL = "runwayml/stable-diffusion-inpainting"


def load_sd_inpaint_pipeline(device=None):
    """Load the Stable Diffusion inpainting pipeline.

    Downloads model weights (~2GB) on first use.
    """
    import torch
    from diffusers import StableDiffusionInpaintPipeline

    if device is None:
        from tools.lib.depth_utils import get_device
        device = get_device()

    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading SD Inpainting ({SD_INPAINT_MODEL}) on {device} (dtype={dtype})...")

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        SD_INPAINT_MODEL, torch_dtype=dtype,
    )
    pipe.to(device)

    if device == "mps":
        pipe.enable_attention_slicing("max")
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    return pipe


def sd_fill(pipe, image_bgr, mask, device="cpu", prompt="background, seamless texture, same style", scale_mode="ai"):
    """Fill masked regions using Stable Diffusion inpainting.

    Args:
        pipe: Loaded SD inpainting pipeline.
        image_bgr: Input image (H, W, 3) BGR uint8.
        mask: Binary mask (H, W) uint8 where 255 = region to fill.
        device: Torch device string.
        prompt: Text prompt to guide generation.
        scale_mode: Upscale method for scaling SD output back to original size.

    Returns:
        Filled image (H, W, 3) BGR uint8.
    """
    import torch
    from PIL import Image

    if mask.max() == 0:
        return image_bgr

    h, w = image_bgr.shape[:2]

    # SD inpainting works at 512x512. Scale down, inpaint, scale back.
    max_dim = 512
    scale = min(max_dim / w, max_dim / h)
    proc_w = int(w * scale) // 8 * 8
    proc_h = int(h * scale) // 8 * 8

    # Convert to PIL RGB.
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb).resize((proc_w, proc_h), Image.LANCZOS)
    pil_mask = Image.fromarray(mask).resize((proc_w, proc_h), Image.NEAREST)

    generator = torch.Generator(device="cpu").manual_seed(42)

    result = pipe(
        prompt=prompt,
        image=pil_img,
        mask_image=pil_mask,
        num_inference_steps=20,
        generator=generator,
        guidance_scale=7.5,
    ).images[0]

    # Scale back to original size using the specified upscaler.
    from tools.lib.scale_utils import upscale
    result_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
    out = upscale(result_bgr, w, h, mode=scale_mode)

    # Only replace masked pixels — keep original pixels untouched.
    fill_region = mask > 0
    output = image_bgr.copy()
    output[fill_region] = out[fill_region]

    return output


# ---------------------------------------------------------------------------
# Back-to-front layer fill (shared logic for all modes)
# ---------------------------------------------------------------------------

def fill_layers_back_to_front(layers, mode="spread", model=None, device="cpu", scale_mode="ai"):
    """Fill transparent regions of layers back-to-front.

    The back layer (index 0) gets ALL transparent pixels filled.
    Each subsequent layer gets its transparent pixels filled.
    The front layer (last index) is left unchanged.

    Args:
        layers: List of RGBA images (back-to-front order).
        mode: Fill method — "spread", "inpaint", or "diffuse".
        model: Loaded model (LaMa for inpaint, SD pipeline for diffuse).
        device: Torch device string.
        scale_mode: Upscale method for diffuse mode ("pixel", "smooth", "ai").

    Returns:
        List of RGBA images with transparent regions filled.
    """
    filled = []
    for i, layer in enumerate(layers):
        if i == len(layers) - 1:
            # Front layer — keep as-is.
            filled.append(layer)
            print(f"    Layer {i + 1}/{len(layers)}: front layer, no fill needed")
            continue

        alpha = layer[:, :, 3]
        transparent_mask = (alpha == 0).astype(np.uint8) * 255

        transparent_pct = (transparent_mask > 0).sum() / transparent_mask.size * 100
        print(f"    Layer {i + 1}/{len(layers)}: {transparent_pct:.1f}% transparent, filling ({mode})...")

        if transparent_mask.max() == 0:
            filled.append(layer)
            continue

        bgr = layer[:, :, :3].copy()

        if mode == "inpaint":
            filled_bgr = lama_fill(model, bgr, transparent_mask, device)
        elif mode == "diffuse":
            filled_bgr = sd_fill(model, bgr, transparent_mask, device, scale_mode=scale_mode)
        else:
            filled_bgr = dilate_fill(bgr, transparent_mask)

        # Build fully opaque RGBA output.
        result = np.dstack([filled_bgr, np.full_like(alpha, 255)])
        filled.append(result)

    return filled

"""Shared image scaling utilities."""

import cv2
import numpy as np


def scale_nearest(img, new_w, new_h, factor):
    """Scale using nearest-neighbor (crisp pixel art)."""
    interp = cv2.INTER_NEAREST if factor > 1 else cv2.INTER_AREA
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def scale_smooth(img, new_w, new_h, factor):
    """Scale using Lanczos (smooth, general-purpose)."""
    interp = cv2.INTER_LANCZOS4 if factor > 1 else cv2.INTER_AREA
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def scale_ai(img, new_w, new_h, factor):
    """Scale using Real-ESRGAN AI super-resolution (py-real-esrgan package)."""
    import torch
    from PIL import Image
    from RealESRGAN import RealESRGAN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Choose model scale: 2x or 4x.
    net_scale = 2 if factor <= 2 else 4

    model = RealESRGAN(device, scale=net_scale)
    model.load_weights(f"weights/RealESRGAN_x{net_scale}.pth", download=True)

    # Handle alpha channel separately — ESRGAN only works on RGB.
    has_alpha = len(img.shape) == 3 and img.shape[2] == 4
    if has_alpha:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
    else:
        bgr = img

    # Convert BGR → RGB → PIL for the model.
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    result = model.predict(pil_img)

    # Convert back to BGR numpy array.
    output = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

    if has_alpha:
        alpha_up = cv2.resize(alpha, (output.shape[1], output.shape[0]),
                              interpolation=cv2.INTER_LANCZOS4)
        output = np.dstack([output, alpha_up])

    # Resize to exact target dimensions if model scale != requested factor.
    if output.shape[1] != new_w or output.shape[0] != new_h:
        output = cv2.resize(output, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    return output


SCALE_MODES = {
    "pixel": scale_nearest,
    "smooth": scale_smooth,
    "ai": scale_ai,
}


def upscale(img, target_w, target_h, mode="ai"):
    """Upscale an image to target dimensions using the specified mode.

    Convenience wrapper for use by other tools.
    """
    h, w = img.shape[:2]
    if w == target_w and h == target_h:
        return img
    factor = target_w / w
    scale_fn = SCALE_MODES.get(mode, scale_smooth)
    return scale_fn(img, target_w, target_h, factor)

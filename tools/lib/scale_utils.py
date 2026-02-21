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
    """Scale using Real-ESRGAN AI super-resolution."""
    import torch
    from PIL import Image
    from huggingface_hub import hf_hub_download
    from py_real_esrgan.rrdbnet_arch import RRDBNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Choose model scale: 2x or 4x.
    net_scale = 2 if factor <= 2 else 4
    num_block = 23 if net_scale == 4 else 16

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=num_block, num_grow_ch=32, scale=net_scale)
    from tools.lib.console import run_with_hf_fallback
    weights_path = run_with_hf_fallback(
        hf_hub_download,
        repo_id="sberbank-ai/Real-ESRGAN",
        filename=f"RealESRGAN_x{net_scale}.pth",
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval().to(device)

    # Handle alpha channel separately — ESRGAN only works on RGB.
    has_alpha = len(img.shape) == 3 and img.shape[2] == 4
    if has_alpha:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
    else:
        bgr = img

    # BGR → RGB, normalize to [0, 1], to tensor (1, 3, H, W).
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = model(tensor)

    # Back to numpy BGR.
    out = output_tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    output = cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

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

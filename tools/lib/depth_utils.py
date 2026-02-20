"""Shared depth estimation utilities using Depth Anything V2."""

import cv2
import numpy as np


# Model name mapping for Depth Anything V2.
MODEL_MAP = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}


def get_device():
    """Select the best available device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_depth_pipeline(model_size):
    """Load the Depth Anything V2 pipeline."""
    from transformers import pipeline as hf_pipeline
    model_name = MODEL_MAP[model_size]
    device = get_device()
    print(f"Loading {model_name} on {device}...")
    pipe = hf_pipeline(task="depth-estimation", model=model_name, device=device)
    return pipe


def compute_depth_map(pipe, image_bgr):
    """Run depth estimation on a BGR opencv image.

    If pipe is None, uses a vertical gradient fallback (for testing without a model).
    Returns a float32 array in [0, 1] where 0 = farthest and 1 = nearest.
    """
    h, w = image_bgr.shape[:2]

    if pipe is None:
        # Fallback: vertical gradient (top=far, bottom=near).
        y_grad = np.linspace(0, 1, h, dtype=np.float32).reshape(-1, 1)
        return np.broadcast_to(y_grad, (h, w)).copy()

    # Convert BGR to RGB PIL Image.
    from PIL import Image
    rgb = cv2.cvtColor(image_bgr[:, :, :3], cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    result = pipe(pil_img)
    depth_pil = result["depth"]

    # Convert to numpy and normalize to [0, 1].
    depth = np.array(depth_pil, dtype=np.float32)
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 0:
        depth = (depth - d_min) / (d_max - d_min)
    else:
        depth = np.zeros_like(depth)

    # Resize to match original image dimensions (pipeline may resize).
    h, w = image_bgr.shape[:2]
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

    return depth

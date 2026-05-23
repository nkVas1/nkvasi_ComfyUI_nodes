"""
Image conversion helpers and compositing utilities.
"""
import numpy as np
import torch
from PIL import Image


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert ComfyUI IMAGE tensor (H,W,C) float32 [0,1] -> PIL RGB."""
    arr = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def pil_to_tensor(pil: Image.Image) -> torch.Tensor:
    """Convert PIL (RGB or RGBA) -> ComfyUI IMAGE tensor (H,W,C) float32 [0,1]."""
    pil_rgb = pil.convert("RGB")
    arr = np.array(pil_rgb).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def pil_mask_to_tensor(pil_mask: Image.Image) -> torch.Tensor:
    """Convert PIL L mask -> ComfyUI MASK tensor (H,W) float32 [0,1]."""
    arr = np.array(pil_mask.convert("L")).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def apply_background(img: Image.Image, mask: Image.Image, background: str) -> Image.Image:
    """Composite foreground (img) over chosen background using mask."""
    img_rgba = img.convert("RGBA")
    mask_l = mask.convert("L").resize(img_rgba.size, Image.LANCZOS)

    if background == "alpha":
        r, g, b, _ = img_rgba.split()
        out = Image.merge("RGBA", (r, g, b, mask_l))
        return out

    w, h = img_rgba.size
    if background == "checkerboard":
        bg = _make_checkerboard(w, h)
    else:
        color_map = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "green": (0, 177, 64),
            "red": (220, 50, 50),
            "blue": (0, 100, 220),
        }
        bg = Image.new("RGB", (w, h), color_map.get(background, (255, 255, 255)))

    bg = bg.convert("RGBA")
    composed = Image.composite(img_rgba, bg, mask_l)
    return composed.convert("RGBA")


def _make_checkerboard(w: int, h: int, tile: int = 16) -> Image.Image:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            c = 200 if (x // tile + y // tile) % 2 == 0 else 255
            arr[y:y+tile, x:x+tile] = c
    return Image.fromarray(arr, mode="RGB")


def refine_foreground_colors(img: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Fast foreground color estimation (Levin-style alpha matting pre-pass).
    Removes background color bleeding on semi-transparent edges.
    Based on the approach used in BiRefNet and Fast-Foreground-Estimation.
    """
    import cv2
    img_np = np.array(img.convert("RGB")).astype(np.float32)
    mask_np = np.array(mask.convert("L")).astype(np.float32) / 255.0
    alpha = mask_np[:, :, np.newaxis]

    # dilate mask to sample background near edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated = cv2.dilate(mask_np, kernel)
    bg_region = (dilated < 0.1).astype(np.float32)[:, :, np.newaxis]

    # estimate background color from surrounding area using guided blur
    bg_color_sum = np.sum(img_np * bg_region, axis=(0, 1))
    bg_count = np.sum(bg_region) + 1e-6
    bg_color = bg_color_sum / bg_count  # (3,)

    # subtract background bleed proportional to (1-alpha)
    fg_est = img_np - bg_color * (1.0 - alpha)
    fg_est = np.clip(fg_est, 0, 255).astype(np.uint8)
    return Image.fromarray(fg_est, mode="RGB")

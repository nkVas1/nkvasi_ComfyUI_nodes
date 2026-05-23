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
    """
    Convert PIL (RGB or RGBA) -> ComfyUI IMAGE tensor (H,W,3) float32 [0,1].
    ComfyUI's IMAGE type is always 3-channel RGB; alpha is carried separately
    via MASK. This converts to RGB (dropping alpha if present).
    """
    arr = np.array(pil.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def pil_rgba_to_tensor(pil: Image.Image) -> torch.Tensor:
    """
    Convert PIL RGBA -> tensor (H,W,4) float32 [0,1].
    Used exclusively by Save Image Alpha node.
    """
    arr = np.array(pil.convert("RGBA")).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def pil_mask_to_tensor(pil_mask: Image.Image) -> torch.Tensor:
    """Convert PIL L mask -> ComfyUI MASK tensor (H,W) float32 [0,1]."""
    arr = np.array(pil_mask.convert("L")).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def apply_background(img: Image.Image, mask: Image.Image, background: str) -> Image.Image:
    """
    Composite foreground (img) over chosen background using mask.
    For 'alpha' returns RGBA; for all others returns RGBA composited over solid/checker.
    """
    img_rgba = img.convert("RGBA")
    mask_l = mask.convert("L").resize(img_rgba.size, Image.LANCZOS)

    if background == "alpha":
        r, g, b, _ = img_rgba.split()
        return Image.merge("RGBA", (r, g, b, mask_l))

    w, h = img_rgba.size
    if background == "checkerboard":
        bg = _make_checkerboard(w, h)
    else:
        color_map = {
            "white":  (255, 255, 255),
            "black":  (0, 0, 0),
            "green":  (0, 177, 64),
            "red":    (220, 50, 50),
            "blue":   (0, 100, 220),
        }
        bg = Image.new("RGB", (w, h), color_map.get(background, (255, 255, 255)))

    composed = Image.composite(img_rgba, bg.convert("RGBA"), mask_l)
    return composed


def _make_checkerboard(w: int, h: int, tile: int = 16) -> Image.Image:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            c = 200 if (x // tile + y // tile) % 2 == 0 else 255
            arr[y:y+tile, x:x+tile] = c
    return Image.fromarray(arr, mode="RGB")


def refine_foreground_colors(img: Image.Image, mask: Image.Image, strength: float = 0.65) -> Image.Image:
    """
    Foreground color decontamination — removes background color bleed
    on semi-transparent edges (especially hair).

    Uses a two-pass approach:
      1. Estimate local background color via large-kernel blur of BG pixels
      2. Subtract bleed proportional to (1 - alpha) * strength

    `strength` controls aggressiveness: 0.0 = off, 1.0 = full removal.
    Lower default (0.65 vs old 1.0) prevents the white halo on bright backgrounds.
    """
    import cv2
    img_np = np.array(img.convert("RGB")).astype(np.float32)
    mask_np = np.array(mask.convert("L")).astype(np.float32) / 255.0

    # --- estimate background color per-pixel via large gaussian blur ---
    # blur the image masked to BG pixels, then fill with the result
    bg_only = img_np * (1.0 - mask_np[:, :, np.newaxis])
    # large kernel to propagate bg color estimate into edge zone
    ksize = 61
    bg_color_map = cv2.GaussianBlur(bg_only, (ksize, ksize), sigmaX=20)
    weight_map = cv2.GaussianBlur((1.0 - mask_np), (ksize, ksize), sigmaX=20)
    # avoid div/0 in pure-foreground areas
    weight_map = np.maximum(weight_map, 1e-6)[:, :, np.newaxis]
    bg_color_map = bg_color_map / weight_map  # per-pixel background estimate

    # --- subtract bleed proportional to (1-alpha) * strength ---
    alpha = mask_np[:, :, np.newaxis]
    fg_est = img_np - bg_color_map * (1.0 - alpha) * strength
    fg_est = np.clip(fg_est, 0, 255).astype(np.uint8)
    return Image.fromarray(fg_est, mode="RGB")

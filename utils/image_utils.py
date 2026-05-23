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
    Convert PIL -> ComfyUI IMAGE tensor (H,W,3) float32 [0,1].
    Always converts to RGB — ComfyUI IMAGE is 3-channel.
    """
    arr = np.array(pil.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def pil_rgba_to_tensor(pil: Image.Image) -> torch.Tensor:
    """Convert PIL RGBA -> tensor (H,W,4) float32 [0,1]."""
    arr = np.array(pil.convert("RGBA")).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def pil_mask_to_tensor(pil_mask: Image.Image) -> torch.Tensor:
    """Convert PIL L mask -> ComfyUI MASK tensor (H,W) float32 [0,1]."""
    arr = np.array(pil_mask.convert("L")).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def apply_background(img: Image.Image, mask: Image.Image, background: str) -> Image.Image:
    """
    Composite foreground over chosen background using mask as alpha.
      "alpha"     → RGBA PNG (transparent)
      any other   → RGB  PNG (composited over solid colour or checkerboard)
    """
    img_rgb = img.convert("RGB")
    mask_l  = mask.convert("L").resize(img_rgb.size, Image.LANCZOS)

    if background == "alpha":
        r, g, b = img_rgb.split()
        return Image.merge("RGBA", (r, g, b, mask_l))

    w, h = img_rgb.size
    if background == "checkerboard":
        bg = _make_checkerboard(w, h)
    else:
        color_map = {
            "white": (255, 255, 255),
            "black": (0,   0,   0  ),
            "green": (0,   177, 64 ),
            "red":   (220, 50,  50 ),
            "blue":  (0,   100, 220),
        }
        bg = Image.new("RGB", (w, h), color_map.get(background, (255, 255, 255)))

    return Image.composite(img_rgb, bg, mask_l)


def _make_checkerboard(w: int, h: int, tile: int = 16) -> Image.Image:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            c = 200 if (x // tile + y // tile) % 2 == 0 else 255
            arr[y:y+tile, x:x+tile] = c
    return Image.fromarray(arr, mode="RGB")


def refine_foreground_colors(
    img: Image.Image,
    mask: Image.Image,
    strength: float = 0.65,
) -> Image.Image:
    """
    Foreground colour decontamination — Gaussian fallback used by Ensemble.
    For maximum quality use NkVasi_MattingRefine with decontaminate=True
    which calls pymatting.estimate_foreground_ml when available.
    """
    import cv2
    img_np  = np.array(img.convert("RGB")).astype(np.float32)
    mask_np = np.array(mask.convert("L")).astype(np.float32) / 255.0

    bg_only     = img_np * (1.0 - mask_np[:, :, None])
    ksize       = 61
    bg_blur     = cv2.GaussianBlur(bg_only,          (ksize, ksize), sigmaX=20)
    weight_blur = cv2.GaussianBlur(1.0 - mask_np,    (ksize, ksize), sigmaX=20)
    weight_blur = np.maximum(weight_blur, 1e-6)[:, :, None]
    bg_map      = bg_blur / weight_blur
    alpha       = mask_np[:, :, None]
    fg_est      = np.clip(img_np - bg_map * (1.0 - alpha) * strength, 0, 255)
    return Image.fromarray(fg_est.astype(np.uint8), mode="RGB")

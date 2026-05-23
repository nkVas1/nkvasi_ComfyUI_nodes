"""
NkVasi_AlphaPreview — composites IMAGE + MASK onto checkerboard for
instant alpha quality preview inside ComfyUI without saving to disk.

Connect:
  image → output of Remove BG Ensemble (or any IMAGE)
  mask  → the MASK output of the same node

The node outputs an IMAGE (checkerboard composite) suitable for connecting
to a Preview Image node or Save Image.
It also outputs the original MASK unchanged for chaining.
"""
import torch
import numpy as np
from PIL import Image

from ..utils.image_utils import tensor_to_pil, pil_to_tensor, pil_mask_to_tensor


class NkVasi_AlphaPreview:
    """Composites image+mask over checkerboard — instant alpha preview."""

    CATEGORY = "🎭 nkVasi/Background Removal"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("preview", "mask")
    FUNCTION = "preview"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask":  ("MASK",),
            },
            "optional": {
                "tile_size": ("INT", {"default": 16, "min": 4, "max": 64, "step": 4}),
                "color_a":  ("INT", {"default": 200, "min": 0, "max": 255, "step": 1}),
                "color_b":  ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
            },
        }

    def preview(self, image, mask, tile_size=16, color_a=200, color_b=255):
        results = []
        result_masks = []

        for i in range(image.shape[0]):
            pil_img = tensor_to_pil(image[i])
            w, h    = pil_img.size

            mask_i   = mask[i] if mask.ndim == 3 else mask
            mask_np  = (mask_i.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_mask = Image.fromarray(mask_np, mode="L")
            if pil_mask.size != (w, h):
                pil_mask = pil_mask.resize((w, h), Image.LANCZOS)

            # Build checkerboard
            checker = _make_checkerboard(w, h, tile_size, color_a, color_b)

            # Composite: foreground over checker using mask as alpha
            fg_rgba = pil_img.convert("RGBA")
            r, g, b, _ = fg_rgba.split()
            fg_rgba = Image.merge("RGBA", (r, g, b, pil_mask))
            composed = Image.composite(fg_rgba, checker.convert("RGBA"), pil_mask)
            composed_rgb = composed.convert("RGB")

            results.append(pil_to_tensor(composed_rgb))
            result_masks.append(pil_mask_to_tensor(pil_mask))

        return (torch.stack(results), torch.stack(result_masks))


def _make_checkerboard(
    w: int, h: int, tile: int = 16,
    color_a: int = 200, color_b: int = 255,
) -> Image.Image:
    arr = np.full((h, w, 3), color_b, dtype=np.uint8)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            if (x // tile + y // tile) % 2 == 0:
                arr[y:y+tile, x:x+tile] = color_a
    return Image.fromarray(arr, mode="RGB")

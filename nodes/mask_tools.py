"""
NkVasi_MaskTools — apply mask to image with advanced background options.
Accepts external mask (from any source) + image -> composited output.
"""
import torch
from PIL import Image

from ..utils.image_utils import (
    tensor_to_pil, pil_to_tensor,
    pil_mask_to_tensor, apply_background,
    refine_foreground_colors,
)

BACKGROUNDS = ["alpha", "white", "black", "green", "red", "blue", "checkerboard"]


class NkVasi_MaskTools:
    """Apply a mask to an image with flexible background options."""

    CATEGORY = "🎭 nkVasi/Background Removal"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_mask"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "background": (BACKGROUNDS, {"default": "alpha"}),
            },
            "optional": {
                "refine_foreground": ("BOOLEAN", {"default": False}),
                "invert_mask": ("BOOLEAN", {"default": False}),
            },
        }

    def apply_mask(self, image, mask, background, refine_foreground=False, invert_mask=False):
        results = []
        for i in range(image.shape[0]):
            pil_img = tensor_to_pil(image[i])
            import numpy as np
            m_np = mask[i].cpu().numpy().astype(np.float32)
            if invert_mask:
                m_np = 1.0 - m_np
            pil_mask = Image.fromarray((m_np * 255).clip(0, 255).astype(np.uint8), mode="L")
            pil_mask = pil_mask.resize(pil_img.size, Image.LANCZOS)
            if refine_foreground:
                pil_img = refine_foreground_colors(pil_img, pil_mask)
            out_img = apply_background(pil_img, pil_mask, background)
            results.append(pil_to_tensor(out_img))
        return (torch.stack(results),)

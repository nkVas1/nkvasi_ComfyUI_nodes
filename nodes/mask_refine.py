"""
NkVasi_MaskRefine — standalone mask post-processing node.
Apply blur, erode/expand, hole fill, island removal, threshold, feather.
Useful when chaining after any external background removal node.
"""
import torch
import numpy as np
from PIL import Image

from ..utils.mask_ops import (
    smooth_mask, erode_expand_mask,
    remove_small_holes, remove_small_islands,
    feather_mask,
)
from ..utils.image_utils import pil_mask_to_tensor


class NkVasi_MaskRefine:
    """Advanced mask refinement — blur, erode, feather, clean."""

    CATEGORY = "🎭 nkVasi/Background Removal"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "refine"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            },
            "optional": {
                "blur_radius": ("INT", {"default": 1, "min": 0, "max": 64, "step": 1}),
                "erode_expand": ("INT", {"default": 0, "min": -30, "max": 30, "step": 1}),
                "feather_edges": ("INT", {"default": 3, "min": 0, "max": 32, "step": 1}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "remove_holes": ("BOOLEAN", {"default": True}),
                "remove_islands": ("BOOLEAN", {"default": True}),
                "min_hole_size": ("INT", {"default": 200, "min": 0, "max": 5000, "step": 50}),
                "min_island_size": ("INT", {"default": 100, "min": 0, "max": 5000, "step": 50}),
            },
        }

    def refine(
        self,
        mask,
        blur_radius=1,
        erode_expand=0,
        feather_edges=3,
        threshold=0.5,
        remove_holes=True,
        remove_islands=True,
        min_hole_size=200,
        min_island_size=100,
    ):
        out_masks = []
        for i in range(mask.shape[0]):
            m_np = mask[i].cpu().numpy().astype(np.float32)

            # binarize with threshold
            m_np = (m_np >= threshold).astype(np.float32)

            if remove_holes:
                m_np = remove_small_holes(m_np, min_hole_size)
            if remove_islands:
                m_np = remove_small_islands(m_np, min_island_size)
            if erode_expand != 0:
                m_np = erode_expand_mask(m_np, erode_expand)
            if blur_radius > 0:
                m_np = smooth_mask(m_np, blur_radius)
            if feather_edges > 0:
                m_np = feather_mask(m_np, feather_edges)

            pil_m = Image.fromarray((m_np * 255).clip(0, 255).astype(np.uint8), mode="L")
            out_masks.append(pil_mask_to_tensor(pil_m))

        return (torch.stack(out_masks),)

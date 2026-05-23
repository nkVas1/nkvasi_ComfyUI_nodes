"""
NkVasi_MaskRefine — standalone mask post-processing node.

Designed for two use cases:
  A) Quick cleanup after any external BG removal node (no image needed)
  B) High-quality edge refinement when the original image is also available
     (connect image input to enable guided filter + foreground decontamination)

All operations preserve semi-transparent edges — no binarisation unless
the user explicitly sets threshold < 1.0.
"""
import torch
import numpy as np
from PIL import Image

from ..utils.mask_ops import (
    smooth_mask, erode_expand_mask,
    soft_remove_holes, soft_remove_islands,
    guided_filter_mask, feather_mask,
)
from ..utils.image_utils import tensor_to_pil, pil_mask_to_tensor


class NkVasi_MaskRefine:
    """Advanced mask refinement with soft alpha preservation."""

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
                # Connect the original image to enable guided filter refinement
                "image":          ("IMAGE",),
                "blur_radius":    ("INT",   {"default": 1,   "min": 0,    "max": 64,   "step": 1}),
                "erode_expand":   ("INT",   {"default": 0,   "min": -30,  "max": 30,   "step": 1}),
                "feather_edges":  ("INT",   {"default": 2,   "min": 0,    "max": 32,   "step": 1}),
                # threshold: set to 0.0 to keep full soft mask from the input;
                # set to 0.5 for hard binarise (legacy behaviour)
                "threshold":      ("FLOAT", {"default": 0.0, "min": 0.0,  "max": 1.0,  "step": 0.01}),
                "remove_holes":   ("BOOLEAN", {"default": True}),
                "remove_islands": ("BOOLEAN", {"default": True}),
                "min_hole_size":  ("INT",   {"default": 500, "min": 0,    "max": 10000, "step": 50}),
                "min_island_size":("INT",   {"default": 400, "min": 0,    "max": 10000, "step": 50}),
                # guided filter (only active when image is connected)
                "guided_filter":  ("BOOLEAN", {"default": True}),
                "guided_radius":  ("INT",   {"default": 7,   "min": 1,    "max": 32,   "step": 1}),
            },
        }

    def refine(
        self,
        mask,
        image=None,
        blur_radius=1,
        erode_expand=0,
        feather_edges=2,
        threshold=0.0,
        remove_holes=True,
        remove_islands=True,
        min_hole_size=500,
        min_island_size=400,
        guided_filter=True,
        guided_radius=7,
    ):
        out_masks = []

        for i in range(mask.shape[0]):
            m_np = mask[i].cpu().numpy().astype(np.float32)

            # Optional hard threshold (default 0.0 = keep soft mask as-is)
            if threshold > 0.0:
                m_np = (m_np >= threshold).astype(np.float32)

            # Guided filter: edge-aware refinement using the original image
            if guided_filter and image is not None:
                img_idx = min(i, image.shape[0] - 1)
                pil_img  = tensor_to_pil(image[img_idx])
                # resize guide to match mask resolution
                h, w     = m_np.shape[:2]
                pil_guide = pil_img.resize((w, h), Image.LANCZOS)
                guide_np  = np.array(pil_guide).astype(np.float32) / 255.0
                m_np = guided_filter_mask(m_np, guide_np, radius=guided_radius, eps=3e-4)

            if remove_holes:
                m_np = soft_remove_holes(m_np, min_hole_size=min_hole_size)
            if remove_islands:
                m_np = soft_remove_islands(m_np, min_island_size=min_island_size)
            if erode_expand != 0:
                m_np = erode_expand_mask(m_np, erode_expand)
            if feather_edges > 0:
                m_np = feather_mask(m_np, feather_edges)
            if blur_radius > 0:
                m_np = smooth_mask(m_np, blur_radius)

            pil_m = Image.fromarray(
                (m_np * 255).clip(0, 255).astype(np.uint8), mode="L"
            )
            out_masks.append(pil_mask_to_tensor(pil_m))

        return (torch.stack(out_masks),)

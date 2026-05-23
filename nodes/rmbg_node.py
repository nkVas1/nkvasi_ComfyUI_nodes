"""
nkVasi_RMBG_Node — Single-model professional background removal.
Supports: BiRefNet-HR, BiRefNet-general, BiRefNet-matting,
          BEN2, RMBG-2.0, InSPyReNet.
Each model is loaded lazily and cached across calls.
"""
import torch
import numpy as np
from PIL import Image

from ..utils.model_loader import load_birefnet, load_ben2, load_rmbg2, load_inspyrenet
from ..utils.image_utils import (
    tensor_to_pil, pil_to_tensor,
    pil_mask_to_tensor, apply_background,
    refine_foreground_colors,
)
from ..utils.mask_ops import (
    smooth_mask, erode_expand_mask,
    remove_small_holes, remove_small_islands,
)

MODELS = [
    "BiRefNet-HR (2048px, best quality)",
    "BiRefNet-dynamic (any resolution)",
    "BiRefNet-matting (hair/transparency)",
    "BiRefNet-general (fast, balanced)",
    "BEN2 (hair/fur specialist)",
    "RMBG-2.0 (BRIA, commercial-free)",
    "InSPyReNet (portrait specialist)",
]

BACKGROUNDS = ["alpha", "white", "black", "green", "red", "blue", "checkerboard"]


class NkVasi_RMBG_Node:
    """Professional single-pass background removal."""

    CATEGORY = "🎭 nkVasi/Background Removal"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "remove_background"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (MODELS, {"default": "BiRefNet-HR (2048px, best quality)"}),
                "process_resolution": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 128}),
                "background": (BACKGROUNDS, {"default": "alpha"}),
            },
            "optional": {
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 32, "step": 1}),
                "mask_offset": ("INT", {"default": 0, "min": -20, "max": 20, "step": 1}),
                "sensitivity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "refine_foreground": ("BOOLEAN", {"default": True}),
                "remove_holes": ("BOOLEAN", {"default": True}),
                "remove_islands": ("BOOLEAN", {"default": True}),
                "fp16": ("BOOLEAN", {"default": True}),
            },
        }

    def remove_background(
        self,
        image,
        model,
        process_resolution,
        background,
        mask_blur=0,
        mask_offset=0,
        sensitivity=0.5,
        refine_foreground=True,
        remove_holes=True,
        remove_islands=True,
        fp16=True,
    ):
        results_img = []
        results_mask = []

        for i in range(image.shape[0]):
            pil_img = tensor_to_pil(image[i])
            orig_w, orig_h = pil_img.size

            # ---- infer mask ----
            if "BiRefNet" in model:
                variant = _birefnet_variant(model)
                mask_pil = load_birefnet(variant).infer(pil_img, process_resolution, fp16)
            elif "BEN2" in model:
                mask_pil = load_ben2().infer(pil_img, process_resolution)
            elif "RMBG-2.0" in model:
                mask_pil = load_rmbg2().infer(pil_img, process_resolution, fp16)
            elif "InSPyReNet" in model:
                mask_pil = load_inspyrenet().infer(pil_img, process_resolution)
            else:
                raise ValueError(f"Unknown model: {model}")

            # ---- mask post-processing ----
            mask_np = np.array(mask_pil.convert("L")).astype(np.float32) / 255.0

            # threshold with sensitivity
            thresh = np.clip(sensitivity, 0.01, 0.99)
            mask_np = np.clip((mask_np - (thresh - 0.5)) / (1.0 - thresh + 1e-6), 0.0, 1.0)

            if remove_holes:
                mask_np = remove_small_holes(mask_np)
            if remove_islands:
                mask_np = remove_small_islands(mask_np)
            if mask_offset != 0:
                mask_np = erode_expand_mask(mask_np, mask_offset)
            if mask_blur > 0:
                mask_np = smooth_mask(mask_np, mask_blur)

            # resize back to original
            mask_final = Image.fromarray((mask_np * 255).clip(0, 255).astype(np.uint8), mode="L")
            mask_final = mask_final.resize((orig_w, orig_h), Image.LANCZOS)

            # ---- foreground refinement ----
            if refine_foreground:
                pil_img = refine_foreground_colors(pil_img, mask_final)

            # ---- compose output ----
            out_img = apply_background(pil_img, mask_final, background)

            results_img.append(pil_to_tensor(out_img))
            results_mask.append(pil_mask_to_tensor(mask_final))

        return (torch.stack(results_img), torch.stack(results_mask))


def _birefnet_variant(model_str: str) -> str:
    if "HR" in model_str:
        return "HR"
    if "dynamic" in model_str:
        return "dynamic"
    if "matting" in model_str:
        return "matting"
    return "general"

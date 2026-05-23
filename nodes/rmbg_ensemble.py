"""
NkVasi_RMBG_Ensemble — multi-model ensemble for maximum precision.
Runs 2-3 SOTA models and merges masks via weighted averaging.
This is the "professional quality" node that beats every single-model approach.
"""
import torch
import numpy as np
from PIL import Image

from ..utils.model_loader import load_birefnet, load_ben2, load_rmbg2
from ..utils.image_utils import (
    tensor_to_pil, pil_to_tensor,
    pil_mask_to_tensor, apply_background,
    refine_foreground_colors,
)
from ..utils.mask_ops import (
    smooth_mask, erode_expand_mask,
    remove_small_holes, remove_small_islands,
)

MERGE_MODES = [
    "weighted_avg (recommended)",
    "intersection (strict, fewer artifacts)",
    "union (keep more detail)",
    "max (aggressive)",
]

BACKGROUNDS = ["alpha", "white", "black", "green", "red", "blue", "checkerboard"]


def _mask_pil_to_np(mask_pil: Image.Image, target_size: int) -> np.ndarray:
    """
    Convert any PIL mask to a normalised float32 numpy array
    at a fixed (target_size x target_size) resolution.
    This guarantees all masks have identical shapes before merging,
    regardless of what internal resolution each model used.
    """
    resized = mask_pil.convert("L").resize((target_size, target_size), Image.LANCZOS)
    return np.array(resized).astype(np.float32) / 255.0


class NkVasi_RMBG_Ensemble:
    """Multi-model ensemble for near-perfect background removal."""

    CATEGORY = "🎭 nkVasi/Background Removal"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "remove_background_ensemble"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "use_birefnet_hr": ("BOOLEAN", {"default": True}),
                "use_birefnet_matting": ("BOOLEAN", {"default": True}),
                "use_ben2": ("BOOLEAN", {"default": True}),
                "use_rmbg2": ("BOOLEAN", {"default": False}),
                "merge_mode": (MERGE_MODES, {"default": "weighted_avg (recommended)"}),
                "process_resolution": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 128}),
                "background": (BACKGROUNDS, {"default": "alpha"}),
            },
            "optional": {
                "birefnet_hr_weight": ("FLOAT", {"default": 0.40, "min": 0.0, "max": 1.0, "step": 0.05}),
                "birefnet_matting_weight": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "ben2_weight": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),
                "rmbg2_weight": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.05}),
                "mask_blur": ("INT", {"default": 1, "min": 0, "max": 32, "step": 1}),
                "mask_offset": ("INT", {"default": 0, "min": -20, "max": 20, "step": 1}),
                "sensitivity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "refine_foreground": ("BOOLEAN", {"default": True}),
                "fp16": ("BOOLEAN", {"default": True}),
            },
        }

    def remove_background_ensemble(
        self,
        image,
        use_birefnet_hr,
        use_birefnet_matting,
        use_ben2,
        use_rmbg2,
        merge_mode,
        process_resolution,
        background,
        birefnet_hr_weight=0.40,
        birefnet_matting_weight=0.35,
        ben2_weight=0.25,
        rmbg2_weight=0.20,
        mask_blur=1,
        mask_offset=0,
        sensitivity=0.5,
        refine_foreground=True,
        fp16=True,
    ):
        results_img = []
        results_mask = []

        # Merge resolution: use process_resolution as the canonical grid.
        # Every model's output is resized to this before blending —
        # this prevents numpy broadcast errors when models return
        # different native sizes (e.g. BiRefNet-HR=2048, BEN2=1024).
        merge_res = process_resolution

        for i in range(image.shape[0]):
            pil_img = tensor_to_pil(image[i])
            orig_w, orig_h = pil_img.size

            masks = []
            weights = []

            if use_birefnet_hr:
                m = load_birefnet("HR").infer(pil_img, process_resolution, fp16)
                masks.append(_mask_pil_to_np(m, merge_res))
                weights.append(birefnet_hr_weight)

            if use_birefnet_matting:
                m = load_birefnet("matting").infer(pil_img, process_resolution, fp16)
                masks.append(_mask_pil_to_np(m, merge_res))
                weights.append(birefnet_matting_weight)

            if use_ben2:
                m = load_ben2().infer(pil_img, process_resolution)
                masks.append(_mask_pil_to_np(m, merge_res))
                weights.append(ben2_weight)

            if use_rmbg2:
                m = load_rmbg2().infer(pil_img, process_resolution, fp16)
                masks.append(_mask_pil_to_np(m, merge_res))
                weights.append(rmbg2_weight)

            if not masks:
                raise ValueError("At least one model must be enabled in the Ensemble node.")

            # ---- merge (all masks are now merge_res x merge_res) ----
            mode = merge_mode.split(" ")[0]
            if mode == "weighted_avg":
                total_w = sum(weights)
                merged = sum(m * w for m, w in zip(masks, weights)) / total_w
            elif mode == "intersection":
                merged = masks[0]
                for m in masks[1:]:
                    merged = np.minimum(merged, m)
            elif mode == "union":
                merged = masks[0]
                for m in masks[1:]:
                    merged = np.maximum(merged, m)
            elif mode == "max":
                merged = np.maximum.reduce(masks)
            else:
                merged = np.mean(masks, axis=0)

            # ---- post-process ----
            thresh = np.clip(sensitivity, 0.01, 0.99)
            merged = np.clip((merged - (thresh - 0.5)) / (1.0 - thresh + 1e-6), 0.0, 1.0)

            merged = remove_small_holes(merged)
            merged = remove_small_islands(merged)

            if mask_offset != 0:
                merged = erode_expand_mask(merged, mask_offset)
            if mask_blur > 0:
                merged = smooth_mask(merged, mask_blur)

            # resize merged mask back to original image dimensions
            mask_final = Image.fromarray(
                (merged * 255).clip(0, 255).astype(np.uint8), mode="L"
            )
            mask_final = mask_final.resize((orig_w, orig_h), Image.LANCZOS)

            if refine_foreground:
                pil_img = refine_foreground_colors(pil_img, mask_final)

            out_img = apply_background(pil_img, mask_final, background)

            results_img.append(pil_to_tensor(out_img))
            results_mask.append(pil_mask_to_tensor(mask_final))

        return (torch.stack(results_img), torch.stack(results_mask))

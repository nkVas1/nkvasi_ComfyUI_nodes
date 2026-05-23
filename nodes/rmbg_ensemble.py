"""
NkVasi_RMBG_Ensemble — multi-model ensemble for maximum precision.

Merge modes:
  weighted_avg          — smooth weighted average
  consensus             — mean × √(fraction_models_agree); best all-around
  soft_intersection     — geometric mean; penalises uncertain pixels
  intersection          — min across all models (strictest)
  union                 — max across all models (most inclusive)

Post-processing pipeline:
  1.  Merge
  2.  Sensitivity soft-remap (no binarisation)
  3.  Guided filter — edge-aware sub-pixel alpha from image gradients
  4a. hair_bg_island_removal (hair_mode) — colour-gated BG patch removal
   b. soft_remove_islands (non-hair mode)
  5.  soft_remove_islands for FG artefacts ALWAYS — removes stray blobs
      outside the main subject regardless of hair_mode
  6.  soft_remove_holes — fills small interior holes
  7.  Geometric ops (offset, feather, blur)
  8.  Resize to original dimensions
  9.  Foreground decontamination (alpha mode only)
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
    smooth_mask, erode_expand_mask, guided_filter_mask,
    soft_remove_holes, soft_remove_islands,
    hair_bg_island_removal, feather_mask,
)

MERGE_MODES = [
    "weighted_avg",
    "consensus (recommended)",
    "soft_intersection",
    "intersection (strictest)",
    "union (most inclusive)",
]

BACKGROUNDS = ["alpha", "white", "black", "green", "red", "blue", "checkerboard"]


def _mask_pil_to_np(mask_pil: Image.Image, target_size: int) -> np.ndarray:
    resized = mask_pil.convert("L").resize((target_size, target_size), Image.LANCZOS)
    return np.array(resized).astype(np.float32) / 255.0


def _merge_masks(masks: list, weights: list, mode: str) -> np.ndarray:
    key = mode.split(" ")[0].lower()

    if key == "weighted_avg":
        total_w = sum(weights) or 1.0
        return np.clip(sum(m * w for m, w in zip(masks, weights)) / total_w, 0, 1)

    if key == "consensus":
        avg   = np.mean(masks, axis=0)
        agree = np.mean([(m > 0.5).astype(np.float32) for m in masks], axis=0)
        return np.clip(avg * (agree ** 0.5), 0, 1)

    if key == "soft_intersection":
        stacked = np.stack(masks, axis=0).clip(1e-6, 1.0)
        return np.exp(np.mean(np.log(stacked), axis=0)).clip(0, 1)

    if key == "intersection":
        result = masks[0].copy()
        for m in masks[1:]:
            result = np.minimum(result, m)
        return result

    if key == "union":
        result = masks[0].copy()
        for m in masks[1:]:
            result = np.maximum(result, m)
        return result

    return np.mean(masks, axis=0)


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
                "use_birefnet_hr":      ("BOOLEAN", {"default": True}),
                "use_birefnet_matting": ("BOOLEAN", {"default": True}),
                "use_ben2":             ("BOOLEAN", {"default": True}),
                "use_rmbg2":            ("BOOLEAN", {"default": False}),
                "merge_mode":         (MERGE_MODES, {"default": "consensus (recommended)"}),
                # 2048 — maximum quality; use 1536 if VRAM is limited
                "process_resolution": ("INT", {"default": 2048, "min": 512, "max": 2048, "step": 128}),
                "background":         (BACKGROUNDS, {"default": "alpha"}),
                # hair_mode: adds colour-gated BG island removal between hair strands
                # ON  — portraits, hair, fur: preserves strand detail, removes BG patches
                # OFF — products, hard objects: standard FG island removal
                # In BOTH modes, stray floating FG artefacts are always removed.
                "hair_mode":          ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "birefnet_hr_weight":      ("FLOAT", {"default": 0.40, "min": 0.0, "max": 1.0, "step": 0.05}),
                "birefnet_matting_weight": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "ben2_weight":             ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),
                "rmbg2_weight":            ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.05}),
                "mask_blur":    ("INT",   {"default": 1,    "min": 0,    "max": 32,    "step": 1}),
                "mask_offset":  ("INT",   {"default": 0,    "min": -20,  "max": 20,    "step": 1}),
                "feather_edges":("INT",   {"default": 2,    "min": 0,    "max": 32,    "step": 1}),
                # sensitivity: lower = keep more (looser); higher = cut more (stricter)
                "sensitivity":  ("FLOAT", {"default": 0.45, "min": 0.0,  "max": 1.0,  "step": 0.01}),
                # hair_mode params — only active when hair_mode=True
                # island_size: max BG patch area (px at process_resolution) to remove
                "island_size":  ("INT",   {"default": 2000, "min": 100,  "max": 30000, "step": 100}),
                # color_thresh: max colour distance to keep as real inter-strand gap
                # lower = more aggressive BG removal; 0.10-0.20 is the sweet spot
                "color_thresh": ("FLOAT", {"default": 0.13, "min": 0.0,  "max": 1.0,  "step": 0.01}),
                # artefact_size: max area of stray FG blobs to remove (both modes)
                "artefact_size":("INT",   {"default": 600,  "min": 0,    "max": 10000, "step": 50}),
                "refine_foreground": ("BOOLEAN", {"default": True}),
                "fp16":              ("BOOLEAN", {"default": True}),
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
        hair_mode=True,
        birefnet_hr_weight=0.40,
        birefnet_matting_weight=0.35,
        ben2_weight=0.25,
        rmbg2_weight=0.20,
        mask_blur=1,
        mask_offset=0,
        feather_edges=2,
        sensitivity=0.45,
        island_size=2000,
        color_thresh=0.13,
        artefact_size=600,
        refine_foreground=True,
        fp16=True,
    ):
        results_img  = []
        results_mask = []
        merge_res    = process_resolution

        for i in range(image.shape[0]):
            pil_img        = tensor_to_pil(image[i])
            orig_w, orig_h = pil_img.size

            pil_guide = pil_img.resize((merge_res, merge_res), Image.LANCZOS)
            guide_np  = np.array(pil_guide).astype(np.float32) / 255.0

            masks, weights = [], []

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
                raise ValueError("[nkVasi] At least one model must be enabled.")

            # === STEP 1: Merge ===
            merged = _merge_masks(masks, weights, merge_mode)

            # === STEP 2: Sensitivity soft-remap (no binarisation) ===
            thresh = np.clip(sensitivity, 0.01, 0.99)
            merged = np.clip(
                (merged - (thresh - 0.5)) / (1.0 - thresh + 0.15),
                0.0, 1.0,
            )

            # === STEP 3: Guided filter ===
            merged = guided_filter_mask(merged, guide_np, radius=7, eps=3e-4)

            # === STEP 4: BG island removal ===
            if hair_mode:
                # Colour-gated removal of BG patches inside the hair region.
                # detect_thresh=0.25 catches semi-opaque BG from guided filter.
                merged = hair_bg_island_removal(
                    merged, guide_np,
                    max_island_size=island_size,
                    color_thresh=color_thresh,
                    detect_thresh=0.25,
                )
            else:
                # For hard objects: standard removal of small FG islands
                merged = soft_remove_islands(merged, min_island_size=400)

            # === STEP 5: Remove stray FG artefacts (ALWAYS, both modes) ===
            # This is separate from BG island removal above.
            # Removes floating FG blobs outside the main subject
            # (e.g. detached hair tips, reflection artefacts).
            # min_island_size uses artefact_size, which is deliberately smaller
            # than the island_size used for BG patches, to avoid cutting hair.
            if artefact_size > 0:
                merged = soft_remove_islands(merged, min_island_size=artefact_size)

            # === STEP 6: Fill small interior holes ===
            merged = soft_remove_holes(merged, min_hole_size=600)

            # === STEP 7: Geometric ops ===
            if mask_offset != 0:
                merged = erode_expand_mask(merged, mask_offset)
            if feather_edges > 0:
                merged = feather_mask(merged, feather_edges)
            if mask_blur > 0:
                merged = smooth_mask(merged, mask_blur)

            # === STEP 8: Resize to original ===
            mask_final_pil = Image.fromarray(
                (merged * 255).clip(0, 255).astype(np.uint8), mode="L"
            ).resize((orig_w, orig_h), Image.LANCZOS)

            # === STEP 9: Foreground decontamination (alpha only) ===
            # Not needed for solid backgrounds — the composited colour covers any bleed.
            if refine_foreground and background == "alpha":
                pil_img = refine_foreground_colors(pil_img, mask_final_pil, strength=0.60)

            out_img = apply_background(pil_img, mask_final_pil, background)

            results_img.append(pil_to_tensor(out_img))
            results_mask.append(pil_mask_to_tensor(mask_final_pil))

        return (torch.stack(results_img), torch.stack(results_mask))

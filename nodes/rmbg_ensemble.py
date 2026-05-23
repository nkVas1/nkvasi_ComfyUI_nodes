"""
NkVasi_RMBG_Ensemble v2.1 — multi-model ensemble for maximum precision.

New in v2.1:
  - Updated defaults to best-tested values (soft_intersection, trimap 4/24).
  - bg_color_suppress: automatically detects uniform backgrounds (blue sky,
    studio colour, etc.) and aggressively removes residual BG colour fringe
    from alpha edges — without touching subject areas with similar colour.

New in v2.0:
  - Confidence Map output: per-pixel agreement between models exported as MASK.
  - Adaptive Trimap mode.
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
    adaptive_bg_cleanup, feather_mask,
)
from ..utils.confidence import (
    confidence_weighted_merge,
    build_adaptive_trimap, confidence_to_pil,
)
from ..utils.bg_color_suppress import auto_bg_color_suppress

MERGE_MODES = [
    "soft_intersection",
    "consensus (recommended)",
    "weighted_avg",
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
    """
    v2.1: Best-tested defaults + bg_color_suppress for uniform backgrounds.
    Connect confidence_map → MattingRefine for optimal hair detail.
    """

    CATEGORY = "🎭 nkVasi/Background Removal"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("image", "mask", "confidence_map")
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
                "merge_mode":           (MERGE_MODES, {"default": "soft_intersection"}),
                "process_resolution":   ("INT",  {"default": 2048, "min": 512, "max": 2048, "step": 128}),
                "background":           (BACKGROUNDS, {"default": "alpha"}),
                "hair_mode":            ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "birefnet_hr_weight":      ("FLOAT", {"default": 0.40, "min": 0.0, "max": 1.0, "step": 0.05}),
                "birefnet_matting_weight": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "ben2_weight":             ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),
                "rmbg2_weight":            ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.05}),
                "mask_blur":         ("INT",   {"default": 1,    "min": 0,   "max": 32,   "step": 1}),
                "mask_offset":       ("INT",   {"default": 0,    "min": -20, "max": 20,   "step": 1}),
                "feather_edges":     ("INT",   {"default": 2,    "min": 0,   "max": 32,   "step": 1}),
                "sensitivity":       ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0,  "step": 0.01}),
                "bg_cleanup_thresh": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 0.5,  "step": 0.01}),
                "artefact_size":     ("INT",   {"default": 600,  "min": 0,   "max": 10000, "step": 50}),
                "refine_foreground": ("BOOLEAN", {"default": True}),
                "fp16":              ("BOOLEAN", {"default": True}),
                # --- Adaptive trimap ---
                "adaptive_trimap": ("BOOLEAN", {"default": True,
                                     "tooltip": "Widen unknown band near uncertain/complex edges"}),
                "trimap_min_px":   ("INT", {"default": 4,  "min": 2, "max": 20, "step": 1,
                                     "tooltip": "Min band width (confident flat areas)"}),
                "trimap_max_px":   ("INT", {"default": 24, "min": 6, "max": 60, "step": 2,
                                     "tooltip": "Max band width (hair / uncertain edges)"}),
                # --- BG color suppression ---
                "bg_color_suppress":      ("BOOLEAN", {"default": True,
                                            "tooltip": "Auto-detect uniform BG colour (sky, studio) and remove colour fringe from edges"}),
                "bg_suppress_strength":   ("FLOAT",   {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05,
                                            "tooltip": "Suppression intensity. 0=off, 1=maximum"}),
                "bg_suppress_fg_overlap": ("FLOAT",   {"default": 0.25, "min": 0.05, "max": 0.60, "step": 0.05,
                                            "tooltip": "Safety: skip suppression if >X fraction of subject contains BG colour"}),
                "bg_hue_std_thresh":      ("FLOAT",   {"default": 0.06, "min": 0.02, "max": 0.20, "step": 0.01,
                                            "tooltip": "Max hue stddev to consider BG uniform. Lower = stricter detection"}),
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
        bg_cleanup_thresh=0.10,
        artefact_size=600,
        refine_foreground=True,
        fp16=True,
        adaptive_trimap=True,
        trimap_min_px=4,
        trimap_max_px=24,
        bg_color_suppress=True,
        bg_suppress_strength=0.65,
        bg_suppress_fg_overlap=0.25,
        bg_hue_std_thresh=0.06,
    ):
        results_img  = []
        results_mask = []
        results_conf = []
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

            # ---- Merge + Confidence Map ----
            merged, confidence = confidence_weighted_merge(masks, weights)
            if not merge_mode.startswith("weighted_avg"):
                merged = _merge_masks(masks, weights, merge_mode)

            # ---- Sensitivity remap ----
            thresh = np.clip(sensitivity, 0.01, 0.99)
            merged = np.clip(
                (merged - (thresh - 0.5)) / (1.0 - thresh + 0.15), 0.0, 1.0)

            # ---- Guided filter ----
            merged = guided_filter_mask(merged, guide_np, radius=7, eps=3e-4)

            # ---- BG cleanup ----
            if hair_mode:
                merged = adaptive_bg_cleanup(
                    merged, guide_np,
                    global_thresh=bg_cleanup_thresh, local_window=31)
            else:
                merged = soft_remove_islands(merged, min_island_size=400)

            if artefact_size > 0:
                merged = soft_remove_islands(merged, min_island_size=artefact_size)
            merged = soft_remove_holes(merged, min_hole_size=600)

            # ---- BG Color Suppression (uniform sky / studio BG) ----
            if bg_color_suppress and bg_suppress_strength > 0.0:
                merged, _bg_info = auto_bg_color_suppress(
                    merged, guide_np,
                    strength=bg_suppress_strength,
                    max_fg_overlap=bg_suppress_fg_overlap,
                    lock_bg=bg_cleanup_thresh,
                    lock_fg=0.92,
                    hue_std_thresh=bg_hue_std_thresh,
                )

            # ---- Adaptive Trimap baked into confidence output ----
            if adaptive_trimap:
                _trimap = build_adaptive_trimap(
                    merged, confidence,
                    min_band_px=trimap_min_px,
                    max_band_px=trimap_max_px,
                )
                conf_out = _trimap.astype(np.float32) / 255.0
            else:
                conf_out = confidence

            # ---- Geometric ops ----
            if mask_offset != 0:
                merged = erode_expand_mask(merged, mask_offset)
            if feather_edges > 0:
                merged = feather_mask(merged, feather_edges)
            if mask_blur > 0:
                merged = smooth_mask(merged, mask_blur)

            # ---- Resize to original ----
            mask_final_pil = Image.fromarray(
                (merged * 255).clip(0, 255).astype(np.uint8), mode="L"
            ).resize((orig_w, orig_h), Image.LANCZOS)

            conf_final_pil = confidence_to_pil(conf_out).resize(
                (orig_w, orig_h), Image.LANCZOS)

            if refine_foreground and background == "alpha":
                pil_img = refine_foreground_colors(pil_img, mask_final_pil, strength=0.60)

            out_img = apply_background(pil_img, mask_final_pil, background)

            results_img.append(pil_to_tensor(out_img))
            results_mask.append(pil_mask_to_tensor(mask_final_pil))
            results_conf.append(pil_mask_to_tensor(conf_final_pil))

        return (
            torch.stack(results_img),
            torch.stack(results_mask),
            torch.stack(results_conf),
        )

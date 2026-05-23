"""
NkVasi_RMBG_Ensemble — multi-model ensemble for maximum precision.

Post-processing pipeline (in order):
  1. Weighted merge of all model masks
  2. Sensitivity threshold (soft remap, NOT binary clamp)
  3. Guided filter — edge-aware soft mask using original image as guide
  4. hair_bg_island_removal — removes background patches between strands
     by color distance analysis (hair_mode only)
  5. soft_remove_holes — fills small opaque holes, preserves soft edges
  6. soft_remove_islands — removes stray FG blobs (non-hair mode)
  7. erode/expand, feather, blur
  8. Resize to original dimensions
  9. Foreground color decontamination

Result: a SOFT float32 alpha mask with semi-transparent hair edges.
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
    "weighted_avg (recommended)",
    "intersection (strict, fewer artifacts)",
    "union (keep more detail)",
    "max (aggressive)",
]

BACKGROUNDS = ["alpha", "white", "black", "green", "red", "blue", "checkerboard"]


def _mask_pil_to_np(mask_pil: Image.Image, target_size: int) -> np.ndarray:
    """
    Convert PIL mask -> float32 numpy at target_size × target_size.
    All masks normalised to the same resolution before merging.
    """
    resized = mask_pil.convert("L").resize((target_size, target_size), Image.LANCZOS)
    return np.array(resized).astype(np.float32) / 255.0


class NkVasi_RMBG_Ensemble:
    """Multi-model ensemble for near-perfect background removal with soft alpha."""

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
                "merge_mode": (MERGE_MODES, {"default": "weighted_avg (recommended)"}),
                # 1536px — BiRefNet-HR sweet spot for portraits; 2048 = max quality
                "process_resolution": ("INT", {"default": 1536, "min": 512, "max": 2048, "step": 128}),
                "background": (BACKGROUNDS, {"default": "alpha"}),
                # hair_mode: guided filter + colour-aware BG island removal
                # critical for portraits; disable for product shots / solid objects
                "hair_mode": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "birefnet_hr_weight":      ("FLOAT", {"default": 0.40, "min": 0.0, "max": 1.0, "step": 0.05}),
                "birefnet_matting_weight": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "ben2_weight":             ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),
                "rmbg2_weight":            ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.05}),
                "mask_blur":    ("INT",   {"default": 1,   "min": 0,    "max": 32, "step": 1}),
                "mask_offset":  ("INT",   {"default": 0,   "min": -20,  "max": 20, "step": 1}),
                "feather_edges":("INT",   {"default": 2,   "min": 0,    "max": 32, "step": 1}),
                # sensitivity: 0.5 = neutral; lower = keep more (looser edges);
                # higher = cut more (stricter, fewer semi-transparent halos)
                "sensitivity":  ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01}),
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
        refine_foreground=True,
        fp16=True,
    ):
        results_img  = []
        results_mask = []
        merge_res    = process_resolution

        for i in range(image.shape[0]):
            pil_img        = tensor_to_pil(image[i])
            orig_w, orig_h = pil_img.size

            # ---- prepare guide image at merge resolution ----
            # computed once, reused for guided filter and island removal
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

            # ================================================================
            # STEP 1: Merge
            # ================================================================
            mode = merge_mode.split(" ")[0]
            if mode == "weighted_avg":
                total_w = sum(weights)
                merged  = sum(m * w for m, w in zip(masks, weights)) / total_w
            elif mode == "intersection":
                merged = masks[0]
                for m in masks[1:]: merged = np.minimum(merged, m)
            elif mode == "union":
                merged = masks[0]
                for m in masks[1:]: merged = np.maximum(merged, m)
            elif mode == "max":
                merged = np.maximum.reduce(masks)
            else:
                merged = np.mean(masks, axis=0)

            # ================================================================
            # STEP 2: Sensitivity — soft remap, NOT binary clamp
            # Maps the confidence range so that:
            #   - pixels the models agree on stay near 0 or 1
            #   - ambiguous edge pixels stay semi-transparent (0.1–0.9)
            # ================================================================
            thresh = np.clip(sensitivity, 0.01, 0.99)
            merged = np.clip(
                (merged - (thresh - 0.5)) / (1.0 - thresh + 0.15),
                0.0, 1.0,
            )

            # ================================================================
            # STEP 3: Guided filter — edge-aware soft mask refinement
            # Must run BEFORE any morphological step to build a proper
            # semi-transparent alpha from image gradients.
            # ================================================================
            merged = guided_filter_mask(merged, guide_np, radius=7, eps=3e-4)

            # ================================================================
            # STEP 4a: Hair BG island removal (hair_mode only)
            # Finds small background patches between strands and removes those
            # whose colour differs from neighbouring hair pixels.
            # Uses the soft mask so removed patches fade rather than hard-cut.
            # ================================================================
            if hair_mode:
                merged = hair_bg_island_removal(
                    merged, guide_np,
                    max_island_size=1200,
                    color_thresh=0.20,
                )
            else:
                # Standard island removal for non-portrait subjects
                merged = soft_remove_islands(merged, min_island_size=400)

            # ================================================================
            # STEP 4b: Fill small interior holes (both modes)
            # Only fills holes fully inside the FG — never touches hair edges.
            # ================================================================
            merged = soft_remove_holes(merged, min_hole_size=600)

            # ================================================================
            # STEP 5: Geometric adjustments
            # ================================================================
            if mask_offset != 0:
                merged = erode_expand_mask(merged, mask_offset)
            if feather_edges > 0:
                merged = feather_mask(merged, feather_edges)
            if mask_blur > 0:
                merged = smooth_mask(merged, mask_blur)

            # ================================================================
            # STEP 6: Resize back to original image dimensions
            # Use LANCZOS to preserve soft sub-pixel alpha values.
            # ================================================================
            mask_final_pil = Image.fromarray(
                (merged * 255).clip(0, 255).astype(np.uint8), mode="L"
            ).resize((orig_w, orig_h), Image.LANCZOS)

            # ================================================================
            # STEP 7: Foreground color decontamination
            # Removes background color bleed on semi-transparent edges.
            # ================================================================
            if refine_foreground:
                pil_img = refine_foreground_colors(pil_img, mask_final_pil, strength=0.60)

            out_img = apply_background(pil_img, mask_final_pil, background)

            results_img.append(pil_to_tensor(out_img))
            results_mask.append(pil_mask_to_tensor(mask_final_pil))

        return (torch.stack(results_img), torch.stack(results_mask))

"""
NkVasi_MattingRefine v2 — commercial-grade alpha matting refinement.

Philosophy (v2 rewrite):
  The ONLY thing that should be changed in a good model-generated mask are
  the edge pixels — the 5-20 px transition band between definite FG and
  definite BG.  Everything else is left exactly as the ensemble produced it.

Pipeline:
  1. LOCK solid zones: pixels >lock_fg (FG core) and <lock_bg (BG core)
     are frozen — they will NEVER be modified regardless of other settings.
  2. DUAL-RADIUS guided filter blending in the narrow edge band:
       a) Coarse pass (r=coarse_r, eps=1e-4): smooths the general edge shape
          and removes jagged staircase artefacts.
       b) Fine   pass (r=fine_r,   eps=1e-8): snaps the mask to image
          gradients, recovering sub-pixel hair strands and eyelashes without
          pulling in image noise (high eps kills noise sensitivity).
       Both passes are blended: result = lerp(coarse, fine, fine_blend)
       and then lerp(original, blended, edge_strength) in the edge zone.
  3. BACKGROUND fringe suppression: pixels just outside the edge band
     (alpha 0.0 – lock_bg) are gently pushed toward zero only where
     the guide image colour is dissimilar to the nearest FG colour.
     This removes semi-transparent BG fringe without cutting hair tips.
  4. Soft hole fill + island removal on the final mask.

NO high-pass / detail_recovery (adds JPEG noise).
NO trimap colour-sampling matting (unstable at wide bands, slow).
Pure cv2 + numpy — no external deps.
"""
import torch
import numpy as np
from PIL import Image

from ..utils.mask_ops import (
    guided_filter_mask,
    soft_remove_islands,
    soft_remove_holes,
    smooth_mask,
)
from ..utils.image_utils import tensor_to_pil, pil_mask_to_tensor


class NkVasi_MattingRefine:
    """
    Narrow-band dual-radius guided filter refinement.
    Plug in after Remove BG Ensemble (or any BG removal node).
    """

    CATEGORY = "🎭 nkVasi/Background Removal"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "refine"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask":  ("MASK",),
            },
            "optional": {
                # --- Edge band controls ---
                # Pixels with alpha > lock_fg are FG core: never modified.
                # Pixels with alpha < lock_bg are BG core: never modified.
                # The band between lock_bg and lock_fg is where refinement happens.
                "lock_fg":        ("FLOAT", {"default": 0.90, "min": 0.50, "max": 1.00, "step": 0.01}),
                "lock_bg":        ("FLOAT", {"default": 0.05, "min": 0.00, "max": 0.40, "step": 0.01}),

                # --- Guided filter passes ---
                # Coarse pass: smooths jagged edges, radius in px
                "coarse_radius":  ("INT",   {"default": 20,   "min": 4,   "max": 80,   "step": 2}),
                # Fine pass: snaps to image gradients (sub-pixel hair detail)
                "fine_radius":    ("INT",   {"default": 3,    "min": 1,   "max": 12,   "step": 1}),
                # 0.0 = use only coarse pass, 1.0 = use only fine pass
                "fine_blend":     ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0,  "step": 0.05}),
                # Overall strength of the refinement in the edge band
                # 0.0 = no change, 1.0 = fully replace with guided result
                "edge_strength":  ("FLOAT", {"default": 0.80, "min": 0.0, "max": 1.0,  "step": 0.05}),

                # --- BG fringe suppression ---
                # Gently suppresses semi-transparent BG fringe just outside the edge band.
                # 0.0 = off.  0.3-0.5 recommended for portraits on blurred/hazy BG.
                "fringe_suppress":("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0,  "step": 0.05}),

                # --- Cleanup ---
                "remove_artefacts":("BOOLEAN", {"default": True}),
                "artefact_size":   ("INT",   {"default": 400, "min": 0,   "max": 5000, "step": 50}),
                "fill_holes":      ("BOOLEAN", {"default": True}),
            },
        }

    def refine(
        self,
        image,
        mask,
        lock_fg=0.90,
        lock_bg=0.05,
        coarse_radius=20,
        fine_radius=3,
        fine_blend=0.55,
        edge_strength=0.80,
        fringe_suppress=0.30,
        remove_artefacts=True,
        artefact_size=400,
        fill_holes=True,
    ):
        out_masks = []

        for i in range(mask.shape[0]):
            img_idx  = min(i, image.shape[0] - 1)
            pil_img  = tensor_to_pil(image[img_idx])

            m_np = mask[i].cpu().numpy().astype(np.float32)
            h, w = m_np.shape[:2]

            # Resize guide to mask resolution
            pil_guide = pil_img.resize((w, h), Image.LANCZOS)
            guide_np  = np.array(pil_guide).astype(np.float32) / 255.0

            m_np = _narrow_band_refine(
                m_np, guide_np,
                lock_fg=lock_fg,
                lock_bg=lock_bg,
                coarse_radius=coarse_radius,
                fine_radius=fine_radius,
                fine_blend=fine_blend,
                edge_strength=edge_strength,
                fringe_suppress=fringe_suppress,
            )

            if remove_artefacts and artefact_size > 0:
                m_np = soft_remove_islands(m_np, min_island_size=artefact_size)
            if fill_holes:
                m_np = soft_remove_holes(m_np, min_hole_size=400)

            pil_m = Image.fromarray(
                (m_np * 255).clip(0, 255).astype(np.uint8), mode="L")
            out_masks.append(pil_mask_to_tensor(pil_m))

        return (torch.stack(out_masks),)


# ---------------------------------------------------------------------------
# Core refinement logic
# ---------------------------------------------------------------------------

def _narrow_band_refine(
    mask: np.ndarray,
    guide: np.ndarray,
    lock_fg: float,
    lock_bg: float,
    coarse_radius: int,
    fine_radius: int,
    fine_blend: float,
    edge_strength: float,
    fringe_suppress: float,
) -> np.ndarray:
    """
    Dual-radius guided filter refinement strictly within the edge band.

    Key invariant: pixels outside [lock_bg, lock_fg] are NEVER modified.
    """
    import cv2

    original = mask.copy()

    # ------------------------------------------------------------------ #
    # 1.  Coarse pass: smooth edge shape (removes jagged staircase)       #
    # ------------------------------------------------------------------ #
    coarse = guided_filter_mask(
        mask, guide,
        radius=coarse_radius,
        eps=1e-4,          # relatively permissive — shape, not detail
    )

    # ------------------------------------------------------------------ #
    # 2.  Fine pass: snap to image gradients for hair/eyelash detail      #
    #     Very low eps = filter stays close to image edges                #
    # ------------------------------------------------------------------ #
    fine = guided_filter_mask(
        mask, guide,
        radius=fine_radius,
        eps=1e-8,          # extremely tight — follows every edge gradient
    )

    # ------------------------------------------------------------------ #
    # 3.  Blend coarse + fine                                             #
    # ------------------------------------------------------------------ #
    blended = (1.0 - fine_blend) * coarse + fine_blend * fine
    blended = np.clip(blended, 0.0, 1.0)

    # ------------------------------------------------------------------ #
    # 4.  Apply ONLY in the edge band, with edge_strength control         #
    # ------------------------------------------------------------------ #
    edge_band = (original > lock_bg) & (original < lock_fg)

    result = original.copy()
    result[edge_band] = (
        original[edge_band] * (1.0 - edge_strength)
        + blended[edge_band] * edge_strength
    )

    # ------------------------------------------------------------------ #
    # 5.  Re-lock FG core and BG core exactly to their original values    #
    #     (guided filter can slightly shift values; we undo that here)    #
    # ------------------------------------------------------------------ #
    result[original >= lock_fg] = original[original >= lock_fg]
    result[original <= lock_bg] = original[original <= lock_bg]

    # ------------------------------------------------------------------ #
    # 6.  BG fringe suppression in the outer part of the edge band        #
    #     (alpha between lock_bg and 0.25)                                #
    # ------------------------------------------------------------------ #
    if fringe_suppress > 0.0:
        result = _suppress_bg_fringe(
            result, original, guide, lock_bg=lock_bg, strength=fringe_suppress)

    return np.clip(result, 0.0, 1.0)


def _suppress_bg_fringe(
    result: np.ndarray,
    original: np.ndarray,
    guide: np.ndarray,
    lock_bg: float,
    strength: float,
    outer_band_max: float = 0.25,
) -> np.ndarray:
    """
    Gently pushes near-transparent fringe pixels toward zero.

    Only operates in the outer sub-band: (lock_bg, outer_band_max).
    Uses local colour distance from the nearest definite-FG neighbourhood
    to decide how aggressively to suppress — pixels that look like the
    background are suppressed, pixels that look like hair are kept.
    """
    import cv2

    fringe_zone = (original > lock_bg) & (original < outer_band_max)
    if fringe_zone.sum() == 0:
        return result

    # Build a "local FG colour" reference by dilating the FG core and
    # sampling the guide image there
    fg_core     = (original >= 0.5).astype(np.uint8) * 255
    k           = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    fg_dilated  = cv2.dilate(fg_core, k)  # generous dilation

    guide_u8    = (guide * 255).clip(0, 255).astype(np.uint8)

    # Mean FG colour in a blurred reference map (cheap approximation)
    fg_ref_r    = cv2.GaussianBlur(
        np.where(fg_dilated[:, :, None] > 0, guide_u8.astype(np.float32),
                 np.nan * np.ones_like(guide_u8, dtype=np.float32)),
        (0, 0), sigmaX=40,
    )
    # NaN propagation is unreliable in cv2 blur — use masked mean instead
    fg_mask_f   = (fg_dilated > 0).astype(np.float32)
    fg_ref_r    = np.zeros_like(guide, dtype=np.float32)
    for c in range(3):
        ch      = guide[:, :, c] * fg_mask_f
        blurred = cv2.GaussianBlur(
            np.ascontiguousarray(ch), (0, 0), sigmaX=40)
        cnt_blur= cv2.GaussianBlur(
            np.ascontiguousarray(fg_mask_f), (0, 0), sigmaX=40)
        fg_ref_r[:, :, c] = np.where(cnt_blur > 0.01, blurred / (cnt_blur + 1e-6), 0.5)

    # Colour distance between fringe pixel and its FG reference
    col_dist    = np.linalg.norm(guide - fg_ref_r, axis=2)  # H×W, range 0..sqrt(3)
    col_dist_n  = np.clip(col_dist / (np.sqrt(3) * 0.5), 0.0, 1.0)  # normalise

    # Suppression weight: more distant from FG colour = more suppressed
    # Multiply by how far below outer_band_max the pixel is
    depth       = 1.0 - (original / outer_band_max)  # 1 near zero, 0 near outer_band_max
    suppress_w  = np.clip(col_dist_n * depth * strength, 0.0, 1.0)

    out         = result.copy()
    out[fringe_zone] = result[fringe_zone] * (1.0 - suppress_w[fringe_zone])
    return out

"""
NkVasi_MattingRefine v3.4

New in v3.4:
  - Updated defaults to best-tested values:
    lock_fg=0.75, lock_bg=0.05, channel_boost=1.00,
    trimap_band_px=4, trimap_max_px=20, edge_strength=0.50,
    coarse_radius=1, fine_blend=1.00, fringe_suppress=1.00
  - confidence_threshold default changed to 0.75

New in v3.3:
  - confidence input from RMBG_Ensemble restricts matting to uncertain zones.
  - Adaptive trimap: per-pixel band width from confidence.
"""
import torch
import numpy as np
from PIL import Image

from ..utils.mask_ops import (
    guided_filter_mask, build_trimap,
    soft_remove_islands, soft_remove_holes,
    smooth_mask,
)
from ..utils.confidence import build_adaptive_trimap
from ..utils.image_utils import tensor_to_pil, pil_mask_to_tensor, pil_to_tensor

try:
    from pymatting import estimate_alpha_cf, estimate_foreground_ml
    _PYMATTING_OK = True
except ImportError:
    _PYMATTING_OK = False


class NkVasi_MattingRefine:
    """
    v3.4: Best-tested defaults. Connect confidence_map from Ensemble to
    focus the matting engine only on uncertain edge zones.
    """

    CATEGORY = "🎭 nkVasi/Background Removal"
    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "image_decontaminated")
    FUNCTION = "refine"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask":  ("MASK",),
            },
            "optional": {
                "confidence":           ("MASK",  {"tooltip": "confidence_map from RMBG_Ensemble — restricts matting to uncertain zones"}),
                "confidence_threshold": ("FLOAT", {"default": 0.75, "min": 0.50, "max": 1.00, "step": 0.01,
                                                    "tooltip": "Pixels with confidence >= this are skipped by the matting engine"}),
                "lock_fg":          ("FLOAT", {"default": 0.75, "min": 0.50, "max": 1.00, "step": 0.01}),
                "lock_bg":          ("FLOAT", {"default": 0.05, "min": 0.00, "max": 0.40, "step": 0.01}),
                "channel_boost":    ("FLOAT", {"default": 1.00, "min": 0.0,  "max": 1.0,  "step": 0.05}),
                "trimap_band_px":   ("INT",   {"default": 4,   "min": 2,  "max": 40, "step": 2,
                                               "tooltip": "Base unknown-band radius. With confidence input = MIN width"}),
                "trimap_max_px":    ("INT",   {"default": 20,  "min": 6,  "max": 60, "step": 2,
                                               "tooltip": "Max band width in low-confidence zones"}),
                "edge_strength":    ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.05}),
                "coarse_radius":    ("INT",   {"default": 1,   "min": 1,  "max": 40, "step": 1}),
                "fine_radius":      ("INT",   {"default": 1,   "min": 1,  "max": 6,  "step": 1}),
                "fine_blend":       ("FLOAT", {"default": 1.00, "min": 0.0, "max": 1.0, "step": 0.05}),
                "smooth_edges":     ("INT",   {"default": 1,   "min": 0,  "max": 3,  "step": 1}),
                "fringe_suppress":  ("FLOAT", {"default": 1.00, "min": 0.0, "max": 1.0, "step": 0.05}),
                "decontaminate":    ("BOOLEAN", {"default": True}),
            },
        }

    def refine(
        self,
        image,
        mask,
        confidence=None,
        confidence_threshold=0.75,
        lock_fg=0.75,
        lock_bg=0.05,
        channel_boost=1.00,
        trimap_band_px=4,
        trimap_max_px=20,
        edge_strength=0.50,
        coarse_radius=1,
        fine_radius=1,
        fine_blend=1.00,
        smooth_edges=1,
        fringe_suppress=1.00,
        decontaminate=True,
    ):
        out_masks, out_images = [], []

        for i in range(mask.shape[0]):
            img_idx  = min(i, image.shape[0] - 1)
            pil_img  = tensor_to_pil(image[img_idx])
            original = mask[i].cpu().numpy().astype(np.float32)
            h, w     = original.shape[:2]

            pil_guide = pil_img.resize((w, h), Image.LANCZOS)
            guide_np  = np.array(pil_guide).astype(np.float32) / 255.0

            # ---- Resolve confidence map ----
            if confidence is not None:
                conf_idx = min(i, confidence.shape[0] - 1)
                conf_np  = confidence[conf_idx].cpu().numpy().astype(np.float32)
                if conf_np.shape != original.shape:
                    from PIL import Image as _PIL
                    conf_pil = _PIL.fromarray(
                        (conf_np * 255).clip(0, 255).astype(np.uint8), mode="L")
                    conf_np = np.array(
                        conf_pil.resize((w, h), Image.LANCZOS)
                    ).astype(np.float32) / 255.0
            else:
                conf_np = None

            m_np = original.copy()

            # ---- 1. Channel Boost ----
            if channel_boost > 0.0:
                m_np = _channel_boost(m_np, guide_np, lock_bg, lock_fg, channel_boost)

            # ---- 2. Build trimap ----
            if conf_np is not None:
                trimap = build_adaptive_trimap(
                    m_np, conf_np,
                    min_band_px=trimap_band_px,
                    max_band_px=trimap_max_px,
                )
            else:
                trimap = build_trimap(m_np, erosion_px=trimap_band_px, dilation_px=trimap_band_px)

            # ---- 3. Matting ----
            if _PYMATTING_OK:
                m_np = _pymatting_alpha(guide_np, m_np, trimap, edge_strength, conf_np, confidence_threshold)
            else:
                m_np = _guided_filter_alpha(
                    m_np, guide_np,
                    coarse_radius, fine_radius, fine_blend,
                    edge_strength, lock_bg, lock_fg,
                    conf_np, confidence_threshold,
                )

            # ---- 4. Re-lock FG/BG cores ----
            m_np[original >= lock_fg] = original[original >= lock_fg]
            m_np[original <= lock_bg] = original[original <= lock_bg]
            m_np = np.clip(m_np, 0.0, 1.0)

            # ---- 5. BG fringe suppression ----
            if fringe_suppress > 0.0:
                m_np = _suppress_bg_fringe(m_np, original, guide_np, lock_bg, fringe_suppress)

            # ---- 6. Edge-band smooth ----
            if smooth_edges > 0:
                edge_band = (original > lock_bg) & (original < lock_fg)
                blurred   = smooth_mask(m_np, smooth_edges)
                m_np[edge_band] = blurred[edge_band]
                m_np[original >= lock_fg] = original[original >= lock_fg]
                m_np[original <= lock_bg] = original[original <= lock_bg]

            m_np = np.clip(m_np, 0.0, 1.0)

            # ---- 7. Foreground decontamination ----
            pil_mask_out = Image.fromarray(
                (m_np * 255).clip(0, 255).astype(np.uint8), mode="L")
            pil_decontam = _decontaminate(pil_img, pil_mask_out) if decontaminate else pil_img

            out_masks.append(pil_mask_to_tensor(pil_mask_out))
            out_images.append(pil_to_tensor(pil_decontam))

        return (torch.stack(out_masks), torch.stack(out_images))


# ================================================================== #
# Channel Boost
# ================================================================== #

def _channel_boost(mask, guide, lock_bg, lock_fg, strength):
    import cv2
    edge_band = (mask > lock_bg) & (mask < lock_fg)
    if edge_band.sum() < 100:
        return mask
    stds    = [float(guide[:, :, c][edge_band].std()) for c in range(3)]
    best_c  = int(np.argmax(stds))
    ch_u8   = (guide[:, :, best_c] * 255).clip(0, 255).astype(np.uint8)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ch_eq   = clahe.apply(ch_u8).astype(np.float32) / 255.0
    grad_x  = cv2.Sobel(ch_u8, cv2.CV_32F, 1, 0, ksize=3)
    grad_y  = cv2.Sobel(ch_u8, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag= np.sqrt(grad_x**2 + grad_y**2)
    grad_mag= grad_mag / (grad_mag.max() + 1e-6)
    fg_core = mask >= lock_fg
    bg_core = mask <= lock_bg
    if fg_core.sum() > 0 and bg_core.sum() > 0 and float(ch_eq[fg_core].mean()) < float(ch_eq[bg_core].mean()):
        ch_signal = 1.0 - ch_eq
    else:
        ch_signal = ch_eq
    blend_w = np.clip((grad_mag ** 1.5) * strength * 0.5, 0.0, 0.35)
    result  = mask.copy()
    result[edge_band] = (
        mask[edge_band] * (1.0 - blend_w[edge_band])
        + ch_signal[edge_band] * blend_w[edge_band]
    )
    return np.clip(result, 0.0, 1.0)


# ================================================================== #
# Pymatting Closed-Form Alpha  (confidence-gated)
# ================================================================== #

def _pymatting_alpha(guide_np, mask, trimap, edge_strength, conf_np, conf_thresh):
    trimap_pm = trimap.astype(np.float32) / 255.0
    trimap_pm = np.where(trimap_pm < 0.3, 0.0,
                np.where(trimap_pm > 0.7, 1.0, 0.5))
    if conf_np is not None:
        certain = conf_np >= conf_thresh
        trimap_pm[certain & (mask >= 0.5)] = 1.0
        trimap_pm[certain & (mask <  0.5)] = 0.0
    try:
        alpha_cf = estimate_alpha_cf(guide_np, trimap_pm)
        alpha_cf = np.clip(alpha_cf, 0.0, 1.0).astype(np.float32)
    except Exception:
        return mask.copy()
    unknown = trimap_pm == 0.5
    result  = mask.copy()
    result[unknown] = (
        mask[unknown]       * (1.0 - edge_strength)
        + alpha_cf[unknown] * edge_strength
    )
    return result


# ================================================================== #
# Guided Filter Fallback  (confidence-gated)
# ================================================================== #

def _guided_filter_alpha(mask, guide, coarse_radius, fine_radius, fine_blend,
                         edge_strength, lock_bg, lock_fg, conf_np, conf_thresh):
    coarse  = guided_filter_mask(mask, guide, radius=coarse_radius, eps=1e-4)
    fine    = guided_filter_mask(mask, guide, radius=fine_radius,   eps=1e-5)
    blended = np.clip((1.0 - fine_blend) * coarse + fine_blend * fine, 0.0, 1.0)
    edge_band = (mask > lock_bg) & (mask < lock_fg)
    if conf_np is not None:
        edge_band = edge_band & (conf_np < conf_thresh)
    result    = mask.copy()
    result[edge_band] = (
        mask[edge_band]      * (1.0 - edge_strength)
        + blended[edge_band] * edge_strength
    )
    return result


# ================================================================== #
# BG Fringe Suppression
# ================================================================== #

def _suppress_bg_fringe(result, original, guide, lock_bg, strength,
                        outer_band_max=0.18):
    import cv2
    fringe_zone = (original > lock_bg) & (original < outer_band_max)
    if fringe_zone.sum() == 0:
        return result
    fg_core    = (original >= 0.5).astype(np.uint8) * 255
    k          = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fg_dilated = cv2.dilate(fg_core, k)
    fg_mask_f  = (fg_dilated > 0).astype(np.float32)
    fg_ref     = np.zeros_like(guide, dtype=np.float32)
    for c in range(3):
        ch       = guide[:, :, c] * fg_mask_f
        blurred  = cv2.GaussianBlur(np.ascontiguousarray(ch), (0, 0), sigmaX=20)
        cnt_blur = cv2.GaussianBlur(np.ascontiguousarray(fg_mask_f), (0, 0), sigmaX=20)
        fg_ref[:, :, c] = np.where(cnt_blur > 0.01, blurred / (cnt_blur + 1e-6), 0.5)
    col_dist   = np.linalg.norm(guide - fg_ref, axis=2)
    col_dist_n = np.clip(col_dist / (np.sqrt(3) * 0.5), 0.0, 1.0)
    depth      = 1.0 - (original / (outer_band_max + 1e-6))
    suppress_w = np.clip(col_dist_n * depth * strength, 0.0, 0.5)
    out        = result.copy()
    out[fringe_zone] = result[fringe_zone] * (1.0 - suppress_w[fringe_zone])
    return np.clip(out, 0.0, 1.0)


# ================================================================== #
# Foreground Decontamination
# ================================================================== #

def _decontaminate(pil_img, pil_mask):
    img_np   = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
    mask_np  = np.array(pil_mask.convert("L")).astype(np.float32) / 255.0
    if _PYMATTING_OK:
        try:
            F = estimate_foreground_ml(
                img_np.astype(np.float64),
                mask_np.astype(np.float64),
            )
            return Image.fromarray(
                (np.clip(F, 0, 1) * 255).astype(np.uint8), mode="RGB")
        except Exception:
            pass
    import cv2
    bg_only     = img_np * (1.0 - mask_np[:, :, None])
    bg_blur     = cv2.GaussianBlur(bg_only, (61, 61), sigmaX=20)
    weight_blur = cv2.GaussianBlur(1.0 - mask_np, (61, 61), sigmaX=20)
    weight_blur = np.maximum(weight_blur, 1e-6)[:, :, None]
    bg_map      = bg_blur / weight_blur
    fg_est      = np.clip(img_np - bg_map * (1.0 - mask_np[:, :, None]) * 0.65, 0, 1)
    return Image.fromarray((fg_est * 255).astype(np.uint8), mode="RGB")

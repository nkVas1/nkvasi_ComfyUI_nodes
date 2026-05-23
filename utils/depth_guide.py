"""
utils/depth_guide.py  v2.1

Depth Pro — guided mask refinement using monocular metric depth.

=== What changed in v2.1 vs v2.0 ===

Problem
-------
  Residual BG fringe survived because:
  1. Pass 1 BG veto could only reduce alpha proportionally (floor=bg_suppress_max)
     → could never fully zero out a stubborn semi-transparent fringe.
  2. Pass 3 Edge crisp blend_w was capped at 0.80 — never drove alpha all the
     way to the depth-derived hard decision.
  3. Hard lock used fixed edge_band_hi=0.95, so pixels with alpha 0.90-0.94
     were locked to ensemble even in obviously-BG depth zones.

v2.1 additions
--------------
  Pass 1+  BG hard cut (NEW):
    After the soft BG veto, a second sweep checks against a tighter threshold
    `depth_bg_hard` (default 0.90).  Any pixel where depth_norm > depth_bg_hard
    AND mask < hard_cut_alpha_max (default 0.70) is driven straight to 0.
    This is the "knife" for obvious fringe that soft suppression can't finish.
    Gated by `bg_hard_cut_strength` (0=off, 1=full hard cut).

  Pass 3+  Edge crisp cap lifted to 1.0:
    blend_w ceiling raised from 0.80 → 1.0 (controlled by edge_crisp_strength).
    At strength=1.0 the depth-edge decision fully replaces ensemble alpha at
    strong depth boundaries.

  Pass 4   Tunable hard-lock threshold:
    `hard_lock_hi` (default 0.90, was fixed 0.95) — pixels with alpha above
    this are hard-locked to ensemble.  Lowering it lets BG veto reach deeper
    into semi-transparent fringes.
"""
from __future__ import annotations
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _normalise_depth(depth: np.ndarray) -> np.ndarray:
    """Normalise any depth array to [0, 1]."""
    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max - d_min < 1e-5:
        return np.zeros_like(depth, dtype=np.float32)
    return ((depth - d_min) / (d_max - d_min)).astype(np.float32)


def _resolve_polarity(
    depth_norm: np.ndarray,
    mask: np.ndarray,
    mask_fg_thresh: float = 0.5,
) -> np.ndarray:
    """
    Ensure depth_norm convention is 0=near(FG), 1=far(BG).

    Uses the ensemble mask as a polarity oracle: compare mean depth under
    definite-FG vs definite-BG pixels.  Fallback to centre/border heuristic
    when either zone has fewer than 100 pixels.
    """
    fg_px = depth_norm[mask > mask_fg_thresh]
    bg_px = depth_norm[mask < (1.0 - mask_fg_thresh)]

    if len(fg_px) < 100 or len(bg_px) < 100:
        h, w = depth_norm.shape
        cy, cx = h // 2, w // 2
        centre = float(depth_norm[
            cy - h // 8 : cy + h // 8,
            cx - w // 8 : cx + w // 8,
        ].mean())
        border = float(np.concatenate([
            depth_norm[: h // 8, :].ravel(),
            depth_norm[-h // 8 :, :].ravel(),
            depth_norm[:, : w // 8].ravel(),
            depth_norm[:, -w // 8 :].ravel(),
        ]).mean())
        return (1.0 - depth_norm) if centre > border else depth_norm

    mean_fg = float(fg_px.mean())
    mean_bg = float(bg_px.mean())
    return (1.0 - depth_norm) if mean_fg > mean_bg else depth_norm


def _fg_depth_threshold(
    depth_norm: np.ndarray,
    mask: np.ndarray,
    fg_percentile: float = 80.0,
) -> float:
    """
    P-th percentile of depth values inside the ensemble-FG zone.
    Pixels beyond this depth are 'probably background'.
    """
    fg_depth = depth_norm[mask > 0.5]
    if len(fg_depth) == 0:
        return 0.5
    return float(np.percentile(fg_depth, fg_percentile))


def _depth_edge_map(depth_norm: np.ndarray, blur_px: int = 1) -> np.ndarray:
    """
    Normalised Sobel gradient magnitude of the depth map.
    Returns float32 H×W in [0, 1].  High = sharp depth boundary = object contour.
    """
    import cv2
    d_u8 = (depth_norm * 255).clip(0, 255).astype(np.uint8)
    gx   = cv2.Sobel(d_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy   = cv2.Sobel(d_u8, cv2.CV_32F, 0, 1, ksize=3)
    mag  = np.sqrt(gx ** 2 + gy ** 2)
    if mag.max() < 1e-6:
        return np.zeros_like(mag)
    mag = mag / mag.max()
    if blur_px > 0:
        k   = blur_px * 2 + 1
        mag = cv2.GaussianBlur(np.ascontiguousarray(mag), (k, k), blur_px / 2)
        mag = np.clip(mag / (mag.max() + 1e-6), 0.0, 1.0)
    return mag.astype(np.float32)


# ---------------------------------------------------------------------------
# Core refinement  v2.1
# ---------------------------------------------------------------------------

def depth_guided_mask(
    mask: np.ndarray,
    depth_norm: np.ndarray,
    strength: float = 0.70,
    fg_percentile: float = 80.0,
    # Hard-lock zone limits
    edge_band_lo: float   = 0.05,
    hard_lock_hi: float   = 0.90,   # v2.1: was fixed 0.95
    # Pass 1 — BG soft veto
    depth_bg_suppress: float = 0.75,
    bg_suppress_max: float   = 0.85,
    # Pass 1+ — BG hard cut (NEW in v2.1)
    depth_bg_hard: float         = 0.90,   # depth above this → hard-zero candidate
    hard_cut_alpha_max: float    = 0.70,   # only cut pixels with alpha < this
    bg_hard_cut_strength: float  = 0.80,   # 0=off, 1=always cut to zero
    # Pass 2 — FG recovery
    depth_fg_recover: float  = 0.35,
    hair_alpha_lo: float     = 0.02,   # API compat, unused
    hair_alpha_hi: float     = 0.50,
    recovery_max: float      = 0.60,
    # Pass 3 — Edge crisp
    edge_crisp_strength: float = 0.50,
) -> np.ndarray:
    """
    Refine alpha mask using a polarity-corrected normalised depth map.

    Five-pass pipeline (v2.1 adds BG hard cut after soft veto):
      Pass 1   BG soft veto   — proportional suppression
      Pass 1+  BG hard cut    — drive deep-BG pixels straight to 0
      Pass 2   FG recovery    — recover missed strands
      Pass 3   Edge crisp     — depth-edge driven boundary sharpening
      Pass 4   Hard lock      — snap safe-zone pixels back to ensemble
    """
    import cv2

    # --- Resize depth to mask resolution ---
    if depth_norm.shape != mask.shape:
        d_pil      = Image.fromarray(
            (depth_norm * 255).clip(0, 255).astype(np.uint8), mode="L")
        depth_norm = np.array(
            d_pil.resize((mask.shape[1], mask.shape[0]), Image.LANCZOS)
        ).astype(np.float32) / 255.0

    # --- A. Polarity correction ---
    depth_norm = _resolve_polarity(depth_norm, mask)

    fg_thresh = _fg_depth_threshold(depth_norm, mask, fg_percentile)
    edge_mag  = _depth_edge_map(depth_norm, blur_px=1)
    result    = mask.copy()

    # ================================================================
    # Pass 1 — BG soft veto
    # Proportionally suppress alpha where depth says clearly far
    # ================================================================
    bg_threshold = max(fg_thresh, depth_bg_suppress)
    bg_veto      = (mask > edge_band_lo) & (depth_norm > bg_threshold)
    if bg_veto.any():
        excess     = np.clip(
            (depth_norm - bg_threshold) / max(1.0 - bg_threshold, 1e-3),
            0.0, 1.0,
        )
        suppress_w = np.clip(excess * strength, 0.0, 1.0 - bg_suppress_max)
        result[bg_veto] = result[bg_veto] * (1.0 - suppress_w[bg_veto])

    # ================================================================
    # Pass 1+  BG hard cut  (v2.1 NEW)
    # Pixels where depth is very far AND alpha is still semi-transparent
    # → drive directly to zero, scaled by bg_hard_cut_strength.
    # Only applied to pixels NOT in the FG safe zone (mask < hard_lock_hi).
    # ================================================================
    if bg_hard_cut_strength > 0.0:
        hard_cut_zone = (
            (depth_norm > depth_bg_hard) &
            (mask < hard_cut_alpha_max) &
            (mask > edge_band_lo)
        )
        if hard_cut_zone.any():
            # How deep into hard-BG territory is this pixel?
            hard_excess = np.clip(
                (depth_norm - depth_bg_hard) / max(1.0 - depth_bg_hard, 1e-3),
                0.0, 1.0,
            )
            # Cut weight: product of depth excess and global strength knob
            cut_w = np.clip(hard_excess * bg_hard_cut_strength, 0.0, 1.0)
            # Drive alpha toward 0 by cut_w; never raise
            result[hard_cut_zone] = result[hard_cut_zone] * (1.0 - cut_w[hard_cut_zone])

    # ================================================================
    # Pass 2 — FG recovery
    # Ensemble missed thin strands; depth says near + strong edge
    # → raise alpha toward recovery_max
    # ================================================================
    recover_cand = (mask < hair_alpha_hi) & (depth_norm < depth_fg_recover)
    if recover_cand.any():
        proximity  = np.clip(
            1.0 - depth_norm / max(depth_fg_recover, 1e-3),
            0.0, 1.0,
        )
        edge_w     = np.clip(edge_mag * 2.0, 0.0, 1.0)
        rec_w      = np.clip(proximity * edge_w * strength, 0.0, 1.0)
        miss_ratio = np.clip(
            1.0 - mask / max(hair_alpha_hi, 1e-3),
            0.0, 1.0,
        )
        target  = recovery_max * miss_ratio
        new_val = result + (target - result) * rec_w
        result[recover_cand] = np.maximum(
            result[recover_cand], new_val[recover_cand]
        )

    # ================================================================
    # Pass 3 — Edge crisp
    # Blend alpha toward hard depth-derived 0/1 at boundary zones.
    # v2.1: blend_w cap raised 0.80 → 1.0
    # ================================================================
    if edge_crisp_strength > 0.0:
        boundary_zone = (mask > edge_band_lo) & (mask < hard_lock_hi)
        if boundary_zone.any():
            depth_fg_mask = (depth_norm < fg_thresh).astype(np.float32)
            blend_w       = np.clip(edge_mag * edge_crisp_strength, 0.0, 1.0)  # v2.1: was 0.80
            result_sharp  = result * (1.0 - blend_w) + depth_fg_mask * blend_w
            result[boundary_zone] = result_sharp[boundary_zone]

    # ================================================================
    # Pass 4 — Hard lock  (applied BEFORE blur)
    # v2.1: uses hard_lock_hi instead of fixed 0.95
    # ================================================================
    result[mask >= hard_lock_hi]  = mask[mask >= hard_lock_hi]
    result[mask <= edge_band_lo]  = mask[mask <= edge_band_lo]

    # --- Final light smooth at transition boundary ---
    changed = np.abs(result - mask) > 0.01
    if changed.any():
        blurred = cv2.GaussianBlur(np.ascontiguousarray(result), (3, 3), 0.8)
        result[changed] = result[changed] * 0.80 + blurred[changed] * 0.20
        result[mask >= hard_lock_hi] = mask[mask >= hard_lock_hi]
        result[mask <= edge_band_lo] = mask[mask <= edge_band_lo]

    return np.clip(result, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Adaptive trimap  v2.1 (unchanged from v2.0)
# ---------------------------------------------------------------------------

def depth_adaptive_trimap(
    mask: np.ndarray,
    depth_norm: np.ndarray,
    confidence: np.ndarray | None = None,
    min_band_px: int = 4,
    max_band_px: int = 24,
    fg_percentile: float = 80.0,
) -> np.ndarray:
    """
    Build trimap with unknown-band width modulated by depth gradient.

    Returns uint8 H×W {0=BG, 128=unknown, 255=FG}
    """
    import cv2

    if depth_norm.shape != mask.shape:
        d_pil      = Image.fromarray(
            (depth_norm * 255).clip(0, 255).astype(np.uint8), mode="L")
        depth_norm = np.array(
            d_pil.resize((mask.shape[1], mask.shape[0]), Image.LANCZOS)
        ).astype(np.float32) / 255.0

    depth_norm = _resolve_polarity(depth_norm, mask)

    edge_mag  = _depth_edge_map(depth_norm, blur_px=max(1, min_band_px // 2))
    fg_thresh = _fg_depth_threshold(depth_norm, mask, fg_percentile)

    depth_uncert = 1.0 - np.clip(
        np.abs(depth_norm - fg_thresh) / max(fg_thresh, 1e-3) * 0.5,
        0.0, 1.0,
    )

    if confidence is not None:
        if confidence.shape != mask.shape:
            c_pil      = Image.fromarray(
                (confidence * 255).clip(0, 255).astype(np.uint8), mode="L")
            confidence = np.array(
                c_pil.resize((mask.shape[1], mask.shape[0]), Image.LANCZOS)
            ).astype(np.float32) / 255.0
        uncertainty = np.maximum(1.0 - confidence, depth_uncert)
    else:
        uncertainty = depth_uncert

    band_signal = np.clip(
        uncertainty * 0.5 + edge_mag * 0.5, 0.0, 1.0
    ).astype(np.float32)

    hard     = (mask > 0.5).astype(np.uint8) * 255
    band_r   = (min_band_px + band_signal * (max_band_px - min_band_px)).astype(np.float32)

    dist_to_bg = cv2.distanceTransform(hard,       cv2.DIST_L2, 5)
    dist_to_fg = cv2.distanceTransform(255 - hard, cv2.DIST_L2, 5)

    is_unknown = (
        (dist_to_bg < band_r) & (dist_to_fg < band_r)
    ) | (dist_to_bg < min_band_px) | (dist_to_fg < min_band_px)

    depth_fg_zone      = (depth_norm < fg_thresh * 0.6).astype(bool)
    missed_by_ensemble = depth_fg_zone & (mask < 0.3)
    k_dil              = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    missed_dilated     = cv2.dilate(
        missed_by_ensemble.astype(np.uint8), k_dil
    ).astype(bool)
    is_unknown = is_unknown | missed_dilated

    k_fg    = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (min_band_px * 2 + 1, min_band_px * 2 + 1))
    fg_core = cv2.erode(hard, k_fg)

    trimap              = np.zeros_like(hard)
    trimap[is_unknown]  = 128
    trimap[fg_core == 255] = 255
    return trimap


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------

def depth_to_pil(depth_norm: np.ndarray) -> Image.Image:
    """float32 H×W [0,1] → grayscale PIL. dark=near(FG), bright=far(BG)."""
    return Image.fromarray(
        (depth_norm * 255).clip(0, 255).astype(np.uint8), mode="L")

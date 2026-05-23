"""
utils/depth_guide.py  v2.0

Depth Pro — guided mask refinement using monocular metric depth.

=== Architecture v2 ===

v1 problems
-----------
  1. Polarity heuristic (centre > border → flip) unreliable for portraits
     where the subject fills most of the frame.
  2. Refinement was only a soft "nudge" inside edge_zone — almost no effect
     on final alpha because MattingRefine overwrites it anyway.
  3. Depth edge map — the strongest signal Depth Pro produces — was never
     used directly.
  4. FG recovery (depth says FG but ensemble missed it) was not implemented.

v2 design
---------
  A. Polarity resolved via mask oracle:
       depth values under ensemble-FG zone are "near", flip if needed so
       that FG = low depth value throughout the pipeline.

  B. Four-pass refinement:
     Pass 1  BG veto     — ensemble says FG but depth is clearly far
                           → suppress alpha proportionally to depth excess
     Pass 2  FG recovery — ensemble missed thin strands (alpha < hair_hi)
                           but depth says near + strong depth edge
                           → recover alpha scaled by proximity × edge_mag
     Pass 3  Edge crisp  — replace soft ensemble boundary with depth-edge
                           driven alpha: Sobel gradient used as 0–1 weight
                           to blend sharp vs. soft alpha at sub-pixel level
     Pass 4  Hard lock   — pixels clearly inside FG core or clearly BG
                           are snapped back to original ensemble values
                           (depth never overrides safe zones)
                           applied BEFORE final blur

  C. depth_adaptive_trimap v2:
       unknown band is EXPANDED beyond ensemble boundary wherever depth
       gradient is strong (thin structure that ensemble may have missed),
       giving MattingRefine the best possible working zone.
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

    Uses the ensemble mask as a polarity oracle:
      - Compute mean depth under definite-FG pixels (mask > mask_fg_thresh)
        and under definite-BG pixels (mask < 1 - mask_fg_thresh).
      - If mean_fg_depth > mean_bg_depth the map is inverted; flip it.

    This is far more reliable than the centre/border heuristic for portraits
    where the subject occupies most of the frame.
    """
    fg_px = depth_norm[mask > mask_fg_thresh]
    bg_px = depth_norm[mask < (1.0 - mask_fg_thresh)]

    if len(fg_px) < 100 or len(bg_px) < 100:
        # Fallback: centre/border heuristic
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
    # FG should be near (low value); if it's high, map is inverted
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
    Compute normalised Sobel gradient magnitude of the depth map.
    Returns float32 H×W in [0, 1].
    High values = sharp depth boundary = probable object contour.
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
# Core refinement  v2
# ---------------------------------------------------------------------------

def depth_guided_mask(
    mask: np.ndarray,
    depth_norm: np.ndarray,
    strength: float = 0.70,
    fg_percentile: float = 80.0,
    # hard-zone limits (pixels outside edge_band are never changed)
    edge_band_lo: float = 0.05,
    edge_band_hi: float = 0.95,
    # Pass 1 — BG veto
    depth_bg_suppress: float = 0.75,
    bg_suppress_max: float   = 0.85,   # alpha floor after suppression
    # Pass 2 — FG recovery
    depth_fg_recover: float  = 0.35,   # depth < this → eligible for recovery
    hair_alpha_lo: float     = 0.02,   # kept for API compat, unused internally
    hair_alpha_hi: float     = 0.50,   # ensemble alpha above this = already found
    recovery_max: float      = 0.60,   # max alpha we'll restore a missed strand to
    # Pass 3 — Edge crisp
    edge_crisp_strength: float = 0.50,
) -> np.ndarray:
    """
    Refine alpha mask using a polarity-corrected normalised depth map.

    Four-pass pipeline — see module docstring for full description.
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
    # Pass 1 — BG veto
    # ensemble says FG (mask > edge_band_lo)
    # depth says clearly far (depth_norm > bg_threshold)
    # → reduce alpha proportionally to depth excess
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
    # Pass 2 — FG recovery
    # ensemble missed (alpha < hair_alpha_hi)
    # depth says near (depth_norm < depth_fg_recover)
    # depth edge is strong → probable thin strand
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
    # at ensemble boundary: use depth-edge magnitude to blend alpha
    # toward a hard 0/1 decision aligned with depth contour
    # ================================================================
    if edge_crisp_strength > 0.0:
        boundary_zone = (mask > edge_band_lo) & (mask < edge_band_hi)
        if boundary_zone.any():
            depth_fg_mask = (depth_norm < fg_thresh).astype(np.float32)
            blend_w       = np.clip(edge_mag * edge_crisp_strength, 0.0, 0.80)
            result_sharp  = result * (1.0 - blend_w) + depth_fg_mask * blend_w
            result[boundary_zone] = result_sharp[boundary_zone]

    # ================================================================
    # Pass 4 — Hard lock  (applied BEFORE blur)
    # safe-zone pixels always restored to original ensemble values
    # ================================================================
    result[mask >= edge_band_hi] = mask[mask >= edge_band_hi]
    result[mask <= edge_band_lo] = mask[mask <= edge_band_lo]

    # --- Final light smooth at transition boundary ---
    changed = np.abs(result - mask) > 0.01
    if changed.any():
        blurred = cv2.GaussianBlur(np.ascontiguousarray(result), (3, 3), 0.8)
        result[changed] = result[changed] * 0.80 + blurred[changed] * 0.20
        # Re-lock after blur to prevent bleed from unsuppressed neighbours
        result[mask >= edge_band_hi] = mask[mask >= edge_band_hi]
        result[mask <= edge_band_lo] = mask[mask <= edge_band_lo]

    return np.clip(result, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Adaptive trimap  v2
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

    v2: polarity resolved via mask oracle; unknown band expanded beyond
    ensemble boundary wherever depth-edge magnitude is high (thin structures
    the ensemble may have missed) → MattingRefine gets the best working zone.

    Returns uint8 H×W {0=BG, 128=unknown, 255=FG}
    """
    import cv2

    if depth_norm.shape != mask.shape:
        d_pil      = Image.fromarray(
            (depth_norm * 255).clip(0, 255).astype(np.uint8), mode="L")
        depth_norm = np.array(
            d_pil.resize((mask.shape[1], mask.shape[0]), Image.LANCZOS)
        ).astype(np.float32) / 255.0

    # Resolve polarity using mask oracle
    depth_norm = _resolve_polarity(depth_norm, mask)

    edge_mag  = _depth_edge_map(depth_norm, blur_px=max(1, min_band_px // 2))
    fg_thresh = _fg_depth_threshold(depth_norm, mask, fg_percentile)

    # Depth-based uncertainty: pixels near fg_thresh boundary are most uncertain
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

    # v2: depth gradient weighted equally with uncertainty
    band_signal = np.clip(
        uncertainty * 0.5 + edge_mag * 0.5, 0.0, 1.0
    ).astype(np.float32)

    # Per-pixel band radius + distance-transform trimap
    hard     = (mask > 0.5).astype(np.uint8) * 255
    band_r   = (min_band_px + band_signal * (max_band_px - min_band_px)).astype(np.float32)

    dist_to_bg = cv2.distanceTransform(hard,       cv2.DIST_L2, 5)
    dist_to_fg = cv2.distanceTransform(255 - hard, cv2.DIST_L2, 5)

    is_unknown = (
        (dist_to_bg < band_r) & (dist_to_fg < band_r)
    ) | (dist_to_bg < min_band_px) | (dist_to_fg < min_band_px)

    # v2: expand unknown zone to cover depth-FG pixels that ensemble missed
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
    """
    float32 H×W [0,1] → grayscale PIL.
    After polarity resolution: dark=near(FG), bright=far(BG).
    """
    return Image.fromarray(
        (depth_norm * 255).clip(0, 255).astype(np.uint8), mode="L")

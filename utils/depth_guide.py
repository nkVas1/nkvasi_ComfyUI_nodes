"""
utils/depth_guide.py

Depth Pro — guided trimap and mask refinement using monocular metric depth.

Why Depth Pro?
--------------
Depth Pro (Apple MLRC, 2024) is a multi-scale ViT that produces
metric absolute depth in <0.3 s on a single GPU.  Unlike relative-depth
models (MiDaS, Depth Anything) it does not need calibration data and
gives physically meaningful distance values.  Most importantly for
segmentation it has a dedicated boundary-sharpness loss that keeps
edge depth very precise — which is exactly what we need for trimap
refinement at hair/fur/transparent object boundaries.

Pipeline position
-----------------
  1. Ensemble produces rough alpha mask + confidence_map.
  2. DepthGuide is called AFTER ensemble merge, BEFORE MattingRefine.
  3. It refines the mask using depth:
       a. Depth-gated BG suppression: pixels that are "far" (depth > FG
          threshold) and inside the edge band → forced toward 0.
       b. Depth-gated FG confidence: pixels "close" and currently
          uncertain → nudged toward 1.
       c. Boundary sharpening: strong depth gradient at edge → narrows
          the uncertain band, giving MattingRefine a cleaner trimap.

Module structure
----------------
  _DepthProWrapper     — lazy-loaded model (cached via model_loader)
  compute_depth_map    — PIL Image → float32 H×W depth in metres (or
                          normalised [0,1] if metric not available)
  depth_guided_mask    — float32 mask × depth → refined float32 mask
  depth_adaptive_trimap— builds trimap with depth-aware band widths
"""
from __future__ import annotations
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEPTH_PRO_HF_ID = "apple/DepthPro"


# ---------------------------------------------------------------------------
# Depth map normalisation helpers
# ---------------------------------------------------------------------------

def _normalise_depth(depth: np.ndarray) -> np.ndarray:
    """
    Normalise depth map to [0, 1] where:
      0 = closest point (foreground)
      1 = farthest point (background)
    """
    d_min = float(depth.min())
    d_max = float(depth.max())
    if d_max - d_min < 1e-5:
        return np.zeros_like(depth, dtype=np.float32)
    return ((depth - d_min) / (d_max - d_min)).astype(np.float32)


def _fg_depth_threshold(
    depth_norm: np.ndarray,
    mask: np.ndarray,
    fg_percentile: float = 80.0,
) -> float:
    """
    Estimate the depth threshold that separates FG from BG.
    Uses fg_percentile-th percentile of depth values inside the FG zone
    (mask > 0.5). Pixels beyond this depth are "probably background".
    """
    fg_depth = depth_norm[mask > 0.5]
    if len(fg_depth) == 0:
        return 0.5
    return float(np.percentile(fg_depth, fg_percentile))


# ---------------------------------------------------------------------------
# Core refinement
# ---------------------------------------------------------------------------

def depth_guided_mask(
    mask: np.ndarray,
    depth_norm: np.ndarray,
    strength: float = 0.70,
    fg_percentile: float = 80.0,
    edge_band_lo: float = 0.05,
    edge_band_hi: float = 0.95,
    depth_bg_suppress: float = 0.85,
    depth_fg_boost:    float = 0.30,
) -> np.ndarray:
    """
    Refine alpha mask using a normalised depth map.

    Args:
        mask            : float32 H×W alpha [0,1]
        depth_norm      : float32 H×W normalised depth [0=near, 1=far]
        strength        : overall influence of depth signal [0,1]
        fg_percentile   : percentile of FG depth used as BG threshold
        edge_band_lo/hi : alpha range considered the "uncertain edge zone"
        depth_bg_suppress: depth_norm > this → suppress toward BG
        depth_fg_boost  : depth_norm < this → boost toward FG

    Returns:
        float32 H×W refined alpha

    Fix log
    -------
    v1.1: Fixed fg_cand threshold formula (was computing fg_thresh²/fg_thresh
          instead of a plain depth_fg_boost constant, making the FG boost
          zone almost empty and allowing bg suppression to dominate even
          near pixels, which made fine hair fringe appear brighter).
          Fixed lock/blur order: hard cores are now locked BEFORE the
          Gaussian smoothing pass so blur cannot raise already-suppressed
          semi-transparent pixels back up.
    """
    import cv2

    # Resize depth to mask resolution if needed
    if depth_norm.shape != mask.shape:
        depth_pil  = Image.fromarray(
            (depth_norm * 255).clip(0, 255).astype(np.uint8), mode="L")
        depth_norm = np.array(
            depth_pil.resize((mask.shape[1], mask.shape[0]), Image.LANCZOS)
        ).astype(np.float32) / 255.0

    fg_thresh = _fg_depth_threshold(depth_norm, mask, fg_percentile)
    edge_zone = (mask > edge_band_lo) & (mask < edge_band_hi)

    result = mask.copy()

    # ---- 1. BG suppression: far pixels in edge zone → push toward 0 ----
    #
    # Threshold = the stricter of the adaptive fg_thresh and the hard
    # depth_bg_suppress slider.  Using max() means we only suppress
    # pixels that are clearly BG by BOTH measures.
    bg_thresh = max(fg_thresh, depth_bg_suppress)
    bg_cand   = edge_zone & (depth_norm > bg_thresh)
    if bg_cand.any():
        over       = np.clip(
            (depth_norm - bg_thresh) / max(1.0 - bg_thresh, 1e-3),
            0.0, 1.0,
        )
        suppress_w = np.clip(over * strength, 0.0, 0.90)
        result[bg_cand] = result[bg_cand] * (1.0 - suppress_w[bg_cand])

    # ---- 2. FG confidence boost: near pixels in edge zone → push toward 1 ----
    #
    # Fixed: threshold is simply depth_fg_boost (a plain constant).
    # Previous code computed `fg_thresh * depth_fg_boost / fg_thresh * fg_thresh`
    # which collapsed to `depth_fg_boost * fg_thresh` — typically ~0.21 instead
    # of 0.30, making this branch nearly a no-op and leaving hair fringe
    # pixels unprotected so BG suppression raised their relative brightness.
    fg_cand = edge_zone & (depth_norm < depth_fg_boost)
    if fg_cand.any():
        under   = np.clip(
            (depth_fg_boost - depth_norm) / max(depth_fg_boost, 1e-3),
            0.0, 1.0,
        )
        boost_w = np.clip(under * strength * 0.5, 0.0, 0.40)
        result[fg_cand] = result[fg_cand] + (
            (1.0 - result[fg_cand]) * boost_w[fg_cand]
        )

    # ---- 3. Lock hard cores BEFORE blur ----
    #
    # Fixed: locking is applied here, before the Gaussian smoothing pass.
    # Previously it was applied after blur, which allowed the blur to raise
    # semi-transparent suppressed pixels (0.85 < mask < 0.95) back toward
    # their original values by averaging with unsuppressed neighbours.
    result[mask >= edge_band_hi] = mask[mask >= edge_band_hi]
    result[mask <= edge_band_lo] = mask[mask <= edge_band_lo]

    # ---- 4. Smooth transition at modified boundary ----
    #
    # Blur only the pixels that actually changed; hard cores are already
    # locked above and won't be touched again.
    changed = np.abs(result - mask) > 0.01
    if changed.any():
        blurred = cv2.GaussianBlur(np.ascontiguousarray(result), (3, 3), 0.8)
        result[changed] = result[changed] * 0.75 + blurred[changed] * 0.25
        # Re-apply lock after blur to prevent bleed from unsuppressed neighbours
        result[mask >= edge_band_hi] = mask[mask >= edge_band_hi]
        result[mask <= edge_band_lo] = mask[mask <= edge_band_lo]

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def depth_adaptive_trimap(
    mask: np.ndarray,
    depth_norm: np.ndarray,
    confidence: np.ndarray | None = None,
    min_band_px: int = 4,
    max_band_px: int = 24,
    fg_percentile: float = 80.0,
) -> np.ndarray:
    """
    Build trimap where the unknown-band width is modulated by:
      - depth gradient (strong gradient = complex boundary = wider band)
      - depth value (pixel far from camera = lean toward BG)
      - confidence from ensemble (optional, as in build_adaptive_trimap)

    Returns uint8 H×W trimap {0=BG, 128=unknown, 255=FG}
    """
    import cv2
    from .confidence import build_adaptive_trimap as _conf_trimap

    if depth_norm.shape != mask.shape:
        depth_pil  = Image.fromarray(
            (depth_norm * 255).clip(0, 255).astype(np.uint8), mode="L")
        depth_norm = np.array(
            depth_pil.resize((mask.shape[1], mask.shape[0]), Image.LANCZOS)
        ).astype(np.float32) / 255.0

    # ---- Depth gradient as boundary-complexity signal ----
    d_u8   = (depth_norm * 255).clip(0, 255).astype(np.uint8)
    gx     = cv2.Sobel(d_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy     = cv2.Sobel(d_u8, cv2.CV_32F, 0, 1, ksize=3)
    d_grad = np.sqrt(gx**2 + gy**2)
    d_grad = d_grad / (d_grad.max() + 1e-6)
    k_sp   = max(3, min_band_px * 2 + 1)
    d_grad = cv2.GaussianBlur(np.ascontiguousarray(d_grad), (k_sp, k_sp), k_sp / 3)
    d_grad = np.clip(d_grad / (d_grad.max() + 1e-6), 0.0, 1.0)

    # ---- Synthesise uncertainty from depth ----
    fg_thresh    = _fg_depth_threshold(depth_norm, mask, fg_percentile)
    depth_uncert = np.clip(
        np.abs(depth_norm - fg_thresh) / max(fg_thresh, 1e-3) * 0.5,
        0.0, 1.0,
    )
    depth_uncert = 1.0 - np.clip(depth_uncert, 0.0, 1.0)

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
        uncertainty * 0.6 + d_grad * 0.4, 0.0, 1.0
    ).astype(np.float32)

    # ---- Per-pixel radius and distance-transform trimap ----
    hard     = (mask > 0.5).astype(np.uint8) * 255
    band_r   = (min_band_px + band_signal * (max_band_px - min_band_px)).astype(np.float32)

    dist_to_bg = cv2.distanceTransform(hard,       cv2.DIST_L2, 5)
    dist_to_fg = cv2.distanceTransform(255 - hard, cv2.DIST_L2, 5)

    is_unknown = (
        (dist_to_bg < band_r) & (dist_to_fg < band_r)
    ) | (dist_to_bg < min_band_px) | (dist_to_fg < min_band_px)

    k_fg     = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (min_band_px * 2 + 1, min_band_px * 2 + 1))
    fg_core  = cv2.erode(hard, k_fg)

    trimap               = np.zeros_like(hard)
    trimap[is_unknown]   = 128
    trimap[fg_core == 255] = 255
    return trimap


# ---------------------------------------------------------------------------
# Convenience: depth map → ComfyUI MASK tensor
# ---------------------------------------------------------------------------

def depth_to_pil(depth_norm: np.ndarray) -> Image.Image:
    """float32 H×W [0,1] depth → grayscale PIL (dark=near, bright=far)."""
    return Image.fromarray(
        (depth_norm * 255).clip(0, 255).astype(np.uint8), mode="L")

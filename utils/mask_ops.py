"""
Mask post-processing operations.
All functions accept/return float32 numpy arrays in [0, 1].

Key design principle:
  Binary morphological analysis (connected components, island removal) is
  performed on a hard-thresholded copy of the mask, but the RESULT is always
  applied as a multiplier on the SOFT (float) mask — so semi-transparent
  edge pixels are never binarised away.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Low-level primitives
# ---------------------------------------------------------------------------

def smooth_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Gaussian blur on mask — softens hard edges."""
    import cv2
    k = radius * 2 + 1
    return cv2.GaussianBlur(
        np.ascontiguousarray(mask.astype(np.float32)), (k, k), sigmaX=radius / 2
    ).clip(0.0, 1.0)


def guided_filter_mask(
    mask: np.ndarray,
    guide: np.ndarray,
    radius: int = 8,
    eps: float = 1e-3,
) -> np.ndarray:
    """
    Edge-aware guided filter — the single most important step for hair quality.

    Preserves fine sub-pixel transitions where the guide image has high-frequency
    detail (hair strands), while smoothing mask in flat areas (solid background).

    guide : float32 H×W×3 [0,1]  — RGB image, same resolution as mask
    mask  : float32 H×W   [0,1]

    Returns a SOFT float mask — do NOT binarise afterwards if you want
    semi-transparent hair edges.
    """
    import cv2

    guide_u8 = np.ascontiguousarray((guide * 255).clip(0, 255).astype(np.uint8))
    guide_f  = np.ascontiguousarray(guide_u8.astype(np.float32) / 255.0)  # H×W×3 CV_32F
    src_f    = np.ascontiguousarray(mask.astype(np.float32))               # H×W   CV_32F

    try:
        import cv2.ximgproc
        refined = cv2.ximgproc.guidedFilter(
            guide=guide_f,
            src=src_f,
            radius=radius,
            eps=eps,
        )
    except Exception:
        src_u8  = (src_f * 255).clip(0, 255).astype(np.uint8)
        refined = cv2.bilateralFilter(
            np.ascontiguousarray(src_u8),
            d=max(1, radius * 2 + 1),
            sigmaColor=20,
            sigmaSpace=20,
        ).astype(np.float32) / 255.0

    return np.clip(refined, 0.0, 1.0)


def erode_expand_mask(mask: np.ndarray, offset: int) -> np.ndarray:
    """Expand (positive) or shrink (negative) the mask boundary."""
    import cv2
    abs_off = abs(offset)
    kernel  = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (abs_off * 2 + 1, abs_off * 2 + 1)
    )
    mask_u8 = (mask * 255).clip(0, 255).astype(np.uint8)
    result  = cv2.dilate(mask_u8, kernel) if offset > 0 else cv2.erode(mask_u8, kernel)
    return result.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Soft morphology helpers
# ---------------------------------------------------------------------------

def _binary_mask(mask: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    """Return uint8 binary mask (0/255) without modifying the float original."""
    return (mask > thresh).astype(np.uint8) * 255


def soft_remove_holes(
    soft_mask: np.ndarray,
    min_hole_size: int = 500,
) -> np.ndarray:
    """
    Fill small transparent holes INSIDE the foreground on the binary mask,
    then restore soft values in the filled region by setting them to 1.0.
    Only touches pixels that are enclosed entirely within the FG boundary.
    """
    import cv2
    hard  = _binary_mask(soft_mask)
    inv   = cv2.bitwise_not(hard)
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    fill_mask = np.zeros_like(hard)
    for lbl in range(1, nlabels):
        if stats[lbl, cv2.CC_STAT_AREA] <= min_hole_size:
            fill_mask[labels == lbl] = 255
    result = soft_mask.copy()
    result[fill_mask == 255] = 1.0
    return result.clip(0.0, 1.0)


def soft_remove_islands(
    soft_mask: np.ndarray,
    min_island_size: int = 400,
) -> np.ndarray:
    """
    Remove isolated foreground blobs smaller than min_island_size.
    Operates on binary copy, zeroes out those pixels in the soft mask.
    Used when hair_mode=False.
    """
    import cv2
    hard = _binary_mask(soft_mask)
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(hard, connectivity=8)
    keep = np.zeros_like(hard)
    for lbl in range(1, nlabels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_island_size:
            keep[labels == lbl] = 255
    result = soft_mask.copy()
    result[keep == 0] = 0.0
    return result.clip(0.0, 1.0)


def hair_bg_island_removal(
    soft_mask: np.ndarray,
    guide: np.ndarray,
    max_island_size: int = 2000,
    color_thresh: float = 0.15,
    detect_thresh: float = 0.25,
) -> np.ndarray:
    """
    Background island removal tuned for hair.

    KEY FIX: uses detect_thresh=0.25 (not 0.5) to find BG islands in the soft
    mask produced by guided_filter. After guided filter, background patches
    between strands have values ~0.3-0.45 — invisible to a 0.5 threshold.

    Algorithm:
      1. Binarise at detect_thresh to find all "possible FG" regions.
      2. Find connected BG regions inside this extended FG area.
      3. For each small BG region, compare its mean colour to nearby FG pixels.
      4. High colour distance → background leak → suppress to ~0.
      5. Low colour distance → real gap between similar-colour strands → keep.

    guide : float32 H×W×3 [0,1], same resolution as soft_mask
    """
    import cv2

    # Use lower threshold to see half-opaque background patches
    fg_loose  = _binary_mask(soft_mask, thresh=detect_thresh)   # 0/255
    bg_in_fg  = cv2.bitwise_not(fg_loose)                       # BG=255 within loose FG

    # We only care about BG regions INSIDE the overall object extent.
    # Compute the "hull" of the loose FG to limit search area.
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fg_closed    = cv2.morphologyEx(fg_loose, cv2.MORPH_CLOSE, kernel_close)
    search_area  = cv2.bitwise_and(bg_in_fg, fg_closed)

    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(
        search_area, connectivity=8
    )

    # For neighbourhood sampling, dilate strict FG border
    kernel_nb = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg_strict = _binary_mask(soft_mask, thresh=0.5)
    fg_dilate = cv2.dilate(fg_strict, kernel_nb)

    guide_u8 = (guide * 255).clip(0, 255).astype(np.uint8)
    result   = soft_mask.copy()

    for lbl in range(1, nlabels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area > max_island_size:
            continue

        island_px = (labels == lbl)
        neigh_px  = (fg_dilate == 255) & ~island_px

        if neigh_px.sum() < 10:
            # Isolated — no neighbours to compare, treat as background
            result[island_px] = result[island_px] * 0.05
            continue

        hole_color  = guide_u8[island_px].mean(axis=0).astype(np.float32) / 255.0
        neigh_color = guide_u8[neigh_px].mean(axis=0).astype(np.float32) / 255.0
        color_dist  = float(np.linalg.norm(hole_color - neigh_color))

        if color_dist > color_thresh:
            # Background leak — fade out smoothly
            result[island_px] = result[island_px] * 0.04
        # else: real inter-strand gap — keep current soft value

    return result.clip(0.0, 1.0)


def feather_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """
    Soft feathering using distance transform.
    Ramps from full opacity at the edge inward to preserve soft exterior values.
    """
    import cv2
    m_u8     = _binary_mask(mask)
    dist_in  = cv2.distanceTransform(m_u8,                  cv2.DIST_L2, 5)
    feather_w = np.clip(dist_in / (radius + 1e-6), 0.0, 1.0)
    return np.clip(mask * feather_w + mask * (1.0 - feather_w) * 0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Legacy aliases
# ---------------------------------------------------------------------------

def remove_small_holes(mask: np.ndarray, min_size: int = 500) -> np.ndarray:
    return soft_remove_holes(mask, min_hole_size=min_size)


def remove_small_islands(mask: np.ndarray, min_size: int = 400) -> np.ndarray:
    return soft_remove_islands(mask, min_island_size=min_size)

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
    # Use full RGB as guide (3-channel) — gives better edge detection than gray
    guide_f  = guide_u8.astype(np.float32) / 255.0      # H×W×3 CV_32F
    src_f    = np.ascontiguousarray(mask.astype(np.float32))  # H×W CV_32F

    try:
        import cv2.ximgproc
        # guidedFilter accepts multi-channel guide natively
        refined = cv2.ximgproc.guidedFilter(
            guide=np.ascontiguousarray(guide_f),
            src=src_f,
            radius=radius,
            eps=eps,
        )
    except Exception:
        # Bilateral fallback — nearly as good for hair
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

    This way the interior of the subject is solid opaque, but hair edges
    retain their semi-transparency from guided_filter_mask.
    """
    import cv2
    hard  = _binary_mask(soft_mask)
    inv   = cv2.bitwise_not(hard)
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    fill_mask = np.zeros_like(hard)
    for lbl in range(1, nlabels):
        if stats[lbl, cv2.CC_STAT_AREA] <= min_hole_size:
            fill_mask[labels == lbl] = 255
    # apply: where we fill a hole, set soft value to 1.0
    result = soft_mask.copy()
    result[fill_mask == 255] = 1.0
    return result.clip(0.0, 1.0)


def soft_remove_islands(
    soft_mask: np.ndarray,
    min_island_size: int = 400,
) -> np.ndarray:
    """
    Remove isolated foreground blobs (BG islands) smaller than min_island_size
    on the binary mask, then zero out the corresponding pixels in the soft mask.

    Used ONLY when hair_mode=False.  In hair_mode the background islands between
    hair strands are removed with the more targeted bg_island_removal() instead.
    """
    import cv2
    hard = _binary_mask(soft_mask)
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(hard, connectivity=8)
    keep = np.zeros_like(hard)
    for lbl in range(1, nlabels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_island_size:
            keep[labels == lbl] = 255
    # zero out pixels belonging to removed islands in the soft mask
    result = soft_mask.copy()
    result[keep == 0] = 0.0
    return result.clip(0.0, 1.0)


def hair_bg_island_removal(
    soft_mask: np.ndarray,
    guide: np.ndarray,
    max_island_size: int = 800,
    color_thresh: float = 0.18,
) -> np.ndarray:
    """
    Background island removal specifically tuned for hair.

    Strategy:
      1. Find connected BG regions (holes) in the foreground binary mask.
      2. For each small hole, measure its mean color distance from the
         surrounding foreground pixels.
      3. If the hole color is clearly different from adjacent hair (high
         contrast = background leak), erase it (set to 0).
      4. If the hole color is similar to adjacent pixels (could be a real
         gap between strands), leave it semi-transparent.

    This gives us the best of both worlds: background patches removed,
    real inter-strand gaps preserved as semi-transparent.

    guide : float32 H×W×3 [0,1] at the same resolution as soft_mask
    """
    import cv2

    hard    = _binary_mask(soft_mask)              # 0/255
    inv     = cv2.bitwise_not(hard)                # BG=255, FG=0
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)

    # dilate FG mask slightly to get the "neighbourhood" of each BG hole
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_dilate = cv2.dilate(hard, kernel)

    guide_u8 = (guide * 255).clip(0, 255).astype(np.uint8)
    result   = soft_mask.copy()

    for lbl in range(1, nlabels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area > max_island_size:
            continue  # large BG region — definitely background, leave to sensitivity

        hole_px   = (labels == lbl)
        # neighbour FG pixels: dilated area minus the hole itself
        neigh_px  = (fg_dilate == 255) & ~hole_px

        if neigh_px.sum() < 5:
            continue

        # mean color of the hole vs mean color of neighbour FG
        hole_color  = guide_u8[hole_px].mean(axis=0).astype(np.float32) / 255.0
        neigh_color = guide_u8[neigh_px].mean(axis=0).astype(np.float32) / 255.0
        color_dist  = np.linalg.norm(hole_color - neigh_color)

        if color_dist > color_thresh:
            # Clearly different from surrounding hair — it’s a background patch
            # Suppress softly rather than hard-zero to avoid sharp edges
            result[hole_px] = result[hole_px] * 0.05
        else:
            # Similar color to hair — preserve as semi-transparent gap
            pass

    return result.clip(0.0, 1.0)


def feather_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """
    Soft feathering using signed distance transform.
    Blends interior (opaque) into edge zone (semi-transparent) over `radius` px.
    The soft float values from guided_filter are preserved in the interior.
    """
    import cv2
    m_u8     = _binary_mask(mask)
    dist_in  = cv2.distanceTransform(m_u8,                  cv2.DIST_L2, 5)
    dist_out = cv2.distanceTransform(cv2.bitwise_not(m_u8), cv2.DIST_L2, 5)
    # feather weight: 1.0 inside, ramps to 0 over `radius` pixels outside the edge
    feather_w = np.clip(dist_in / (radius + 1e-6), 0.0, 1.0)
    # blend: inside stays as-is, edge zone gets feathered
    return np.clip(mask * feather_w + mask * (1.0 - feather_w) * (dist_in / (dist_in + dist_out + 1e-6)), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Legacy aliases (kept for backward compatibility with Mask Refine node)
# ---------------------------------------------------------------------------

def remove_small_holes(mask: np.ndarray, min_size: int = 500) -> np.ndarray:
    return soft_remove_holes(mask, min_hole_size=min_size)


def remove_small_islands(mask: np.ndarray, min_size: int = 400) -> np.ndarray:
    return soft_remove_islands(mask, min_island_size=min_size)

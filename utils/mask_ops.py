"""
Mask post-processing operations.
All functions accept/return float32 numpy arrays in [0, 1].
"""
import numpy as np


def smooth_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Gaussian blur on mask edges."""
    import cv2
    k = radius * 2 + 1
    return cv2.GaussianBlur(mask, (k, k), sigmaX=radius / 2).clip(0.0, 1.0)


def guided_filter_mask(mask: np.ndarray, guide: np.ndarray, radius: int = 8, eps: float = 1e-3) -> np.ndarray:
    """
    Edge-aware guided filter refinement.
    Uses the source image as guide to preserve fine hair/fur detail
    while smoothing flat background regions.

    guide: float32 H×W×3 in [0,1] (original RGB image, same resolution as mask)
    mask:  float32 H×W   in [0,1]

    cv2.ximgproc.guidedFilter requires:
      - both arrays contiguous (C-order)
      - depth CV_32F or CV_8U
      - guide can be multi-channel, src must be single-channel
    """
    import cv2

    # Convert guide RGB -> gray, ensure contiguous float32
    guide_gray = cv2.cvtColor(
        np.ascontiguousarray((guide * 255).clip(0, 255).astype(np.uint8)),
        cv2.COLOR_RGB2GRAY,
    ).astype(np.float32) / 255.0
    guide_f = np.ascontiguousarray(guide_gray)   # CV_32F, single-channel
    src_f   = np.ascontiguousarray(mask.astype(np.float32))  # CV_32F, single-channel

    try:
        refined = cv2.ximgproc.guidedFilter(
            guide=guide_f,
            src=src_f,
            radius=radius,
            eps=eps,
        )
    except (AttributeError, cv2.error):
        # Fallback: bilateral filter (ximgproc not available or failed)
        src_u8  = (src_f * 255).clip(0, 255).astype(np.uint8)
        refined = cv2.bilateralFilter(
            np.ascontiguousarray(src_u8),
            d=radius * 2 + 1,
            sigmaColor=25,
            sigmaSpace=25,
        ).astype(np.float32) / 255.0

    return np.clip(refined, 0.0, 1.0)


def erode_expand_mask(mask: np.ndarray, offset: int) -> np.ndarray:
    """Positive offset = expand (dilate), negative = shrink (erode)."""
    import cv2
    abs_off = abs(offset)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs_off * 2 + 1, abs_off * 2 + 1))
    mask_u8 = (mask * 255).clip(0, 255).astype(np.uint8)
    result = cv2.dilate(mask_u8, kernel) if offset > 0 else cv2.erode(mask_u8, kernel)
    return result.astype(np.float32) / 255.0


def remove_small_holes(mask: np.ndarray, min_size: int = 500) -> np.ndarray:
    """
    Fill small transparent holes inside the foreground object.
    Default min_size 500 — smaller values delete real gaps between hair strands.
    """
    import cv2
    m_u8 = (mask > 0.5).astype(np.uint8) * 255
    inv = cv2.bitwise_not(m_u8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    result = m_u8.copy()
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] <= min_size:
            result[labels == label] = 255
    return result.astype(np.float32) / 255.0


def remove_small_islands(mask: np.ndarray, min_size: int = 300) -> np.ndarray:
    """
    Remove tiny floating foreground islands (artifacts).
    Default min_size 300 — previous value of 100 removed real hair strands.
    """
    import cv2
    m_u8 = (mask > 0.5).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m_u8, connectivity=8)
    result = np.zeros_like(m_u8)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_size:
            result[labels == label] = 255
    return result.astype(np.float32) / 255.0


def feather_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """
    Soft feathering using signed distance transform.
    Only affects edges within `radius` pixels; interior stays fully opaque.
    """
    import cv2
    m_u8 = (mask > 0.5).astype(np.uint8) * 255
    dist_in  = cv2.distanceTransform(m_u8,                  cv2.DIST_L2, 5)
    dist_out = cv2.distanceTransform(cv2.bitwise_not(m_u8), cv2.DIST_L2, 5)
    signed   = dist_in - dist_out
    return np.clip(signed / (radius + 1e-6) * 0.5 + 0.5, 0.0, 1.0)

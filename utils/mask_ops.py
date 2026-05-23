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

    guide: float32 H×W×3 in [0,1] (original RGB image)
    mask:  float32 H×W in [0,1]
    """
    import cv2
    guide_gray = cv2.cvtColor((guide * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    guide_f = guide_gray.astype(np.float32) / 255.0
    # cv2.ximgproc is optional; fall back to bilateral if unavailable
    try:
        import cv2.ximgproc
        refined = cv2.ximgproc.guidedFilter(
            guide=guide_f, src=mask, radius=radius, eps=eps
        )
    except AttributeError:
        # bilateral filter as fallback (almost as good for hair)
        mask_u8 = (mask * 255).clip(0, 255).astype(np.uint8)
        refined = cv2.bilateralFilter(mask_u8, d=radius * 2 + 1, sigmaColor=25, sigmaSpace=25)
        refined = refined.astype(np.float32) / 255.0
    return refined.clip(0.0, 1.0)


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
    Default min_size raised to 500 — smaller values were deleting
    real semi-transparent gaps between hair strands.
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
    Default min_size raised to 300 — previous value of 100 was
    eating real tiny hair strands far from the head.
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
    Soft feathering using distance transform.
    Only applies feathering near edges (within `radius` pixels),
    leaving interior fully opaque.
    """
    import cv2
    m_u8 = (mask > 0.5).astype(np.uint8) * 255
    dist_in = cv2.distanceTransform(m_u8, cv2.DIST_L2, 5)
    dist_out = cv2.distanceTransform(cv2.bitwise_not(m_u8), cv2.DIST_L2, 5)
    # signed distance: positive inside, negative outside
    signed = dist_in - dist_out
    feathered = np.clip(signed / (radius + 1e-6) * 0.5 + 0.5, 0.0, 1.0)
    return feathered

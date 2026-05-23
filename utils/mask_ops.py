"""
Mask post-processing operations.
All functions accept/return float32 numpy arrays in [0, 1].
"""
import numpy as np


def smooth_mask(mask: np.ndarray, radius: int) -> np.ndarray:
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
    Edge-aware guided filter.
    guide : float32 H×W×3 [0,1]
    mask  : float32 H×W   [0,1]

    Critical anti-aliasing fix:
      DO NOT quantize guide to uint8 before filtering.
      Keeping full float precision preserves sub-pixel gradients and avoids
      staircase edges caused by 8-bit guide banding.
    """
    import cv2
    guide_f = np.ascontiguousarray(guide.astype(np.float32))
    src_f = np.ascontiguousarray(mask.astype(np.float32))
    try:
        import cv2.ximgproc
        refined = cv2.ximgproc.guidedFilter(
            guide=guide_f, src=src_f, radius=radius, eps=eps)
    except Exception:
        src_u8 = (src_f * 255).clip(0, 255).astype(np.uint8)
        refined = cv2.bilateralFilter(
            np.ascontiguousarray(src_u8),
            d=max(1, radius * 2 + 1), sigmaColor=20, sigmaSpace=20,
        ).astype(np.float32) / 255.0
    return np.clip(refined, 0.0, 1.0)


def erode_expand_mask(mask: np.ndarray, offset: int) -> np.ndarray:
    import cv2
    abs_off = abs(offset)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (abs_off * 2 + 1, abs_off * 2 + 1))
    mask_u8 = (mask * 255).clip(0, 255).astype(np.uint8)
    result = (cv2.dilate(mask_u8, kernel) if offset > 0 else cv2.erode(mask_u8, kernel))
    return result.astype(np.float32) / 255.0


def _binary_mask(mask: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    return (mask > thresh).astype(np.uint8) * 255


def build_trimap(mask: np.ndarray, erosion_px: int = 10, dilation_px: int = 10) -> np.ndarray:
    import cv2
    hard = _binary_mask(mask, thresh=0.5)
    k_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_px * 2 + 1, erosion_px * 2 + 1))
    k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1))
    fg = cv2.erode(hard, k_e)
    ext = cv2.dilate(hard, k_d)
    trimap = np.zeros_like(hard)
    trimap[ext == 255] = 128
    trimap[fg == 255] = 255
    return trimap


def soft_remove_holes(soft_mask: np.ndarray, min_hole_size: int = 500) -> np.ndarray:
    import cv2
    hard = _binary_mask(soft_mask)
    inv = cv2.bitwise_not(hard)
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    fill_mask = np.zeros_like(hard)
    for lbl in range(1, nlabels):
        if stats[lbl, cv2.CC_STAT_AREA] <= min_hole_size:
            fill_mask[labels == lbl] = 255
    result = soft_mask.copy()
    result[fill_mask == 255] = 1.0
    return result.clip(0.0, 1.0)


def soft_remove_islands(soft_mask: np.ndarray, min_island_size: int = 400) -> np.ndarray:
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

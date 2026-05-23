"""
Mask post-processing operations.
All functions accept/return float32 numpy arrays in [0, 1].
"""
import numpy as np


def smooth_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Gaussian blur on mask edges."""
    import cv2
    k = radius * 2 + 1
    blurred = cv2.GaussianBlur(mask, (k, k), sigmaX=radius / 2)
    return blurred.clip(0.0, 1.0)


def erode_expand_mask(mask: np.ndarray, offset: int) -> np.ndarray:
    """Positive offset = expand (dilate), negative = shrink (erode)."""
    import cv2
    abs_off = abs(offset)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs_off * 2 + 1, abs_off * 2 + 1))
    mask_u8 = (mask * 255).clip(0, 255).astype(np.uint8)
    if offset > 0:
        result = cv2.dilate(mask_u8, kernel)
    else:
        result = cv2.erode(mask_u8, kernel)
    return result.astype(np.float32) / 255.0


def remove_small_holes(mask: np.ndarray, min_size: int = 200) -> np.ndarray:
    """Fill small transparent holes inside the foreground object."""
    import cv2
    m_u8 = (mask > 0.5).astype(np.uint8) * 255
    # invert: holes become blobs
    inv = cv2.bitwise_not(m_u8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    result = m_u8.copy()
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area <= min_size:
            result[labels == label] = 255  # fill hole
    return result.astype(np.float32) / 255.0


def remove_small_islands(mask: np.ndarray, min_size: int = 100) -> np.ndarray:
    """Remove tiny floating foreground islands (artifacts)."""
    import cv2
    m_u8 = (mask > 0.5).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m_u8, connectivity=8)
    result = np.zeros_like(m_u8)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_size:
            result[labels == label] = 255
    return result.astype(np.float32) / 255.0


def feather_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Soft feathering of edges via distance transform blending."""
    import cv2
    m_u8 = (mask > 0.5).astype(np.uint8) * 255
    dist = cv2.distanceTransform(m_u8, cv2.DIST_L2, 5)
    feathered = np.clip(dist / (radius + 1e-6), 0.0, 1.0)
    # blend original soft mask with feathered
    blended = mask * 0.5 + feathered * 0.5
    return blended.clip(0.0, 1.0)

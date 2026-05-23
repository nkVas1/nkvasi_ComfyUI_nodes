"""
utils/confidence.py

Confidence Map and Adaptive Trimap utilities.

Confidence Map
--------------
When multiple segmentation models run on the same image their outputs rarely
agree perfectly.  The *disagreement* between them is a gold signal that tells
you exactly where the mask is uncertain — usually along hair, fur, or
transparent edges.  We turn that signal into a per-pixel confidence score in
[0, 1] where:

    1.0  = all models agree  (safe to leave as-is)
    0.0  = maximum disagreement (must be processed by the matting engine)

Formula (per pixel):
    agreement = mean( |m_i - avg_mask| )   over all N models   [mean abs. dev.]
    confidence = 1 - clip(agreement / 0.5, 0, 1)

Adaptive Trimap
---------------
Classic build_trimap uses a FIXED erosion/dilation radius everywhere.
Adaptive trimap adjusts the unknown-band width per pixel using two signals:
    1. Low confidence  → wider unknown band (more work for matting engine)
    2. High local curvature of the contour  → wider band (complex boundary)
    3. High confidence + low curvature  → narrower band (trust the mask)

Result: fewer "over-matted" zones in flat BG areas and better hair coverage.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Confidence Map
# ---------------------------------------------------------------------------

def build_confidence_map(masks: list) -> np.ndarray:
    """
    Compute per-pixel confidence from a list of float32 H×W masks in [0,1].

    Returns a float32 H×W array in [0, 1]:
        1 = perfect agreement among all models
        0 = maximum disagreement
    """
    if len(masks) == 1:
        # Single model — uncertainty estimated from distance to decision boundary
        m = masks[0]
        dist_to_half = np.abs(m - 0.5)
        return np.clip(dist_to_half * 2.0, 0.0, 1.0).astype(np.float32)

    stacked = np.stack(masks, axis=0)       # N × H × W
    avg     = stacked.mean(axis=0)          # H × W
    mad     = np.abs(stacked - avg).mean(axis=0)   # mean absolute deviation
    confidence = np.clip(1.0 - mad / 0.5, 0.0, 1.0).astype(np.float32)
    return confidence


def confidence_weighted_merge(masks: list, weights: list) -> tuple:
    """
    Merge masks AND return confidence map in one pass.

    Returns:
        merged     : float32 H×W  weighted-average merged mask
        confidence : float32 H×W  per-pixel confidence [0,1]
    """
    total_w = sum(weights) or 1.0
    merged  = np.clip(
        sum(m * w for m, w in zip(masks, weights)) / total_w,
        0.0, 1.0,
    ).astype(np.float32)
    confidence = build_confidence_map(masks)
    return merged, confidence


# ---------------------------------------------------------------------------
# Adaptive Trimap
# ---------------------------------------------------------------------------

def build_adaptive_trimap(
    mask: np.ndarray,
    confidence: np.ndarray,
    min_band_px: int = 4,
    max_band_px: int = 24,
    curvature_weight: float = 0.5,
) -> np.ndarray:
    """
    Build a 3-value trimap with spatially-varying unknown-band width.

    Args:
        mask            : float32 H×W [0,1] merged alpha mask
        confidence      : float32 H×W [0,1] per-pixel confidence
        min_band_px     : minimum unknown-band half-width in pixels
        max_band_px     : maximum unknown-band half-width in pixels
        curvature_weight: 0=confidence only, 1=confidence+curvature equally

    Returns:
        trimap: uint8 H×W  {0=BG, 128=unknown, 255=FG}

    Algorithm:
        1. Compute local contour curvature via Laplacian of the binary mask edge.
        2. Blend (1-confidence) and curvature → band_signal in [0,1].
        3. Convert band_signal to per-pixel radius r in [min, max].
        4. For each pixel p on the boundary, its influence radius is r[p].
           Implemented efficiently:
           a. Erode  the binary FG mask by min_band_px → definite FG
           b. Dilate the binary FG mask by max_band_px → outer extent
           c. In the outer ring, weight erosion threshold by band_signal
              to get a spatially-varying cut that mimics per-pixel dilation.
    """
    import cv2
    hard = (mask > 0.5).astype(np.uint8) * 255

    # ---- 1. Contour curvature proxy: Laplacian of edge image ----
    edge     = cv2.Canny(hard, 50, 150)
    lap      = cv2.Laplacian(edge.astype(np.float32), cv2.CV_32F, ksize=3)
    curvature= np.abs(lap)
    curvature= curvature / (curvature.max() + 1e-6)
    # Spread curvature to a local neighbourhood
    k_spread = max(3, min_band_px * 2 + 1)
    curvature= cv2.GaussianBlur(
        np.ascontiguousarray(curvature), (k_spread, k_spread), sigmaX=k_spread / 3)
    curvature= np.clip(curvature / (curvature.max() + 1e-6), 0.0, 1.0)

    # ---- 2. Band signal: high where uncertain or curved ----
    uncertainty  = 1.0 - confidence
    band_signal  = np.clip(
        uncertainty * (1.0 - curvature_weight) + curvature * curvature_weight,
        0.0, 1.0,
    ).astype(np.float32)

    # ---- 3. Per-pixel radius map ----
    band_r = (min_band_px + band_signal * (max_band_px - min_band_px)).astype(np.float32)

    # ---- 4. Efficient approximation via distance transform ----
    # dist_fg[p]  = distance from p to nearest FG pixel
    # dist_bg[p]  = distance from p to nearest BG pixel
    dist_to_bg = cv2.distanceTransform(hard,              cv2.DIST_L2, 5)
    dist_to_fg = cv2.distanceTransform(255 - hard,        cv2.DIST_L2, 5)

    # A pixel is in the unknown zone when it is within band_r of the boundary
    # from either side.  The boundary is where dist_to_bg and dist_to_fg meet.
    is_unknown = (
        (dist_to_bg < band_r) & (dist_to_fg < band_r)
    ) | (
        # also include narrow boundary strip even if band is thin
        (dist_to_bg < min_band_px) | (dist_to_fg < min_band_px)
    )

    k_fg = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (min_band_px * 2 + 1, min_band_px * 2 + 1),
    )
    fg_core = cv2.erode(hard, k_fg)

    trimap               = np.zeros_like(hard)
    trimap[is_unknown]   = 128
    trimap[fg_core == 255] = 255    # definite FG overwrites unknown
    return trimap


# ---------------------------------------------------------------------------
# Confidence mask to ComfyUI tensor  (optional output from Ensemble)
# ---------------------------------------------------------------------------

def confidence_to_pil(confidence: np.ndarray):
    """Convert float32 H×W confidence [0,1] to grayscale PIL Image."""
    from PIL import Image
    return Image.fromarray(
        (confidence * 255).clip(0, 255).astype(np.uint8), mode="L"
    )

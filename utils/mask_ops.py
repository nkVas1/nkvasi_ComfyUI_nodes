"""
Mask post-processing operations.
All functions accept/return float32 numpy arrays in [0, 1].

Key design principle:
  Binary morphological analysis (connected components, island removal) is
  performed on a hard-thresholded copy of the mask, but the RESULT is always
  applied as a multiplier on the SOFT (float) mask — so semi-transparent
  edge pixels are never binarised away.

New in v0.5:
  trimap_guided_matting  — proper alpha matting in the unknown zone using
                           local colour sampling (no external dependencies)
  edge_detail_recovery   — restores fine details lost by morphological ops
                           using high-frequency image structure
  adaptive_bg_cleanup    — spatially-adaptive background suppression that
                           accounts for local contrast, not just global threshold
"""
import numpy as np


# ---------------------------------------------------------------------------
# Low-level primitives
# ---------------------------------------------------------------------------

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
    """
    import cv2
    guide_f = np.ascontiguousarray((guide * 255).clip(0, 255).astype(np.uint8)
                                   .astype(np.float32) / 255.0)
    src_f   = np.ascontiguousarray(mask.astype(np.float32))
    try:
        import cv2.ximgproc
        refined = cv2.ximgproc.guidedFilter(
            guide=guide_f, src=src_f, radius=radius, eps=eps)
    except Exception:
        src_u8  = (src_f * 255).clip(0, 255).astype(np.uint8)
        refined = cv2.bilateralFilter(
            np.ascontiguousarray(src_u8),
            d=max(1, radius * 2 + 1), sigmaColor=20, sigmaSpace=20,
        ).astype(np.float32) / 255.0
    return np.clip(refined, 0.0, 1.0)


def erode_expand_mask(mask: np.ndarray, offset: int) -> np.ndarray:
    import cv2
    abs_off = abs(offset)
    kernel  = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (abs_off * 2 + 1, abs_off * 2 + 1))
    mask_u8 = (mask * 255).clip(0, 255).astype(np.uint8)
    result  = cv2.dilate(mask_u8, kernel) if offset > 0 else cv2.erode(mask_u8, kernel)
    return result.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Binary helpers
# ---------------------------------------------------------------------------

def _binary_mask(mask: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    return (mask > thresh).astype(np.uint8) * 255


# ---------------------------------------------------------------------------
# Trimap construction
# ---------------------------------------------------------------------------

def build_trimap(
    mask: np.ndarray,
    erosion_px: int = 10,
    dilation_px: int = 10,
) -> np.ndarray:
    """
    Build a trimap from a soft mask.

    Returns uint8 array with three values:
      0   — definite background
      128 — unknown (transition zone)
      255 — definite foreground

    The unknown band width is controlled by erosion_px + dilation_px.
    Wider band = safer but slower matting.
    """
    import cv2
    hard = _binary_mask(mask, thresh=0.5)

    k_erode  = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erosion_px * 2 + 1, erosion_px * 2 + 1))
    k_dilate = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1))

    fg = cv2.erode(hard,  k_erode)
    ex = cv2.dilate(hard, k_dilate)

    trimap        = np.zeros_like(hard)
    trimap[ex == 255] = 128   # unknown
    trimap[fg == 255] = 255   # definite FG
    return trimap


# ---------------------------------------------------------------------------
# Trimap-guided alpha matting  (no external deps — pure numpy + cv2)
# ---------------------------------------------------------------------------

def trimap_guided_matting(
    soft_mask: np.ndarray,
    guide: np.ndarray,
    erosion_px: int = 12,
    dilation_px: int = 12,
    sample_radius: int = 20,
    n_best: int = 5,
) -> np.ndarray:
    """
    Alpha matting in the trimap unknown zone using local colour sampling.

    For each unknown pixel we:
      1. Sample candidate foreground colours from the nearby definite-FG band.
      2. Sample candidate background colours from the nearby definite-BG band.
      3. Find the best FG/BG pair that minimises the colour reconstruction
         error:  C_p ≈ alpha * F + (1-alpha) * B
      4. Estimate alpha from that pair.

    This is a fast approximation of Bayesian / closed-form matting that
    requires only numpy and OpenCV — no pymatting, no trimaps as images.

    Returns a refined float32 [0,1] mask with sub-pixel hair/edge detail.

    guide : float32 H×W×3 [0,1]  (original image at same resolution)
    """
    import cv2

    trimap = build_trimap(soft_mask, erosion_px=erosion_px, dilation_px=dilation_px)
    result = soft_mask.copy()

    unknown_mask = (trimap == 128)
    fg_mask      = (trimap == 255)
    bg_mask      = (trimap == 0)

    if unknown_mask.sum() == 0:
        return result

    # Pre-fetch guide pixels in the unknown zone
    unk_ys, unk_xs = np.where(unknown_mask)

    # Build KD-like lookup: fg and bg pixel positions + colours
    fg_ys, fg_xs = np.where(fg_mask)
    bg_ys, bg_xs = np.where(bg_mask)

    if len(fg_ys) < n_best or len(bg_ys) < n_best:
        # Trimap too tight — fall back to soft mask as-is
        return result

    fg_colors = guide[fg_ys, fg_xs]  # N_fg x 3
    bg_colors = guide[bg_ys, bg_xs]  # N_bg x 3
    fg_pos    = np.stack([fg_ys, fg_xs], axis=1).astype(np.float32)
    bg_pos    = np.stack([bg_ys, bg_xs], axis=1).astype(np.float32)

    # Process in batches to keep memory reasonable
    BATCH = 4096
    alphas = np.zeros(len(unk_ys), dtype=np.float32)

    for start in range(0, len(unk_ys), BATCH):
        end   = min(start + BATCH, len(unk_ys))
        py    = unk_ys[start:end].astype(np.float32)
        px    = unk_xs[start:end].astype(np.float32)
        pos_p = np.stack([py, px], axis=1)            # B x 2
        col_p = guide[unk_ys[start:end],
                      unk_xs[start:end]]              # B x 3

        # Spatial distances to all FG / BG pixels
        # Use top-K nearest by Euclidean position
        d_fg  = np.linalg.norm(
            pos_p[:, None, :] - fg_pos[None, :, :], axis=2)  # B x N_fg
        d_bg  = np.linalg.norm(
            pos_p[:, None, :] - bg_pos[None, :, :], axis=2)  # B x N_bg

        # Limit to spatial neighbourhood
        mask_fg_near = d_fg < sample_radius
        mask_bg_near = d_bg < sample_radius

        for bi in range(end - start):
            fg_near_idx = np.where(mask_fg_near[bi])[0]
            bg_near_idx = np.where(mask_bg_near[bi])[0]

            if len(fg_near_idx) < 2 or len(bg_near_idx) < 2:
                # No local samples — keep guided filter value
                alphas[start + bi] = float(soft_mask[
                    unk_ys[start + bi], unk_xs[start + bi]])
                continue

            # Pick the n_best nearest FG and BG candidates
            fg_k = fg_near_idx[np.argsort(d_fg[bi, fg_near_idx])[:n_best]]
            bg_k = bg_near_idx[np.argsort(d_bg[bi, bg_near_idx])[:n_best]]

            F_cands = fg_colors[fg_k]  # k x 3
            B_cands = bg_colors[bg_k]  # k x 3
            C_p     = col_p[bi]        # 3

            # Exhaustive alpha estimate for each F,B pair
            best_alpha = float(soft_mask[unk_ys[start+bi], unk_xs[start+bi]])
            best_err   = 1e9
            for F in F_cands:
                for B in B_cands:
                    dFB = F - B
                    denom = float(np.dot(dFB, dFB))
                    if denom < 1e-6:
                        continue
                    a = float(np.clip(np.dot(C_p - B, dFB) / denom, 0.0, 1.0))
                    err = float(np.linalg.norm(C_p - (a * F + (1 - a) * B)))
                    if err < best_err:
                        best_err   = err
                        best_alpha = a
            alphas[start + bi] = best_alpha

    result[unknown_mask] = alphas
    # Smooth only the unknown region slightly to avoid harsh transitions
    blur_r = 2
    blurred = cv2.GaussianBlur(
        np.ascontiguousarray(result), (blur_r * 2 + 1, blur_r * 2 + 1), sigmaX=1.0)
    result[unknown_mask] = blurred[unknown_mask]
    return np.clip(result, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Edge detail recovery
# ---------------------------------------------------------------------------

def edge_detail_recovery(
    mask: np.ndarray,
    guide: np.ndarray,
    strength: float = 0.4,
    radius: int = 3,
) -> np.ndarray:
    """
    Restore fine edge detail lost during morphological cleanup.

    Works by computing the high-frequency luminance structure of the guide
    image at edge pixels (where mask is 0.05-0.95), then modulating the
    mask by that structure.

    This recovers individual hair strands, eyelashes, and fabric fibres
    that were averaged away by Gaussian/bilateral smoothing.

    strength : 0.0 = no recovery, 1.0 = maximum (may re-introduce noise)
    """
    import cv2

    # High-pass of guide luminance
    gray  = cv2.cvtColor(
        (guide * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
    ).astype(np.float32) / 255.0
    k     = radius * 2 + 1
    blurred = cv2.GaussianBlur(
        np.ascontiguousarray(gray), (k, k), sigmaX=radius / 2)
    highpass = gray - blurred  # signed, range ~ -0.3 .. +0.3

    # Modulate only the uncertain/edge zone (not solid FG or solid BG)
    edge_zone = (mask > 0.05) & (mask < 0.95)
    result    = mask.copy()
    result[edge_zone] = np.clip(
        mask[edge_zone] + highpass[edge_zone] * strength, 0.0, 1.0
    )
    return result


# ---------------------------------------------------------------------------
# Adaptive background cleanup
# ---------------------------------------------------------------------------

def adaptive_bg_cleanup(
    soft_mask: np.ndarray,
    guide: np.ndarray,
    global_thresh: float = 0.10,
    local_window: int = 31,
) -> np.ndarray:
    """
    Suppresses near-zero mask values (background leaks) more aggressively
    where the local image contrast is LOW (uniform background areas),
    and more conservatively where contrast is HIGH (edge / hair zones).

    This prevents killing real semi-transparent hair pixels near high-
    frequency regions while still cleaning up solid background leaks.

    global_thresh : pixels below this value are candidate background
    local_window  : neighbourhood size for local contrast estimation
    """
    import cv2

    gray = cv2.cvtColor(
        (guide * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
    ).astype(np.float32) / 255.0

    # Local std deviation = local contrast
    k    = local_window | 1  # ensure odd
    mean = cv2.GaussianBlur(np.ascontiguousarray(gray), (k, k), sigmaX=k / 4)
    sq_mean = cv2.GaussianBlur(
        np.ascontiguousarray(gray ** 2), (k, k), sigmaX=k / 4)
    local_std = np.sqrt(np.clip(sq_mean - mean ** 2, 0, None))

    # Adaptive threshold: lower in flat areas (easy BG), higher near edges
    # local_std ~ 0 in flat BG → threshold stays at global_thresh
    # local_std ~ 0.1+ near hair edges → threshold rises, protecting soft alpha
    adaptive_thresh = global_thresh + local_std * 1.5

    result = soft_mask.copy()
    # Zero out pixels that are below the adaptive threshold for their location
    suppress = (soft_mask < adaptive_thresh) & (soft_mask < 0.5)
    # Smooth suppression boundary to avoid hard-edge ring
    suppress_f = suppress.astype(np.float32)
    suppress_f = cv2.GaussianBlur(
        np.ascontiguousarray(suppress_f), (5, 5), sigmaX=2.0)
    result = result * (1.0 - np.clip(suppress_f * 2.0, 0, 1))
    return np.clip(result, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Soft morphology
# ---------------------------------------------------------------------------

def soft_remove_holes(
    soft_mask: np.ndarray,
    min_hole_size: int = 500,
) -> np.ndarray:
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
    color_thresh: float = 0.13,
    detect_thresh: float = 0.25,
) -> np.ndarray:
    """
    Background island removal tuned for hair.
    Uses detect_thresh=0.25 to find semi-opaque BG patches from guided filter.
    """
    import cv2

    fg_loose  = _binary_mask(soft_mask, thresh=detect_thresh)
    bg_in_fg  = cv2.bitwise_not(fg_loose)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fg_closed    = cv2.morphologyEx(fg_loose, cv2.MORPH_CLOSE, kernel_close)
    search_area  = cv2.bitwise_and(bg_in_fg, fg_closed)

    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(
        search_area, connectivity=8)

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
            result[island_px] = result[island_px] * 0.05
            continue

        hole_color  = guide_u8[island_px].mean(axis=0).astype(np.float32) / 255.0
        neigh_color = guide_u8[neigh_px].mean(axis=0).astype(np.float32) / 255.0
        color_dist  = float(np.linalg.norm(hole_color - neigh_color))

        if color_dist > color_thresh:
            result[island_px] = result[island_px] * 0.04

    return result.clip(0.0, 1.0)


def feather_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    import cv2
    m_u8      = _binary_mask(mask)
    dist_in   = cv2.distanceTransform(m_u8, cv2.DIST_L2, 5)
    feather_w = np.clip(dist_in / (radius + 1e-6), 0.0, 1.0)
    return np.clip(mask * feather_w, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Legacy aliases
# ---------------------------------------------------------------------------

def remove_small_holes(mask: np.ndarray, min_size: int = 500) -> np.ndarray:
    return soft_remove_holes(mask, min_hole_size=min_size)


def remove_small_islands(mask: np.ndarray, min_size: int = 400) -> np.ndarray:
    return soft_remove_islands(mask, min_island_size=min_size)

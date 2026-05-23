"""
Mask post-processing operations.
All functions accept/return float32 numpy arrays in [0, 1].

Key design principle:
  Binary morphological analysis is performed on a hard-thresholded copy,
  but results are always applied to the SOFT (float) mask.

v0.5 additions:
  trimap_guided_matting  — O(N) memory alpha matting via KD-tree spatial lookup
  edge_detail_recovery   — high-freq luminance modulation on edge pixels
  adaptive_bg_cleanup    — spatially-adaptive contrast-aware BG suppression
  build_trimap           — construct FG/Unknown/BG trimap from soft mask
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
    guide_f = np.ascontiguousarray(
        (guide * 255).clip(0, 255).astype(np.uint8).astype(np.float32) / 255.0
    )
    src_f = np.ascontiguousarray(mask.astype(np.float32))
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
    result  = (cv2.dilate(mask_u8, kernel) if offset > 0
               else cv2.erode(mask_u8, kernel))
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
    Build a 3-value trimap from a soft mask.
      255 — definite foreground
      128 — unknown (transition band)
        0 — definite background
    """
    import cv2
    hard = _binary_mask(mask, thresh=0.5)
    k_e  = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erosion_px  * 2 + 1, erosion_px  * 2 + 1))
    k_d  = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1))
    fg   = cv2.erode (hard, k_e)
    ext  = cv2.dilate(hard, k_d)
    trimap           = np.zeros_like(hard)
    trimap[ext == 255] = 128
    trimap[fg  == 255] = 255
    return trimap


# ---------------------------------------------------------------------------
# Trimap-guided alpha matting  (O(N) memory — KD-tree spatial lookup)
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
    Alpha matting in the unknown trimap zone via local colour sampling.

    Memory-safe implementation:
      • Uses scipy.spatial.cKDTree for O(N log N) spatial nearest-neighbour
        queries — never builds an (N_unknown × N_fg) distance matrix.
      • Falls back to a grid-subsampled approximation if scipy is unavailable.

    For each unknown pixel p:
      1. Query the K nearest definite-FG pixels within sample_radius.
      2. Query the K nearest definite-BG pixels within sample_radius.
      3. For each (F, B) candidate pair, solve:
            alpha = clip( dot(C_p - B, F - B) / dot(F - B, F - B), 0, 1 )
      4. Keep the alpha with lowest reconstruction error
            err = ||C_p - alpha*F - (1-alpha)*B||.

    guide : float32 H×W×3 [0,1] at the same resolution as soft_mask.
    """
    trimap = build_trimap(soft_mask, erosion_px=erosion_px, dilation_px=dilation_px)
    result = soft_mask.copy()

    unk_ys, unk_xs = np.where(trimap == 128)
    fg_ys,  fg_xs  = np.where(trimap == 255)
    bg_ys,  bg_xs  = np.where(trimap == 0)

    if len(unk_ys) == 0 or len(fg_ys) < n_best or len(bg_ys) < n_best:
        return result

    fg_pos    = np.stack([fg_ys, fg_xs], axis=1).astype(np.float32)
    bg_pos    = np.stack([bg_ys, bg_xs], axis=1).astype(np.float32)
    unk_pos   = np.stack([unk_ys, unk_xs], axis=1).astype(np.float32)
    fg_colors = guide[fg_ys, fg_xs].astype(np.float32)   # N_fg × 3
    bg_colors = guide[bg_ys, bg_xs].astype(np.float32)   # N_bg × 3
    unk_colors= guide[unk_ys, unk_xs].astype(np.float32)  # N_unk × 3

    # ---------- KD-tree path (preferred) ----------
    try:
        from scipy.spatial import cKDTree
        fg_tree = cKDTree(fg_pos)
        bg_tree = cKDTree(bg_pos)
        _matting_kdtree(
            result, unk_ys, unk_xs, unk_colors,
            fg_tree, fg_colors, bg_tree, bg_colors,
            soft_mask, sample_radius, n_best,
        )
    except ImportError:
        # ---------- Fallback: subsample FG/BG to at most MAX_SAMPLES points ----------
        MAX_SAMPLES = 4000
        if len(fg_ys) > MAX_SAMPLES:
            idx = np.random.choice(len(fg_ys), MAX_SAMPLES, replace=False)
            fg_pos_s    = fg_pos[idx];    fg_colors_s = fg_colors[idx]
        else:
            fg_pos_s    = fg_pos;         fg_colors_s = fg_colors
        if len(bg_ys) > MAX_SAMPLES:
            idx = np.random.choice(len(bg_ys), MAX_SAMPLES, replace=False)
            bg_pos_s    = bg_pos[idx];    bg_colors_s = bg_colors[idx]
        else:
            bg_pos_s    = bg_pos;         bg_colors_s = bg_colors

        _matting_bruteforce(
            result, unk_ys, unk_xs, unk_colors,
            fg_pos_s, fg_colors_s, bg_pos_s, bg_colors_s,
            soft_mask, sample_radius, n_best,
        )

    # Light blur only in unknown zone to remove per-pixel noise
    import cv2
    blurred = cv2.GaussianBlur(
        np.ascontiguousarray(result.astype(np.float32)), (5, 5), sigmaX=1.5)
    result[trimap == 128] = blurred[trimap == 128]
    return np.clip(result, 0.0, 1.0)


def _matting_kdtree(
    result, unk_ys, unk_xs, unk_colors,
    fg_tree, fg_colors, bg_tree, bg_colors,
    soft_mask, sample_radius, n_best,
):
    """KD-tree alpha estimation — O(N_unk × K log N) time, O(N_unk × K) memory."""
    K         = min(n_best, len(fg_colors), len(bg_colors))
    BATCH     = 2048
    n_unk     = len(unk_ys)

    for start in range(0, n_unk, BATCH):
        end     = min(start + BATCH, n_unk)
        pos_b   = np.stack([unk_ys[start:end],
                            unk_xs[start:end]], axis=1).astype(np.float32)
        col_b   = unk_colors[start:end]  # B × 3

        # Query K nearest in each tree within sample_radius
        fg_dists, fg_idx = fg_tree.query(pos_b, k=K, distance_upper_bound=sample_radius + 1e-3)
        bg_dists, bg_idx = bg_tree.query(pos_b, k=K, distance_upper_bound=sample_radius + 1e-3)

        for bi in range(end - start):
            # Valid hits only (distance < radius)
            fg_valid = fg_idx[bi][fg_dists[bi] < sample_radius]
            bg_valid = bg_idx[bi][bg_dists[bi] < sample_radius]

            if len(fg_valid) == 0 or len(bg_valid) == 0:
                # No local samples — keep the guided filter value
                result[unk_ys[start + bi], unk_xs[start + bi]] = \
                    float(soft_mask[unk_ys[start + bi], unk_xs[start + bi]])
                continue

            F_cands = fg_colors[fg_valid[:n_best]]  # k × 3
            B_cands = bg_colors[bg_valid[:n_best]]  # k × 3
            C_p     = col_b[bi]                      # 3

            best_a   = float(soft_mask[unk_ys[start + bi], unk_xs[start + bi]])
            best_err = 1e9
            for F in F_cands:
                for B in B_cands:
                    dFB   = F - B
                    denom = float(np.dot(dFB, dFB))
                    if denom < 1e-6:
                        continue
                    a   = float(np.clip(np.dot(C_p - B, dFB) / denom, 0.0, 1.0))
                    err = float(np.linalg.norm(C_p - (a * F + (1.0 - a) * B)))
                    if err < best_err:
                        best_err = err
                        best_a   = a
            result[unk_ys[start + bi], unk_xs[start + bi]] = best_a


def _matting_bruteforce(
    result, unk_ys, unk_xs, unk_colors,
    fg_pos, fg_colors, bg_pos, bg_colors,
    soft_mask, sample_radius, n_best,
):
    """Fallback brute-force matting on a subsampled FG/BG set."""
    BATCH = 512   # smaller batch since scipy unavailable, memory is tighter
    n_unk = len(unk_ys)

    for start in range(0, n_unk, BATCH):
        end   = min(start + BATCH, n_unk)
        pos_b = np.stack([unk_ys[start:end],
                          unk_xs[start:end]], axis=1).astype(np.float32)
        col_b = unk_colors[start:end]

        # Brute-force distances on the already-subsampled sets
        d_fg = np.linalg.norm(pos_b[:, None, :] - fg_pos[None, :, :], axis=2)  # B × N_fg
        d_bg = np.linalg.norm(pos_b[:, None, :] - bg_pos[None, :, :], axis=2)  # B × N_bg

        for bi in range(end - start):
            fg_near = np.where(d_fg[bi] < sample_radius)[0]
            bg_near = np.where(d_bg[bi] < sample_radius)[0]

            if len(fg_near) == 0 or len(bg_near) == 0:
                result[unk_ys[start + bi], unk_xs[start + bi]] = \
                    float(soft_mask[unk_ys[start + bi], unk_xs[start + bi]])
                continue

            fg_k  = fg_near[np.argsort(d_fg[bi, fg_near])[:n_best]]
            bg_k  = bg_near[np.argsort(d_bg[bi, bg_near])[:n_best]]
            C_p   = col_b[bi]

            best_a, best_err = float(soft_mask[unk_ys[start+bi], unk_xs[start+bi]]), 1e9
            for F in fg_colors[fg_k]:
                for B in bg_colors[bg_k]:
                    dFB   = F - B
                    denom = float(np.dot(dFB, dFB))
                    if denom < 1e-6:
                        continue
                    a   = float(np.clip(np.dot(C_p - B, dFB) / denom, 0.0, 1.0))
                    err = float(np.linalg.norm(C_p - (a * F + (1.0 - a) * B)))
                    if err < best_err:
                        best_err = err
                        best_a   = a
            result[unk_ys[start + bi], unk_xs[start + bi]] = best_a


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
    Restore fine edge detail (hair, eyelashes) lost during smoothing.
    Modulates mask in the edge zone (0.05–0.95) by the high-pass luminance
    of the guide image.
    """
    import cv2
    gray    = cv2.cvtColor(
        (guide * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
    ).astype(np.float32) / 255.0
    k       = radius * 2 + 1
    blurred = cv2.GaussianBlur(np.ascontiguousarray(gray), (k, k), sigmaX=radius / 2)
    highpass= gray - blurred  # signed
    edge_zone = (mask > 0.05) & (mask < 0.95)
    result    = mask.copy()
    result[edge_zone] = np.clip(
        mask[edge_zone] + highpass[edge_zone] * strength, 0.0, 1.0)
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
    Spatially-adaptive BG suppression.
    • Low local contrast (flat background) → aggressive threshold.
    • High local contrast (hair/edge zone) → conservative, preserves soft alpha.
    """
    import cv2
    gray     = cv2.cvtColor(
        (guide * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
    ).astype(np.float32) / 255.0
    k        = local_window | 1
    mean     = cv2.GaussianBlur(np.ascontiguousarray(gray),     (k, k), sigmaX=k / 4)
    sq_mean  = cv2.GaussianBlur(np.ascontiguousarray(gray**2),  (k, k), sigmaX=k / 4)
    local_std= np.sqrt(np.clip(sq_mean - mean**2, 0, None))
    adaptive_thresh = global_thresh + local_std * 1.5
    suppress = (soft_mask < adaptive_thresh) & (soft_mask < 0.5)
    suppress_f = cv2.GaussianBlur(
        np.ascontiguousarray(suppress.astype(np.float32)), (5, 5), sigmaX=2.0)
    result = soft_mask * (1.0 - np.clip(suppress_f * 2.0, 0, 1))
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
    hard    = _binary_mask(soft_mask)
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(hard, connectivity=8)
    keep    = np.zeros_like(hard)
    for lbl in range(1, nlabels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_island_size:
            keep[labels == lbl] = 255
    result  = soft_mask.copy()
    result[keep == 0] = 0.0
    return result.clip(0.0, 1.0)


def hair_bg_island_removal(
    soft_mask: np.ndarray,
    guide: np.ndarray,
    max_island_size: int = 2000,
    color_thresh: float = 0.13,
    detect_thresh: float = 0.25,
) -> np.ndarray:
    """Colour-gated BG patch removal tuned for hair (legacy, kept for compatibility)."""
    import cv2
    fg_loose     = _binary_mask(soft_mask, thresh=detect_thresh)
    bg_in_fg     = cv2.bitwise_not(fg_loose)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fg_closed    = cv2.morphologyEx(fg_loose, cv2.MORPH_CLOSE, kernel_close)
    search_area  = cv2.bitwise_and(bg_in_fg, fg_closed)
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(
        search_area, connectivity=8)
    kernel_nb = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg_strict = _binary_mask(soft_mask, thresh=0.5)
    fg_dilate = cv2.dilate(fg_strict, kernel_nb)
    guide_u8  = (guide * 255).clip(0, 255).astype(np.uint8)
    result    = soft_mask.copy()
    for lbl in range(1, nlabels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area > max_island_size:
            continue
        island_px = (labels == lbl)
        neigh_px  = (fg_dilate == 255) & ~island_px
        if neigh_px.sum() < 10:
            result[island_px] *= 0.05
            continue
        hole_color  = guide_u8[island_px].mean(axis=0).astype(np.float32) / 255.0
        neigh_color = guide_u8[neigh_px].mean(axis=0).astype(np.float32) / 255.0
        if float(np.linalg.norm(hole_color - neigh_color)) > color_thresh:
            result[island_px] *= 0.04
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

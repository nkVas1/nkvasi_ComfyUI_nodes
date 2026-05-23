"""
utils/bg_color_suppress.py

Background Color Suppression for uniform backgrounds.

Problem
-------
When the background is a simple uniform color (blue sky, studio white, green
screen, solid gradient) the segmentation models leave a thin fringe of
residual BG color baked into the alpha edge. On a neutral grey composite it
appears as a faint colored halo ("blue tint on hair ends", etc.).

Approach
--------
1. Detect background color: sample pixels where mask < bg_mask_thresh,
   compute median color in HSV space.
2. Measure uniformity: stddev of Hue channel in BG zone. If stddev < hue_std_thresh
   the BG is considered "uniform" and suppression is activated.
3. Safety check: count FG pixels (mask > fg_mask_thresh) that fall within the
   BG hue range. If that fraction > max_fg_overlap (default 0.25) the
   suppression is skipped entirely to avoid eating object parts.
4. Suppression: for each pixel in the edge zone (lock_bg < mask < lock_fg),
   compute HSV color distance to median BG color. Weight = proximity to BG
   color × (1 - mask) × strength. Subtract weight from alpha.
5. The suppressed mask is then re-locked at original FG/BG cores.

Outputs a modified float32 H×W mask.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorised RGB [0,1] H×W×3 → HSV [0,1] H×W×3. Pure numpy, no cv2."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc + 1e-8

    h = np.zeros_like(maxc)
    mask_r = (maxc == r) & (delta > 1e-7)
    mask_g = (maxc == g) & (delta > 1e-7)
    mask_b = (maxc == b) & (delta > 1e-7)
    h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6
    h[mask_g] = (b[mask_g] - r[mask_g]) / delta[mask_g] + 2
    h[mask_b] = (r[mask_b] - g[mask_b]) / delta[mask_b] + 4
    h = h / 6.0

    s = np.where(maxc > 1e-6, delta / (maxc + 1e-8), 0.0)
    v = maxc
    return np.stack([h, s, v], axis=-1).astype(np.float32)


def _hue_distance(h1: np.ndarray, h2: float) -> np.ndarray:
    """Angular distance between hue values in [0,1]. Returns [0, 0.5]."""
    d = np.abs(h1 - h2)
    return np.minimum(d, 1.0 - d)   # wrap around


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_uniform_bg(
    guide_rgb: np.ndarray,
    mask: np.ndarray,
    bg_mask_thresh: float = 0.10,
    hue_std_thresh: float = 0.06,
) -> dict | None:
    """
    Analyse the background region and return a descriptor if it is uniform.

    Args:
        guide_rgb      : float32 H×W×3 image in [0,1]
        mask           : float32 H×W alpha in [0,1]
        bg_mask_thresh : pixels with mask < this are considered background
        hue_std_thresh : if BG Hue stddev < this → uniform BG detected

    Returns:
        dict with keys: median_hsv (3,), hue_std, sat_mean, hue_range_lo/hi
        OR None if BG is not uniform enough.
    """
    bg_pixels_rgb = guide_rgb[mask < bg_mask_thresh]   # N×3
    if len(bg_pixels_rgb) < 200:
        return None

    bg_hsv = _rgb_to_hsv(bg_pixels_rgb.reshape(-1, 1, 3)).reshape(-1, 3)
    h, s, v = bg_hsv[:, 0], bg_hsv[:, 1], bg_hsv[:, 2]

    # Low saturation BG (white/grey/black) → no fringe color to suppress
    if float(np.median(s)) < 0.10:
        return None

    hue_std = float(np.std(h))
    if hue_std > hue_std_thresh:
        return None

    med_h = float(np.median(h))
    med_s = float(np.median(s))
    med_v = float(np.median(v))
    hue_spread = min(hue_std * 3.0, 0.15)   # ±spread defines the suppression band

    return {
        "median_hsv":  np.array([med_h, med_s, med_v], dtype=np.float32),
        "hue_std":     hue_std,
        "sat_mean":    float(np.mean(s)),
        "hue_lo":      med_h - hue_spread,
        "hue_hi":      med_h + hue_spread,
    }


def suppress_bg_color(
    mask: np.ndarray,
    guide_rgb: np.ndarray,
    bg_info: dict,
    strength: float = 0.65,
    max_fg_overlap: float = 0.25,
    lock_bg: float = 0.04,
    lock_fg: float = 0.92,
    edge_feather: float = 0.30,
) -> np.ndarray:
    """
    Suppress residual BG color from alpha edge zone.

    Args:
        mask          : float32 H×W current alpha
        guide_rgb     : float32 H×W×3 original image
        bg_info       : dict returned by detect_uniform_bg()
        strength      : suppression intensity [0,1]
        max_fg_overlap: safety — abort if BG color covers > this fraction of FG
        lock_bg/fg    : core zones never touched
        edge_feather  : how far into the FG to extend suppression (0=edge only)

    Returns:
        float32 H×W modified alpha
    """
    import cv2

    if bg_info is None or strength <= 0.0:
        return mask

    hsv = _rgb_to_hsv(guide_rgb)           # H×W×3
    img_hue = hsv[..., 0]
    img_sat = hsv[..., 1]

    med_h   = float(bg_info["median_hsv"][0])
    med_s   = float(bg_info["median_hsv"][1])
    hue_lo  = float(bg_info["hue_lo"])
    hue_hi  = float(bg_info["hue_hi"])
    spread  = max(abs(hue_hi - med_h), 0.04)

    # ---- Safety check: FG overlap ----
    fg_zone  = mask > 0.5
    in_range = (_hue_distance(img_hue, med_h) < spread) & (img_sat > 0.15)
    fg_in_bg_color = (fg_zone & in_range).sum()
    fg_total = fg_zone.sum()
    if fg_total > 0 and (fg_in_bg_color / fg_total) > max_fg_overlap:
        # Too much of the subject has the same color as the BG → skip
        return mask

    # ---- Per-pixel color proximity to BG ----
    hue_dist    = _hue_distance(img_hue, med_h)           # [0, 0.5]
    hue_prox    = np.clip(1.0 - hue_dist / spread, 0.0, 1.0)
    sat_prox    = np.clip(img_sat / (med_s + 1e-6), 0.0, 1.0)
    color_score = (hue_prox * sat_prox).astype(np.float32)  # high = looks like BG

    # ---- Edge zone: slightly extend into FG side for fringe removal ----
    fg_edge_limit = min(lock_fg, 0.5 + edge_feather)
    edge_zone     = (mask > lock_bg) & (mask < fg_edge_limit)

    # ---- Soft weight: fade off as we move deeper into FG ----
    # depth_in_fg in [0,1]: 0 at lock_bg boundary, 1 at fg_edge_limit
    depth_in_fg = np.clip(
        (mask - lock_bg) / max(fg_edge_limit - lock_bg, 1e-3), 0.0, 1.0)
    fade = (1.0 - depth_in_fg) ** 1.5   # steep roll-off deeper in FG

    suppress_w = np.clip(color_score * fade * strength, 0.0, 0.8)

    result = mask.copy()
    result[edge_zone] = np.clip(
        mask[edge_zone] - suppress_w[edge_zone] * mask[edge_zone],
        0.0, 1.0,
    )

    # ---- Re-lock cores ----
    result[mask >= lock_fg] = mask[mask >= lock_fg]
    result[mask <= lock_bg] = mask[mask <= lock_bg]

    # ---- Smooth transition to avoid hard steps ----
    k = 3
    blurred = cv2.GaussianBlur(np.ascontiguousarray(result), (k, k), sigmaX=1)
    blend_zone = edge_zone
    result[blend_zone] = (
        result[blend_zone] * 0.7 + blurred[blend_zone] * 0.3
    )
    result[mask >= lock_fg] = mask[mask >= lock_fg]
    result[mask <= lock_bg] = mask[mask <= lock_bg]

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def auto_bg_color_suppress(
    mask: np.ndarray,
    guide_rgb: np.ndarray,
    strength: float = 0.65,
    max_fg_overlap: float = 0.25,
    lock_bg: float = 0.04,
    lock_fg: float = 0.92,
    hue_std_thresh: float = 0.06,
) -> tuple:
    """
    One-call convenience: detect + suppress.

    Returns:
        (modified_mask, bg_info_or_None)
        bg_info is None if suppression was not applied.
    """
    bg_info = detect_uniform_bg(
        guide_rgb, mask,
        bg_mask_thresh=lock_bg * 2.5,
        hue_std_thresh=hue_std_thresh,
    )
    if bg_info is None:
        return mask, None

    result = suppress_bg_color(
        mask, guide_rgb, bg_info,
        strength=strength,
        max_fg_overlap=max_fg_overlap,
        lock_bg=lock_bg,
        lock_fg=lock_fg,
    )
    return result, bg_info

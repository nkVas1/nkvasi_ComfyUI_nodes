# Changelog

## [0.3.0] — 2026-05-23
### Added
- **Soft alpha pipeline** — all mask operations now preserve semi-transparent edge pixels;
  hair/fur edges output real partial opacity instead of binary cut
- `hair_bg_island_removal()` — colour-distance-gated removal of background patches
  between hair strands; patches similar in colour to adjacent hair are kept semi-transparent
- `soft_remove_holes()` / `soft_remove_islands()` — morphological cleanup that operates
  on binary analysis but applies results as multipliers on the soft float mask
- `NkVasi_SaveImageAlpha` node — saves IMAGE + MASK as proper RGBA PNG with transparency;
  embeds workflow metadata like the built-in Save Image node
- `guided_filter_mask()` now uses full RGB guide (3-channel) for better edge detection
- `NkVasi_MaskRefine` now accepts optional IMAGE input to enable guided filter refinement

### Changed
- `rmbg_ensemble.py` post-processing pipeline fully rewritten (7 documented steps)
- Default `process_resolution` raised 1024 → 1536 (BiRefNet-HR full quality)
- Default `feather_edges` changed 0 → 2 (soft transition on all outputs by default)
- Default `sensitivity` changed 0.5 → 0.45 (retain more semi-transparent hair pixels)
- `refine_foreground_colors()` rewritten with per-pixel BG estimation via large-kernel blur;
  `strength` parameter added (default 0.60); removes white halo on bright backgrounds
- `NkVasi_MaskRefine` default `threshold` changed 0.5 → 0.0 (pass-through soft mask)

### Fixed
- `guided_filter_mask()` crash — added `np.ascontiguousarray` on both arrays before
  passing to `cv2.ximgproc.guidedFilter` (OpenCV assertion `-215`)
- `except` clause widened to `(AttributeError, cv2.error)` for robust bilateral fallback

## [0.1.0] — 2026-05-23
### Added
- `NkVasi_RMBG_Node` — single-model background removal
- `NkVasi_RMBG_Ensemble` — multi-model ensemble
- `NkVasi_MaskRefine` — standalone mask post-processing
- `NkVasi_MaskTools` — apply mask to image with background options
- Foreground color estimation, FP16 support, checkerboard background
- Lazy model loading with in-process caching

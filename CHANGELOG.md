# Changelog

## [0.4.0] — 2026-05-23
### Fixed
- `hair_bg_island_removal`: was using `thresh=0.5` to binarise — missed all
  semi-opaque BG patches (values 0.25–0.45) left by guided filter.
  Now uses `detect_thresh=0.25` + closes the loose FG hull to limit search area.
- `union` merge mode was identical to `max` — both computed `np.maximum.reduce`.
  `union` is now correctly `max over all masks`; old `max (aggressive)` removed.
- `feather_mask` formula corrected; was accidentally zeroing out exterior pixels.

### Added
- `consensus` merge mode: `mean(masks) × √fraction_models_agree`.
  Best all-around mode — keeps hair detail while removing background leaks.
  Set as new default merge mode.
- `soft_intersection` merge mode: geometric mean of all masks.
  Penalises low-confidence pixels without hard intersection cutoff.
- `NkVasi_AlphaPreview` node: composites IMAGE+MASK over adjustable
  checkerboard for instant alpha quality preview in ComfyUI.
- `island_size` and `color_thresh` exposed as UI params in Ensemble node
  for per-image fine-tuning.

## [0.3.0] — 2026-05-23
### Added
- Soft alpha pipeline — semi-transparent hair edges
- `hair_bg_island_removal()` with colour-distance gating
- `soft_remove_holes()` / `soft_remove_islands()`
- `NkVasi_SaveImageAlpha` node — real RGBA PNG output
- `NkVasi_MaskRefine` now accepts IMAGE for guided filter refinement

### Changed
- `process_resolution` default 1024 → 1536
- `feather_edges` default 0 → 2
- `sensitivity` default 0.5 → 0.45
- `refine_foreground_colors` rewritten with per-pixel BG estimation

### Fixed
- `guided_filter_mask` OpenCV assertion `-215` via `np.ascontiguousarray`

## [0.1.0] — 2026-05-23
### Added
- `NkVasi_RMBG_Node`, `NkVasi_RMBG_Ensemble`, `NkVasi_MaskRefine`,
  `NkVasi_MaskTools`, foreground decontamination, FP16, lazy model loading

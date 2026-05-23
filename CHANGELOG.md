# Changelog

## [0.1.0] — 2026-05-23
### Added
- `NkVasi_RMBG_Node` — single-model background removal (BiRefNet-HR/dynamic/matting/general, BEN2, RMBG-2.0, InSPyReNet)
- `NkVasi_RMBG_Ensemble` — multi-model ensemble with weighted_avg / intersection / union / max merge modes
- `NkVasi_MaskRefine` — standalone mask post-processing (blur, erode/expand, feather, hole fill, island removal)
- `NkVasi_MaskTools` — apply mask to image with background options
- Foreground color estimation to remove background bleed from edges
- FP16 inference support for BiRefNet and RMBG-2.0
- Checkerboard background output option
- Full lazy model loading with in-process caching

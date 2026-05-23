# 🎭 nkVasi ComfyUI — Professional Background Removal Nodes

> **Goal:** Near-professional-grade background removal inside ComfyUI — clean edges, preserved hair/fur/glass, zero color bleeding, no artifacts.

---

## Why these nodes?

Most existing nodes use a single model with default settings. They leave:
- Jagged edges on hair / fur / transparent materials
- Color bleeding (background color "leaks" into edge pixels)
- Small floating islands and holes inside the mask
- Quality loss due to downscaling without post-processing

**nkVasi nodes solve all of this** by combining:
1. **SOTA models** — BiRefNet-HR, BiRefNet-matting, BEN2, RMBG-2.0, InSPyReNet
2. **Multi-model ensemble** — merge masks from 2-4 models for maximum precision
3. **Advanced post-processing** — hole fill, island removal, feathering, edge refinement
4. **Foreground color estimation** — removes background color bleed on semi-transparent edges

---

## Nodes

### 🎭 Remove BG (nkVasi)
Single-model background removal with full control.

| Parameter | Description |
|-----------|-------------|
| `model` | Choose from BiRefNet-HR, BiRefNet-dynamic, BiRefNet-matting, BEN2, RMBG-2.0, InSPyReNet |
| `process_resolution` | Inference resolution (256–2048). Higher = better detail, more VRAM |
| `sensitivity` | Threshold tuning — raise to be stricter, lower to keep more |
| `mask_blur` | Gaussian blur on mask edges (0 = sharp) |
| `mask_offset` | Expand (+) or shrink (–) mask boundary |
| `refine_foreground` | Remove background color bleed from edges |
| `remove_holes` | Fill small transparent holes in the subject |
| `remove_islands` | Remove tiny floating artifact blobs |
| `fp16` | Use FP16 inference (faster, less VRAM, ~0% quality loss) |

**Outputs:** `IMAGE` (composited), `MASK` (float, 0–1)

---

### 🎭 Remove BG Ensemble (nkVasi) ⭐ Recommended
Runs 2–4 models simultaneously and merges their masks.

This is the **highest quality** mode. By combining BiRefNet-HR + BiRefNet-matting + BEN2:
- Hair/fur: BiRefNet-matting and BEN2 excel
- Hard edges: BiRefNet-HR excels
- The ensemble catches what any single model misses

| Merge Mode | When to use |
|------------|-------------|
| `weighted_avg` | Default — best balance |
| `intersection` | Strict — removes all ambiguous areas |
| `union` | Keep maximum detail |
| `max` | Most aggressive foreground detection |

---

### 🔬 Mask Refine (nkVasi)
Standalone mask post-processor. Chain after **any** background removal node.

- `blur_radius` — smooth edges
- `erode_expand` — pixel-level boundary adjustment
- `feather_edges` — distance-transform feathering
- `threshold` — rebinarize with custom threshold
- `remove_holes` / `remove_islands` — clean artifacts
- `min_hole_size` / `min_island_size` — control cleanup sensitivity

---

### 🛠️ Mask Tools (nkVasi)
Apply any external mask to an image with background color/checkerboard options.
Supports `invert_mask` and `refine_foreground`.

---

## Installation

```bash
# Clone into ComfyUI custom_nodes directory
cd ComfyUI/custom_nodes
git clone https://github.com/nkVas1/nkvasi_ComfyUI_nodes.git

# Install dependencies
pip install -r nkvasi_ComfyUI_nodes/requirements.txt
```

All models are **downloaded automatically** from HuggingFace on first use.

---

## Recommended Workflows

### Maximum Quality (portraits / products)
```
Load Image → Remove BG Ensemble (BiRefNet-HR + BiRefNet-matting + BEN2)
          → Mask Refine (feather=3, remove_holes=True)
          → Preview / Save
```

### Fast / Batch Processing
```
Load Image → Remove BG (BiRefNet-general, res=512, fp16=True)
```

### Chaining with existing nodes
```
Any external RMBG node → Mask Refine (nkVasi) → Mask Tools (nkVasi)
```

---

## Model Comparison

| Model | Best for | Resolution | Notes |
|-------|----------|------------|-------|
| BiRefNet-HR | General, products, sharp edges | up to 2048 | Best all-round |
| BiRefNet-dynamic | Any resolution input | 256–2304 | Robust, flexible |
| BiRefNet-matting | Hair, fur, transparent objects | 1024–2048 | Soft edge specialist |
| BiRefNet-general | Fast batch processing | 1024 | Balanced |
| BEN2 | Hair, 4K, complex edges | 1024 | CGM pipeline |
| RMBG-2.0 | Products, objects (non-commercial) | 1024 | BRIA AI, non-commercial license |
| InSPyReNet | Human portraits | 1024 | Portrait specialist |

---

## License

MIT License — code is free to use commercially.

Note: **RMBG-2.0 model** (briaai) has a **non-commercial license**. Use BiRefNet models for commercial projects.

---

## Credits

- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) — ZhengPeng7 et al., CAAI AIR 2024
- [BEN2](https://huggingface.co/PramaLLC/BEN2) — PramaLLC, Confidence Guided Matting
- [RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) — BRIA AI
- [InSPyReNet / transparent-background](https://github.com/plemeri/InSPyReNet) — plemeri
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — comfyanonymous

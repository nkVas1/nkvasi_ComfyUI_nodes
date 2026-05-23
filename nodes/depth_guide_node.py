"""
NkVasi_DepthGuide v2.1

Depth Pro — guided mask refinement node.

Workflow position:
  IMAGE ─┬─> RMBG_Ensemble ──> mask ──────────────────> DepthGuide ─> mask_refined ─> MattingRefine
         └────────────────────────────────────────────> DepthGuide
                                  confidence_map ─────────> DepthGuide (optional)

v2.1 additions
--------------
  • BG hard cut: pixels where depth is very far + alpha still semi-transparent
    are driven straight to alpha=0 (controlled by depth_bg_hard / bg_hard_cut_strength).
    This eliminates stubborn fringe that soft suppression couldn't finish.
  • Edge crisp cap raised to 1.0: at strength=1.0 the depth-edge decision
    fully replaces ensemble alpha at strong depth boundaries.
  • Tunable hard-lock threshold (hard_lock_hi): default 0.90 instead of
    fixed 0.95 — lets BG veto reach deeper into semi-transparent fringe zones.
"""
import torch
import numpy as np
from PIL import Image

from ..utils.model_loader   import load_depth_pro
from ..utils.depth_guide    import (
    depth_guided_mask, depth_adaptive_trimap,
    depth_to_pil, _normalise_depth,
)
from ..utils.image_utils    import tensor_to_pil, pil_mask_to_tensor


class NkVasi_DepthGuide:
    """
    v2.1: Depth Pro guided mask refinement.
    Insert between RMBG_Ensemble and MattingRefine for best results.
    """

    CATEGORY = "🎭 nkVasi/Background Removal"
    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("mask_refined", "depth_map")
    FUNCTION = "refine_with_depth"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask":  ("MASK",),
            },
            "optional": {
                "confidence": ("MASK", {
                    "tooltip": "confidence_map from RMBG_Ensemble — combined with depth for better trimap"
                }),

                # ── Pass 1: BG soft suppression ──────────────────────────
                "bg_suppress_strength": ("FLOAT", {
                    "default": 0.70, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Soft BG suppression strength — proportional alpha reduction for far-depth pixels"
                }),
                "fg_depth_percentile": ("FLOAT", {
                    "default": 80.0, "min": 50.0, "max": 99.0, "step": 1.0,
                    "tooltip": "Percentile of FG depth used as soft-BG threshold. Higher = more conservative"
                }),
                "depth_bg_threshold": ("FLOAT", {
                    "default": 0.75, "min": 0.50, "max": 1.00, "step": 0.01,
                    "tooltip": "Soft-veto threshold: depth above this starts proportional suppression"
                }),

                # ── Pass 1+: BG hard cut (NEW v2.1) ──────────────────────
                "depth_bg_hard": ("FLOAT", {
                    "default": 0.90, "min": 0.60, "max": 1.00, "step": 0.01,
                    "tooltip": "Hard-cut threshold: depth above this drives alpha straight to 0. Lower = more aggressive"
                }),
                "bg_hard_cut_strength": ("FLOAT", {
                    "default": 0.80, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Hard-cut strength. 0=disabled, 1=always zero out deep-BG pixels. Main control for stubborn fringe"
                }),
                "hard_cut_alpha_max": ("FLOAT", {
                    "default": 0.70, "min": 0.10, "max": 1.00, "step": 0.05,
                    "tooltip": "Only apply hard cut to pixels with alpha below this. Protects FG core from accidental erasure"
                }),

                # ── Pass 2: FG recovery ───────────────────────────────────
                "depth_fg_threshold": ("FLOAT", {
                    "default": 0.35, "min": 0.05, "max": 0.60, "step": 0.01,
                    "tooltip": "Depth below which missed strands are eligible for FG recovery"
                }),
                "recovery_max": ("FLOAT", {
                    "default": 0.60, "min": 0.10, "max": 1.00, "step": 0.05,
                    "tooltip": "Maximum alpha to restore when recovering a missed FG strand"
                }),

                # ── Pass 3: Edge crisp ────────────────────────────────────
                "edge_crisp_strength": ("FLOAT", {
                    "default": 0.50, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Depth-edge driven boundary sharpening. At 1.0 fully replaces ensemble alpha at depth contours"
                }),

                # ── Pass 4: Hard lock threshold ───────────────────────────
                "hard_lock_hi": ("FLOAT", {
                    "default": 0.90, "min": 0.70, "max": 0.99, "step": 0.01,
                    "tooltip": "Alpha above this is always locked to ensemble value. Lower = depth can suppress more fringe"
                }),

                # ── Adaptive trimap ───────────────────────────────────────
                "use_depth_trimap": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Build adaptive trimap using depth gradient as boundary-complexity signal"
                }),
                "trimap_min_px": ("INT", {
                    "default": 4, "min": 2, "max": 20, "step": 1,
                    "tooltip": "Min unknown-band half-width (pixels)"
                }),
                "trimap_max_px": ("INT", {
                    "default": 24, "min": 6, "max": 60, "step": 2,
                    "tooltip": "Max unknown-band half-width (hair / complex boundary)"
                }),

                # ── Output ────────────────────────────────────────────────
                "invert_depth_vis": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert depth_map visualisation (near=white instead of near=dark)"
                }),
            },
        }

    def refine_with_depth(
        self,
        image,
        mask,
        confidence=None,
        bg_suppress_strength=0.70,
        fg_depth_percentile=80.0,
        depth_bg_threshold=0.75,
        depth_bg_hard=0.90,
        bg_hard_cut_strength=0.80,
        hard_cut_alpha_max=0.70,
        depth_fg_threshold=0.35,
        recovery_max=0.60,
        edge_crisp_strength=0.50,
        hard_lock_hi=0.90,
        use_depth_trimap=True,
        trimap_min_px=4,
        trimap_max_px=24,
        invert_depth_vis=False,
    ):
        depth_model = load_depth_pro()
        out_masks   = []
        out_depths  = []

        for i in range(mask.shape[0]):
            img_idx = min(i, image.shape[0] - 1)
            pil_img = tensor_to_pil(image[img_idx])
            m_np    = mask[i].cpu().numpy().astype(np.float32)

            # ---- 1. Run Depth Pro ----
            depth_norm = depth_model.infer(pil_img)

            if depth_norm.shape != m_np.shape:
                d_pil      = Image.fromarray(
                    (depth_norm * 255).clip(0, 255).astype(np.uint8), mode="L")
                depth_norm = np.array(
                    d_pil.resize((m_np.shape[1], m_np.shape[0]), Image.LANCZOS)
                ).astype(np.float32) / 255.0

            # ---- 2. Resolve confidence ----
            conf_np = None
            if confidence is not None:
                c_idx   = min(i, confidence.shape[0] - 1)
                conf_np = confidence[c_idx].cpu().numpy().astype(np.float32)
                if conf_np.shape != m_np.shape:
                    c_pil   = Image.fromarray(
                        (conf_np * 255).clip(0, 255).astype(np.uint8), mode="L")
                    conf_np = np.array(
                        c_pil.resize((m_np.shape[1], m_np.shape[0]), Image.LANCZOS)
                    ).astype(np.float32) / 255.0

            # ---- 3. Depth-guided mask refinement (v2.1 five-pass) ----
            m_refined = depth_guided_mask(
                m_np, depth_norm,
                strength=bg_suppress_strength,
                fg_percentile=fg_depth_percentile,
                depth_bg_suppress=depth_bg_threshold,
                depth_bg_hard=depth_bg_hard,
                bg_hard_cut_strength=bg_hard_cut_strength,
                hard_cut_alpha_max=hard_cut_alpha_max,
                depth_fg_recover=depth_fg_threshold,
                recovery_max=recovery_max,
                edge_crisp_strength=edge_crisp_strength,
                hard_lock_hi=hard_lock_hi,
            )

            # ---- 4. Adaptive trimap baked into refined mask ----
            if use_depth_trimap:
                trimap = depth_adaptive_trimap(
                    m_refined, depth_norm,
                    confidence=conf_np,
                    min_band_px=trimap_min_px,
                    max_band_px=trimap_max_px,
                    fg_percentile=fg_depth_percentile,
                )
                t_norm   = trimap.astype(np.float32) / 255.0
                hard_fg  = t_norm >= 0.9
                hard_bg  = t_norm <= 0.1
                m_refined[hard_fg] = np.maximum(m_refined[hard_fg], 0.95)
                m_refined[hard_bg] = np.minimum(m_refined[hard_bg], 0.05)
                m_refined = np.clip(m_refined, 0.0, 1.0)

            # ---- 5. Depth visualisation ----
            vis = 1.0 - depth_norm if invert_depth_vis else depth_norm
            depth_pil = depth_to_pil(vis)
            orig_w, orig_h = pil_img.size
            depth_pil = depth_pil.resize((orig_w, orig_h), Image.LANCZOS)
            mask_pil  = Image.fromarray(
                (m_refined * 255).clip(0, 255).astype(np.uint8), mode="L"
            ).resize((orig_w, orig_h), Image.LANCZOS)

            out_masks.append(pil_mask_to_tensor(mask_pil))
            out_depths.append(pil_mask_to_tensor(depth_pil))

        return (torch.stack(out_masks), torch.stack(out_depths))

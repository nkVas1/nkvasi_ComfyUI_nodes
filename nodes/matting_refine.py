"""
NkVasi_MattingRefine — high-quality alpha matting refinement node.

Insert between any BG removal node and Save Image Alpha to achieve
commercial-quality results with individual hair strands and zero artefacts.

Pipeline:
  1. Guided filter pre-pass (clean up coarse mask edges)
  2. Trimap construction from the input mask
  3. trimap_guided_matting — local colour-sampling alpha estimation
     in the unknown zone (approximation of Bayesian matting)
  4. edge_detail_recovery — high-frequency detail restoration
  5. adaptive_bg_cleanup  — contrast-aware BG leak suppression
  6. Guided filter post-pass (smooth matting artefacts)
  7. Remove stray FG islands

No external matting libraries needed — pure cv2 + numpy.
"""
import torch
import numpy as np
from PIL import Image

from ..utils.mask_ops import (
    guided_filter_mask,
    trimap_guided_matting,
    edge_detail_recovery,
    adaptive_bg_cleanup,
    soft_remove_islands,
    soft_remove_holes,
    smooth_mask,
)
from ..utils.image_utils import tensor_to_pil, pil_mask_to_tensor


class NkVasi_MattingRefine:
    """Commercial-grade alpha matting refinement. Plug in after Remove BG Ensemble."""

    CATEGORY = "🎭 nkVasi/Background Removal"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "refine"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask":  ("MASK",),
            },
            "optional": {
                # Width of the unknown zone (px). Wider = safer, slower.
                "trimap_band_px":    ("INT",   {"default": 14,   "min": 4,   "max": 60,  "step": 2}),
                # High-freq detail recovery strength (hair strands, eyelashes)
                "detail_strength":   ("FLOAT", {"default": 0.35,  "min": 0.0, "max": 1.0, "step": 0.05}),
                # Adaptive BG leak threshold
                "bg_cleanup_thresh": ("FLOAT", {"default": 0.08,  "min": 0.0, "max": 0.5,  "step": 0.01}),
                "guided_radius":     ("INT",   {"default": 7,     "min": 2,   "max": 24,  "step": 1}),
                "remove_artefacts":  ("BOOLEAN", {"default": True}),
                "artefact_size":     ("INT",   {"default": 500,  "min": 0,   "max": 5000, "step": 50}),
                "final_blur":        ("INT",   {"default": 1,    "min": 0,   "max": 8,   "step": 1}),
            },
        }

    def refine(
        self,
        image,
        mask,
        trimap_band_px=14,
        detail_strength=0.35,
        bg_cleanup_thresh=0.08,
        guided_radius=7,
        remove_artefacts=True,
        artefact_size=500,
        final_blur=1,
    ):
        out_masks = []

        for i in range(mask.shape[0]):
            img_idx  = min(i, image.shape[0] - 1)
            pil_img  = tensor_to_pil(image[img_idx])

            m_np = mask[i].cpu().numpy().astype(np.float32)
            h, w = m_np.shape[:2]

            pil_guide = pil_img.resize((w, h), Image.LANCZOS)
            guide_np  = np.array(pil_guide).astype(np.float32) / 255.0

            # 1. Pre-pass guided filter
            m_np = guided_filter_mask(m_np, guide_np, radius=guided_radius, eps=3e-4)

            # 2+3. Trimap + colour-sampling alpha matting
            m_np = trimap_guided_matting(
                m_np, guide_np,
                erosion_px=trimap_band_px,
                dilation_px=trimap_band_px,
                sample_radius=max(trimap_band_px * 3, 30),
                n_best=5,
            )

            # 4. Edge detail recovery
            if detail_strength > 0.0:
                m_np = edge_detail_recovery(
                    m_np, guide_np, strength=detail_strength, radius=3)

            # 5. Adaptive BG cleanup
            if bg_cleanup_thresh > 0.0:
                m_np = adaptive_bg_cleanup(
                    m_np, guide_np,
                    global_thresh=bg_cleanup_thresh, local_window=31)

            # 6. Post-pass guided filter
            m_np = guided_filter_mask(m_np, guide_np, radius=guided_radius, eps=3e-4)

            # 7. Artefact removal + hole fill
            if remove_artefacts and artefact_size > 0:
                m_np = soft_remove_islands(m_np, min_island_size=artefact_size)
            m_np = soft_remove_holes(m_np, min_hole_size=400)

            if final_blur > 0:
                m_np = smooth_mask(m_np, final_blur)

            pil_m = Image.fromarray(
                (m_np * 255).clip(0, 255).astype(np.uint8), mode="L")
            out_masks.append(pil_mask_to_tensor(pil_m))

        return (torch.stack(out_masks),)

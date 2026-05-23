"""
NkVasi_SaveImageAlpha — saves IMAGE + MASK as PNG with real alpha channel.

Behaviour by bg_mode:
  alpha    — saves as RGBA PNG (transparent background)
  any other— saves as RGB PNG (image already has background composited
              by the Ensemble node — do NOT re-apply the mask or you lose
              the background colour again)

COMPATIBILITY NOTE
  When using Remove BG Ensemble with background=alpha → connect both
  image+mask outputs here and set bg_mode=alpha.
  When using background=white/blue/etc. → connect only the image output
  and set bg_mode=composited (or use any standard Save Image node).
"""
import os
import json
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths


BG_MODES = ["alpha", "composited (solid/checker background)"]


class NkVasi_SaveImageAlpha:
    """
    Saves IMAGE (and optional MASK) as PNG.
    • bg_mode=alpha          → RGBA PNG, mask used as alpha channel
    • bg_mode=composited     → RGB  PNG, image saved as-is (background
                               was already baked in by the Ensemble node)
    """

    CATEGORY = "🎭 nkVasi/Background Removal"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save_image_alpha"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":           ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "nkVasi_alpha"}),
                "bg_mode":         (BG_MODES,  {"default": "alpha"}),
            },
            "optional": {
                # mask is only used when bg_mode=alpha
                "mask": ("MASK",),
            },
            "hidden": {
                "prompt":       "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    def save_image_alpha(
        self,
        image,
        filename_prefix="nkVasi_alpha",
        bg_mode="alpha",
        mask=None,
        prompt=None,
        extra_pnginfo=None,
    ):
        output_dir = folder_paths.get_output_directory()
        results    = []
        save_alpha = bg_mode.startswith("alpha")

        for i in range(image.shape[0]):
            img_np  = (image[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_rgb = Image.fromarray(img_np, mode="RGB")

            if save_alpha:
                # Need a mask — use supplied mask or full-opaque fallback
                if mask is not None:
                    mask_i  = mask[i] if mask.ndim == 3 else mask
                    mask_np = (mask_i.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    pil_a   = Image.fromarray(mask_np, mode="L")
                    if pil_a.size != pil_rgb.size:
                        pil_a = pil_a.resize(pil_rgb.size, Image.LANCZOS)
                else:
                    # No mask connected in alpha mode — full opaque
                    pil_a = Image.new("L", pil_rgb.size, 255)

                r, g, b = pil_rgb.split()
                pil_out = Image.merge("RGBA", (r, g, b, pil_a))
            else:
                # Composited mode: background is already baked into the RGB image
                # Just save it as-is — do NOT touch the alpha channel
                pil_out = pil_rgb

            counter  = _next_counter(output_dir, filename_prefix)
            filename = f"{filename_prefix}_{counter:05d}.png"
            filepath = os.path.join(output_dir, filename)

            pnginfo = PngInfo()
            if prompt:
                pnginfo.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo:
                for k, v in extra_pnginfo.items():
                    pnginfo.add_text(k, json.dumps(v))

            pil_out.save(filepath, format="PNG", pnginfo=pnginfo, compress_level=6)
            results.append({"filename": filename, "subfolder": "", "type": "output"})

        return {"ui": {"images": results}}


def _next_counter(directory: str, prefix: str) -> int:
    existing = [f for f in os.listdir(directory)
                if f.startswith(prefix) and f.endswith(".png")]
    if not existing:
        return 1
    counters = []
    for name in existing:
        stem = name[len(prefix):].lstrip("_").replace(".png", "")
        if stem.isdigit():
            counters.append(int(stem))
    return max(counters) + 1 if counters else 1

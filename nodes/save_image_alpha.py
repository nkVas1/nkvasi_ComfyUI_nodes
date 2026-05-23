"""
NkVasi_SaveImageAlpha — saves IMAGE + MASK as PNG with true alpha channel.

ComfyUI's built-in Save Image node always converts to RGB (no transparency).
This node merges the image and mask into a proper RGBA PNG.
"""
import os
import json
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths  # ComfyUI built-in


class NkVasi_SaveImageAlpha:
    """
    Saves IMAGE + MASK as a PNG with a real alpha channel.
    Connect the 'image' output and 'mask' output from Remove BG Ensemble
    (or any other node) to get a proper transparent PNG.
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
                "mask":            ("MASK",),
                "filename_prefix": ("STRING", {"default": "nkVasi_alpha"}),
            },
            "hidden": {
                "prompt":    "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    def save_image_alpha(
        self,
        image,
        mask,
        filename_prefix="nkVasi_alpha",
        prompt=None,
        extra_pnginfo=None,
    ):
        output_dir = folder_paths.get_output_directory()
        results = []

        for i in range(image.shape[0]):
            # Convert IMAGE tensor (H,W,3) -> uint8
            img_np = (image[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_rgb = Image.fromarray(img_np, mode="RGB")

            # Convert MASK tensor (H,W) -> uint8 L
            mask_i = mask[i] if mask.ndim == 3 else mask
            mask_np = (mask_i.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_alpha = Image.fromarray(mask_np, mode="L")

            # Resize alpha to match image if different (e.g. from external mask node)
            if pil_alpha.size != pil_rgb.size:
                pil_alpha = pil_alpha.resize(pil_rgb.size, Image.LANCZOS)

            # Merge into RGBA
            pil_rgba = pil_rgb.convert("RGBA")
            r, g, b, _ = pil_rgba.split()
            pil_rgba = Image.merge("RGBA", (r, g, b, pil_alpha))

            # Build output filename with auto-increment
            counter = _next_counter(output_dir, filename_prefix)
            filename = f"{filename_prefix}_{counter:05d}.png"
            filepath = os.path.join(output_dir, filename)

            # Embed workflow metadata (same as ComfyUI built-in)
            pnginfo = PngInfo()
            if prompt:
                pnginfo.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo:
                for k, v in extra_pnginfo.items():
                    pnginfo.add_text(k, json.dumps(v))

            pil_rgba.save(filepath, format="PNG", pnginfo=pnginfo, compress_level=6)

            results.append({
                "filename": filename,
                "subfolder": "",
                "type": "output",
            })

        return {"ui": {"images": results}}


def _next_counter(directory: str, prefix: str) -> int:
    """Find the next unused integer counter for the given prefix."""
    existing = [
        f for f in os.listdir(directory)
        if f.startswith(prefix) and f.endswith(".png")
    ]
    if not existing:
        return 1
    counters = []
    for name in existing:
        stem = name[len(prefix):].lstrip("_").replace(".png", "")
        if stem.isdigit():
            counters.append(int(stem))
    return max(counters) + 1 if counters else 1

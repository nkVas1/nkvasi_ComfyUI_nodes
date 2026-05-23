"""
Lazy model loader with in-process caching.
Each model is loaded once and reused across ComfyUI sessions.
"""
import os
import torch
import numpy as np
from PIL import Image
from functools import lru_cache

# --------------------------------------------------------------------------
# Cache dict — prevents reloading models within one ComfyUI session
# --------------------------------------------------------------------------
_CACHE: dict = {}


# ==========================================================================
# BiRefNet wrapper
# ==========================================================================
class _BiRefNetWrapper:
    VARIANTS = {
        "HR": "ZhengPeng7/BiRefNet_HR",
        "dynamic": "ZhengPeng7/BiRefNet_dynamic",
        "matting": "ZhengPeng7/BiRefNet-matting",
        "general": "ZhengPeng7/BiRefNet",
    }

    def __init__(self, variant: str = "HR"):
        from transformers import AutoModelForImageSegmentation
        from torchvision.transforms.functional import normalize

        hf_id = self.VARIANTS.get(variant, self.VARIANTS["HR"])
        print(f"[nkVasi] Loading BiRefNet-{variant} from {hf_id} ...")
        self.model = AutoModelForImageSegmentation.from_pretrained(
            hf_id, trust_remote_code=True
        )
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self._normalize = normalize

    @torch.inference_mode()
    def infer(self, pil_img: Image.Image, resolution: int = 1024, fp16: bool = True) -> Image.Image:
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        inp = transform(pil_img.convert("RGB")).unsqueeze(0)
        device = next(self.model.parameters()).device
        inp = inp.to(device)

        if fp16 and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                preds = self.model(inp)[-1].sigmoid()
        else:
            preds = self.model(inp)[-1].sigmoid()

        mask_arr = preds[0, 0].cpu().numpy()
        mask_arr = (mask_arr * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(mask_arr, mode="L")


# ==========================================================================
# BEN2 wrapper
# ==========================================================================
class _BEN2Wrapper:
    def __init__(self):
        print("[nkVasi] Loading BEN2 from PramaLLC/BEN2 ...")
        from transformers import AutoModelForImageSegmentation
        self.model = AutoModelForImageSegmentation.from_pretrained(
            "PramaLLC/BEN2", trust_remote_code=True
        )
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    @torch.inference_mode()
    def infer(self, pil_img: Image.Image, resolution: int = 1024) -> Image.Image:
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        inp = transform(pil_img.convert("RGB")).unsqueeze(0)
        device = next(self.model.parameters()).device
        inp = inp.to(device)

        preds = self.model(inp)[-1].sigmoid()
        mask_arr = preds[0, 0].cpu().numpy()
        mask_arr = (mask_arr * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(mask_arr, mode="L")


# ==========================================================================
# RMBG-2.0 wrapper (BRIA)
# ==========================================================================
class _RMBG2Wrapper:
    def __init__(self):
        print("[nkVasi] Loading RMBG-2.0 from briaai/RMBG-2.0 ...")
        from transformers import AutoModelForImageSegmentation
        self.model = AutoModelForImageSegmentation.from_pretrained(
            "briaai/RMBG-2.0", trust_remote_code=True
        )
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    @torch.inference_mode()
    def infer(self, pil_img: Image.Image, resolution: int = 1024, fp16: bool = True) -> Image.Image:
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        inp = transform(pil_img.convert("RGB")).unsqueeze(0)
        device = next(self.model.parameters()).device
        inp = inp.to(device)

        if fp16 and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                preds = self.model(inp)[-1].sigmoid()
        else:
            preds = self.model(inp)[-1].sigmoid()

        mask_arr = preds[0, 0].cpu().numpy()
        mask_arr = (mask_arr * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(mask_arr, mode="L")


# ==========================================================================
# InSPyReNet wrapper
# ==========================================================================
class _InSPyReNetWrapper:
    def __init__(self):
        print("[nkVasi] Loading InSPyReNet via transparent-background ...")
        from transparent_background import Remover
        self.remover = Remover()  # downloads model automatically

    def infer(self, pil_img: Image.Image, resolution: int = 1024) -> Image.Image:
        # transparent-background returns RGBA
        result = self.remover.process(pil_img.convert("RGB"), type="rgba")
        _, _, _, alpha = result.split()
        return alpha


# ==========================================================================
# Public accessors
# ==========================================================================
def load_birefnet(variant: str = "HR") -> _BiRefNetWrapper:
    key = f"birefnet_{variant}"
    if key not in _CACHE:
        _CACHE[key] = _BiRefNetWrapper(variant)
    return _CACHE[key]


def load_ben2() -> _BEN2Wrapper:
    if "ben2" not in _CACHE:
        _CACHE["ben2"] = _BEN2Wrapper()
    return _CACHE["ben2"]


def load_rmbg2() -> _RMBG2Wrapper:
    if "rmbg2" not in _CACHE:
        _CACHE["rmbg2"] = _RMBG2Wrapper()
    return _CACHE["rmbg2"]


def load_inspyrenet() -> _InSPyReNetWrapper:
    if "inspyrenet" not in _CACHE:
        _CACHE["inspyrenet"] = _InSPyReNetWrapper()
    return _CACHE["inspyrenet"]

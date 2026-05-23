"""
Lazy model loader with in-process caching.
Each model is loaded once and reused across ComfyUI sessions.
"""
import os
import sys
import time
import threading
import torch
import numpy as np
from PIL import Image

_CACHE: dict = {}


# ==========================================================================
# Terminal UI — spinner + progress bar
# ==========================================================================
class _SpinnerCtx:
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    C = {
        "reset": "\033[0m", "green": "\033[92m", "cyan": "\033[96m",
        "yellow": "\033[93m", "bold": "\033[1m", "dim": "\033[2m",
        "magenta": "\033[95m", "red": "\033[91m",
    }

    def __init__(self, label: str):
        self.label = label[:32].ljust(32)
        self._stop = threading.Event()
        self._start = time.time()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self):
        c = self.C
        i = 0
        while not self._stop.is_set():
            elapsed = time.time() - self._start
            frame = self.FRAMES[i % len(self.FRAMES)]
            sys.stdout.write(
                f"\r  {c['bold']}{c['cyan']}[nkVasi]{c['reset']} "
                f"{c['bold']}{self.label}{c['reset']} "
                f"{c['yellow']}{frame}{c['reset']} "
                f"{c['dim']}{elapsed:.1f}s{c['reset']}   "
            )
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, *_):
        self._stop.set()
        self._thread.join()
        c = self.C
        elapsed = time.time() - self._start
        if exc_type is None:
            status = f"{c['green']}{c['bold']}✓ loaded{c['reset']}"
        else:
            status = f"{c['red']}{c['bold']}✗ failed{c['reset']}"
        sys.stdout.write(
            f"\r  {c['bold']}{c['cyan']}[nkVasi]{c['reset']} "
            f"{c['bold']}{self.label}{c['reset']} "
            f"{status} {c['dim']}{elapsed:.1f}s{c['reset']}\n"
        )
        sys.stdout.flush()
        return False


def _header(label: str, source: str):
    c = _SpinnerCtx.C
    sep = "─" * 62
    print(f"\n  {c['dim']}{sep}{c['reset']}")
    print(
        f"  {c['bold']}{c['cyan']}[nkVasi]{c['reset']} "
        f"Loading {c['bold']}{c['magenta']}{label}{c['reset']} "
        f"{c['dim']}← {source}{c['reset']}"
    )
    print(f"  {c['dim']}{sep}{c['reset']}")


# ==========================================================================
# BiRefNet wrapper
# ==========================================================================
class _BiRefNetWrapper:
    VARIANTS = {
        "HR":      "ZhengPeng7/BiRefNet_HR",
        "dynamic": "ZhengPeng7/BiRefNet_dynamic",
        "matting": "ZhengPeng7/BiRefNet-matting",
        "general": "ZhengPeng7/BiRefNet",
    }

    def __init__(self, variant: str = "HR"):
        from transformers import AutoModelForImageSegmentation
        hf_id = self.VARIANTS.get(variant, self.VARIANTS["HR"])
        _header(f"BiRefNet-{variant}", hf_id)
        with _SpinnerCtx(f"BiRefNet-{variant} · init          "):
            self.model = AutoModelForImageSegmentation.from_pretrained(
                hf_id, trust_remote_code=True
            )
            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda()

    @torch.inference_mode()
    def infer(self, pil_img: Image.Image, resolution: int = 1024, fp16: bool = True) -> Image.Image:
        from torchvision import transforms
        t = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        inp = t(pil_img.convert("RGB")).unsqueeze(0)
        device = next(self.model.parameters()).device
        inp = inp.to(device)
        if fp16 and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                preds = self.model(inp)[-1].sigmoid()
        else:
            preds = self.model(inp)[-1].sigmoid()
        return Image.fromarray(
            (preds[0, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8), mode="L"
        )


# ==========================================================================
# BEN2 wrapper
# Two loading strategies (tried in order):
#   1. pip package  `ben2`  (cleanest, AutoModel.from_pretrained)
#   2. Dynamic import of BEN2.py from snapshot_download cache
#
# The HF repo contains:  BEN2.py  +  BEN2_Base.pth  (NOT BEN2_Base.py)
# ==========================================================================
class _BEN2Wrapper:
    HF_ID = "PramaLLC/BEN2"

    def __init__(self):
        _header("BEN2", self.HF_ID)
        c = _SpinnerCtx.C

        # ---- Strategy 1: use official ben2 pip package ----
        try:
            print(f"  {c['dim']}Strategy 1/2 · loading via ben2 pip package{c['reset']}")
            with _SpinnerCtx("BEN2 · AutoModel.from_pretrained"):
                from ben2 import AutoModel
                self.model = AutoModel.from_pretrained(self.HF_ID)
                self.model.eval()
                if torch.cuda.is_available():
                    self.model = self.model.cuda()
            self._use_automodel = True
            return
        except Exception as e:
            print(f"  {c['dim']}ben2 package not available ({e}), trying Strategy 2…{c['reset']}")

        # ---- Strategy 2: dynamic import of BEN2.py from HF snapshot ----
        print(f"  {c['dim']}Strategy 2/2 · snapshot_download + dynamic import{c['reset']}")
        try:
            from huggingface_hub import snapshot_download
            with _SpinnerCtx("BEN2 · snapshot_download       "):
                local_dir = snapshot_download(repo_id=self.HF_ID)
        except Exception as e:
            raise RuntimeError(f"[nkVasi] BEN2 download failed: {e}") from e

        # The file in the HF repo is  BEN2.py  (not BEN2_Base.py)
        ben2_py = os.path.join(local_dir, "BEN2.py")
        pth_file = os.path.join(local_dir, "BEN2_Base.pth")

        if not os.path.exists(ben2_py):
            # list what we actually got so the user can report it
            files = os.listdir(local_dir) if os.path.isdir(local_dir) else []
            raise FileNotFoundError(
                f"[nkVasi] BEN2.py not found in snapshot: {local_dir}\n"
                f"Files present: {files}"
            )

        if not os.path.exists(pth_file):
            files = os.listdir(local_dir) if os.path.isdir(local_dir) else []
            raise FileNotFoundError(
                f"[nkVasi] BEN2_Base.pth not found in snapshot: {local_dir}\n"
                f"Files present: {files}"
            )

        with _SpinnerCtx("BEN2 · load weights            "):
            import importlib.util
            spec = importlib.util.spec_from_file_location("BEN2_module", ben2_py)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            self.model = module.BEN_Base()
            self.model.loadcheckpoints(pth_file)
            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda()

        self._use_automodel = False

    @torch.inference_mode()
    def infer(self, pil_img: Image.Image, resolution: int = 1024) -> Image.Image:
        result = self.model.inference(
            pil_img.convert("RGB"),
            refine_foreground=False,  # nkVasi does its own post-processing
        )
        # Both strategies return an RGBA PIL image
        if isinstance(result, Image.Image):
            if result.mode == "RGBA":
                _, _, _, alpha = result.split()
                return alpha
            return result.convert("L")
        # fallback: tensor mask
        if isinstance(result, torch.Tensor):
            arr = result.squeeze().cpu().numpy()
            return Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8), mode="L")
        return result


# ==========================================================================
# RMBG-2.0 wrapper (BRIA)
# ==========================================================================
class _RMBG2Wrapper:
    HF_ID = "briaai/RMBG-2.0"

    def __init__(self):
        from transformers import AutoModelForImageSegmentation
        _header("RMBG-2.0", self.HF_ID)
        with _SpinnerCtx("RMBG-2.0 · init              "):
            self.model = AutoModelForImageSegmentation.from_pretrained(
                self.HF_ID, trust_remote_code=True
            )
            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda()

    @torch.inference_mode()
    def infer(self, pil_img: Image.Image, resolution: int = 1024, fp16: bool = True) -> Image.Image:
        from torchvision import transforms
        t = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        inp = t(pil_img.convert("RGB")).unsqueeze(0)
        device = next(self.model.parameters()).device
        inp = inp.to(device)
        if fp16 and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                preds = self.model(inp)[-1].sigmoid()
        else:
            preds = self.model(inp)[-1].sigmoid()
        return Image.fromarray(
            (preds[0, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8), mode="L"
        )


# ==========================================================================
# InSPyReNet wrapper
# ==========================================================================
class _InSPyReNetWrapper:
    def __init__(self):
        _header("InSPyReNet", "transparent-background (PyPI)")
        with _SpinnerCtx("InSPyReNet · init             "):
            from transparent_background import Remover
            self.remover = Remover()

    def infer(self, pil_img: Image.Image, resolution: int = 1024) -> Image.Image:
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

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

# --------------------------------------------------------------------------
# Cache dict — prevents reloading models within one ComfyUI session
# --------------------------------------------------------------------------
_CACHE: dict = {}


# ==========================================================================
# Terminal progress bar (pure stdlib, no extra deps)
# ==========================================================================
class _TermProgress:
    """
    Compact animated progress bar for terminal.
    Shows: label | bar | % | speed | elapsed

    Example:
      [nkVasi] BiRefNet-HR  ████████████░░░░░░░░  61%  2.3 MB/s  00:14
    """
    BAR_WIDTH = 22
    COLORS = {
        "reset": "\033[0m",
        "green": "\033[92m",
        "cyan": "\033[96m",
        "yellow": "\033[93m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "magenta": "\033[95m",
    }

    def __init__(self, label: str, total_bytes: int = 0):
        self.label = label[:22].ljust(22)
        self.total = total_bytes
        self.done = 0
        self._start = time.time()
        self._last_done = 0
        self._last_time = time.time()
        self._finished = False
        self._lock = threading.Lock()
        # Print initial line
        self._print_bar(0, 0.0)

    def _c(self, name: str) -> str:
        return self.COLORS.get(name, "")

    def _fmt_time(self, seconds: float) -> str:
        s = int(seconds)
        return f"{s//60:02d}:{s%60:02d}"

    def _fmt_bytes(self, b: int) -> str:
        if b < 1024:
            return f"{b} B"
        elif b < 1024 ** 2:
            return f"{b/1024:.1f} KB"
        elif b < 1024 ** 3:
            return f"{b/1024**2:.1f} MB"
        return f"{b/1024**3:.2f} GB"

    def _print_bar(self, pct: float, speed_bps: float):
        filled = int(self.BAR_WIDTH * pct)
        bar = "█" * filled + "░" * (self.BAR_WIDTH - filled)
        elapsed = time.time() - self._start
        speed_str = self._fmt_bytes(int(speed_bps)) + "/s" if speed_bps > 0 else "  ----  "
        size_str = f" {self._fmt_bytes(self.done)}/{self._fmt_bytes(self.total)}" if self.total > 0 else ""
        c = self.COLORS
        line = (
            f"  {c['bold']}{c['cyan']}[nkVasi]{c['reset']} "
            f"{c['bold']}{self.label}{c['reset']} "
            f"{c['green']}{bar}{c['reset']} "
            f"{c['yellow']}{pct*100:5.1f}%{c['reset']} "
            f"{c['dim']}{speed_str}{c['reset']}"
            f"{c['magenta']}{size_str}{c['reset']} "
            f"{c['dim']}{self._fmt_time(elapsed)}{c['reset']}"
        )
        sys.stdout.write("\r" + line)
        sys.stdout.flush()

    def update(self, chunk_bytes: int):
        with self._lock:
            now = time.time()
            self.done += chunk_bytes
            dt = now - self._last_time
            speed = (self.done - self._last_done) / dt if dt > 0.1 else 0
            self._last_done = self.done
            self._last_time = now
            pct = (self.done / self.total) if self.total > 0 else 0.0
            self._print_bar(pct, speed)

    def finish(self, status: str = "✓ done"):
        with self._lock:
            if self._finished:
                return
            self._finished = True
            elapsed = time.time() - self._start
            avg_speed = self.done / elapsed if elapsed > 0 else 0
            c = self.COLORS
            line = (
                f"  {c['bold']}{c['cyan']}[nkVasi]{c['reset']} "
                f"{c['bold']}{self.label}{c['reset']} "
                f"{c['green']}{'█' * self.BAR_WIDTH}{c['reset']} "
                f"{c['yellow']}100.0%{c['reset']} "
                f"{c['dim']}{self._fmt_bytes(int(avg_speed))}/s{c['reset']} "
                f"{c['magenta']}{self._fmt_bytes(self.done)}{c['reset']} "
                f"{c['green']}{c['bold']}{status}{c['reset']} "
                f"{c['dim']}{self._fmt_time(elapsed)}{c['reset']}"
            )
            sys.stdout.write("\r" + line + "\n")
            sys.stdout.flush()

    def fail(self, msg: str = "✗ error"):
        with self._lock:
            c = self.COLORS
            elapsed = time.time() - self._start
            line = (
                f"  {c['bold']}{c['cyan']}[nkVasi]{c['reset']} "
                f"{c['bold']}{self.label}{c['reset']} "
                f"\033[91m{'▓' * self.BAR_WIDTH} {msg}\033[0m "
                f"{c['dim']}{self._fmt_time(elapsed)}{c['reset']}"
            )
            sys.stdout.write("\r" + line + "\n")
            sys.stdout.flush()


def _spinner_loader(label: str):
    """
    Indeterminate spinner for phases where total size is unknown
    (e.g. model init, trust_remote_code compile).
    Returns a context manager.
    """
    return _SpinnerCtx(label)


class _SpinnerCtx:
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, label: str):
        self.label = label[:30].ljust(30)
        self._stop = threading.Event()
        self._start = time.time()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self):
        c = _TermProgress.COLORS
        i = 0
        while not self._stop.is_set():
            elapsed = time.time() - self._start
            frame = self.FRAMES[i % len(self.FRAMES)]
            line = (
                f"  {c['bold']}{c['cyan']}[nkVasi]{c['reset']} "
                f"{c['bold']}{self.label}{c['reset']} "
                f"{c['yellow']}{frame}{c['reset']} "
                f"{c['dim']}{elapsed:.1f}s{c['reset']}"
            )
            sys.stdout.write("\r" + line + "   ")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop.set()
        self._thread.join()
        c = _TermProgress.COLORS
        elapsed = time.time() - self._start
        if exc_type is None:
            msg = f"{c['green']}{c['bold']}✓ loaded{c['reset']} {c['dim']}{elapsed:.1f}s{c['reset']}"
        else:
            msg = f"\033[91m✗ failed\033[0m {c['dim']}{elapsed:.1f}s{c['reset']}"
        sys.stdout.write("\r  " + c['bold'] + c['cyan'] + "[nkVasi]" + c['reset'] +
                         " " + c['bold'] + self.label + c['reset'] + " " + msg + "\n")
        sys.stdout.flush()
        return False  # don't suppress exceptions


def _hf_download_with_progress(repo_id: str, label: str, **kwargs):
    """
    Download from HuggingFace with live progress bar.
    Wraps huggingface_hub.snapshot_download with tqdm progress.
    """
    from huggingface_hub import snapshot_download
    import huggingface_hub
    c = _TermProgress.COLORS
    print(
        f"\n  {c['bold']}{c['cyan']}[nkVasi]{c['reset']} "
        f"Downloading {c['bold']}{label}{c['reset']} "
        f"← {c['dim']}{repo_id}{c['reset']}"
    )
    # snapshot_download shows its own tqdm progress per file
    # we wrap it cleanly
    cache_dir = kwargs.pop("cache_dir", None)
    local_dir = kwargs.pop("local_dir", None)
    return snapshot_download(
        repo_id,
        cache_dir=cache_dir,
        local_dir=local_dir,
        **kwargs,
    )


# ==========================================================================
# Header printer
# ==========================================================================
def _print_load_header(label: str, source: str):
    c = _TermProgress.COLORS
    bar = "─" * 60
    print(f"\n  {c['dim']}{bar}{c['reset']}")
    print(
        f"  {c['bold']}{c['cyan']}[nkVasi]{c['reset']} "
        f"Loading {c['bold']}{c['magenta']}{label}{c['reset']} "
        f"{c['dim']}← {source}{c['reset']}"
    )
    print(f"  {c['dim']}{bar}{c['reset']}")


# ==========================================================================
# BiRefNet wrapper
# ==========================================================================
class _BiRefNetWrapper:
    VARIANTS = {
        "HR":       "ZhengPeng7/BiRefNet_HR",
        "dynamic":  "ZhengPeng7/BiRefNet_dynamic",
        "matting":  "ZhengPeng7/BiRefNet-matting",
        "general":  "ZhengPeng7/BiRefNet",
    }

    def __init__(self, variant: str = "HR"):
        from transformers import AutoModelForImageSegmentation
        hf_id = self.VARIANTS.get(variant, self.VARIANTS["HR"])
        _print_load_header(f"BiRefNet-{variant}", hf_id)

        with _spinner_loader(f"BiRefNet-{variant} · init model"):
            self.model = AutoModelForImageSegmentation.from_pretrained(
                hf_id, trust_remote_code=True
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
        return Image.fromarray((mask_arr * 255).clip(0, 255).astype(np.uint8), mode="L")


# ==========================================================================
# BEN2 wrapper  —  uses native BEN2 class, NOT AutoModelForImageSegmentation
# BEN2 has no model_type in config.json and must be loaded via its own API.
# ==========================================================================
class _BEN2Wrapper:
    HF_ID = "PramaLLC/BEN2"

    def __init__(self):
        _print_load_header("BEN2", self.HF_ID)

        # Step 1: download repo files (weights + model code)
        try:
            from huggingface_hub import snapshot_download
            c = _TermProgress.COLORS
            print(
                f"  {c['dim']}Step 1/2 · downloading weights (≈750 MB on first run){c['reset']}"
            )
            with _spinner_loader("BEN2 · snapshot_download  "):
                local_dir = snapshot_download(repo_id=self.HF_ID)
        except Exception as e:
            raise RuntimeError(f"[nkVasi] BEN2 download failed: {e}") from e

        # Step 2: load model using BEN2's own class
        c = _TermProgress.COLORS
        print(f"  {c['dim']}Step 2/2 · initializing model weights on device{c['reset']}")
        with _spinner_loader("BEN2 · load weights        "):
            import importlib.util, sys as _sys
            spec = importlib.util.spec_from_file_location(
                "BEN2_model", os.path.join(local_dir, "BEN2_Base.py")
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            self.model = module.BEN_Base()
            self.model.loadcheckpoints(
                os.path.join(local_dir, "BEN2_Base.pth")
            )
            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda()

    @torch.inference_mode()
    def infer(self, pil_img: Image.Image, resolution: int = 1024) -> Image.Image:
        # BEN2's native matting pipeline — returns RGBA
        device = next(self.model.parameters()).device
        result = self.model.inference(
            pil_img.convert("RGB"),
            refine_foreground=False,  # we do our own refinement
        )
        if isinstance(result, Image.Image) and result.mode == "RGBA":
            _, _, _, alpha = result.split()
            return alpha
        # fallback: result is already a mask tensor
        if isinstance(result, torch.Tensor):
            mask_arr = result.squeeze().cpu().numpy()
            return Image.fromarray((mask_arr * 255).clip(0, 255).astype(np.uint8), mode="L")
        return result


# ==========================================================================
# RMBG-2.0 wrapper (BRIA)
# ==========================================================================
class _RMBG2Wrapper:
    HF_ID = "briaai/RMBG-2.0"

    def __init__(self):
        from transformers import AutoModelForImageSegmentation
        _print_load_header("RMBG-2.0", self.HF_ID)
        with _spinner_loader("RMBG-2.0 · init model      "):
            self.model = AutoModelForImageSegmentation.from_pretrained(
                self.HF_ID, trust_remote_code=True
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
        return Image.fromarray((mask_arr * 255).clip(0, 255).astype(np.uint8), mode="L")


# ==========================================================================
# InSPyReNet wrapper
# ==========================================================================
class _InSPyReNetWrapper:
    def __init__(self):
        _print_load_header("InSPyReNet", "transparent-background (PyPI)")
        with _spinner_loader("InSPyReNet · init remover  "):
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

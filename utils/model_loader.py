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
# ==========================================================================
class _BEN2Wrapper:
    HF_ID = "PramaLLC/BEN2"

    def __init__(self):
        _header("BEN2", self.HF_ID)
        c = _SpinnerCtx.C

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

        print(f"  {c['dim']}Strategy 2/2 · snapshot_download + dynamic import{c['reset']}")
        try:
            from huggingface_hub import snapshot_download
            with _SpinnerCtx("BEN2 · snapshot_download       "):
                local_dir = snapshot_download(repo_id=self.HF_ID)
        except Exception as e:
            raise RuntimeError(f"[nkVasi] BEN2 download failed: {e}") from e

        ben2_py  = os.path.join(local_dir, "BEN2.py")
        pth_file = os.path.join(local_dir, "BEN2_Base.pth")

        if not os.path.exists(ben2_py):
            files = os.listdir(local_dir) if os.path.isdir(local_dir) else []
            raise FileNotFoundError(
                f"[nkVasi] BEN2.py not found in snapshot: {local_dir}\nFiles present: {files}")

        if not os.path.exists(pth_file):
            files = os.listdir(local_dir) if os.path.isdir(local_dir) else []
            raise FileNotFoundError(
                f"[nkVasi] BEN2_Base.pth not found in snapshot: {local_dir}\nFiles present: {files}")

        with _SpinnerCtx("BEN2 · load weights            "):
            import importlib.util
            spec   = importlib.util.spec_from_file_location("BEN2_module", ben2_py)
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
            refine_foreground=False,
        )
        if isinstance(result, Image.Image):
            if result.mode == "RGBA":
                _, _, _, alpha = result.split()
                return alpha
            return result.convert("L")
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
# Depth Pro wrapper  (Apple MLRC, 2024)
#
# Loading strategies (tried in order):
#   1. depth_pro pip package   (pip install git+https://github.com/apple/ml-depth-pro)
#   2. HuggingFace transformers AutoModelForDepthEstimation  (apple/DepthPro)
#   3. snapshot_download + robust src/ discovery + importlib.util load
# ==========================================================================

def _find_depth_pro_src(root: str) -> str | None:
    """
    Recursively search `root` for a directory containing depth_pro/__init__.py.
    Returns the parent directory (the one to add to sys.path), or None.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        if "depth_pro" in dirnames:
            candidate = os.path.join(dirpath, "depth_pro", "__init__.py")
            if os.path.exists(candidate):
                return dirpath  # add dirpath so `import depth_pro` works
    return None


def _find_checkpoint(root: str) -> str | None:
    """
    Search for the Depth Pro checkpoint file inside `root`.
    Apple repo uses 'depth_pro.pt'; HF LFS may store it under blobs/.
    """
    for name in ("depth_pro.pt", "pytorch_model.bin", "model.safetensors"):
        for dirpath, _, filenames in os.walk(root):
            if name in filenames:
                return os.path.join(dirpath, name)
    return None


def _importlib_load_depth_pro(src_parent: str):
    """
    Load depth_pro as a module via importlib so we don't depend on sys.path
    being clean, and avoid conflicts with a previously-cached failed import.
    """
    import importlib
    import importlib.util

    # Remove any stale cached entry (e.g. from a failed bare import attempt)
    for key in list(sys.modules.keys()):
        if key == "depth_pro" or key.startswith("depth_pro."):
            del sys.modules[key]

    init_path = os.path.join(src_parent, "depth_pro", "__init__.py")
    spec   = importlib.util.spec_from_file_location(
        "depth_pro", init_path,
        submodule_search_locations=[os.path.join(src_parent, "depth_pro")],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["depth_pro"] = module
    spec.loader.exec_module(module)
    return module


class _DepthProWrapper:
    HF_ID = "apple/DepthPro"

    def __init__(self):
        _header("Depth Pro", self.HF_ID)
        c = _SpinnerCtx.C
        self._strategy: str = ""

        # ---- Strategy 1: official depth-pro pip package ----
        try:
            print(f"  {c['dim']}Strategy 1/3 · depth_pro pip package{c['reset']}")
            with _SpinnerCtx("DepthPro · pip package         "):
                import depth_pro
                self._model, self._transform = depth_pro.create_model_and_transforms()
                self._model.eval()
                if torch.cuda.is_available():
                    self._model = self._model.cuda()
            self._strategy = "pip"
            return
        except Exception as e:
            print(f"  {c['dim']}depth_pro not available ({e}), trying Strategy 2…{c['reset']}")

        # ---- Strategy 2: HuggingFace transformers pipeline ----
        try:
            print(f"  {c['dim']}Strategy 2/3 · transformers AutoModel{c['reset']}")
            with _SpinnerCtx("DepthPro · transformers         "):
                from transformers import AutoImageProcessor, AutoModelForDepthEstimation
                self._processor = AutoImageProcessor.from_pretrained(self.HF_ID)
                self._model     = AutoModelForDepthEstimation.from_pretrained(
                    self.HF_ID,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                )
                self._model.eval()
                if torch.cuda.is_available():
                    self._model = self._model.cuda()
            self._strategy = "transformers"
            return
        except Exception as e:
            print(f"  {c['dim']}transformers strategy failed ({e}), trying Strategy 3…{c['reset']}")

        # ---- Strategy 3: snapshot_download + robust dynamic import ----
        print(f"  {c['dim']}Strategy 3/3 · snapshot_download + importlib{c['reset']}")
        try:
            from huggingface_hub import snapshot_download
            with _SpinnerCtx("DepthPro · snapshot_download   "):
                local_dir = snapshot_download(repo_id=self.HF_ID)
        except Exception as e:
            raise RuntimeError(f"[nkVasi] Depth Pro download failed: {e}") from e

        c2 = _SpinnerCtx.C
        print(f"  {c2['dim']}Snapshot path: {local_dir}{c2['reset']}")

        # --- Find src/depth_pro anywhere in the snapshot tree ---
        src_parent = _find_depth_pro_src(local_dir)
        if src_parent is None:
            # Snapshot may not include Python source (weights-only repo).
            # Try HF hub cache blobs structure one level up.
            parent = os.path.dirname(local_dir)
            src_parent = _find_depth_pro_src(parent)

        if src_parent is None:
            files_found = []
            for dp, dn, fn in os.walk(local_dir):
                for f in fn:
                    files_found.append(os.path.relpath(os.path.join(dp, f), local_dir))
            raise RuntimeError(
                f"[nkVasi] Depth Pro: could not find depth_pro/__init__.py in snapshot.\n"
                f"Files in snapshot:\n" + "\n".join(files_found[:40])
            )

        print(f"  {c2['dim']}Found depth_pro source at: {src_parent}{c2['reset']}")

        # --- Find checkpoint ---
        ckpt_path = _find_checkpoint(local_dir)
        if ckpt_path is None:
            ckpt_path = _find_checkpoint(os.path.dirname(local_dir))
        print(f"  {c2['dim']}Checkpoint: {ckpt_path}{c2['reset']}")

        try:
            with _SpinnerCtx("DepthPro · dynamic import      "):
                depth_pro = _importlib_load_depth_pro(src_parent)

                if ckpt_path is not None:
                    # Build config with explicit checkpoint path
                    cfg = depth_pro.DepthProConfig(checkpoint_uri=ckpt_path)
                    self._model, self._transform = \
                        depth_pro.create_model_and_transforms(config=cfg)
                else:
                    # Let the library find the checkpoint itself
                    self._model, self._transform = \
                        depth_pro.create_model_and_transforms()

                self._model.eval()
                if torch.cuda.is_available():
                    self._model = self._model.cuda()
            self._strategy = "snapshot"
        except Exception as e:
            raise RuntimeError(
                f"[nkVasi] Depth Pro all strategies failed: {e}\n"
                f"Try: pip install git+https://github.com/apple/ml-depth-pro"
            ) from e

    @torch.inference_mode()
    def infer(self, pil_img: Image.Image) -> np.ndarray:
        """
        Run Depth Pro on a PIL image.
        Returns float32 H×W normalised depth: 0=nearest (FG), 1=farthest (BG).
        """
        rgb = pil_img.convert("RGB")

        if self._strategy in ("pip", "snapshot"):
            device = next(self._model.parameters()).device
            inp    = self._transform(rgb).unsqueeze(0).to(device)
            result = self._model.infer(inp)
            depth  = result["depth"].squeeze().cpu().numpy().astype(np.float32)

        elif self._strategy == "transformers":
            device = next(self._model.parameters()).device
            inputs = self._processor(images=rgb, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.float32,
            ):
                outputs = self._model(**inputs)
            depth = outputs.predicted_depth.squeeze().cpu().numpy().astype(np.float32)

        else:
            raise RuntimeError("[nkVasi] DepthPro: no strategy loaded")

        # Normalise to [0, 1]
        d_min, d_max = float(depth.min()), float(depth.max())
        if d_max - d_min < 1e-5:
            return np.zeros_like(depth)
        norm = (depth - d_min) / (d_max - d_min)

        # Auto-detect and fix inverted depth (centre should be FG = low value)
        h, w   = norm.shape
        cy, cx = h // 2, w // 2
        centre_val = float(norm[cy - h // 8:cy + h // 8, cx - w // 8:cx + w // 8].mean())
        border_val = float(np.concatenate([
            norm[:h // 8, :].ravel(),
            norm[-h // 8:, :].ravel(),
            norm[:, :w // 8].ravel(),
            norm[:, -w // 8:].ravel(),
        ]).mean())
        if centre_val > border_val:
            norm = 1.0 - norm

        return norm.astype(np.float32)


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


def load_depth_pro() -> _DepthProWrapper:
    if "depth_pro" not in _CACHE:
        _CACHE["depth_pro"] = _DepthProWrapper()
    return _CACHE["depth_pro"]

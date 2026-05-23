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

# Directory used to cache Depth Pro source code (downloaded from GitHub)
_DEPTH_PRO_SRC_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".cache", "depth_pro_src",
)


# ==========================================================================
# Terminal UI — spinner
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
        status = (
            f"{c['green']}{c['bold']}✓ loaded{c['reset']}" if exc_type is None
            else f"{c['red']}{c['bold']}✗ failed{c['reset']}"
        )
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
        result = self.model.inference(pil_img.convert("RGB"), refine_foreground=False)
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
# The HuggingFace repo apple/DepthPro is WEIGHTS-ONLY (depth_pro.pt + README).
# Python source lives at https://github.com/apple/ml-depth-pro
#
# depth_pro/__init__.py exports ONLY:
#   create_model_and_transforms, load_rgb
# DepthProConfig lives in depth_pro/depth_pro.py (the submodule).
#
# Loading strategies:
#   1. depth_pro already installed as pip package
#   2. transformers AutoModelForDepthEstimation (falls back gracefully)
#   3a. Download source ZIP from GitHub → extract to .cache/depth_pro_src/
#   3b. Patch optional dependencies (pillow_heif etc.) in extracted source
#   3c. Download weights via hf_hub_download
#   3d. Load package + submodule via importlib, pass ckpt_path to config
# ==========================================================================

_DEPTH_PRO_GITHUB_ZIP = (
    "https://github.com/apple/ml-depth-pro/archive/refs/heads/main.zip"
)

_PILLOW_HEIF_STUB = '''
# --- nkVasi patch: make pillow_heif optional ---
try:
    import pillow_heif as pillow_heif
except ImportError:
    import types as _types
    pillow_heif = _types.ModuleType("pillow_heif")
    pillow_heif.register_heif_opener = lambda *a, **kw: None
    import sys as _sys
    _sys.modules["pillow_heif"] = pillow_heif
# --- end nkVasi patch ---
'''


def _patch_depth_pro_src(src_parent: str) -> None:
    utils_path = os.path.join(src_parent, "depth_pro", "utils.py")
    if not os.path.exists(utils_path):
        return
    with open(utils_path, "r", encoding="utf-8") as fh:
        src = fh.read()
    if "nkVasi patch" in src:
        return
    if "import pillow_heif" in src:
        patched = src.replace("import pillow_heif", _PILLOW_HEIF_STUB, 1)
        with open(utils_path, "w", encoding="utf-8") as fh:
            fh.write(patched)


def _ensure_depth_pro_src() -> str:
    src_parent = os.path.join(_DEPTH_PRO_SRC_CACHE, "ml-depth-pro-main", "src")
    marker     = os.path.join(src_parent, "depth_pro", "__init__.py")

    if not os.path.exists(marker):
        import urllib.request
        import zipfile
        c = _SpinnerCtx.C

        os.makedirs(_DEPTH_PRO_SRC_CACHE, exist_ok=True)
        zip_path = os.path.join(_DEPTH_PRO_SRC_CACHE, "ml-depth-pro-main.zip")

        print(f"  {c['dim']}Downloading Depth Pro source from GitHub…{c['reset']}")
        print(f"  {c['dim']}URL: {_DEPTH_PRO_GITHUB_ZIP}{c['reset']}")
        try:
            urllib.request.urlretrieve(_DEPTH_PRO_GITHUB_ZIP, zip_path)
        except Exception as e:
            raise RuntimeError(
                f"[nkVasi] Failed to download Depth Pro source ZIP: {e}\n"
                f"Please install manually: pip install git+https://github.com/apple/ml-depth-pro"
            ) from e

        print(f"  {c['dim']}Extracting…{c['reset']}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(_DEPTH_PRO_SRC_CACHE)
        os.remove(zip_path)

        if not os.path.exists(marker):
            for dp, dn, _ in os.walk(_DEPTH_PRO_SRC_CACHE):
                if "depth_pro" in dn:
                    candidate = os.path.join(dp, "depth_pro", "__init__.py")
                    if os.path.exists(candidate):
                        src_parent = dp
                        break
            else:
                raise RuntimeError(
                    f"[nkVasi] Could not find depth_pro/__init__.py after extraction.\n"
                    f"Cache dir: {_DEPTH_PRO_SRC_CACHE}"
                )

    _patch_depth_pro_src(src_parent)
    return src_parent


def _importlib_load_depth_pro(src_parent: str):
    """
    Load the depth_pro package from src_parent via importlib.
    Also explicitly loads the depth_pro.depth_pro submodule so that
    DepthProConfig and create_model_and_transforms are fully available.
    Returns the top-level depth_pro module.
    """
    import importlib.util

    # Purge any stale entries
    for key in list(sys.modules.keys()):
        if key == "depth_pro" or key.startswith("depth_pro."):
            del sys.modules[key]

    # Pre-register pillow_heif stub so submodules that import it don't crash
    if "pillow_heif" not in sys.modules:
        try:
            import pillow_heif  # noqa: F401
        except ImportError:
            import types
            stub = types.ModuleType("pillow_heif")
            stub.register_heif_opener = lambda *a, **kw: None  # type: ignore
            sys.modules["pillow_heif"] = stub

    dp_dir = os.path.join(src_parent, "depth_pro")

    def _load_submodule(name: str, rel_path: str):
        path = os.path.join(dp_dir, rel_path)
        spec = importlib.util.spec_from_file_location(
            name, path,
            submodule_search_locations=[dp_dir],
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    # 1. Load package root (__init__.py)
    init_spec = importlib.util.spec_from_file_location(
        "depth_pro",
        os.path.join(dp_dir, "__init__.py"),
        submodule_search_locations=[dp_dir],
    )
    pkg = importlib.util.module_from_spec(init_spec)
    sys.modules["depth_pro"] = pkg
    init_spec.loader.exec_module(pkg)

    # 2. Explicitly load depth_pro.depth_pro submodule (contains DepthProConfig)
    #    This is needed because __init__.py only re-exports create_model_and_transforms
    #    but DepthProConfig is defined in the submodule itself.
    dp_sub = _load_submodule("depth_pro.depth_pro", "depth_pro.py")
    pkg.depth_pro = dp_sub  # attach as attribute for convenience

    return pkg


class _DepthProWrapper:
    HF_ID = "apple/DepthPro"

    def __init__(self):
        _header("Depth Pro", self.HF_ID)
        c = _SpinnerCtx.C
        self._strategy: str = ""

        # ---- Strategy 1: pip package already installed ----
        try:
            print(f"  {c['dim']}Strategy 1/3 · depth_pro pip package{c['reset']}")
            with _SpinnerCtx("DepthPro · pip package         "):
                import depth_pro as _dp
                self._model, self._transform = _dp.create_model_and_transforms()
                self._model.eval()
                if torch.cuda.is_available():
                    self._model = self._model.cuda()
            self._strategy = "pip"
            return
        except Exception as e:
            print(f"  {c['dim']}pip strategy failed ({e}), trying Strategy 2…{c['reset']}")

        # ---- Strategy 2: transformers pipeline ----
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

        # ---- Strategy 3: GitHub source ZIP + HF weights ----
        print(f"  {c['dim']}Strategy 3/3 · GitHub source ZIP + HF weights{c['reset']}")

        try:
            with _SpinnerCtx("DepthPro · source from GitHub  "):
                src_parent = _ensure_depth_pro_src()
        except Exception as e:
            raise RuntimeError(str(e)) from e

        print(f"  {c['dim']}Source ready at: {src_parent}{c['reset']}")

        try:
            from huggingface_hub import hf_hub_download
            with _SpinnerCtx("DepthPro · weights from HF     "):
                ckpt_path = hf_hub_download(
                    repo_id=self.HF_ID,
                    filename="depth_pro.pt",
                )
        except Exception as e:
            raise RuntimeError(
                f"[nkVasi] Depth Pro: failed to download weights: {e}"
            ) from e

        print(f"  {c['dim']}Weights: {ckpt_path}{c['reset']}")

        # 3d — Load via importlib.
        # DepthProConfig lives in depth_pro.depth_pro submodule, not in package root.
        # create_model_and_transforms accepts a config kwarg with checkpoint_uri.
        try:
            with _SpinnerCtx("DepthPro · dynamic import      "):
                dp_pkg = _importlib_load_depth_pro(src_parent)
                # DepthProConfig is in the submodule, attached as pkg.depth_pro
                DepthProConfig = dp_pkg.depth_pro.DepthProConfig
                cfg = DepthProConfig(checkpoint_uri=ckpt_path)
                self._model, self._transform = \
                    dp_pkg.create_model_and_transforms(config=cfg)
                self._model.eval()
                if torch.cuda.is_available():
                    self._model = self._model.cuda()
            self._strategy = "snapshot"
        except Exception as e:
            raise RuntimeError(
                f"[nkVasi] Depth Pro all strategies failed: {e}\n"
                f"Manual fix: pip install git+https://github.com/apple/ml-depth-pro"
            ) from e

    @torch.inference_mode()
    def infer(self, pil_img: Image.Image) -> np.ndarray:
        """Returns float32 H×W normalised depth: 0=nearest (FG), 1=farthest (BG)."""
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

        d_min, d_max = float(depth.min()), float(depth.max())
        if d_max - d_min < 1e-5:
            return np.zeros_like(depth)
        norm = (depth - d_min) / (d_max - d_min)

        h, w   = norm.shape
        cy, cx = h // 2, w // 2
        centre_val = float(norm[cy - h // 8:cy + h // 8, cx - w // 8:cx + w // 8].mean())
        border_val = float(np.concatenate([
            norm[:h // 8, :].ravel(), norm[-h // 8:, :].ravel(),
            norm[:, :w // 8].ravel(), norm[:, -w // 8:].ravel(),
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

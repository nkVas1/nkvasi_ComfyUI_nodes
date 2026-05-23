"""
Safe auto-installer for optional dependencies.

Design rules:
  1. NEVER downgrade an already-installed package.
  2. Install with --no-deps to avoid cascade dependency resolution.
  3. Pin the exact version known to work without conflicts in ComfyUI.
  4. Run silently in background thread so ComfyUI startup is not blocked.
  5. Log results to ComfyUI console with [nkVasi] prefix.
  6. Re-check availability after install so the node picks it up
     in the same session without restart (via importlib.reload).
"""
import sys
import subprocess
import importlib
import threading
from typing import Callable, Optional


# ------------------------------------------------------------------ #
# Package specs: (import_name, pip_name, version_constraint)          #
# version_constraint is passed to pip only when the package is ABSENT #
# i.e. we never force-reinstall or downgrade anything.                #
# ------------------------------------------------------------------ #
_OPTIONAL_PACKAGES = [
    (
        "pymatting",          # import name
        "pymatting",          # pip name
        ">=1.1.8",            # minimum version we need
        # --no-deps: pymatting's only real dep is numpy which is already present
        ["--no-deps"],
    ),
]


def _is_installed(import_name: str, min_version: Optional[str] = None) -> bool:
    """Return True if the package is importable (and optionally meets version)."""
    try:
        mod = importlib.import_module(import_name)
        if min_version and hasattr(mod, "__version__"):
            from packaging.version import Version
            required = min_version.lstrip(">=").lstrip(">").strip()
            if Version(mod.__version__) < Version(required):
                return False
        return True
    except ImportError:
        return False


def _install_one(
    import_name: str,
    pip_name: str,
    version_constraint: str,
    extra_flags: list,
) -> bool:
    """
    Install a single package using the same Python executable that is
    running ComfyUI.  Returns True on success.
    """
    pip_spec = f"{pip_name}{version_constraint}"
    cmd = [
        sys.executable, "-m", "pip", "install",
        pip_spec,
        "--quiet",
        "--no-warn-script-location",
    ] + extra_flags

    print(f"[nkVasi] Auto-installing {pip_spec} ...", flush=True)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            print(f"[nkVasi] {pip_name} installed ✓", flush=True)
            return True
        else:
            print(
                f"[nkVasi] {pip_name} install FAILED:\n{result.stderr.strip()}",
                flush=True,
            )
            return False
    except Exception as exc:
        print(f"[nkVasi] {pip_name} install exception: {exc}", flush=True)
        return False


def _refresh_import(import_name: str) -> bool:
    """Try to import / reload the module after installation."""
    try:
        if import_name in sys.modules:
            importlib.reload(sys.modules[import_name])
        else:
            importlib.import_module(import_name)
        return True
    except Exception:
        return False


def ensure_optional_deps(callback: Optional[Callable] = None) -> None:
    """
    Check and install missing optional dependencies in a background thread.

    After all installs are done, calls `callback()` if provided so the
    caller can refresh availability flags (e.g. re-check _PYMATTING_OK).

    Usage in __init__.py::

        from .utils.auto_install import ensure_optional_deps
        ensure_optional_deps()
    """
    def _worker():
        any_installed = False
        for import_name, pip_name, version_constraint, extra_flags in _OPTIONAL_PACKAGES:
            if _is_installed(import_name):
                # Already present — do nothing, no version downgrade ever
                continue
            ok = _install_one(import_name, pip_name, version_constraint, extra_flags)
            if ok:
                _refresh_import(import_name)
                any_installed = True

        if any_installed and callback is not None:
            try:
                callback()
            except Exception:
                pass

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

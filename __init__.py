from .nodes.rmbg_node import NkVasi_RMBG_Node
from .nodes.rmbg_ensemble import NkVasi_RMBG_Ensemble
from .nodes.mask_refine import NkVasi_MaskRefine
from .nodes.mask_tools import NkVasi_MaskTools
from .nodes.save_image_alpha import NkVasi_SaveImageAlpha
from .nodes.alpha_preview import NkVasi_AlphaPreview
from .nodes.matting_refine import NkVasi_MattingRefine

NODE_CLASS_MAPPINGS = {
    "NkVasi_RMBG":           NkVasi_RMBG_Node,
    "NkVasi_RMBG_Ensemble":  NkVasi_RMBG_Ensemble,
    "NkVasi_MaskRefine":     NkVasi_MaskRefine,
    "NkVasi_MaskTools":      NkVasi_MaskTools,
    "NkVasi_SaveImageAlpha": NkVasi_SaveImageAlpha,
    "NkVasi_AlphaPreview":   NkVasi_AlphaPreview,
    "NkVasi_MattingRefine":  NkVasi_MattingRefine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NkVasi_RMBG":           "🎭 Remove BG (nkVasi)",
    "NkVasi_RMBG_Ensemble":  "🎭 Remove BG Ensemble (nkVasi)",
    "NkVasi_MaskRefine":     "🔬 Mask Refine (nkVasi)",
    "NkVasi_MaskTools":      "🛠️ Mask Tools (nkVasi)",
    "NkVasi_SaveImageAlpha": "💾 Save Image Alpha (nkVasi)",
    "NkVasi_AlphaPreview":   "🔍 Alpha Preview (nkVasi)",
    "NkVasi_MattingRefine":  "✨ Matting Refine (nkVasi)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

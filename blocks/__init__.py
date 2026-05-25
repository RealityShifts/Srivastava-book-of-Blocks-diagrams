"""Built-in block library.

Each submodule defines a single category and exports:

    CATEGORY      -- short slug (matches subdirectory under ``./diagrams``)
    CATEGORY_DESC -- one-line description
    BLOCKS        -- mapping of ``BlockName -> Spec`` tuples

This package collects them all into ``CATEGORIES`` in a stable order.
Adding a new built-in category is a two-step process:

    1. Drop ``blocks/<slug>.py`` next to the others, exposing the three names above.
    2. Add the module name to ``_MODULES`` below so that ordering is explicit.
"""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

from dsl import Spec

_MODULES = (
    "core",
    "attention",
    "transformer",
    "cnn_vision",
    "unet_diffusion",
    "gan",
    "vit",
    "sequence",
    "gnn",
    "generative",
    "rl",
    "memory_retrieval",
    "embedding",
    "optimization",
    "multimodal",
    "efficient",
    "specialized",
)


def _build() -> Dict[str, Tuple[str, Dict[str, Spec]]]:
    out: Dict[str, Tuple[str, Dict[str, Spec]]] = {}
    for mod_name in _MODULES:
        mod = importlib.import_module(f"{__name__}.{mod_name}")
        slug = getattr(mod, "CATEGORY", mod_name)
        desc = getattr(mod, "CATEGORY_DESC", "")
        blocks = getattr(mod, "BLOCKS")
        if slug in out:
            raise ValueError(f"duplicate built-in category slug: {slug!r}")
        out[slug] = (desc, blocks)
    return out


CATEGORIES: Dict[str, Tuple[str, Dict[str, Spec]]] = _build()

__all__ = ["CATEGORIES"]

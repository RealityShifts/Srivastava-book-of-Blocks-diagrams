"""Dump the built-in block library into editor/public/library.json.

The React app loads this manifest at runtime so the palette can offer every
built-in block as a draggable ``_ref``, and the canvas can inline-expand any
ref into its sub-graph. Re-run whenever blocks/*.py changes::

    python editor/scripts/extract_library.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from blocks import CATEGORIES  # noqa: E402
from dsl import KINDS  # noqa: E402


def _node_to_dict(node: Tuple[Any, ...]) -> Dict[str, Any]:
    if len(node) == 2:
        kind, label = node
        return {"kind": kind, "label": label}
    if len(node) == 3:
        kind, label, meta = node
        return {"kind": kind, "label": label, "meta": meta}
    raise ValueError(f"unexpected node tuple shape: {node!r}")


def _spec_to_dict(spec: Sequence[Any]) -> Dict[str, Any]:
    if len(spec) == 3:
        desc, shapes, rows = spec
        skips: List[Tuple[int, int, int, int]] = []
        edges = None
    elif len(spec) == 4:
        desc, shapes, rows, skips = spec
        edges = None
    elif len(spec) == 5:
        desc, shapes, rows, skips, edges = spec
    else:
        raise ValueError(f"unexpected spec arity {len(spec)}")

    norm_rows = [[_node_to_dict(n) for n in row] for row in rows]
    for row in norm_rows:
        for n in row:
            assert n["kind"] in KINDS, f"unknown kind {n['kind']!r}"

    norm_edges: List[List[Any]] | None
    if edges is None:
        norm_edges = None
    else:
        norm_edges = [list(e) for e in edges]

    return {
        "desc": desc,
        "shapes": shapes,
        "rows": norm_rows,
        "skips": [list(s) for s in skips],
        "edges": norm_edges,
    }


def main() -> None:
    out: Dict[str, Any] = {"categories": {}}
    n_blocks = 0
    for cat, (cat_desc, blocks) in CATEGORIES.items():
        out["categories"][cat] = {
            "desc": cat_desc,
            "blocks": {name: _spec_to_dict(spec) for name, spec in blocks.items()},
        }
        n_blocks += len(blocks)

    dst = HERE.parent / "public" / "library.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"wrote {n_blocks} blocks across {len(out['categories'])} categories -> {dst}")


if __name__ == "__main__":
    main()

"""Snapshot the public class surface of RealityShifts/Srivastava-book-of-Blocks
into a JSON registry consumed by `_codegen.py` (Python) and
`src/generators/pytorchCodegen.ts` / `flaxCodegen.ts` (TS).

For each `pytorch_blocks/*_blocks.py` and `flax_blocks/*_blocks.py` we
record every top-level `class` and (if present) its `__init__` and
`forward` (or `__call__`) parameter lists. This is enough for the
codegen to:

  - know which submodule to import any block from,
  - emit a sensible kwargs-call placeholder with the actual parameter
    names and defaults the constructor expects,
  - know how many positional tensors `forward()` needs (1 → `self.f(x)`,
    2 → `self.f(x, y)`, etc.) so the wiring loop can be correct.

Usage::

    python editor/scripts/extract_block_registry.py /path/to/Srivastava-book-of-Blocks
    # → writes editor/public/pytorch_blocks.json and flax_blocks.json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _ann(node: Optional[ast.AST]) -> Optional[str]:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def _default(node: Optional[ast.AST]) -> Optional[str]:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def _params(func: ast.FunctionDef) -> List[Dict[str, Any]]:
    """Extract positional + keyword params from a function def, skipping
    self/cls. Returns a list of `{name, annotation?, default?, kind}` dicts.
    """
    out: List[Dict[str, Any]] = []
    args = func.args

    # Build defaults right-aligned to positional args.
    pos = args.args
    defaults = [None] * (len(pos) - len(args.defaults)) + list(args.defaults)
    for a, d in zip(pos, defaults):
        if a.arg in ("self", "cls"):
            continue
        out.append({
            "name": a.arg,
            "annotation": _ann(a.annotation),
            "default": _default(d),
            "kind": "positional",
        })

    for a, d in zip(args.kwonlyargs, args.kw_defaults):
        out.append({
            "name": a.arg,
            "annotation": _ann(a.annotation),
            "default": _default(d),
            "kind": "keyword",
        })

    return out


def _docstring_first_line(node: ast.ClassDef) -> Optional[str]:
    doc = ast.get_docstring(node)
    if not doc:
        return None
    return doc.split("\n", 1)[0].strip() or None


def extract_module(path: Path) -> Dict[str, Any]:
    tree = ast.parse(path.read_text(), filename=str(path))
    classes: Dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        init: Optional[List[Dict[str, Any]]] = None
        forward_arity: Optional[int] = None
        for sub in node.body:
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if sub.name == "__init__":
                    init = _params(sub)
                elif sub.name in ("forward", "__call__"):
                    # Count positional, non-self tensor params (annotation is
                    # not always present, but param count is what we need).
                    pos = [a for a in sub.args.args if a.arg not in ("self", "cls")]
                    forward_arity = len(pos)
        classes[node.name] = {
            "doc": _docstring_first_line(node),
            "init": init,
            "forward_arity": forward_arity,
        }
    return classes


def extract_framework(root: Path, framework: str) -> Dict[str, Any]:
    fw_dir = root / framework
    if not fw_dir.is_dir():
        raise FileNotFoundError(f"missing {fw_dir}")
    modules: Dict[str, Dict[str, Any]] = {}
    for path in sorted(fw_dir.glob("*_blocks.py")):
        modules[path.stem] = extract_module(path)
    # Flat index: ClassName -> module stem (e.g. "ResidualBlock" -> "core_blocks").
    # When the same name appears in multiple modules we keep the first.
    index: Dict[str, str] = {}
    for mod_name, classes in modules.items():
        for cls_name in classes:
            index.setdefault(cls_name, mod_name)
    return {"framework": framework, "modules": modules, "index": index}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "root", type=Path,
        help="Path to a local checkout of RealityShifts/Srivastava-book-of-Blocks "
             "(contains pytorch_blocks/ and flax_blocks/ subdirs).",
    )
    p.add_argument(
        "--out-dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "public",
        help="Where to write the JSON files (default: editor/public/).",
    )
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for fw in ("pytorch_blocks", "flax_blocks"):
        data = extract_framework(args.root, fw)
        out_path = args.out_dir / f"{fw}.json"
        out_path.write_text(json.dumps(data, indent=2) + "\n")
        n_modules = len(data["modules"])
        n_classes = len(data["index"])
        print(f"wrote {out_path}: {n_modules} modules, {n_classes} classes",
              file=sys.stderr)


if __name__ == "__main__":
    main()

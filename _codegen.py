"""Compile a DSL spec.py into runnable PyTorch ``nn.Module`` code by binding
``_ref(...)`` nodes to classes in ``RealityShifts/Srivastava-book-of-Blocks``.

What gets generated, per block in the input spec:

* one ``nn.Module`` subclass whose ``__init__`` instantiates one attribute
  per ``_ref`` node from the matching ``pytorch_blocks.<category>.<Class>``;
* a topologically-ordered ``forward(...)`` whose positional args are the
  block's source IO nodes (no incoming edge) and which returns the sink IO
  nodes (no outgoing edge), with one local per intermediate node;
* inline torch ops for primitives whose label is unambiguous
  (``+``, ``concat``, ``ReLU``, ``GELU``, ``softmax``, …) and
  ``# TODO`` placeholders for everything else.

Constructor kwargs are *placeholders*: we use the actual parameter names
extracted from the upstream source, default values where available, and
``...`` for non-defaulted args. The author fills those in before training.

Usage::

    python _codegen.py --specs examples/my_arch.py            # → my_arch_model.py next to it
    python _codegen.py --specs FOO.py -o bar.py               # explicit out
    python _codegen.py --specs FOO.py -o -                    # stdout
    python _codegen.py --specs FOO.py --registry editor/public/pytorch_blocks.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Reuse the same spec-loading + unpacking dance _generate.py uses.
from _generate import _spec_unpack, _node_parts  # type: ignore


REGISTRY_DEFAULT = Path(__file__).parent / "editor" / "public" / "pytorch_blocks.json"


# ---------------------------------------------------------------------------
# Inline primitive mapping
# ---------------------------------------------------------------------------
# Labels (case-insensitive, alpha-num-only key) -> python expression template.
# `{ins}` is replaced by a comma-joined input list; `{a}` and `{b}` by the
# first two inputs when present.

_ACTIVATIONS = {
    "relu": "F.relu({a})",
    "gelu": "F.gelu({a})",
    "silu": "F.silu({a})",
    "swish": "F.silu({a})",
    "sigmoid": "torch.sigmoid({a})",
    "tanh": "torch.tanh({a})",
    "softplus": "F.softplus({a})",
    "softmax": "F.softmax({a}, dim=-1)",
    "logsoftmax": "F.log_softmax({a}, dim=-1)",
    "mish": "({a} * torch.tanh(F.softplus({a})))",
    "identity": "{a}",
    "leakyrelu": "F.leaky_relu({a}, 0.2)",
    "elu": "F.elu({a})",
}

_MERGES = {
    "+": "({ins_plus})",
    "add": "({ins_plus})",
    "sum": "({ins_plus})",
    "*": "({ins_star})",
    "mul": "({ins_star})",
    "concat": "torch.cat([{ins}], dim=-1)",
    "cat": "torch.cat([{ins}], dim=-1)",
    "stack": "torch.stack([{ins}], dim=1)",
}


def _norm_label(label: str) -> str:
    """Reduce a free-form node label to a lookup key: lowercased alpha-num."""
    return re.sub(r"[^a-z0-9+*]", "", label.lower())


def _inline_op(label: str, inputs: List[str]) -> Optional[str]:
    """Return a Python expression for a primitive node, or None to defer."""
    key = _norm_label(label)
    if not inputs:
        return None

    if key in _ACTIVATIONS:
        # All activations consume a single tensor; if multiple inputs were
        # wired in, fall through so the caller emits a TODO.
        if len(inputs) == 1:
            return _ACTIVATIONS[key].format(a=inputs[0])
        return None

    if key in _MERGES:
        return _MERGES[key].format(
            ins=", ".join(inputs),
            ins_plus=" + ".join(inputs),
            ins_star=" * ".join(inputs),
        )

    # Heuristic fallbacks for compact labels seen in diagrams.
    if "matmul" in key or ("·" in label and len(inputs) == 1):
        return None  # truly stateful → user must wire a Linear
    return None


# ---------------------------------------------------------------------------
# Spec loader
# ---------------------------------------------------------------------------


def _load_specs(path: Path) -> Tuple[str, str, Dict[str, Any]]:
    """Load a spec.py module exactly the way _generate.py does."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    if hasattr(mod, "CATEGORIES"):
        # Flatten — codegen doesn't care about per-category framing.
        blocks: Dict[str, Any] = {}
        category = "mixed"
        category_desc = "Mixed categories from a multi-category spec."
        for cat_name, cat_data in mod.CATEGORIES.items():
            blocks.update(cat_data.get("blocks", {}))
            category = cat_name
            category_desc = cat_data.get("desc", category_desc)
        return category, category_desc, blocks
    if not hasattr(mod, "BLOCKS"):
        raise RuntimeError(f"{path} exposes neither BLOCKS nor CATEGORIES")
    return (
        getattr(mod, "CATEGORY", "user"),
        getattr(mod, "CATEGORY_DESC", "User-authored blocks."),
        mod.BLOCKS,
    )


# ---------------------------------------------------------------------------
# Per-block graph extraction
# ---------------------------------------------------------------------------


class _G:
    """Lightweight per-block working graph."""

    def __init__(self) -> None:
        self.uids: List[str] = []                                # in row-major declaration order
        self.kind: Dict[str, str] = {}
        self.label: Dict[str, str] = {}
        self.ref: Dict[str, str] = {}                            # uid -> ref block name
        self.row: Dict[str, int] = {}
        self.col: Dict[str, int] = {}
        self.row_width: Dict[int, int] = {}
        self.preds: Dict[str, List[str]] = defaultdict(list)
        self.succs: Dict[str, List[str]] = defaultdict(list)


def _uid_for(meta: Dict[str, Any], r: int, c: int) -> str:
    """Fall back to a synthetic id when the spec didn't set one."""
    explicit = meta.get("id") if meta else None
    return explicit or f"r{r}c{c}"


def _build_graph(spec: Tuple[Any, ...]) -> _G:
    desc, shapes, rows, skips, edges_explicit = _spec_unpack(spec)
    g = _G()
    g.desc = desc            # type: ignore[attr-defined]
    g.shapes = shapes        # type: ignore[attr-defined]

    # Pass 1: register nodes; honour spec-supplied uids when present.
    for r, row in enumerate(rows):
        g.row_width[r] = len(row)
        for c, node in enumerate(row):
            kind, label, meta = _node_parts(node)
            uid = _uid_for(meta, r, c)
            g.uids.append(uid)
            g.kind[uid] = kind
            g.label[uid] = label
            g.row[uid] = r
            g.col[uid] = c
            if kind == "ref":
                g.ref[uid] = label

    # Pass 2: edges. Mirror _generate.py's auto-wiring rule when none supplied.
    def _add(src: str, dst: str) -> None:
        g.preds[dst].append(src)
        g.succs[src].append(dst)

    row_uids = [[g.uids[sum(g.row_width[k] for k in range(r)) + c] for c in range(g.row_width[r])]
                for r in range(len(rows))]

    if edges_explicit:
        # Edge specs are (src, dst[, label]) tuples that reference uids.
        for e in edges_explicit:
            src, dst = e[0], e[1]
            if src in g.kind and dst in g.kind:
                _add(src, dst)
    else:
        for r in range(len(row_uids) - 1):
            a, b = row_uids[r], row_uids[r + 1]
            if len(a) > 1 and len(a) == len(b):
                for i in range(len(a)):
                    _add(a[i], b[i])
            else:
                for s in a:
                    for t in b:
                        _add(s, t)

    for r1, c1, r2, c2 in skips:
        _add(row_uids[r1][c1], row_uids[r2][c2])

    return g


# ---------------------------------------------------------------------------
# Code emission
# ---------------------------------------------------------------------------


def _safe_attr(s: str) -> str:
    """Lower-snake, suffix to avoid collisions / keywords."""
    out = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    if not out or out[0].isdigit():
        out = "x_" + out
    return out


def _safe_class(s: str) -> str:
    parts = re.split(r"[^A-Za-z0-9]+", s)
    parts = [p[:1].upper() + p[1:] for p in parts if p]
    name = "".join(parts) or "Block"
    if name[0].isdigit():
        name = "B" + name
    return name


def _var(uid: str) -> str:
    return "t_" + re.sub(r"[^A-Za-z0-9]+", "_", uid)


def _placeholder_kwargs(init: Optional[List[Dict[str, Any]]]) -> Tuple[str, List[str]]:
    """Render an ``__init__`` parameter list into a kwargs call expression.

    Defaulted params get their default; required ones get ``...``. Inline
    comments are NOT used per-arg (they'd swallow the trailing comma).
    Returns ``(call_expr, required_names)`` so the caller can emit a single
    summary ``# TODO`` line that names the required params.
    """
    if not init:
        return "", []
    parts: List[str] = []
    required: List[str] = []
    for p in init:
        if p["default"] is not None:
            parts.append(f"{p['name']}={p['default']}")
        else:
            parts.append(f"{p['name']}=...")
            required.append(p["name"])
    return ", ".join(parts), required


def _topo(g: _G) -> List[str]:
    indeg = {u: len(g.preds[u]) for u in g.uids}
    q = [u for u in g.uids if indeg[u] == 0]
    out: List[str] = []
    head = 0
    while head < len(q):
        u = q[head]; head += 1
        out.append(u)
        for v in g.succs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    # Cycle tail (preserve declaration order so we never silently drop nodes).
    for u in g.uids:
        if u not in out:
            out.append(u)
    return out


def _emit_block(
    block_name: str,
    spec: Tuple[Any, ...],
    registry_index: Dict[str, str],
    registry_modules: Dict[str, Any],
    framework: str,
) -> str:
    g = _build_graph(spec)
    order = _topo(g)

    sources = [u for u in g.uids if g.kind[u] == "io" and not g.preds[u]]
    sinks = [u for u in g.uids if g.kind[u] == "io" and not g.succs[u]]
    if not sources:
        # No IO declared at the top — treat the first node as the only input.
        sources = [g.uids[0]] if g.uids else []
    if not sinks:
        sinks = [g.uids[-1]] if g.uids else []

    # Sub-module instantiation lines.
    init_lines: List[str] = []
    attr_of: Dict[str, str] = {}
    used: Dict[str, int] = {}
    for u in g.uids:
        if g.kind[u] != "ref":
            continue
        ref_name = g.ref[u]
        attr_base = _safe_attr(ref_name)
        used[attr_base] = used.get(attr_base, 0) + 1
        attr = attr_base if used[attr_base] == 1 else f"{attr_base}_{used[attr_base]}"
        attr_of[u] = attr
        mod_name = registry_index.get(ref_name)
        if mod_name is None:
            init_lines.append(
                f"        # TODO: '{ref_name}' is not in {framework}_blocks; "
                f"replace with a local nn.Module."
            )
            init_lines.append(f"        self.{attr} = nn.Identity()")
            continue
        init = registry_modules[mod_name][ref_name]["init"]
        kwargs, required = _placeholder_kwargs(init)
        alias = mod_name.removesuffix("_blocks")
        if required:
            init_lines.append(f"        # TODO: set required kwargs {required}")
        init_lines.append(f"        self.{attr} = {alias}.{ref_name}({kwargs})")

    if not init_lines:
        init_lines.append("        pass  # no _ref sub-modules in this block")

    fwd_lines: List[str] = []
    arg_list = ", ".join(_var(u) for u in sources) or "x"
    fwd_sig = f"    def forward(self, {arg_list}):"

    for u in order:
        var = _var(u)
        ins = [_var(p) for p in g.preds[u]]

        if u in sources:
            continue

        kind = g.kind[u]
        label = g.label[u]

        if kind == "ref" and u in attr_of:
            attr = attr_of[u]
            # Pass exactly the wired predecessors; the registry's forward_arity
            # counts optional params too, so trusting it would lead to
            # duplicated args (e.g. attn_mask).
            call = ", ".join(ins) if ins else "..."
            note = "" if ins else "  # TODO: '{label}' has no wired inputs".format(label=label)
            fwd_lines.append(f"        {var} = self.{attr}({call}){note}")
            continue

        if kind == "io":
            # Pass-through; collapse to the first predecessor.
            if ins:
                fwd_lines.append(f"        {var} = {ins[0]}")
            else:
                fwd_lines.append(f"        {var} = ...  # TODO: unconnected IO '{label}'")
            continue

        # Inline primitives (op/norm/act/merge/...) — try the table.
        expr = _inline_op(label, ins)
        if expr is not None:
            fwd_lines.append(f"        {var} = {expr}")
            continue

        if not ins:
            fwd_lines.append(
                f"        {var} = ...  # TODO: implement '{label}' (no inputs wired)"
            )
            continue
        first = ins[0]
        more = f"  # NOTE: other inputs {ins[1:]} unused" if len(ins) > 1 else ""
        fwd_lines.append(
            f"        {var} = {first}  # TODO: implement '{label}' (kind={kind}){more}"
        )

    ret = ", ".join(_var(u) for u in sinks) if sinks else "x"
    fwd_lines.append(f"        return {ret}")

    class_name = _safe_class(block_name)
    desc = getattr(g, "desc", "")
    shapes = getattr(g, "shapes", "")
    doc = f'    """{desc}\n\n    Shapes: ``{shapes}``\n    """' if desc else ""

    body = "\n".join([
        f"class {class_name}(nn.Module):",
        doc if doc else "",
        "    def __init__(self):",
        "        super().__init__()",
        *init_lines,
        "",
        fwd_sig,
        *fwd_lines,
    ])
    return body


def _imports_for(
    blocks_by_name: Dict[str, Tuple[Any, ...]],
    registry_index: Dict[str, str],
    framework: str,
) -> List[str]:
    """Collect ``import pytorch_blocks.X as alias`` lines for the
    sub-modules actually used by any ref node across all blocks."""
    needed_modules: set[str] = set()
    for spec in blocks_by_name.values():
        g = _build_graph(spec)
        for u in g.ref:
            mod = registry_index.get(g.ref[u])
            if mod:
                needed_modules.add(mod)
    pkg = "pytorch_blocks" if framework == "pytorch" else "flax_blocks"
    out = []
    for mod in sorted(needed_modules):
        alias = mod.removesuffix("_blocks")
        out.append(f"from {pkg} import {mod} as {alias}")
    return out


def generate_module(
    blocks: Dict[str, Tuple[Any, ...]],
    *,
    registry: Dict[str, Any],
    framework: str,
    source_path: Optional[Path],
) -> str:
    if framework != "pytorch":
        raise ValueError(f"v1 only supports framework='pytorch', got {framework!r}")

    fw_module_alias = "torch.nn as nn"
    fw_extras = "import torch\nimport torch.nn.functional as F"

    index = registry["index"]
    modules = registry["modules"]

    imports = _imports_for(blocks, index, framework)

    blocks_src = "\n\n\n".join(
        _emit_block(name, spec, index, modules, framework)
        for name, spec in blocks.items()
    )

    src_line = f"# Source spec: {source_path}\n" if source_path else ""
    header = (
        '"""Auto-generated by _codegen.py from a Block Diagrams DSL spec.\n\n'
        "Built against RealityShifts/Srivastava-book-of-Blocks "
        f"({framework}_blocks).\n"
        f"{src_line}"
        "\n"
        "Constructor kwargs marked ``...  # TODO`` need real values before\n"
        "training; nodes annotated ``# TODO: implement '<label>'`` are\n"
        "primitive ops the codegen couldn't map automatically (free-form\n"
        "labels like 'matmul x · Wᵀ') — wire in a real ``nn.Linear`` etc.\n"
        '"""\n\n'
        f"from __future__ import annotations\n\n"
        f"{fw_extras}\n"
        f"import {fw_module_alias}\n"
    )

    return header + "\n".join(imports) + "\n\n\n" + blocks_src + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--specs", type=Path, required=True,
                   help="Path to a spec.py exposing BLOCKS or CATEGORIES.")
    p.add_argument("-o", "--out", default=None,
                   help="Output .py path, or '-' for stdout. "
                        "Default: <specs_stem>_model.py next to the input.")
    p.add_argument("--registry", type=Path, default=None,
                   help="Override the JSON block registry path. "
                        "Default: editor/public/pytorch_blocks.json")
    args = p.parse_args()

    framework = "pytorch"  # Flax registry is snapshotted but codegen is PT-only in v1.
    reg_path = args.registry
    if reg_path is None:
        reg_path = Path(__file__).parent / "editor" / "public" / "pytorch_blocks.json"
    if not reg_path.is_file():
        print(f"error: registry not found at {reg_path}. "
              f"Run editor/scripts/extract_block_registry.py first.",
              file=sys.stderr)
        sys.exit(2)
    registry = json.loads(reg_path.read_text())

    _, _, blocks = _load_specs(args.specs)
    if not blocks:
        print(f"error: no BLOCKS in {args.specs}", file=sys.stderr)
        sys.exit(2)

    out_src = generate_module(
        blocks,
        registry=registry,
        framework=framework,
        source_path=args.specs,
    )

    if args.out == "-":
        sys.stdout.write(out_src)
    else:
        out_path = Path(args.out) if args.out else args.specs.with_name(f"{args.specs.stem}_model.py")
        out_path.write_text(out_src)
        # Compile-check: catches obvious emit bugs without needing torch installed.
        try:
            compile(out_src, str(out_path), "exec")
            print(f"wrote {len(blocks)} block(s) -> {out_path} ({framework})",
                  file=sys.stderr)
        except SyntaxError as e:
            print(f"error: generated file has a syntax error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()

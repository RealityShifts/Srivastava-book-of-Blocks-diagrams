"""Generate Mermaid architecture diagrams for every block.

Run ``python _generate.py`` to (re)build everything. Use
``python _generate.py --depth N`` to recursively expand ``_ref`` nodes up
to ``N`` levels deep (default: 1).

The 122 blocks shipped here act as a *built-in library*: you can author
your own architecture spec file and ``_ref()`` any built-in block by
name. Run::

    python _generate.py --specs my_arch.py --out ./out

to generate diagrams for your blocks; pass ``--specs`` multiple times to
merge several files. Add ``--no-builtins`` to skip regenerating the 122
library files (they remain available as resolvable references).

By default, generated files land under ``./diagrams/<category>/<Name>.md``
(repo-root ``diagrams/`` keeps the source tree uncluttered). Pass ``--out``
to send them elsewhere.

Each block produces a self-contained Markdown file under
``<category>/<BlockName>.md`` containing one
Mermaid ``flowchart TD`` diagram. Markdown files render natively on GitHub
and most note apps; they import into draw.io via
*Arrange > Insert > Advanced > Mermaid* and into Excalidraw via
*Mermaid to Excalidraw*. Render to standalone SVG with::

    npx -y @mermaid-js/mermaid-cli -i input.md -o output.svg

DSL
---
Each spec is ``(desc, shapes, rows[, skips])`` where:

* ``rows`` is a list of rows; each row is a list of ``(kind, label)`` nodes
  drawn left-to-right.
* Edges between consecutive rows are auto-emitted: 1:1 when row sizes
  match, otherwise full bipartite (so parallel branches fan-out from a
  single node and fan-in into a single merge node).
* ``skips`` (optional) is a list of ``(from_row, from_col, to_row, to_col)``
  rendered as dashed arrows, useful for residuals.

Recursion (``_ref``)
--------------------
A node of kind ``ref`` whose label exactly matches a registered block name
is expanded inline as a Mermaid ``subgraph`` (up to ``--depth`` levels).
Beyond the depth limit, or for an unresolved name, the ref renders as a
single styled box. Cycles are detected and broken automatically.

User spec files
---------------
A user spec file is a regular Python file that exposes one of:

* ``CATEGORIES = {cat_name: (cat_desc, {block_name: spec, ...}), ...}``
  -- same shape as the built-in registry, for multi-category libraries.
* ``BLOCKS = {block_name: spec, ...}`` -- single category. Optionally set
  module-level ``CATEGORY = "..."`` and ``CATEGORY_DESC = "..."``;
  otherwise the file's stem is used as the category name.

Inside the file, write ``from _generate import _io, _op, _ref, ...`` to
get the DSL helpers; this script automatically prepends its own directory
to ``sys.path`` before loading user files.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from collections import namedtuple
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple  # noqa: F401

# Re-exported so user spec files can keep doing ``from _generate import _io``.
from dsl import (  # noqa: F401
    KINDS,
    Edge,
    Meta,
    Node,
    Row,
    Skip,
    Spec,
    _io,
    _op,
    _norm,
    _act,
    _attn,
    _merge,
    _emb,
    _loss,
    _ref,
    _edge,
)

DEFAULT_OUT_SUBDIR = "diagrams"

# Mermaid flowchart layout defaults. ``rankSpacing`` controls the gap (in
# pixels) between consecutive ranks (rows in TD flow); ``nodeSpacing`` is
# the gap between nodes in the same rank. Both are overridable per-run via
# ``--rank-spacing`` / ``--node-spacing`` on the CLI and via kwargs on
# :func:`render_diagram`.
DEFAULT_RANK_SPACING = 10
DEFAULT_NODE_SPACING = 30


# ---------------------------------------------------------------------------
# Mermaid styling shared by every diagram
# ---------------------------------------------------------------------------

STYLE = """    classDef io fill:#f1f5f9,stroke:#334155,stroke-width:1.4px,color:#0f172a
    classDef op fill:#dbeafe,stroke:#1d4ed8,stroke-width:1.4px,color:#1e3a8a
    classDef norm fill:#dcfce7,stroke:#15803d,stroke-width:1.4px,color:#14532d
    classDef act fill:#ffedd5,stroke:#c2410c,stroke-width:1.4px,color:#7c2d12
    classDef attn fill:#ede9fe,stroke:#6d28d9,stroke-width:1.4px,color:#4c1d95
    classDef merge fill:#fef3c7,stroke:#b45309,stroke-width:1.4px,color:#78350f
    classDef emb fill:#fef9c3,stroke:#a16207,stroke-width:1.4px,color:#713f12
    classDef loss fill:#fee2e2,stroke:#b91c1c,stroke-width:1.4px,color:#7f1d1d
    classDef ctrl fill:#f5f5f4,stroke:#52525b,stroke-width:1.4px,color:#27272a
    classDef ref fill:#e0f2fe,stroke:#0369a1,stroke-width:2px,color:#0c4a6e,stroke-dasharray: 4 2"""

Ports = Tuple[str, ...]
Slot = namedtuple("Slot", ["in_ports", "out_ports", "in_shape", "out_shape", "uid"])
SlotIds = List[List[Slot]]
EdgeRec = namedtuple("EdgeRec", ["src", "dst", "style", "label"])
# style ∈ {"solid", "skip", "mismatch"}


MISMATCH_STYLE = "stroke:#dc2626,stroke-width:2.2px,color:#991b1b"


# ---------------------------------------------------------------------------
# Mermaid renderer (with recursive `_ref` expansion via subgraph)
# ---------------------------------------------------------------------------


def _label(text: str) -> str:
    # Mermaid quoted labels accept most characters; escape stray quotes
    # and avoid confusing the parser with backticks.
    return text.replace('"', "&quot;").replace("`", "&#96;")


def _node_parts(node: Node) -> Tuple[str, str, Meta]:
    if len(node) == 2:
        return node[0], node[1], {}
    return node[0], node[1], node[2]


def _spec_unpack(spec: Spec) -> Tuple[str, str, List[Row], Sequence[Skip], Optional[Sequence[Edge]]]:
    if len(spec) == 3:
        desc, shapes, rows = spec
        return desc, shapes, rows, (), None
    if len(spec) == 4:
        desc, shapes, rows, skips = spec
        return desc, shapes, rows, skips, None
    desc, shapes, rows, skips, edges = spec
    return desc, shapes, rows, skips, edges


def _norm_shape(s: Optional[str]) -> Optional[str]:
    return None if s is None else "".join(s.split())


def _shape_check(src: Slot, dst: Slot) -> Tuple[bool, Optional[str]]:
    """Return (mismatch?, label_text). Only flags when both ends declare shapes."""
    if src.out_shape is None or dst.in_shape is None:
        return False, None
    if _norm_shape(src.out_shape) == _norm_shape(dst.in_shape):
        return False, None
    return True, f"{src.out_shape} ≠ {dst.in_shape}"


def _build_edges(
    src: Slot,
    dst: Slot,
    *,
    style: str = "solid",
    label: Optional[str] = None,
) -> List[EdgeRec]:
    mismatch, mlabel = _shape_check(src, dst)
    if mismatch and style != "skip":
        style = "mismatch"
        label = mlabel
    return [
        EdgeRec(s, d, style, label)
        for s in src.out_ports
        for d in dst.in_ports
    ]


def _render_rows(
    rows: List[Row],
    skips: Sequence[Skip],
    edges_explicit: Optional[Sequence[Edge]],
    registry: Optional[Dict[str, Spec]],
    max_depth: int,
    depth: int,
    prefix: str,
    ancestors: FrozenSet[str],
) -> Tuple[List[str], List[EdgeRec], SlotIds]:
    node_lines: List[str] = []
    edge_recs: List[EdgeRec] = []
    slot_ids: SlotIds = []
    id_to_pos: Dict[str, Tuple[int, int]] = {}

    for r, row in enumerate(rows):
        row_slots: List[Slot] = []
        for c, node in enumerate(row):
            kind, label, meta = _node_parts(node)
            assert kind in KINDS, f"unknown kind {kind!r} in row {r} col {c}"
            nid = f"{prefix}{r}_{c}"
            uid: Optional[str] = meta.get("id")
            in_shape: Optional[str] = meta.get("in")
            out_shape: Optional[str] = meta.get("out")

            sub: Optional[Spec] = None
            if (
                kind == "ref"
                and registry is not None
                and depth < max_depth
                and label in registry
                and label not in ancestors
            ):
                sub = registry[label]

            if sub is not None:
                _, _, sub_rows, sub_skips, sub_edges = _spec_unpack(sub)
                sub_node_lines, sub_edges_out, sub_slots = _render_rows(
                    sub_rows, sub_skips, sub_edges, registry, max_depth,
                    depth + 1, f"{nid}_", ancestors | {label},
                )
                node_lines.append(f'subgraph {nid}["{_label(label)}"]')
                node_lines.extend("    " + ln for ln in sub_node_lines)
                node_lines.append("end")
                # Inner edges bubble up to the flat list so linkStyle indices
                # remain consistent across the whole diagram.
                edge_recs.extend(sub_edges_out)
                # Aggregate every port of the first row (entry) and last row (exit)
                # so parents fan in / out across the full IO boundary of the sub-block.
                in_ports = tuple(p for slot in sub_slots[0] for p in slot.in_ports)
                out_ports = tuple(p for slot in sub_slots[-1] for p in slot.out_ports)
                row_slots.append(Slot(in_ports, out_ports, in_shape, out_shape, uid))
            else:
                node_lines.append(f'{nid}["{_label(label)}"]:::{kind}')
                row_slots.append(Slot((nid,), (nid,), in_shape, out_shape, uid))

            if uid is not None:
                if uid in id_to_pos:
                    raise ValueError(
                        f"duplicate node id {uid!r} in block at scope {prefix!r}"
                    )
                id_to_pos[uid] = (r, c)
        slot_ids.append(row_slots)

    # Explicit edges short-circuit auto wiring entirely; otherwise fall back
    # to the row-adjacency rule (column-aligned when widths match, else fan).
    if edges_explicit is None:
        for r in range(len(rows) - 1):
            a, b = slot_ids[r], slot_ids[r + 1]
            if len(a) == len(b) and len(a) > 1:
                for c in range(len(a)):
                    edge_recs.extend(_build_edges(a[c], b[c]))
            else:
                for ci in range(len(a)):
                    for cj in range(len(b)):
                        edge_recs.extend(_build_edges(a[ci], b[cj]))
    else:
        for spec_edge in edges_explicit:
            if len(spec_edge) == 2:
                src_id, dst_id = spec_edge
                e_label: Optional[str] = None
            else:
                src_id, dst_id, e_label = spec_edge
            if src_id not in id_to_pos:
                raise ValueError(f"edge references unknown id: {src_id!r}")
            if dst_id not in id_to_pos:
                raise ValueError(f"edge references unknown id: {dst_id!r}")
            sr, sc = id_to_pos[src_id]
            dr, dc = id_to_pos[dst_id]
            edge_recs.extend(
                _build_edges(slot_ids[sr][sc], slot_ids[dr][dc], label=e_label)
            )

    # Skip / residual edges (dashed). Shape mismatches on skips don't recolour
    # the line (skips are usually "same tensor, different path").
    for r1, c1, r2, c2 in skips:
        edge_recs.extend(
            _build_edges(slot_ids[r1][c1], slot_ids[r2][c2], style="skip", label="skip")
        )

    return node_lines, edge_recs, slot_ids


def _edge_line(rec: EdgeRec) -> str:
    if rec.style == "solid":
        if rec.label:
            return f'{rec.src} -->|"{_label(rec.label)}"| {rec.dst}'
        return f"{rec.src} --> {rec.dst}"
    if rec.style == "skip":
        return f'{rec.src} -.->|"{_label(rec.label or "skip")}"| {rec.dst}'
    if rec.style == "mismatch":
        return f'{rec.src} ==>|"{_label(rec.label or "shape mismatch")}"| {rec.dst}'
    raise ValueError(f"unknown edge style: {rec.style!r}")


def _init_directive(rank_spacing: int, node_spacing: int) -> str:
    """Mermaid ``%%{init}%%`` line that sets flowchart layout spacing.

    See https://mermaid.js.org/syntax/flowchart.html#configuration — these
    knobs are picked up by Mermaid >= 8 and tighten the vertical / horizontal
    gaps between rendered nodes so dense block diagrams stay readable.
    """
    return (
        "%%{init: {'flowchart': {"
        f"'rankSpacing': {rank_spacing}, 'nodeSpacing': {node_spacing}"
        "}}}%%"
    )


def render_diagram(
    rows: List[Row],
    skips: Sequence[Skip] = (),
    edges: Optional[Sequence[Edge]] = None,
    *,
    registry: Optional[Dict[str, Spec]] = None,
    max_depth: int = 0,
    block_name: str = "<unnamed>",
    rank_spacing: int = DEFAULT_RANK_SPACING,
    node_spacing: int = DEFAULT_NODE_SPACING,
) -> str:
    node_lines, edge_recs, _ = _render_rows(
        rows, skips, edges, registry, max_depth,
        depth=0, prefix="n", ancestors=frozenset(),
    )

    body: List[str] = list(node_lines)
    mismatch_indices: List[int] = []
    for i, rec in enumerate(edge_recs):
        body.append(_edge_line(rec))
        if rec.style == "mismatch":
            mismatch_indices.append(i)
            print(
                f"  shape mismatch in {block_name}: "
                f"{rec.src} → {rec.dst}  [{rec.label}]",
                file=sys.stderr,
            )
    if mismatch_indices:
        idx = ",".join(str(i) for i in mismatch_indices)
        body.append(f"linkStyle {idx} {MISMATCH_STYLE}")

    return "\n".join(
        [
            _init_directive(rank_spacing, node_spacing),
            "flowchart TD",
            *("    " + ln for ln in body),
            STYLE,
        ]
    )


def render_markdown(name: str, desc: str, shapes: str, body: str) -> str:
    return (
        f"# {name}\n\n"
        f"> {desc}\n\n"
        f"**Shapes:** `{shapes}`\n\n"
        f"```mermaid\n{body}\n```\n"
    )


# ---------------------------------------------------------------------------
# Block library (each spec lives in ./blocks/<category>.py and is exposed via
# blocks/__init__.py; see DSL docs in dsl.py)
# ---------------------------------------------------------------------------

from blocks import CATEGORIES  # noqa: E402


README_BODY = """\
# Architecture diagrams

One Mermaid `flowchart TD` per public block, organised by category. Specs
live in [`blocks/`](./blocks), the DSL helpers in [`dsl.py`](./dsl.py),
and the renderer in [`_generate.py`](./_generate.py). All generated
diagrams live under [`diagrams/`](./diagrams) — every block in the index
below links to its own `.md` file there. Regenerate everything with

```bash
python _generate.py
```

## Visual editor

A React Flow-based web editor lives in [`editor/`](./editor) — drag from
a palette of the 10 primitive kinds + all 122 built-in blocks, wire them
together, get live shape-checking (mismatched edges turn red), then export
to **Mermaid**, **DSL `.py`** (round-trips through `python _generate.py
--specs ...`), or **Graph JSON**.

```bash
cd editor
npm install
npm run extract-library   # snapshot blocks/ -> public/library.json
npm run dev               # http://localhost:5173
```

## Where these render

| Tool | What to do |
| ---- | ---------- |
| **GitHub** | Renders the `.md` files inline — just open them in the browser. |
| **draw.io / diagrams.net** | *Arrange → Insert → Advanced → Mermaid*, paste the fenced block. |
| **Excalidraw** | *Library → Mermaid to Excalidraw*, paste the fenced block. |
| **Notion / Obsidian / GitLab / VS Code** | Native Mermaid in markdown preview. |
| **Standalone SVG** | `npx -y @mermaid-js/mermaid-cli -i path/to/Block.md -o Block.svg` |

## Recommended diagramming tools (alternatives to draw.io)

* **Excalidraw** — beautiful hand-drawn look, exports SVG, supports Mermaid import.
* **D2** (`d2lang.com`) — declarative, very clean output, good for hierarchies.
* **Mermaid Live Editor** (`mermaid.live`) — paste & download SVG/PNG.
* **TikZ / PGF** — gold standard for paper-quality figures (LaTeX).
* **PlotNeuralNet** — tex-based 3-D blocks for deep nets.
* **NN-SVG** — quick SVGs for classic CNN / FCN / LeNet shapes.
* **Penrose** — declarative diagram constraints if you need bespoke layouts.

## Color legend

| Class | Used for |
| --- | --- |
| `io`     | Inputs and outputs |
| `op`     | Generic differentiable op (matmul, conv, …) |
| `norm`   | Normalisation layers |
| `act`    | Activation functions |
| `attn`   | Attention operators |
| `merge`  | Sum / concat / element-wise combine |
| `emb`    | Embedding tables / encoded representations |
| `loss`   | Loss / objective |
| `ctrl`   | Control / non-differentiable flow |
| `ref`    | Reference to another block (expanded as a subgraph) |

Dashed arrows (`-. skip .->`) mark residual / skip connections.

## Recursive expansion

Specs may use `_ref("BlockName")` to point at another registered block.
Run `python _generate.py --depth N` to inline references up to `N` levels
deep as nested Mermaid `subgraph` blocks (default: 1). At depth 0 each
ref renders as a single `ref`-styled box; cycles are detected and broken.

## Custom architectures (`--specs`)

The 122 built-in blocks act as a reusable library. To diagram your own
architecture, write a Python file that exposes either `BLOCKS` (single
category) or `CATEGORIES` (multi-category), reusing the DSL helpers and
referencing built-ins by name:

```python
# my_arch.py
from _generate import _io, _op, _ref

CATEGORY = "myarch"
CATEGORY_DESC = "A 12-layer transformer wired from built-in blocks."

BLOCKS = {
    "MyTransformer": (
        "Stacked TransformerEncoderBlocks fed by a token embedding.",
        "(B, T) → (B, T, D)",
        [
            [_io("ids  (B, T)")],
            [_ref("TokenEmbedding")],
            [_ref("TransformerEncoderBlock")],
            [_ref("TransformerEncoderBlock")],
            [_io("y  (B, T, D)")],
        ],
    ),
}
```

Then generate:

```bash
# writes into ./diagrams/<your category>/<BlockName>.md by default
python _generate.py --specs my_arch.py --depth 1
# only your blocks, library kept as registered references:
python _generate.py --specs my_arch.py --no-builtins
# or send the output anywhere else
python _generate.py --specs my_arch.py --out ./out
```

Pass `--specs` multiple times to merge several files. User block names
shadow built-ins of the same name.
"""


CategoriesMap = Dict[str, Tuple[str, Dict[str, Spec]]]


def build_registry(*category_maps: CategoriesMap) -> Dict[str, Spec]:
    """Flatten one or more category dicts into a single ``name -> spec`` map.

    Built-in categories are passed first; later maps (e.g. user-supplied)
    override earlier ones, so users can shadow a built-in block by reusing
    its name. Duplicate names *within* a single map raise — that's a bug.
    """
    reg: Dict[str, Spec] = {}
    for i, cats in enumerate(category_maps):
        seen_in_map: set = set()
        for _cat, (_desc, blocks) in cats.items():
            for name, spec in blocks.items():
                if name in seen_in_map:
                    raise ValueError(
                        f"duplicate block name within map #{i}: {name!r}"
                    )
                seen_in_map.add(name)
                reg[name] = spec
    return reg


def load_user_specs(path: Path) -> CategoriesMap:
    """Import a user spec file and normalise its exports to ``CategoriesMap``.

    Accepts either ``CATEGORIES`` (preferred, multi-category) or ``BLOCKS``
    (single category, with optional ``CATEGORY`` / ``CATEGORY_DESC``).
    """
    here = str(Path(__file__).parent.resolve())
    if here not in sys.path:
        # so user files can `from _generate import _io, _op, _ref, ...`
        sys.path.insert(0, here)

    spec = importlib.util.spec_from_file_location(f"_user_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if hasattr(mod, "CATEGORIES"):
        return dict(mod.CATEGORIES)
    if hasattr(mod, "BLOCKS"):
        cat = getattr(mod, "CATEGORY", path.stem)
        desc = getattr(
            mod, "CATEGORY_DESC", f"User-supplied blocks from {path.name}.",
        )
        return {cat: (desc, dict(mod.BLOCKS))}
    raise ValueError(
        f"{path}: must define BLOCKS (dict) or CATEGORIES (dict) at module level"
    )


def _safe_rel(target: Path, root: Path) -> Optional[str]:
    """Return a relative link from ``root`` to ``target`` only if ``target``
    lives inside ``root``. Avoids leaking absolute filesystem paths into
    INDEX/markdown when the user's output dir is separate from the built-in
    library tree.
    """
    try:
        target.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    return os.path.relpath(target, root)


def _readme_index(
    categories: CategoriesMap,
    repo_root: Path,
    paths: Dict[str, Path],
) -> str:
    """Build the per-category, collapsible block index that gets appended to
    the static README body. Links are repo-relative so they work on GitHub.
    """
    lines = ["", "## Index", "", f"{sum(len(b) for _, b in categories.values())} blocks across {len(categories)} categories. Click a section to expand.", ""]
    for cat, (desc, blocks) in categories.items():
        lines.append(
            f"<details><summary><b>{cat}</b> &middot; {desc} &middot; "
            f"{len(blocks)} block{'s' if len(blocks) != 1 else ''}</summary>"
        )
        lines.append("")
        for name in blocks:
            target = paths.get(name)
            rel = _safe_rel(target, repo_root) if target is not None else None
            lines.append(f"- [{name}]({rel})" if rel else f"- {name}")
        lines.append("")
        lines.append("</details>")
        lines.append("")
    return "\n".join(lines)


def write_index(out_root: Path, categories: CategoriesMap, paths: Dict[str, Path]) -> None:
    """Write INDEX.md at ``out_root`` listing every block.

    Block entries get a clickable link only when the file lives within
    ``out_root``; cross-tree references render as plain text so the index
    stays portable.
    """
    lines = ["# Index", ""]
    for cat, (desc, blocks) in categories.items():
        lines.append(f"## {cat}")
        lines.append("")
        lines.append(desc)
        lines.append("")
        for name in blocks:
            target = paths.get(name)
            rel = _safe_rel(target, out_root) if target is not None else None
            lines.append(f"* [{name}]({rel})" if rel else f"* {name}")
        lines.append("")
    (out_root / "INDEX.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Mermaid architecture diagrams. Without --specs the 122 "
            "built-in blocks are regenerated; with --specs the named files are "
            "loaded, merged into the registry, and rendered to --out."
        ),
    )
    parser.add_argument(
        "--depth", type=int, default=1,
        help="Max recursion depth for _ref nodes (0 = render refs as leaves).",
    )
    parser.add_argument(
        "--specs", type=Path, action="append", default=[],
        metavar="PATH",
        help="Path to a user spec file (Python). May be passed multiple times.",
    )
    parser.add_argument(
        "--out", type=Path, default=None, metavar="DIR",
        help="Output directory (default: ./diagrams next to this script).",
    )
    parser.add_argument(
        "--no-builtins", action="store_true",
        help="Don't regenerate built-in block .md files; keep them as registered "
             "references only.",
    )
    parser.add_argument(
        "--rank-spacing", type=int, default=DEFAULT_RANK_SPACING, metavar="PX",
        help=f"Vertical gap (px) between Mermaid ranks. Default: {DEFAULT_RANK_SPACING}.",
    )
    parser.add_argument(
        "--node-spacing", type=int, default=DEFAULT_NODE_SPACING, metavar="PX",
        help=f"Horizontal gap (px) between nodes in the same rank. "
             f"Default: {DEFAULT_NODE_SPACING}.",
    )
    args = parser.parse_args()

    builtin_root = Path(__file__).parent.resolve()
    builtin_diagrams = builtin_root / DEFAULT_OUT_SUBDIR
    out_root = (args.out.resolve() if args.out else builtin_diagrams)
    out_root.mkdir(parents=True, exist_ok=True)

    user_cats: CategoriesMap = {}
    for path in args.specs:
        user_cats.update(load_user_specs(path.resolve()))

    # User maps override built-ins; built-ins are checked for internal duplicates.
    registry = build_registry(CATEGORIES, user_cats) if user_cats else build_registry(CATEGORIES)
    all_cats: CategoriesMap = {**CATEGORIES, **user_cats}

    # Decide what gets rendered to disk.
    to_generate: CategoriesMap = (
        dict(user_cats) if args.no_builtins else dict(all_cats)
    )

    # Map every known block name to the .md file users will navigate to.
    # Generated files live under out_root; if --no-builtins, we point at the
    # original built-in tree so cross-references still resolve to a real file.
    name_to_path: Dict[str, Path] = {}
    for cat, (_, blocks) in to_generate.items():
        for name in blocks:
            name_to_path[name] = (out_root / cat / f"{name}.md").resolve()
    if args.no_builtins:
        for cat, (_, blocks) in CATEGORIES.items():
            for name in blocks:
                name_to_path.setdefault(
                    name, (builtin_diagrams / cat / f"{name}.md").resolve(),
                )

    total = 0
    for cat, (_, blocks) in to_generate.items():
        out_dir = out_root / cat
        out_dir.mkdir(parents=True, exist_ok=True)
        # only wipe .md inside categories we are actually regenerating
        for old in out_dir.glob("*.md"):
            old.unlink()
        for name, spec in blocks.items():
            desc, shapes, rows, skips, edges = _spec_unpack(spec)
            body = render_diagram(
                rows, skips, edges,
                registry=registry,
                max_depth=args.depth,
                block_name=name,
                rank_spacing=args.rank_spacing,
                node_spacing=args.node_spacing,
            )
            md = render_markdown(name, desc, shapes, body)
            name_to_path[name].write_text(md)
            total += 1

    # README only belongs to the built-in repo; don't clobber arbitrary --out
    # trees. We regenerate it whenever we're rendering the built-in library
    # to its default home (no --specs and no custom --out).
    rebuild_readme = (
        not args.specs and out_root == builtin_diagrams
    )
    if rebuild_readme:
        (builtin_root / "README.md").write_text(
            README_BODY + "\n" + _readme_index(CATEGORIES, builtin_root, name_to_path)
        )

    write_index(out_root, all_cats, name_to_path)

    user_count = sum(len(b) for _, b in user_cats.values())
    print(
        f"wrote {total} diagrams across {len(to_generate)} categories "
        f"({user_count} user, {total - user_count} built-in) → {out_root}"
    )


if __name__ == "__main__":
    main()

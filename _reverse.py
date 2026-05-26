"""Reverse a rendered Mermaid diagram back into a DSL ``spec.py`` file.

The output is consumable by ``python _generate.py --specs <file>`` so you
can round-trip:

    md  ->  _reverse.py  ->  spec.py  ->  _generate.py  ->  md  (functionally identical)

Scope: this parser only handles the shape of Mermaid that ``_generate.py``
itself emits — node declarations like ``nN_N["label"]:::kind``, edges of
the four supported styles (solid / labelled / dotted-skip / thick-mismatch),
``subgraph nID["RefName"] ... end`` blocks (collapsed back to a single
``_ref(...)`` node), plus the optional ``%%{init}%%`` directive and trailing
``classDef`` / ``linkStyle`` lines (ignored — they're recomputed at render).

Per-port shape metadata (``id``/``in``/``out`` kwargs) is **lossy through
Mermaid**: it lives only in the source spec, not in the rendered diagram.
After reversing, every node gets a fresh ``id=rNcN`` so the explicit
``edges=[...]`` list keeps working; ``in_``/``out`` are left off.

Usage::

    python _reverse.py diagrams/core/Linear.md            # writes Linear_spec.py next to it
    python _reverse.py diagrams/core/*.md -o core_spec.py  # multi-block bundle
    python _reverse.py FOO.md -o -                         # print to stdout
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Markdown front-matter extractors
# ---------------------------------------------------------------------------

_MERMAID_FENCE = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)
_TITLE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_DESC = re.compile(r"^>\s+(.+?)\s*$", re.MULTILINE)
_SHAPES = re.compile(r"^\*\*Shapes:\*\*\s+`(.+?)`\s*$", re.MULTILINE)


def _frontmatter(text: str) -> Tuple[str, str, str]:
    title = _TITLE.search(text)
    desc = _DESC.search(text)
    shapes = _SHAPES.search(text)
    return (
        (title.group(1) if title else "ReversedBlock"),
        (desc.group(1) if desc else "Reversed from a Mermaid diagram."),
        (shapes.group(1) if shapes else "(*) → (*)"),
    )


def _mermaid_body(text: str) -> str:
    m = _MERMAID_FENCE.search(text)
    if not m:
        raise ValueError("no ```mermaid ... ``` block found")
    return m.group(1)


# ---------------------------------------------------------------------------
# Mermaid body parser (scoped to _generate.py's output)
# ---------------------------------------------------------------------------

_NODE = re.compile(r'^(\S+?)\["(.+)"\]:::(\w+)\s*$')
_SUBGRAPH_START = re.compile(r'^subgraph\s+(\S+?)\["(.+)"\]\s*$')
_SUBGRAPH_END = re.compile(r"^end\s*$")
_EDGE = re.compile(
    r'^(\S+)\s+(-->|-\.->|==>)'
    r'(?:\|"([^"]*)"\|)?'
    r'\s+(\S+)\s*$'
)
_INIT = re.compile(r"^%%\{.*\}%%\s*$")
_FLOWCHART_HDR = re.compile(r"^flowchart\s+\w+\s*$")
_NOISE_PREFIXES = ("classDef ", "linkStyle ", "%%")


class Parsed:
    """Intermediate graph extracted from a Mermaid body."""

    def __init__(self) -> None:
        self.nodes: "OrderedDict[str, Tuple[str, str]]" = OrderedDict()  # id -> (kind, label)
        # edges keyed for dedup: (src, dst, style) -> label (last wins, but
        # all duplicates from subgraph-collapse share the same effective edge)
        self.edges: "OrderedDict[Tuple[str, str, str], Optional[str]]" = OrderedDict()


def _parse_body(body: str) -> Parsed:
    p = Parsed()
    sub_stack: List[str] = []
    # Map every node id seen inside a subgraph back to its OUTERMOST containing
    # subgraph id. We collapse the whole expansion to a single _ref node, so
    # any edge touching an inner port becomes an edge to/from the ref's id.
    remap: Dict[str, str] = {}

    for raw in body.splitlines():
        line = raw.strip()
        if not line or _INIT.match(line) or _FLOWCHART_HDR.match(line):
            continue
        if any(line.startswith(pre) for pre in _NOISE_PREFIXES):
            continue

        m = _SUBGRAPH_START.match(line)
        if m:
            sid, ref_name = m.group(1), m.group(2)
            if sub_stack:
                # Nested subgraph — still collapse to the OUTERMOST ref.
                remap[sid] = remap.get(sub_stack[0], sub_stack[0])
            else:
                # Top-level subgraph -> register as a ref node.
                p.nodes[sid] = ("ref", ref_name)
            sub_stack.append(sid)
            continue

        if _SUBGRAPH_END.match(line):
            if sub_stack:
                sub_stack.pop()
            continue

        m = _NODE.match(line)
        if m:
            nid, label, kind = m.group(1), m.group(2), m.group(3)
            if sub_stack:
                remap[nid] = remap.get(sub_stack[0], sub_stack[0])
            else:
                p.nodes[nid] = (kind, label)
            continue

        m = _EDGE.match(line)
        if m:
            src, op, label, dst = m.group(1), m.group(2), m.group(3), m.group(4)
            src = remap.get(src, src)
            dst = remap.get(dst, dst)
            if src == dst:
                continue  # self-loop after collapse (internal subgraph edge)
            style = {"-->": "solid", "-.->": "skip", "==>": "mismatch"}[op]
            if style == "mismatch":
                # Mismatch is a render-time decoration of a solid edge — drop
                # the auto-generated "(...) ≠ (...)" label so the regenerator
                # can recompute it from per-port shapes (which Mermaid lost).
                style = "solid"
                label = None
            p.edges[(src, dst, style)] = label
            continue
        # silently skip anything else (the renderer doesn't emit much beyond
        # the cases handled above)

    return p


# ---------------------------------------------------------------------------
# Layering: longest-path depth -> rows
# ---------------------------------------------------------------------------


def _layer(parsed: Parsed) -> List[List[str]]:
    """Group nodes into rows by longest-path depth from any source."""
    ids = list(parsed.nodes.keys())
    fwd: Dict[str, List[str]] = defaultdict(list)
    indeg: Dict[str, int] = {i: 0 for i in ids}
    for (s, t, _style), _lbl in parsed.edges.items():
        if s in indeg and t in indeg:
            fwd[s].append(t)
            indeg[t] += 1

    level: Dict[str, int] = {}
    queue: List[str] = []
    # Preserve declaration order for stable layout.
    for i in ids:
        if indeg[i] == 0:
            level[i] = 0
            queue.append(i)

    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        lu = level[u]
        for v in fwd[u]:
            level[v] = max(level.get(v, 0), lu + 1)
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)

    max_lvl = max(level.values(), default=0)
    rows: List[List[str]] = [[] for _ in range(max_lvl + 1)]
    for i in ids:
        if i in level:
            rows[level[i]].append(i)

    # Any cycle stragglers go on a tail row so we never silently drop nodes.
    stragglers = [i for i in ids if i not in level]
    if stragglers:
        rows.append(stragglers)
    return rows


# ---------------------------------------------------------------------------
# DSL emitter
# ---------------------------------------------------------------------------

_KIND_FN: Dict[str, str] = {
    "io": "_io", "op": "_op", "norm": "_norm", "act": "_act", "attn": "_attn",
    "merge": "_merge", "emb": "_emb", "loss": "_loss", "ref": "_ref",
    # ctrl has no helper today; fall back to _op so it still parses.
    "ctrl": "_op",
}


def _py_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _emit_node(kind: str, label: str, uid: str) -> str:
    fn = _KIND_FN[kind]
    return f'{fn}({_py_str(label)}, id={_py_str(uid)})'


def _block_to_spec(parsed: Parsed, desc: str, shapes: str) -> str:
    rows = _layer(parsed)

    uid_of: Dict[str, str] = {}
    pos_of: Dict[str, Tuple[int, int]] = {}
    for r, row in enumerate(rows):
        for c, nid in enumerate(row):
            uid_of[nid] = f"r{r}c{c}"
            pos_of[nid] = (r, c)

    rows_lines = []
    for row in rows:
        items = ", ".join(
            _emit_node(parsed.nodes[nid][0], parsed.nodes[nid][1], uid_of[nid])
            for nid in row
        )
        rows_lines.append(f"            [{items}]")

    skips: List[Tuple[int, int, int, int]] = []
    explicit_edges: List[Tuple[str, str, Optional[str]]] = []
    for (s, t, style), label in parsed.edges.items():
        if s not in uid_of or t not in uid_of:
            continue
        if style == "skip":
            r1, c1 = pos_of[s]
            r2, c2 = pos_of[t]
            skips.append((r1, c1, r2, c2))
        else:
            explicit_edges.append((uid_of[s], uid_of[t], label))

    skips_lit = "[" + ", ".join(f"({a}, {b}, {c}, {d})" for a, b, c, d in skips) + "]"
    edges_lines = []
    for s, t, label in explicit_edges:
        if label:
            edges_lines.append(f'            _edge({_py_str(s)}, {_py_str(t)}, {_py_str(label)})')
        else:
            edges_lines.append(f'            _edge({_py_str(s)}, {_py_str(t)})')
    edges_body = ",\n".join(edges_lines)

    return (
        "        (\n"
        f"            {_py_str(desc)},\n"
        f"            {_py_str(shapes)},\n"
        "            [\n"
        + ",\n".join(rows_lines)
        + "\n            ],\n"
        f"            {skips_lit},  # positional skip / residual edges\n"
        "            [\n"
        + edges_body
        + ("\n" if edges_body else "")
        + "            ],\n"
        "        )"
    )


def reverse_file(path: Path) -> Tuple[str, str]:
    """Read a Mermaid .md and return ``(BlockName, BlockSpecLiteral)``."""
    text = path.read_text()
    name, desc, shapes = _frontmatter(text)
    body = _mermaid_body(text)
    parsed = _parse_body(body)
    spec = _block_to_spec(parsed, desc, shapes)
    return name, spec


def emit_module(
    entries: List[Tuple[str, str]],
    *,
    category: str,
    category_desc: str,
) -> str:
    items = ",\n".join(
        f"    {_py_str(name)}: " + spec.lstrip()
        for name, spec in entries
    )
    return (
        '"""Auto-generated by _reverse.py from rendered Mermaid diagrams.\n\n'
        "Drop straight into ``python _generate.py --specs <this_file.py>`` to\n"
        "re-render — the output should be functionally identical to the\n"
        "original .md inputs (per-port shape metadata is lossy through\n"
        "Mermaid and is not recovered).\n"
        '"""\n\n'
        "from _generate import _io, _op, _norm, _act, _attn, _merge, _emb, _loss, _ref, _edge  # noqa: F401\n\n"
        f"CATEGORY = {_py_str(category)}\n"
        f"CATEGORY_DESC = {_py_str(category_desc)}\n\n"
        "BLOCKS = {\n"
        + items
        + ",\n}\n"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Reverse rendered Mermaid .md diagrams back into a DSL spec.py "
            "consumable by `python _generate.py --specs ...`."
        ),
    )
    p.add_argument("inputs", type=Path, nargs="+", help="One or more .md files.")
    p.add_argument(
        "-o", "--out", default=None,
        help="Output spec.py path, or '-' for stdout. "
             "Default: <first_input_stem>_spec.py next to the first input.",
    )
    p.add_argument(
        "--category", default=None,
        help="CATEGORY slug for the emitted module. "
             "Default: parent directory of the first input, or 'reversed'.",
    )
    p.add_argument(
        "--category-desc", default="Reversed from rendered Mermaid diagrams.",
        help="CATEGORY_DESC string for the emitted module.",
    )
    args = p.parse_args()

    entries: List[Tuple[str, str]] = []
    seen: set = set()
    for path in args.inputs:
        name, spec = reverse_file(path)
        if name in seen:
            print(f"warning: duplicate block name {name!r} (skipping later occurrence)",
                  file=sys.stderr)
            continue
        seen.add(name)
        entries.append((name, spec))

    first = args.inputs[0]
    category = args.category or (first.parent.name if first.parent.name not in ("", ".") else "reversed")
    module_src = emit_module(entries, category=category, category_desc=args.category_desc)

    if args.out == "-":
        sys.stdout.write(module_src)
    else:
        out_path = Path(args.out) if args.out else first.with_name(f"{first.stem}_spec.py")
        out_path.write_text(module_src)
        print(f"wrote {len(entries)} block(s) -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

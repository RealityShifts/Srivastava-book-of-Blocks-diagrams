"""DSL helpers and types shared by the renderer and every block file.

Block authors should ``from dsl import _io, _op, _ref, ...`` and return
``Spec`` tuples shaped ``(desc, shapes, rows[, skips[, edges[, notes]]])``.

Every node helper accepts the same set of optional keyword arguments:

* ``id``    -- unique handle within the block so the node can be referenced
              from the ``edges`` list.
* ``shape`` -- shorthand: sets both the node's expected input shape and its
              produced output shape to the same value.
* ``in_``   -- explicit input shape (overrides ``shape`` for the inbound side).
* ``out``   -- explicit output shape (overrides ``shape`` for the outbound side).

Nodes that don't pass any of these still produce the original ``(kind, label)``
2-tuple, so every existing spec keeps working unchanged.

The optional 5th spec element ``edges`` is a list of ``(src_id, dst_id)`` or
``(src_id, dst_id, label)`` tuples (use :func:`_edge` for readability). When
``edges`` is provided the renderer skips its row-adjacency auto-wiring and
draws only the explicit edges (plus skip edges). When two shape-annotated
nodes are connected and their shapes disagree, the edge is rendered as a red
error arrow and a warning is printed.

The optional 6th spec element ``notes`` is a dict of long-form context that
the renderer turns into extra markdown sections under each diagram. Use
:func:`_notes` for an ergonomic constructor. Supported keys (all optional):

* ``used_in``   -- famous architectures / systems where the block appears.
* ``tasks``     -- problem domains / what you'd reach for this block to solve.
* ``pitfalls``  -- common gotchas, instabilities, or things to verify.
* ``see_also``  -- papers / refs / related blocks (markdown links allowed).

Any other key passed to :func:`_notes` is preserved verbatim and rendered as
its own section. Pass ``None`` or an empty dict to skip the block entirely.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

KINDS = {"io", "op", "norm", "act", "attn", "merge", "emb", "loss", "ctrl", "ref"}

Meta = Dict[str, Any]
Node = Union[Tuple[str, str], Tuple[str, str, Meta]]
Row = List[Node]
Skip = Tuple[int, int, int, int]
Edge = Union[Tuple[str, str], Tuple[str, str, str]]
Notes = Dict[str, List[str]]
Spec = Tuple


def _meta(
    *,
    id: Optional[str],
    shape: Optional[str],
    in_: Optional[str],
    out: Optional[str],
) -> Meta:
    m: Meta = {}
    if id is not None:
        m["id"] = id
    in_val = in_ if in_ is not None else shape
    out_val = out if out is not None else shape
    if in_val is not None:
        m["in"] = in_val
    if out_val is not None:
        m["out"] = out_val
    return m


def _node(
    kind: str,
    label: str,
    *,
    id: Optional[str] = None,
    shape: Optional[str] = None,
    in_: Optional[str] = None,
    out: Optional[str] = None,
) -> Node:
    m = _meta(id=id, shape=shape, in_=in_, out=out)
    return (kind, label, m) if m else (kind, label)


def _io(label, *, id=None, shape=None, in_=None, out=None):
    return _node("io", label, id=id, shape=shape, in_=in_, out=out)


def _op(label, *, id=None, shape=None, in_=None, out=None):
    return _node("op", label, id=id, shape=shape, in_=in_, out=out)


def _norm(label, *, id=None, shape=None, in_=None, out=None):
    return _node("norm", label, id=id, shape=shape, in_=in_, out=out)


def _act(label, *, id=None, shape=None, in_=None, out=None):
    return _node("act", label, id=id, shape=shape, in_=in_, out=out)


def _attn(label, *, id=None, shape=None, in_=None, out=None):
    return _node("attn", label, id=id, shape=shape, in_=in_, out=out)


def _merge(label, *, id=None, shape=None, in_=None, out=None):
    return _node("merge", label, id=id, shape=shape, in_=in_, out=out)


def _emb(label, *, id=None, shape=None, in_=None, out=None):
    return _node("emb", label, id=id, shape=shape, in_=in_, out=out)


def _loss(label, *, id=None, shape=None, in_=None, out=None):
    return _node("loss", label, id=id, shape=shape, in_=in_, out=out)


def _ref(name, *, id=None, shape=None, in_=None, out=None):
    return _node("ref", name, id=id, shape=shape, in_=in_, out=out)


def _edge(src: str, dst: str, label: Optional[str] = None) -> Edge:
    """Edge tuple for the spec's optional 5th element. ``label`` is rendered
    on the arrow when provided (shape-mismatch edges always carry their own
    label and override anything supplied here)."""
    return (src, dst, label) if label is not None else (src, dst)


def _notes(
    *,
    used_in: Optional[List[str]] = None,
    tasks: Optional[List[str]] = None,
    pitfalls: Optional[List[str]] = None,
    see_also: Optional[List[str]] = None,
    **extras: List[str],
) -> Notes:
    """Build the spec's optional 6th-element ``notes`` dict.

    Each kwarg is a list of strings; entries may contain markdown (including
    links like ``[Title](url)``). Only non-empty keys end up in the output,
    so passing nothing yields an empty dict.

    The four named slots render in a fixed order under the diagram; ``extras``
    keep their declaration order and render after them with title-cased
    headings (e.g. ``related_blocks=[...]`` → "Related blocks").
    """
    out: Notes = {}
    if used_in:
        out["used_in"] = list(used_in)
    if tasks:
        out["tasks"] = list(tasks)
    if pitfalls:
        out["pitfalls"] = list(pitfalls)
    if see_also:
        out["see_also"] = list(see_also)
    for k, v in extras.items():
        if v:
            out[k] = list(v)
    return out


__all__ = [
    "KINDS", "Meta", "Node", "Row", "Skip", "Edge", "Notes", "Spec",
    "_io", "_op", "_norm", "_act", "_attn",
    "_merge", "_emb", "_loss", "_ref", "_edge", "_notes",
]

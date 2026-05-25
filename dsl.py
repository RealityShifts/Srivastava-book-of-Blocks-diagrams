"""DSL helpers and types shared by the renderer and every block file.

Block authors should ``from dsl import _io, _op, _ref, ...`` and
return ``Spec`` tuples shaped ``(desc, shapes, rows[, skips])``.
The renderer in ``_generate.py`` consumes the same DSL.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

KINDS = {"io", "op", "norm", "act", "attn", "merge", "emb", "loss", "ctrl", "ref"}

Node = Tuple[str, str]
Row = List[Node]
Skip = Tuple[int, int, int, int]
Spec = Tuple


def _io(label: str) -> Node: return ("io", label)
def _op(label: str) -> Node: return ("op", label)
def _norm(label: str) -> Node: return ("norm", label)
def _act(label: str) -> Node: return ("act", label)
def _attn(label: str) -> Node: return ("attn", label)
def _merge(label: str) -> Node: return ("merge", label)
def _emb(label: str) -> Node: return ("emb", label)
def _loss(label: str) -> Node: return ("loss", label)
def _ref(name: str) -> Node: return ("ref", name)


__all__ = [
    "KINDS", "Node", "Row", "Skip", "Spec",
    "_io", "_op", "_norm", "_act", "_attn",
    "_merge", "_emb", "_loss", "_ref",
]

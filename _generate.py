"""Generate Mermaid architecture diagrams for every block.

Run ``python _generate.py`` to (re)build everything. Use
``python _generate.py --depth N`` to recursively expand ``_ref`` nodes up
to ``N`` levels deep (default: 1).

The 122 blocks shipped here act as a *built-in library*: you can author
your own architecture spec file and ``_ref()`` any built-in block by
name. Run::

    python _generate.py --specs my_arch.py --out ./diagrams

to generate diagrams for your blocks; pass ``--specs`` multiple times to
merge several files. Add ``--no-builtins`` to skip regenerating the 122
library files (they remain available as resolvable references).

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
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

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

KINDS = {"io", "op", "norm", "act", "attn", "merge", "emb", "loss", "ctrl", "ref"}

Node = Tuple[str, str]
Row = List[Node]
Skip = Tuple[int, int, int, int]
Spec = Tuple
Ports = Tuple[str, ...]
SlotIds = List[List[Tuple[Ports, Ports]]]  # per row, per slot: (in_ports, out_ports)


# ---------------------------------------------------------------------------
# Mermaid renderer (with recursive `_ref` expansion via subgraph)
# ---------------------------------------------------------------------------


def _label(text: str) -> str:
    # Mermaid quoted labels accept most characters; escape stray quotes
    # and avoid confusing the parser with backticks.
    return text.replace('"', "&quot;").replace("`", "&#96;")


def _spec_rows_skips(spec: Spec) -> Tuple[List[Row], Sequence[Skip]]:
    if len(spec) == 3:
        return spec[2], ()
    return spec[2], spec[3]


def _render_rows(
    rows: List[Row],
    skips: Sequence[Skip],
    registry: Optional[Dict[str, Spec]],
    max_depth: int,
    depth: int,
    prefix: str,
    ancestors: FrozenSet[str],
) -> Tuple[List[str], SlotIds]:
    lines: List[str] = []
    slot_ids: SlotIds = []

    for r, row in enumerate(rows):
        row_slots: List[Tuple[Ports, Ports]] = []
        for c, (kind, label) in enumerate(row):
            assert kind in KINDS, f"unknown kind {kind!r} in row {r} col {c}"
            nid = f"{prefix}{r}_{c}"
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
                sub_rows, sub_skips = _spec_rows_skips(sub)
                sub_lines, sub_slots = _render_rows(
                    sub_rows, sub_skips, registry, max_depth,
                    depth + 1, f"{nid}_", ancestors | {label},
                )
                lines.append(f'subgraph {nid}["{_label(label)}"]')
                lines.extend("    " + ln for ln in sub_lines)
                lines.append("end")
                # Aggregate every port of the first row (entry) and last row (exit)
                # so parents fan in / out across the full IO boundary of the sub-block.
                in_ports = tuple(p for slot in sub_slots[0] for p in slot[0])
                out_ports = tuple(p for slot in sub_slots[-1] for p in slot[1])
                row_slots.append((in_ports, out_ports))
            else:
                lines.append(f'{nid}["{_label(label)}"]:::{kind}')
                row_slots.append(((nid,), (nid,)))
        slot_ids.append(row_slots)

    # auto edges between consecutive rows
    for r in range(len(rows) - 1):
        a, b = slot_ids[r], slot_ids[r + 1]
        if len(a) == len(b) and len(a) > 1:
            for c in range(len(a)):
                for src in a[c][1]:
                    for dst in b[c][0]:
                        lines.append(f"{src} --> {dst}")
        else:
            for ci in range(len(a)):
                for cj in range(len(b)):
                    for src in a[ci][1]:
                        for dst in b[cj][0]:
                            lines.append(f"{src} --> {dst}")

    # skip / residual edges (dashed)
    for r1, c1, r2, c2 in skips:
        for src in slot_ids[r1][c1][1]:
            for dst in slot_ids[r2][c2][0]:
                lines.append(f"{src} -. skip .-> {dst}")

    return lines, slot_ids


def render_diagram(
    rows: List[Row],
    skips: Sequence[Skip] = (),
    *,
    registry: Optional[Dict[str, Spec]] = None,
    max_depth: int = 0,
) -> str:
    body, _ = _render_rows(
        rows, skips, registry, max_depth,
        depth=0, prefix="n", ancestors=frozenset(),
    )
    return "\n".join(["flowchart TD", *("    " + ln for ln in body), STYLE])


def render_markdown(name: str, desc: str, shapes: str, body: str) -> str:
    return (
        f"# {name}\n\n"
        f"> {desc}\n\n"
        f"**Shapes:** `{shapes}`\n\n"
        f"```mermaid\n{body}\n```\n"
    )


# ---------------------------------------------------------------------------
# Block specs (grouped by category subfolder)
# ---------------------------------------------------------------------------

# Each entry: name -> (desc, shapes, rows) or (desc, shapes, rows, skips)


def _io(label: str) -> Node: return ("io", label)
def _op(label: str) -> Node: return ("op", label)
def _norm(label: str) -> Node: return ("norm", label)
def _act(label: str) -> Node: return ("act", label)
def _attn(label: str) -> Node: return ("attn", label)
def _merge(label: str) -> Node: return ("merge", label)
def _emb(label: str) -> Node: return ("emb", label)
def _loss(label: str) -> Node: return ("loss", label)
def _ref(name: str) -> Node: return ("ref", name)


CORE: Dict[str, Spec] = {
    "Linear": (
        "Affine projection y = x · Wᵀ + b.",
        "(B, in) → (B, out)",
        [
            [_io("x  (B, in)")],
            [_op("matmul  x · Wᵀ")],
            [_op("+ bias")],
            [_io("y  (B, out)")],
        ],
    ),
    "ConvBlock": (
        "Conv → Norm → Activation, the canonical CNN unit.",
        "(B, C, H, W) → (B, C', H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("Conv 2D")],
            [_norm("BatchNorm / GroupNorm / LayerNorm")],
            [_act("ReLU / GELU / SiLU / Mish")],
            [_io("y  (B, C', H, W)")],
        ],
    ),
    "DepthwiseSeparableConv2d": (
        "Depthwise spatial conv followed by 1×1 pointwise conv (MobileNet).",
        "(B, C, H, W) → (B, C', H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("Depthwise Conv K×K  (groups=C)")],
            [_op("Pointwise Conv 1×1  (C → C')")],
            [_io("y  (B, C', H, W)")],
        ],
    ),
    "DilatedConv2d": (
        "Standard conv with dilation > 1 (atrous), enlarges receptive field for free.",
        "(B, C, H, W) → (B, C', H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("Conv K×K  (dilation = d)")],
            [_io("y  (B, C', H, W)")],
        ],
    ),
    "GroupConv2d": (
        "Conv2d with channel groups (ResNeXt cardinality).",
        "(B, C, H, W) → (B, C', H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("Conv K×K  (groups = g)")],
            [_io("y  (B, C', H, W)")],
        ],
    ),
    "Conv1d": (
        "1-D convolution wrapper.",
        "(B, C, T) → (B, C', T)",
        [
            [_io("x  (B, C, T)")],
            [_op("Conv 1D K  (stride, padding)")],
            [_io("y  (B, C', T)")],
        ],
    ),
    "Conv3d": (
        "3-D convolution wrapper.",
        "(B, C, T, H, W) → (B, C', T, H, W)",
        [
            [_io("x  (B, C, T, H, W)")],
            [_op("Conv 3D K  (stride, padding)")],
            [_io("y  (B, C', T, H, W)")],
        ],
    ),
    "Mish": (
        "Self-gated activation: x · tanh(softplus(x)).",
        "(*) → (*)",
        [
            [_io("x  (*)")],
            [_op("softplus(x)")],
            [_op("tanh(·)")],
            [_merge("× x")],
            [_io("y  (*)")],
        ],
        [(0, 0, 3, 0)],
    ),
    "RMSNorm": (
        "Root-mean-square normalisation, scale-only (no mean subtraction).",
        "(*, D) → (*, D)",
        [
            [_io("x  (*, D)")],
            [_op("RMS = √mean(x²)")],
            [_op("x / (RMS + ε)")],
            [_op("× learned g")],
            [_io("y  (*, D)")],
        ],
    ),
    "AdaIN": (
        "Adaptive Instance Normalisation: replace x's per-channel stats with style stats.",
        "x:(B, C, H, W), s:(B, C) → y:(B, C, H, W)",
        [
            [_io("x  content  (B, C, H, W)"), _io("s  style  (B, C)")],
            [_op("μ_x, σ_x  per channel"), _op("γ, β = MLP(s)")],
            [_op("(x − μ_x) / σ_x  · γ + β")],
            [_io("y  (B, C, H, W)")],
        ],
    ),
    "SPADE": (
        "Spatially-adaptive denormalisation: γ, β come from a segmentation map.",
        "x:(B, C, H, W), seg:(B, K, H, W) → y:(B, C, H, W)",
        [
            [_io("x  (B, C, H, W)"), _io("seg map  (B, K, H, W)")],
            [_norm("BatchNorm(x)"), _op("Conv → γ(spatial), β(spatial)")],
            [_op("γ · x_norm + β")],
            [_io("y  (B, C, H, W)")],
        ],
    ),
    "ResidualBlock": (
        "Two conv-norm-act stack with identity skip and post-add activation.",
        "(B, C, H, W) → (B, C, H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("Conv 3×3")],
            [_norm("Norm")],
            [_act("ReLU")],
            [_op("Conv 3×3")],
            [_norm("Norm")],
            [_merge("+")],
            [_act("ReLU")],
            [_io("y  (B, C, H, W)")],
        ],
        [(0, 0, 6, 0)],
    ),
    "SkipConnection": (
        "Generic identity skip around any sub-module f.",
        "(*) → (*)",
        [
            [_io("x  (*)")],
            [_op("f(x)")],
            [_merge("+")],
            [_io("y  (*)")],
        ],
        [(0, 0, 2, 0)],
    ),
}


ATTENTION: Dict[str, Spec] = {
    "MultiHeadAttention": (
        "Standard scaled dot-product attention with H heads and an output projection.",
        "Q,K,V:(B, T, D) → (B, T, D)",
        [
            [_io("Q  (B, T, D)"), _io("K  (B, T, D)"), _io("V  (B, T, D)")],
            [_op("linear Q"), _op("linear K"), _op("linear V")],
            [_op("split into H heads")],
            [_op("Q · Kᵀ / √d_h")],
            [_act("softmax (+ optional mask)")],
            [_op("· V")],
            [_op("concat heads")],
            [_op("output linear")],
            [_io("y  (B, T, D)")],
        ],
    ),
    "SelfAttention": (
        "MultiHeadAttention applied with Q = K = V = x.",
        "(B, T, D) → (B, T, D)",
        [
            [_io("x  (B, T, D)")],
            [_op("derive Q, K, V from x")],
            [_ref("MultiHeadAttention")],
            [_io("y  (B, T, D)")],
        ],
    ),
    "CausalSelfAttention": (
        "SelfAttention with an upper-triangular mask preventing future leakage.",
        "(B, T, D) → (B, T, D)",
        [
            [_io("x  (B, T, D)")],
            [_op("Q, K, V from x")],
            [_attn("MultiHeadAttention + causal mask")],
            [_io("y  (B, T, D)")],
        ],
    ),
    "CrossAttention": (
        "Q comes from x, K and V come from a separate context tensor.",
        "x:(B, T_q, D), ctx:(B, T_k, D) → (B, T_q, D)",
        [
            [_io("x  (B, T_q, D)"), _io("context  (B, T_k, D)")],
            [_op("Q from x"), _op("K, V from context")],
            [_ref("MultiHeadAttention")],
            [_io("y  (B, T_q, D)")],
        ],
    ),
    "WindowAttention": (
        "Self-attention restricted to non-overlapping w×w windows (Swin).",
        "(B, H, W, C) → (B, H, W, C)",
        [
            [_io("x  (B, H, W, C)")],
            [_op("window partition  (w × w)")],
            [_attn("MHA + relative pos bias")],
            [_op("window reverse")],
            [_io("y  (B, H, W, C)")],
        ],
    ),
    "LinearAttention": (
        "Kernelised attention with O(N) cost via Q · (Kᵀ V).",
        "(B, T, D) → (B, T, D)",
        [
            [_io("Q, K, V  (B, T, D)")],
            [_op("φ(Q),  φ(K)")],
            [_op("Kᵀ · V")],
            [_op("Q · (Kᵀ V)")],
            [_op("normalise by Q · Σ K")],
            [_io("y  (B, T, D)")],
        ],
    ),
    "FlashAttention": (
        "Mathematically identical to MHA, but uses tiled IO-aware kernels.",
        "(B, T, D) → (B, T, D)",
        [
            [_io("Q, K, V  (B, T, D)")],
            [_attn("flash kernel  (tiled softmax, no materialised attn matrix)")],
            [_io("y  (B, T, D)")],
        ],
    ),
    "RotaryEmbedding": (
        "Inject absolute position into Q, K via per-pair rotations.",
        "Q,K:(B, H, T, d) → rotated Q,K",
        [
            [_io("Q, K  (B, H, T, d)")],
            [_op("freqs θ_pos = 10000^(−2i/d)")],
            [_op("rotate (even, odd) dim pairs")],
            [_io("Q', K'  (B, H, T, d)")],
        ],
    ),
    "RelativePositionBias": (
        "Learnable bias added to attention logits as a function of (i − j).",
        "logits:(*, T, T) → biased logits",
        [
            [_io("position pairs (i, j)  (T, T)")],
            [_op("bias table lookup B[i − j]")],
            [_merge("+ to attn logits")],
            [_io("biased logits  (*, T, T)")],
        ],
    ),
    "AttentionPooling": (
        "A learnable query attends over a sequence to produce a single pooled vector.",
        "(B, T, D) → (B, D)",
        [
            [_io("learnable query  (1, D)"), _io("x  (B, T, D)")],
            [_attn("MHA(query, x, x)")],
            [_io("pooled  (B, D)")],
        ],
    ),
}


TRANSFORMER: Dict[str, Spec] = {
    "FeedForward": (
        "Two-layer MLP with an inner hidden dim r × D.",
        "(B, T, D) → (B, T, D)",
        [
            [_io("x  (B, T, D)")],
            [_op("linear up  (D → r·D)")],
            [_act("GELU")],
            [_op("linear down  (r·D → D)")],
            [_io("y  (B, T, D)")],
        ],
    ),
    "SwiGLU": (
        "Gated FFN where the gate path uses SiLU (LLaMA-style).",
        "(B, T, D) → (B, T, D)",
        [
            [_io("x  (B, T, D)")],
            [_op("linear → (a, b)")],
            [_op("silu(a) ⊙ b")],
            [_op("linear out")],
            [_io("y  (B, T, D)")],
        ],
    ),
    "GEGLU": (
        "Same gated FFN as SwiGLU but with GELU on the gate.",
        "(B, T, D) → (B, T, D)",
        [
            [_io("x  (B, T, D)")],
            [_op("linear → (a, b)")],
            [_op("gelu(a) ⊙ b")],
            [_op("linear out")],
            [_io("y  (B, T, D)")],
        ],
    ),
    "TransformerEncoderBlock": (
        "Pre-norm transformer block: LN → MHA → +; LN → FFN → +.",
        "(B, T, D) → (B, T, D)",
        [
            [_io("x  (B, T, D)")],
            [_norm("LayerNorm")],
            [_ref("MultiHeadAttention")],
            [_merge("+")],
            [_norm("LayerNorm")],
            [_ref("FeedForward")],
            [_merge("+")],
            [_io("y  (B, T, D)")],
        ],
        [(0, 0, 3, 0), (3, 0, 6, 0)],
    ),
    "TransformerDecoderBlock": (
        "Pre-norm decoder: causal self-attn, cross-attn over context, FFN.",
        "x:(B, T, D), ctx:(B, M, D) → (B, T, D)",
        [
            [_io("x  (B, T, D)")],
            [_norm("LayerNorm")],
            [_ref("CausalSelfAttention")],
            [_merge("+")],
            [_norm("LayerNorm")],
            [_ref("CrossAttention")],
            [_merge("+")],
            [_norm("LayerNorm")],
            [_ref("FeedForward")],
            [_merge("+")],
            [_io("y  (B, T, D)")],
        ],
        [(0, 0, 3, 0), (3, 0, 6, 0), (6, 0, 9, 0)],
    ),
    "MixtureOfExperts": (
        "Top-k gated mixture: router scores experts, top-k run, weighted sum.",
        "(B, T, D) → (B, T, D)",
        [
            [_io("x  (B, T, D)")],
            [_op("router (linear → softmax → top-k)")],
            [_op("Expert 1"), _op("Expert 2"), _op("…"), _op("Expert E")],
            [_merge("weighted sum (gates · experts)")],
            [_io("y  (B, T, D)")],
        ],
    ),
    "SwitchMoE": (
        "MoE specialisation with k = 1: a single expert per token.",
        "(B, T, D) → (B, T, D)",
        [
            [_io("x  (B, T, D)")],
            [_op("router (top-1)")],
            [_op("Expert 1"), _op("…"), _op("Expert E")],
            [_op("selected expert (one per token)")],
            [_io("y  (B, T, D)")],
        ],
    ),
}


CNN_VISION: Dict[str, Spec] = {
    "InceptionBlock": (
        "Parallel multi-scale convs concatenated channel-wise (GoogLeNet).",
        "(B, C, H, W) → (B, C', H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("1×1 Conv"), _op("1×1 → 3×3"), _op("1×1 → 5×5"), _op("3×3 MaxPool → 1×1")],
            [_merge("concat (channel)")],
            [_io("y  (B, C', H, W)")],
        ],
    ),
    "DenseBlock": (
        "Each layer's input is the concatenation of all earlier layers' outputs (DenseNet).",
        "(B, C, H, W) → (B, C + k·L, H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("BN-ReLU-Conv  (layer 1)")],
            [_merge("concat with x")],
            [_op("BN-ReLU-Conv  (layer 2)")],
            [_merge("concat with all prev")],
            [_op("…  (L layers)")],
            [_io("y  (B, C + k·L, H, W)")],
        ],
    ),
    "SqueezeExcitation": (
        "Channel-wise gating via global pool → bottleneck MLP → sigmoid.",
        "(B, C, H, W) → (B, C, H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("Global AvgPool")],
            [_op("FC down  (C → C/r)")],
            [_act("ReLU")],
            [_op("FC up  (C/r → C)")],
            [_act("Sigmoid")],
            [_merge("· x  (channel scale)")],
            [_io("y  (B, C, H, W)")],
        ],
        [(0, 0, 6, 0)],
    ),
    "CBAM": (
        "Sequential channel attention then spatial attention.",
        "(B, C, H, W) → (B, C, H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("Channel Attention  (MLP on max+avg pool)")],
            [_merge("· x")],
            [_op("Spatial Attention  (Conv on max+avg over C)")],
            [_merge("· x")],
            [_io("y  (B, C, H, W)")],
        ],
        [(0, 0, 2, 0), (2, 0, 4, 0)],
    ),
    "SpatialPyramidPooling": (
        "Multi-scale fixed-output pooling, concatenated.",
        "(B, C, H, W) → (B, C·Σbins²,)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("AvgPool 1×1"), _op("AvgPool 2×2"), _op("AvgPool 4×4")],
            [_merge("flatten + concat")],
            [_io("y  (B, C·Σbins²,)")],
        ],
    ),
    "FeaturePyramidNetwork": (
        "Build multi-scale feature maps via top-down upsampling + lateral 1×1 connections.",
        "(C3, C4, C5) → (P3, P4, P5)",
        [
            [_io("C3, C4, C5  (B, Cᵢ, H/2ⁱ, W/2ⁱ)")],
            [_op("1×1 lateral on each level")],
            [_op("top-down: upsample + add")],
            [_op("3×3 smooth")],
            [_io("P3, P4, P5  (B, C, H/2ⁱ, W/2ⁱ)")],
        ],
    ),
    "ASPP": (
        "Atrous Spatial Pyramid Pooling: parallel atrous convs + image pool, concatenated.",
        "(B, C, H, W) → (B, C', H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("1×1 Conv"), _op("3×3 dil=6"), _op("3×3 dil=12"), _op("3×3 dil=18"), _op("Image pool")],
            [_merge("concat")],
            [_op("1×1 Conv")],
            [_io("y  (B, C', H, W)")],
        ],
    ),
    "PixelShuffleUpsample": (
        "Sub-pixel upsampling: rearrange r² channels into r×r spatial blocks.",
        "(B, C, H, W) → (B, C, r·H, r·W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("Conv → C·r² channels")],
            [_op("PixelShuffle r")],
            [_io("y  (B, C, r·H, r·W)")],
        ],
    ),
    "DeformableConv2d": (
        "Conv whose sample locations are shifted by a learned offset field.",
        "(B, C, H, W) → (B, C', H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("offset Conv → Δp")],
            [_op("bilinear sample at p + Δp")],
            [_op("Conv on sampled features")],
            [_io("y  (B, C', H, W)")],
        ],
    ),
    "DeformableAttention": (
        "Attention that samples a small set of keys at learned offsets per query (Deformable DETR).",
        "(B, N, C) → (B, N, C)",
        [
            [_io("x  (B, N, C)")],
            [_op("MLP → reference + sampling offsets")],
            [_op("bilinear sample features")],
            [_attn("attention weights (per head, per point)")],
            [_merge("weighted sum + output proj")],
            [_io("y  (B, N, C)")],
        ],
    ),
}


UNET_DIFFUSION: Dict[str, Spec] = {
    "SinusoidalTimeEmbedding": (
        "Sin/cos positional embedding of the diffusion timestep t.",
        "t:(B,) → emb:(B, D)",
        [
            [_io("t  (B,)")],
            [_op("freqs = 10000^(−2i/D)")],
            [_op("[sin(t·f),  cos(t·f)]")],
            [_emb("emb  (B, D)")],
        ],
    ),
    "TimestepMLP": (
        "Two-layer MLP applied to the time embedding before injection.",
        "(B, D) → (B, D')",
        [
            [_io("t_emb  (B, D)")],
            [_op("linear")],
            [_act("SiLU")],
            [_op("linear")],
            [_io("y  (B, D')")],
        ],
    ),
    "DownsampleBlock": (
        "Spatial downsample by 2× via a strided 3×3 conv.",
        "(B, C, H, W) → (B, C, H/2, W/2)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("Conv 3×3, stride 2")],
            [_io("y  (B, C, H/2, W/2)")],
        ],
    ),
    "UpsampleBlock": (
        "Nearest-neighbour upsample by 2× followed by a 3×3 conv.",
        "(B, C, H, W) → (B, C, 2H, 2W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("nearest upsample 2×")],
            [_op("Conv 3×3")],
            [_io("y  (B, C, 2H, 2W)")],
        ],
    ),
    "UNetResBlock": (
        "Residual block conditioned on a time embedding (added between the two convs).",
        "x:(B, C, H, W), t:(B, D) → (B, C', H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_norm("GroupNorm")],
            [_act("SiLU")],
            [_op("Conv 3×3")],
            [_merge("+ MLP(t_emb)  (broadcast)")],
            [_norm("GroupNorm")],
            [_act("SiLU")],
            [_op("Conv 3×3")],
            [_merge("+")],
            [_io("y  (B, C', H, W)")],
        ],
        [(0, 0, 8, 0)],
    ),
    "UNet": (
        "Encoder-decoder with skip connections at matching resolutions; bottleneck attention.",
        "(B, C, H, W) → (B, C, H, W)",
        [
            [_io("x  noised  (B, C, H, W)")],
            [_op("Encoder Block 1")],
            [_op("Down 2×")],
            [_op("Encoder Block 2")],
            [_op("Down 2×")],
            [_op("Bottleneck (+ self-attn)")],
            [_op("Up 2×  + skip")],
            [_op("Decoder Block 2")],
            [_op("Up 2×  + skip")],
            [_op("Decoder Block 1")],
            [_op("Conv → ε̂")],
            [_io("y  noise pred  (B, C, H, W)")],
        ],
        [(1, 0, 9, 0), (3, 0, 7, 0)],
    ),
    "NoisePredictor": (
        "End-to-end ε-prediction network used by DDPM/DDIM samplers.",
        "x_t:(B, C, H, W), t:(B,) → ε̂:(B, C, H, W)",
        [
            [_io("x_t  (B, C, H, W)"), _io("t  (B,)")],
            [_op("image stem"), _ref("SinusoidalTimeEmbedding")],
            [_ref("UNet")],
            [_io("ε̂  (B, C, H, W)")],
        ],
    ),
    "ZeroConv2d": (
        "Conv2d initialised to zero — outputs zero at start so it can be added safely.",
        "(B, C, H, W) → (B, C', H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("Conv  (W = 0, b = 0 at init)")],
            [_io("y  (B, C', H, W)  (= 0 at init)")],
        ],
    ),
    "ControlNetBlock": (
        "Trainable copy of UNet encoder + ZeroConvs; outputs are added to the frozen UNet decoder.",
        "x, control → ΔUNet decoder features",
        [
            [_io("x  noised  (B, C, H, W)"), _io("control image  (B, 3, H, W)")],
            [_op("frozen UNet encoder"), _op("trainable encoder copy")],
            [_op("ZeroConv on each level")],
            [_merge("add into UNet decoder skips")],
            [_io("Δ feats  (multi-scale, matches UNet decoder)")],
        ],
    ),
    "LoRALinear": (
        "Frozen linear plus a low-rank residual B·A·x scaled by α/r.",
        "(B, in) → (B, out)",
        [
            [_io("x  (B, in)")],
            [_op("frozen W · x"), _op("A · x   (in → r)")],
            [_op("identity"), _op("B · (A · x)   (r → out)")],
            [_merge("+  α/r ·")],
            [_io("y  (B, out)")],
        ],
    ),
    "LoRAConv2d": (
        "Same low-rank residual idea applied to 2-D convolutions.",
        "(B, C, H, W) → (B, C', H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("frozen Conv"), _op("Conv A  (C → r, k×k)")],
            [_op("identity"), _op("Conv B  (r → C', 1×1)")],
            [_merge("+  α/r ·")],
            [_io("y  (B, C', H, W)")],
        ],
    ),
    "HyperNetwork": (
        "A meta-network that emits weights consumed by a target network.",
        "cond:(B, D_c) → params → y",
        [
            [_io("condition  (B, D_c)")],
            [_op("meta MLP")],
            [_op("generated weights θ")],
            [_op("target net  (uses θ on input x)")],
            [_io("y  (task-dependent)")],
        ],
    ),
    "IPAdapterCrossAttention": (
        "Two parallel cross-attentions (text and image) summed into the residual stream.",
        "x:(B, T, D), text/image features → (B, T, D)",
        [
            [_io("x  Q  (B, T, D)")],
            [_attn("Cross-Attn  (text K, V)"), _attn("Cross-Attn  (image K, V)")],
            [_merge("+")],
            [_io("y  (B, T, D)")],
        ],
    ),
}


GAN: Dict[str, Spec] = {
    "EqualLinear": (
        "Linear with equalized learning rate: weight scaled at runtime by gain/√fan_in.",
        "(B, in) → (B, out)",
        [
            [_io("x  (B, in)")],
            [_op("matmul  x · (W · s)")],
            [_op("+ bias · lr_mul")],
            [_io("y  (B, out)")],
        ],
    ),
    "EqualConv2d": (
        "Conv2d with equalized learning rate (StyleGAN family).",
        "(B, C, H, W) → (B, C', H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("Conv with W · (gain/√(k²·C))")],
            [_op("+ bias")],
            [_io("y  (B, C', H, W)")],
        ],
    ),
    "GeneratorBlock": (
        "Vanilla generator block: upsample → conv → norm → activation.",
        "(B, C, H, W) → (B, C', 2H, 2W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("Upsample 2×")],
            [_op("Conv 3×3")],
            [_norm("BatchNorm")],
            [_act("ReLU")],
            [_io("y  (B, C', 2H, 2W)")],
        ],
    ),
    "DiscriminatorBlock": (
        "Vanilla discriminator block: strided conv → norm → leaky ReLU.",
        "(B, C, H, W) → (B, C', H/2, W/2)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("Conv 3×3, stride 2")],
            [_norm("InstanceNorm")],
            [_act("LeakyReLU 0.2")],
            [_io("y  (B, C', H/2, W/2)")],
        ],
    ),
    "MappingNetwork": (
        "8-layer MLP with reduced LR that maps z → w (StyleGAN).",
        "z:(B, D) → w:(B, D)",
        [
            [_io("z  (B, D)")],
            [_op("PixelNorm")],
            [_op("EqualLinear × 8  (lr_mul = 0.01)")],
            [_io("w  (B, D)")],
        ],
    ),
    "StyleBlock": (
        "Conv + per-pixel noise + AdaIN modulated by style w (StyleGAN v1).",
        "x:(B, C, H, W), w:(B, D) → y",
        [
            [_io("x  (B, C, H, W)"), _io("w  (B, D)")],
            [_op("Conv 3×3"), _op("A: linear w → (γ, β)")],
            [_op("+ Gaussian noise · learned scale")],
            [_op("AdaIN(γ, β)")],
            [_act("LeakyReLU")],
            [_io("y  (B, C, H, W)")],
        ],
    ),
    "ModulatedConv2d": (
        "StyleGAN2 modulated conv: scale weights by style, demodulate, then convolve.",
        "x:(B, C, H, W), w:(B, D) → y",
        [
            [_io("x  (B, C, H, W)"), _io("w  (B, D)")],
            [_op("weights W"), _op("A: linear w → s")],
            [_op("W' = W · s")],
            [_op("demod: W'' = W' / ||W'||")],
            [_op("Conv with W''")],
            [_io("y  (B, C', H, W)")],
        ],
    ),
    "MinibatchStdDev": (
        "Append a per-batch standard-deviation map as an extra channel (PGGAN/StyleGAN).",
        "(B, C, H, W) → (B, C+1, H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("std over batch  (per channel, per pixel)")],
            [_op("average → scalar")],
            [_op("tile → (B, 1, H, W)")],
            [_merge("concat as new channel")],
            [_io("y  (B, C+1, H, W)")],
        ],
        [(0, 0, 4, 0)],
    ),
    "ProgressiveGrowing": (
        "Fade in a new high-res block via α-blend with the previous resolution (PGGAN).",
        "x → upsampled / new block → y",
        [
            [_io("x  (B, C, H, W)")],
            [_op("old layers (already trained)")],
            [_op("new high-res layer")],
            [_merge("(1−α) old  +  α new")],
            [_io("y  (B, C', 2H, 2W)")],
        ],
        [(1, 0, 3, 0)],
    ),
}


VIT: Dict[str, Spec] = {
    "PatchEmbedding": (
        "Strided p×p conv that turns an image into a sequence of patch tokens.",
        "(B, C, H, W) → (B, N, D),  N = (H/p)·(W/p)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("Conv  kernel = stride = p")],
            [_op("flatten spatial")],
            [_op("transpose → (B, N, D)")],
            [_io("tokens  (B, N, D)")],
        ],
    ),
    "CLSToken": (
        "Prepend a learnable [CLS] token to each sequence (used as the global representation).",
        "(B, N, D) → (B, 1+N, D)",
        [
            [_io("tokens  (B, N, D)")],
            [_op("prepend [CLS]  (learnable, broadcast over batch)")],
            [_io("tokens'  (B, 1+N, D)")],
        ],
    ),
    "SwinWindowAttention": (
        "Self-attention restricted to non-overlapping w×w windows with relative-pos bias.",
        "(B, N, C) → (B, N, C)  with implicit (H, W)",
        [
            [_io("x  (B, H, W, C)")],
            [_op("window partition  (w × w)")],
            [_attn("MHA + relative position bias")],
            [_op("window reverse")],
            [_io("y  (B, H, W, C)")],
        ],
    ),
    "ShiftedWindowAttention": (
        "Cyclic-shift variant that lets adjacent windows exchange information.",
        "(B, N, C) → (B, N, C)  with implicit (H, W)",
        [
            [_io("x  (B, H, W, C)")],
            [_op("cyclic shift (−w/2)")],
            [_op("window partition")],
            [_attn("MHA + shifted-window mask")],
            [_op("window reverse")],
            [_op("cyclic shift (+w/2)")],
            [_io("y  (B, H, W, C)")],
        ],
    ),
    "MaskedImageModeling": (
        "MAE/BEiT objective: random-mask patches, encode visible tokens, reconstruct.",
        "(B, N, D) → masked, (B, N, p²·C) reconstructed",
        [
            [_io("image patches  (B, N, p²·C)")],
            [_op("random mask (≈ 75 %)")],
            [_op("encoder  (visible tokens only)")],
            [_op("insert [MASK] tokens")],
            [_op("decoder")],
            [_io("reconstructed patches  (B, N, p²·C)")],
        ],
    ),
}


SEQUENCE: Dict[str, Spec] = {
    "RNNCell": (
        "Vanilla recurrent cell: h_t = tanh(W_x x_t + W_h h_{t-1} + b).",
        "x_t:(B, D_x), h_{t-1}:(B, D_h) → h_t:(B, D_h)",
        [
            [_io("x_t  (B, D_x)"), _io("h_{t−1}  (B, D_h)")],
            [_op("W_x · x_t"), _op("W_h · h_{t−1}")],
            [_merge("+ b")],
            [_act("tanh")],
            [_io("h_t  (B, D_h)")],
        ],
    ),
    "LSTMCell": (
        "Long short-term memory cell with input/forget/cell/output gates.",
        "x_t, h_{t-1}, c_{t-1} → h_t, c_t",
        [
            [_io("x_t  (B, D_x)"), _io("h_{t−1}  (B, D_h)"), _io("c_{t−1}  (B, D_h)")],
            [_op("linear → (i, f, g, o)")],
            [_op("c_t = f ⊙ c_{t−1} + i ⊙ g")],
            [_op("h_t = o ⊙ tanh(c_t)")],
            [_io("h_t, c_t  (B, D_h)")],
        ],
    ),
    "GRUCell": (
        "Gated recurrent unit: reset (r), update (z) and candidate (n) gates.",
        "x_t, h_{t-1} → h_t",
        [
            [_io("x_t  (B, D_x)"), _io("h_{t−1}  (B, D_h)")],
            [_op("linear → (r, z, n)")],
            [_op("h_t = (1 − z) ⊙ h_{t−1} + z ⊙ n")],
            [_io("h_t  (B, D_h)")],
        ],
    ),
    "StateSpaceModel": (
        "Linear SSM with discretized A, B and an output mapping y = C h + D u.",
        "u:(B, T, D) → y:(B, T, D)",
        [
            [_io("u_t  (B, T, D)")],
            [_op("discretize  Ā = exp(Δ·A),  B̄ = (Ā − I)·A⁻¹·B")],
            [_op("scan  h_t = Ā h_{t−1} + B̄ u_t")],
            [_op("y_t = C h_t + D u_t")],
            [_io("y_t  (B, T, D)")],
        ],
    ),
    "MambaBlock": (
        "Selective state-space block: gates, conv, input-dependent SSM, gated output.",
        "(B, T, D) → (B, T, D)",
        [
            [_io("x  (B, T, D)")],
            [_op("in_proj  (D → 2·D')")],
            [_op("Conv 1D"), _act("SiLU (gate path)")],
            [_op("selective SSM  (A, Δ, B, C from x)")],
            [_merge("⊙ SiLU(gate)")],
            [_op("out_proj  (D' → D)")],
            [_io("y  (B, T, D)")],
        ],
    ),
}


GNN: Dict[str, Spec] = {
    "MessagePassing": (
        "Generic message-passing skeleton: message → aggregate → update.",
        "X:(N, F), edges:(2, E) → X':(N, F')",
        [
            [_io("node feats X  (N, F)"), _io("edges (i, j)  (2, E)")],
            [_op("message  m_ij = φ(h_i, h_j, e_ij)")],
            [_op("aggregate  m_j = Σ_{i ∈ N(j)} m_ij")],
            [_op("update  h_j' = ψ(h_j, m_j)")],
            [_io("X'  (N, F')")],
        ],
    ),
    "GraphConv": (
        "GCN propagation: H' = Â · X · W with symmetric normalisation.",
        "X:(N, F), A:(N, N) → H':(N, F')",
        [
            [_io("X  (N, F)"), _io("A  (N, N)")],
            [_op("Â = D^(−1/2) (A + I) D^(−1/2)")],
            [_op("H' = Â · X · W")],
            [_io("H'  (N, F')")],
        ],
    ),
    "GraphAttention": (
        "GAT: per-edge attention coefficients followed by neighbour aggregation.",
        "X:(N, F), edges → H':(N, F')",
        [
            [_io("X  (N, F)"), _io("edges  (2, E)")],
            [_op("linear projection W·h")],
            [_attn("attention coef α_ij per edge")],
            [_act("softmax over neighbours")],
            [_op("aggregate  Σ α_ij · W·h_j")],
            [_io("H'  (N, F')")],
        ],
    ),
}


GENERATIVE: Dict[str, Spec] = {
    "VAE": (
        "Encoder → posterior (μ, σ) → reparameterised z → decoder.",
        "(B, C, H, W) → (B, C, H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("encoder")],
            [_op("linear → (μ, log σ²)")],
            [_op("z = μ + ε ⊙ σ,   ε ∼ N(0, I)")],
            [_op("decoder")],
            [_io("x̂  (B, C, H, W)")],
        ],
    ),
    "MaskedConv2d": (
        "Conv2d with an autoregressive mask preventing each pixel from seeing future pixels (PixelCNN).",
        "(B, C, H, W) → (B, C', H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("Conv  (kernel multiplied by AR mask)")],
            [_io("y  (B, C', H, W)")],
        ],
    ),
    "AutoregressiveBlock": (
        "Stack of masked convs producing a per-pixel distribution over the next pixel.",
        "(B, C, H, W) → (B, K, H, W)  logits",
        [
            [_io("x  (B, C, H, W)")],
            [_op("MaskedConv  type A")],
            [_op("MaskedConv  type B  × N")],
            [_op("Conv 1×1 → logits")],
            [_io("logits  (B, K, H, W)")],
        ],
    ),
    "AffineCouplingLayer": (
        "Normalising-flow coupling: split, predict scale/shift from one half, transform the other.",
        "x = (a, b) → y = (a, b · exp(s) + t)",
        [
            [_io("x = (a, b)  (B, D)")],
            [_op("NN(a) → (s, t)")],
            [_op("a' = a,  b' = b ⊙ exp(s) + t")],
            [_io("y = (a', b')  (B, D)")],
        ],
    ),
    "EnergyBasedModel": (
        "A network that maps x to a scalar energy E(x); samples drawn via Langevin/MCMC.",
        "x → E(x):(B,)",
        [
            [_io("x  (B, …)")],
            [_op("network")],
            [_io("E(x)  (B,)")],
        ],
    ),
    "DDPMScheduler": (
        "Forward diffusion: x_t = √α̅_t · x_0 + √(1 − α̅_t) · ε.",
        "x_0, t → x_t",
        [
            [_io("x_0  (B, C, H, W)"), _io("t  (B,)")],
            [_op("β schedule → α̅_t = ∏(1 − β)")],
            [_op("x_t = √α̅_t · x_0 + √(1 − α̅_t) · ε")],
            [_io("x_t  (B, C, H, W)")],
        ],
    ),
    "DDIMScheduler": (
        "Deterministic non-Markovian sampler that reuses the predicted x̂_0.",
        "x_t, ε̂ → x_{t-1}",
        [
            [_io("x_t  (B, C, H, W)"), _io("ε̂  (B, C, H, W)")],
            [_op("x̂_0 = (x_t − √(1 − α̅_t) ε̂) / √α̅_t")],
            [_op("x_{t−1} = √α̅_{t−1} · x̂_0 + √(1 − α̅_{t−1}) · ε̂")],
            [_io("x_{t−1}  (B, C, H, W)")],
        ],
    ),
}


RL: Dict[str, Spec] = {
    "PolicyNetwork": (
        "Maps a state to an action distribution / Gaussian (μ, σ) / discrete logits.",
        "s:(B, D_s) → π(a | s)",
        [
            [_io("state  (B, D_s)")],
            [_op("MLP trunk")],
            [_op("head → π(a | s)")],
            [_io("action / dist  (B, D_a)")],
        ],
    ),
    "ValueNetwork": (
        "Estimates V(s).",
        "s:(B, D_s) → V:(B,)",
        [
            [_io("state  (B, D_s)")],
            [_op("MLP")],
            [_op("head → V(s)")],
            [_io("value  (B,)")],
        ],
    ),
    "QNetwork": (
        "Estimates Q(s, a) — either by concatenating (s, a) or with one head per discrete action.",
        "s, a → Q:(B,)",
        [
            [_io("state, action  (B, D_s) + (B, D_a)")],
            [_op("concat / one-hot")],
            [_op("MLP")],
            [_io("Q(s, a)  (B,)")],
        ],
    ),
    "ActorCritic": (
        "Shared trunk that branches into a policy head and a value head.",
        "s → π(a | s),  V(s)",
        [
            [_io("state  (B, D_s)")],
            [_op("shared MLP trunk")],
            [_op("policy head"), _op("value head")],
            [_io("π, V  (B, D_a), (B,)")],
        ],
    ),
    "ReplayBuffer": (
        "Ring buffer storing transitions; sampled in mini-batches for off-policy learning.",
        "(s, a, r, s', d) → batch",
        [
            [_io("transition (s, a, r, s', d)  (·,)")],
            [_op("ring buffer  (capacity N)")],
            [_op("uniform / prioritised sample")],
            [_io("mini-batch  (B, ·)")],
        ],
    ),
    "TargetNetwork": (
        "Slowly-tracking copy of the online network used to stabilise bootstrap targets.",
        "θ_online → θ_target",
        [
            [_io("θ_online  (params)")],
            [_op("θ_target ← τ · θ_target + (1 − τ) · θ_online")],
            [_io("θ_target  (params)")],
        ],
    ),
}


MEMORY_RETRIEVAL: Dict[str, Spec] = {
    "ExternalMemory": (
        "Soft attention over an external memory bank acts as a read operation.",
        "q:(B, D), mem:(M, D) → r:(B, D)",
        [
            [_io("query  (B, D)"), _io("memory bank  (M, D)")],
            [_attn("attention(query, mem, mem)")],
            [_io("retrieved  (B, D)")],
        ],
    ),
    "VectorStore": (
        "Encode a corpus into a vector index, then look up the top-k nearest neighbours.",
        "corpus → index;  q → top-k",
        [
            [_io("corpus  (N docs)")],
            [_op("encoder → embeddings")],
            [_op("ANN index  (FAISS / HNSW / IVFPQ)")],
            [_io("query → top-k  (k, D)")],
        ],
    ),
    "RAGModule": (
        "Retrieval-Augmented Generation: retrieve top-k docs, condition the LM on them.",
        "query → answer",
        [
            [_io("query  (T_q,)")],
            [_op("VectorStore.retrieve  top-k")],
            [_op("encode docs (token-level)")],
            [_attn("cross-attention (LM ← docs)")],
            [_io("generated answer  (T_a,)")],
        ],
    ),
    "KVCache": (
        "Cache K, V tensors per layer per token to skip re-encoding history.",
        "x_t → K_{1..t}, V_{1..t}",
        [
            [_io("x_t  (B, 1, D)")],
            [_op("compute K_t, V_t")],
            [_op("append to cache")],
            [_io("cached K, V  (L, B, T, D)")],
        ],
    ),
}


EMBEDDING: Dict[str, Spec] = {
    "TokenEmbedding": (
        "Lookup-table embedding: id → vector.",
        "(B, T) → (B, T, D)",
        [
            [_io("ids  (B, T)")],
            [_op("lookup  E[id]")],
            [_emb("token emb  (B, T, D)")],
        ],
    ),
    "LearnedPositionalEmbedding": (
        "A separate learned vector per absolute position, added to the token embedding.",
        "(B, T, D) → (B, T, D)",
        [
            [_io("positions 0..T−1  (T,)")],
            [_op("lookup  P[pos]")],
            [_merge("+ token emb")],
            [_emb("pos-encoded tokens  (B, T, D)")],
        ],
    ),
    "SinusoidalPositionalEmbedding": (
        "Fixed sin/cos position encoding (Vaswani et al.).",
        "(B, T, D) → (B, T, D)",
        [
            [_io("position p  (T,)")],
            [_op("freqs = 10000^(−2i/D)")],
            [_op("[sin(p·f),  cos(p·f)]")],
            [_merge("+ token emb")],
            [_emb("pos-encoded tokens  (B, T, D)")],
        ],
    ),
    "ProjectionHead": (
        "Two-layer MLP + L2 normalisation, used for contrastive / representation learning.",
        "(B, D) → (B, D')",
        [
            [_io("x  (B, D)")],
            [_op("linear")],
            [_act("GELU")],
            [_op("linear")],
            [_op("L2 normalize")],
            [_emb("z  (B, D')")],
        ],
    ),
    "CLIPLoss": (
        "Symmetric InfoNCE between matched image / text embeddings.",
        "z_img, z_txt:(B, D) → loss",
        [
            [_io("image emb  (B, D)"), _io("text emb  (B, D)")],
            [_op("L2 normalize")],
            [_op("logits = z_img · z_txtᵀ / τ")],
            [_loss("symmetric cross-entropy (rows + cols)")],
            [_io("loss  (1,)")],
        ],
    ),
    "info_nce": (
        "Pairwise InfoNCE used by SimCLR / MoCo.",
        "z₁, z₂:(B, D) → loss",
        [
            [_io("z₁, z₂  (B, D)")],
            [_op("L2 normalize")],
            [_op("logits = z₁ · z₂ᵀ / τ")],
            [_loss("cross-entropy with diagonal targets")],
            [_io("loss  (1,)")],
        ],
    ),
}


OPTIMIZATION: Dict[str, Spec] = {
    "Lion": (
        "Sign-of-momentum optimiser (Chen et al. 2023).",
        "g, θ → θ'",
        [
            [_io("grad g  (params)")],
            [_op("m ← β₁·m + (1 − β₁)·g")],
            [_op("update = sign(β₂·m + (1 − β₂)·g)")],
            [_op("θ ← θ − lr · update")],
            [_io("θ'  (params)")],
        ],
    ),
    "Sophia": (
        "Hessian-clipped second-order optimiser.",
        "g, h, θ → θ'",
        [
            [_io("g, h  (params)  (Hessian estimate)")],
            [_op("m ← β₁·m + (1 − β₁)·g")],
            [_op("update = clip(m / max(h, ε), ρ)")],
            [_op("θ ← θ − lr · update")],
            [_io("θ'  (params)")],
        ],
    ),
    "EMA": (
        "Exponential moving average of model weights.",
        "θ_online → θ_ema",
        [
            [_io("θ_online  (params)")],
            [_op("θ_ema ← τ · θ_ema + (1 − τ) · θ_online")],
            [_io("θ_ema  (params)")],
        ],
    ),
    "MixedPrecisionTrainer": (
        "Loss-scaled fp16 forward, fp32 master weights, gradient unscale + clip + step.",
        "x → updated θ",
        [
            [_io("x  (B, …)")],
            [_op("fp16 forward")],
            [_op("loss × scale")],
            [_op("backward (fp32 master grads)")],
            [_op("unscale + clip")],
            [_op("optimizer step")],
            [_io("updated θ  (params)")],
        ],
    ),
    "CheckpointedSequential": (
        "Sequential whose forward is rematerialised on the backward pass to save memory.",
        "(*) → (*)",
        [
            [_io("x  (*)")],
            [_op("layer 1   (recompute on backward)")],
            [_op("layer 2   (recompute on backward)")],
            [_op("…  (k layers)")],
            [_io("y  (*)")],
        ],
    ),
}


MULTIMODAL: Dict[str, Spec] = {
    "CLIPEncoder": (
        "Dual-tower image / text encoders with projection heads into a shared space.",
        "image, text → (z_img, z_txt)",
        [
            [_io("image  (B, 3, H, W)"), _io("text tokens  (B, T)")],
            [_op("image transformer"), _op("text transformer")],
            [_op("projection"), _op("projection")],
            [_op("L2 norm"), _op("L2 norm")],
            [_emb("z_img  (B, D)"), _emb("z_txt  (B, D)")],
        ],
    ),
    "PerceiverResampler": (
        "Fixed-length learnable latents cross-attend over arbitrary-length media features.",
        "media:(B, M, D), latents:(L, D) → (B, L, D)",
        [
            [_io("learnable latents  (L, D)"), _io("media features  (B, M, D)")],
            [_attn("cross-attn  (latents query)")],
            [_op("FFN")],
            [_op("× N layers")],
            [_io("resampled latents  (B, L, D)")],
        ],
    ),
    "QFormer": (
        "BLIP-2 querying transformer: queries self-attend, then cross-attend to image features.",
        "queries, image feats → queries",
        [
            [_io("learnable queries  (Q, D)"), _io("image features  (B, M, D)")],
            [_attn("Self-Attention")],
            [_attn("Cross-Attention (image)")],
            [_op("FFN")],
            [_op("× N layers")],
            [_io("queries  (B, Q, D)")],
        ],
    ),
    "ToolUseBlock": (
        "Soft selector over a set of tools, each invoked, results merged back into the state.",
        "state → updated state",
        [
            [_io("state  (B, D)")],
            [_op("tool selector  (softmax)")],
            [_op("Tool 1"), _op("…"), _op("Tool K")],
            [_merge("merge tool outputs")],
            [_io("new state  (B, D)")],
        ],
    ),
    "MemoryAttention": (
        "Cross-attention layer that reads from an external memory bank.",
        "x:(B, T, D), mem:(M, D) → (B, T, D)",
        [
            [_io("x  (B, T, D)"), _io("memory bank  (M, D)")],
            [_attn("cross-attention over memory")],
            [_io("y  (B, T, D)")],
        ],
    ),
}


EFFICIENT: Dict[str, Spec] = {
    "QuantizedLinearInt8": (
        "Linear with weights stored in int8 + per-output-channel scale.",
        "(B, in) → (B, out)",
        [
            [_io("x  (B, in)  fp")],
            [_op("W_int8  (per-channel scale s)")],
            [_op("matmul x · W_int8")],
            [_op("× s   (dequantise)")],
            [_op("+ bias")],
            [_io("y  (B, out)")],
        ],
    ),
    "QuantizedLinear4bit": (
        "Linear with weights packed to 4-bit + grouped scales / zeros (NF4 / GPTQ style).",
        "(B, in) → (B, out)",
        [
            [_io("x  (B, in)  fp")],
            [_op("unpack 4-bit → int")],
            [_op("dequantise per group  (s, z)")],
            [_op("matmul x · W")],
            [_op("+ bias")],
            [_io("y  (B, out)")],
        ],
    ),
    "MagnitudePruner": (
        "Zero-out weights with |W| below a threshold τ.",
        "W → sparse W'",
        [
            [_io("weights W  (out, in)")],
            [_op("|W| < τ → mask = 0")],
            [_op("W' = W ⊙ mask")],
            [_io("sparse W'  (out, in)")],
        ],
    ),
    "TokenPruner": (
        "Drop low-importance tokens before further attention layers.",
        "(B, N, D) → (B, K, D),  K < N",
        [
            [_io("tokens  (B, N, D)")],
            [_op("importance score per token")],
            [_op("top-k keep")],
            [_io("reduced tokens  (B, K, D)")],
        ],
    ),
    "LowRankLinear": (
        "Two stacked linears whose product approximates a full matrix W ≈ U Vᵀ.",
        "(B, in) → (B, out)",
        [
            [_io("x  (B, in)")],
            [_op("down: linear  in → r")],
            [_op("up: linear   r → out")],
            [_io("y  (B, out)")],
        ],
    ),
    "ColumnParallelLinear": (
        "Linear whose output columns are sharded across tensor-parallel ranks.",
        "x → y  (gathered)",
        [
            [_io("x  (B, in)  replicated")],
            [_op("split W cols across ranks")],
            [_op("local matmul")],
            [_op("all-gather")],
            [_io("y  (B, out)  gathered")],
        ],
    ),
    "RowParallelLinear": (
        "Linear whose input is sharded across ranks; outputs all-reduced.",
        "x  (sharded) → y",
        [
            [_io("x  (B, in/r)  sharded")],
            [_op("local matmul")],
            [_op("all-reduce sum")],
            [_io("y  (B, out)")],
        ],
    ),
    "PipelineStage": (
        "One stage of pipeline parallelism — receives micro-batches, computes, sends forward.",
        "x_chunk → y_chunk",
        [
            [_io("x_chunk  (μb, …)")],
            [_op("stage_i forward")],
            [_op("send to stage_{i+1}")],
            [_io("y_chunk  (μb, …)")],
        ],
    ),
}


SPECIALIZED: Dict[str, Spec] = {
    "NeuralODE": (
        "Treat depth as continuous time and integrate dx/dt = f_θ(x, t).",
        "x_0 → x_T",
        [
            [_io("x_0  (B, D)"), _io("f_θ(x, t)  (·)→(·)")],
            [_op("ODE solver  (Euler / RK4 / dopri5)")],
            [_op("x(T) = x_0 + ∫₀ᵀ f dt")],
            [_io("x_T  (B, D)")],
        ],
    ),
    "SpectralConv2d": (
        "Multiply low-frequency Fourier modes by learned weights (Fourier Neural Operator).",
        "(B, C, H, W) → (B, C', H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("FFT2")],
            [_op("× learned weights  (low modes only)")],
            [_op("IFFT2")],
            [_io("y  (B, C', H, W)")],
        ],
    ),
    "FNOBlock": (
        "FNO block: spectral conv + 1×1 conv, summed and activated.",
        "(B, C, H, W) → (B, C, H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_ref("SpectralConv2d"), _op("1×1 Conv")],
            [_merge("+")],
            [_act("GELU")],
            [_io("y  (B, C, H, W)")],
        ],
    ),
    "KANLayer": (
        "Kolmogorov–Arnold layer: every edge has a learned univariate spline; node = sum of edges.",
        "(B, in) → (B, out)",
        [
            [_io("x  (B, in)")],
            [_op("learned spline  φ_ij(x_i)  per edge")],
            [_op("sum over inputs at each output node")],
            [_io("y  (B, out)")],
        ],
    ),
    "CapsuleLayer": (
        "Capsule routing layer (Sabour et al.): predict votes, dynamic routing-by-agreement, squash.",
        "primary caps → output caps",
        [
            [_io("primary capsules u_i  (N₁, d₁)")],
            [_op("votes  û_{j|i} = W_ij · u_i")],
            [_op("dynamic routing × T")],
            [_op("squash")],
            [_io("output capsules v_j  (N₂, d₂)")],
        ],
    ),
    "SlotAttention": (
        "Iterated cross-attention from a small set of slots into per-pixel features (SlotAttention).",
        "feats:(B, N, D), slots:(B, K, D) → slots",
        [
            [_io("features  (B, N, D)"), _io("slots  (B, K, D)")],
            [_attn("cross-attn  (slots query)")],
            [_op("GRU update")],
            [_op("× T iterations")],
            [_io("updated slots  (B, K, D)")],
        ],
    ),
}


# ---------------------------------------------------------------------------
# Top-level registry + runner
# ---------------------------------------------------------------------------

CATEGORIES: Dict[str, Tuple[str, Dict[str, Spec]]] = {
    "core":              ("Core neural-network primitives.", CORE),
    "attention":         ("Attention mechanisms.", ATTENTION),
    "transformer":       ("Transformer encoder / decoder, FFN variants, MoE.", TRANSFORMER),
    "cnn_vision":        ("CNN and vision-specific blocks.", CNN_VISION),
    "unet_diffusion":    ("UNet, time conditioning, ControlNet, LoRA, hypernets.", UNET_DIFFUSION),
    "gan":               ("GAN building blocks: StyleGAN, PGGAN, equalised LR.", GAN),
    "vit":               ("Vision Transformer blocks.", VIT),
    "sequence":          ("Recurrent and state-space sequence models.", SEQUENCE),
    "gnn":               ("Graph neural network layers.", GNN),
    "generative":        ("VAE, autoregressive, normalising-flow, EBM, diffusion schedulers.", GENERATIVE),
    "rl":                ("Reinforcement-learning building blocks.", RL),
    "memory_retrieval":  ("External memory, vector stores, RAG, KV caches.", MEMORY_RETRIEVAL),
    "embedding":         ("Token / positional / projection embeddings, contrastive losses.", EMBEDDING),
    "optimization":      ("Optimisers, schedulers, EMA, mixed-precision, checkpointing.", OPTIMIZATION),
    "multimodal":        ("Multimodal / agentic blocks.", MULTIMODAL),
    "efficient":         ("Sparsity, quantisation, parallelism, low-rank.", EFFICIENT),
    "specialized":       ("Specialised research blocks (NeuralODE, FNO, KAN, capsules, slots).", SPECIALIZED),
}


README_BODY = """\
# Architecture diagrams

One Mermaid `flowchart TD` per public block, organised by category. The
specs live in [`_generate.py`](./_generate.py); regenerate everything with

```bash
python _generate.py
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
python _generate.py --specs my_arch.py --out ./diagrams --depth 1
# only your blocks, library kept as registered references:
python _generate.py --specs my_arch.py --out ./diagrams --no-builtins
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


def _spec_unpack(spec: Spec) -> Tuple[str, str, List[Row], Sequence[Skip]]:
    if len(spec) == 3:
        desc, shapes, rows = spec
        return desc, shapes, rows, ()
    desc, shapes, rows, skips = spec
    return desc, shapes, rows, skips


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
        help="Output directory (default: directory of this script).",
    )
    parser.add_argument(
        "--no-builtins", action="store_true",
        help="Don't regenerate built-in block .md files; keep them as registered "
             "references only.",
    )
    args = parser.parse_args()

    builtin_root = Path(__file__).parent.resolve()
    out_root = (args.out.resolve() if args.out else builtin_root)
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
                    name, (builtin_root / cat / f"{name}.md").resolve(),
                )

    total = 0
    for cat, (_, blocks) in to_generate.items():
        out_dir = out_root / cat
        out_dir.mkdir(parents=True, exist_ok=True)
        # only wipe .md inside categories we are actually regenerating
        for old in out_dir.glob("*.md"):
            old.unlink()
        for name, spec in blocks.items():
            desc, shapes, rows, skips = _spec_unpack(spec)
            body = render_diagram(rows, skips, registry=registry, max_depth=args.depth)
            md = render_markdown(name, desc, shapes, body)
            name_to_path[name].write_text(md)
            total += 1

    # README only belongs to the built-in repo; don't clobber arbitrary --out trees.
    if not args.specs and out_root == builtin_root:
        (builtin_root / "README.md").write_text(README_BODY)

    write_index(out_root, all_cats, name_to_path)

    user_count = sum(len(b) for _, b in user_cats.values())
    print(
        f"wrote {total} diagrams across {len(to_generate)} categories "
        f"({user_count} user, {total - user_count} built-in) → {out_root}"
    )


if __name__ == "__main__":
    main()

"""Attention mechanisms."""

from typing import Dict

from dsl import Spec, _io, _op, _norm, _act, _attn, _merge, _emb, _loss, _ref  # noqa: F401

CATEGORY = "attention"
CATEGORY_DESC = "Attention mechanisms."

BLOCKS: Dict[str, Spec] = {
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

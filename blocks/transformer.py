"""Transformer encoder / decoder, FFN variants, MoE."""

from typing import Dict

from dsl import Spec, _io, _op, _norm, _act, _attn, _merge, _emb, _loss, _ref  # noqa: F401

CATEGORY = "transformer"
CATEGORY_DESC = "Transformer encoder / decoder, FFN variants, MoE."

BLOCKS: Dict[str, Spec] = {
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

"""Multimodal / agentic blocks."""

from typing import Dict

from dsl import Spec, _io, _op, _norm, _act, _attn, _merge, _emb, _loss, _ref  # noqa: F401

CATEGORY = "multimodal"
CATEGORY_DESC = "Multimodal / agentic blocks."

BLOCKS: Dict[str, Spec] = {
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

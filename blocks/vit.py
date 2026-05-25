"""Vision Transformer blocks."""

from typing import Dict

from dsl import Spec, _io, _op, _norm, _act, _attn, _merge, _emb, _loss, _ref  # noqa: F401

CATEGORY = "vit"
CATEGORY_DESC = "Vision Transformer blocks."

BLOCKS: Dict[str, Spec] = {
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

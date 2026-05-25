"""Token / positional / projection embeddings, contrastive losses."""

from typing import Dict

from dsl import Spec, _io, _op, _norm, _act, _attn, _merge, _emb, _loss, _ref  # noqa: F401

CATEGORY = "embedding"
CATEGORY_DESC = "Token / positional / projection embeddings, contrastive losses."

BLOCKS: Dict[str, Spec] = {
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

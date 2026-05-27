"""Vision Transformer blocks."""

from typing import Dict

from dsl import (  # noqa: F401
    Spec, _io, _op, _norm, _act, _attn, _merge, _emb, _loss, _ref, _notes,
)

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
        [],
        None,
        _notes(
            used_in=[
                "ViT, DeiT, BEiT, DINO — all standard image transformers",
                "Stable Diffusion latent ViT components",
                "Video / 3D transformers (3D conv kernel for patch-time tokens)",
            ],
            tasks=[
                "Converting an image into a sequence for a Transformer",
                "Adjustable trade-off between spatial detail and sequence length via patch size",
            ],
            pitfalls=[
                "Patch size p must divide H and W exactly; resize or pad first.",
                "Smaller p quadratically increases compute (more tokens, more attention).",
                "Pretrained ViT checkpoints lock in p — fine-tuning at a different resolution needs "
                "position-embedding interpolation.",
            ],
            see_also=[
                "[ViT (Dosovitskiy et al. 2020)](https://arxiv.org/abs/2010.11929)",
                "[DeiT (Touvron et al. 2020)](https://arxiv.org/abs/2012.12877)",
            ],
        ),
    ),
    "CLSToken": (
        "Prepend a learnable [CLS] token to each sequence (used as the global representation).",
        "(B, N, D) → (B, 1+N, D)",
        [
            [_io("tokens  (B, N, D)")],
            [_op("prepend [CLS]  (learnable, broadcast over batch)")],
            [_io("tokens'  (B, 1+N, D)")],
        ],
        [],
        None,
        _notes(
            used_in=[
                "BERT — [CLS] for sentence classification",
                "ViT, DeiT — image classification head reads only the CLS token",
                "DINO — CLS distillation between teacher and student",
            ],
            tasks=[
                "Producing a single global representation from a Transformer encoder",
                "Acting as a per-input prompt position when fine-tuning",
            ],
            pitfalls=[
                "CLS pooling can be sub-optimal vs mean-pooling of patch tokens — verify empirically.",
                "When using register tokens (Darcet et al. 2023), additional learnable tokens reduce "
                "attention artefacts and may replace CLS for retrieval.",
                "Position embedding must reserve slot 0 for CLS — off-by-one bugs corrupt all positions.",
            ],
            see_also=[
                "[BERT (Devlin et al. 2018)](https://arxiv.org/abs/1810.04805)",
                "[ViT (Dosovitskiy et al. 2020)](https://arxiv.org/abs/2010.11929)",
                "[Register Tokens (Darcet et al. 2023)](https://arxiv.org/abs/2309.16588)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "Swin Transformer v1 / v2 backbones",
                "Mask2Former, SwinUNETR (medical) detection / segmentation heads",
            ],
            tasks=[
                "Hierarchical vision backbones that scale to detection / segmentation resolutions",
                "Replacing global attention when O(N²) is infeasible",
            ],
            pitfalls=[
                "Each window is independent — must alternate with a shifted-window block to mix.",
                "Relative position bias table has shape (2w−1)², becomes large for big windows.",
                "Inputs must be padded so H, W are multiples of the window size.",
            ],
            see_also=[
                "[Swin Transformer (Liu et al. 2021)](https://arxiv.org/abs/2103.14030)",
                "[Swin v2 (Liu et al. 2021)](https://arxiv.org/abs/2111.09883)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "Swin Transformer's odd-numbered blocks (SW-MSA after W-MSA)",
                "Hybrid CNN-transformer detectors using Swin as backbone",
            ],
            tasks=[
                "Letting adjacent windows exchange tokens without paying global-attention cost",
            ],
            pitfalls=[
                "The shift mask is non-trivial — getting the connectivity wrong silently halves quality.",
                "Cyclic shift must be exactly undone after attention; off-by-one rolls leak features.",
                "Combine carefully with relative position bias — shift changes which (i,j) pairs are valid.",
            ],
            see_also=[
                "[Swin Transformer (Liu et al. 2021)](https://arxiv.org/abs/2103.14030)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "MAE — Masked Autoencoders pre-training (He et al. 2021)",
                "BEiT v1 / v2 — predicts discrete tokens instead of pixels",
                "VideoMAE, SiT — video and audio extensions",
            ],
            tasks=[
                "Self-supervised pre-training of Vision Transformers",
                "Few-label fine-tuning where labeled data is scarce",
            ],
            pitfalls=[
                "75 % masking ratio is empirically best for images; lower ratios under-train the encoder.",
                "Decoder is intentionally tiny — making it deeper does NOT help downstream tasks.",
                "Pixel-reconstruction loss is unnormalised — pre-normalise patches for stable training.",
                "Heavy memory at sequence-level reordering; tensor-shuffle indexing is easy to get wrong.",
            ],
            see_also=[
                "[MAE (He et al. 2021)](https://arxiv.org/abs/2111.06377)",
                "[BEiT (Bao et al. 2021)](https://arxiv.org/abs/2106.08254)",
                "[VideoMAE (Tong et al. 2022)](https://arxiv.org/abs/2203.12602)",
            ],
        ),
    ),
}

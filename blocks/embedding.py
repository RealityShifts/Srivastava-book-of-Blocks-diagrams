"""Token / positional / projection embeddings, contrastive losses."""

from typing import Dict

from dsl import (  # noqa: F401
    Spec, _io, _op, _norm, _act, _attn, _merge, _emb, _loss, _ref, _notes,
)

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
        [],
        None,
        _notes(
            used_in=[
                "Every LLM input layer (BERT, GPT, T5, LLaMA)",
                "Word2Vec, GloVe — classical static embeddings",
                "Code / protein / DNA models with custom tokenizers",
            ],
            tasks=[
                "Mapping discrete tokens to continuous vectors for downstream layers",
                "Output classification (when tied to the input weight matrix)",
            ],
            pitfalls=[
                "Vocabulary mismatch — embedding rows must align with the tokenizer that produced ids.",
                "Forgetting to scale by √D in original Transformer breaks position-encoding magnitude.",
                "Tied input/output embeddings save parameters but require careful gradient scaling.",
                "OOV token handling (UNK / fallback bytes) must be explicit; silent fallbacks bias evals.",
            ],
            see_also=[
                "[Attention Is All You Need (Vaswani et al. 2017)](https://arxiv.org/abs/1706.03762)",
                "[Tied Embeddings (Press & Wolf 2016)](https://arxiv.org/abs/1608.05859)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "BERT, RoBERTa, GPT-2 — original Transformer-era position encoding",
                "ViT — learned position embeddings per patch",
            ],
            tasks=[
                "Position injection for fixed-length sequences",
            ],
            pitfalls=[
                "Cannot extrapolate beyond the training length without interpolation/extension hacks.",
                "Adds D × T_max parameters — large at long contexts.",
                "Largely replaced by RoPE / ALiBi in modern LLMs for length extrapolation reasons.",
            ],
            see_also=[
                "[BERT (Devlin et al. 2018)](https://arxiv.org/abs/1810.04805)",
                "[ViT (Dosovitskiy et al. 2020)](https://arxiv.org/abs/2010.11929)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "Original Transformer (Vaswani et al. 2017)",
                "Diffusion timestep encoders (DDPM / DDIM noise predictor)",
                "Music / audio Transformers",
            ],
            tasks=[
                "Position injection without learned parameters",
                "Encoding any continuous scalar (timestep, frequency, depth)",
            ],
            pitfalls=[
                "Encoding magnitudes match a specific D — copying the impl with the wrong base (10000) "
                "or missing the /2 in the exponent silently breaks things.",
                "Extrapolation beyond training T degrades smoothly but not magically.",
                "Even/odd dim split must match between sin and cos to be invertible.",
            ],
            see_also=[
                "[Attention Is All You Need (Vaswani et al. 2017)](https://arxiv.org/abs/1706.03762)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "SimCLR, MoCo, BYOL — contrastive image SSL",
                "CLIP, BLIP — image-text alignment heads",
                "Sentence-BERT, GTE, BGE — text embedding models",
            ],
            tasks=[
                "Mapping features to a contrastive / retrieval space (often discarded after pretraining)",
                "Stable training of contrastive losses (the head absorbs feature distortion)",
            ],
            pitfalls=[
                "L2 normalisation is essential before computing cosine similarity / temperature softmax.",
                "Projector is typically DISCARDED at inference — use the pre-projection features instead "
                "(SimCLR §6, MoCo v2).",
                "Width and depth of the projector matter more than people expect (BYOL has 4096-d hidden).",
            ],
            see_also=[
                "[SimCLR (Chen et al. 2020)](https://arxiv.org/abs/2002.05709)",
                "[MoCo v2 (Chen et al. 2020)](https://arxiv.org/abs/2003.04297)",
                "[BYOL (Grill et al. 2020)](https://arxiv.org/abs/2006.07733)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "CLIP (Radford et al. 2021) — 400M image-text pair pre-training",
                "OpenCLIP, LAION CLIP releases (DataComp, MetaCLIP)",
                "SigLIP — sigmoid-loss replacement",
            ],
            tasks=[
                "Image-text retrieval, zero-shot classification",
                "Foundation embeddings for downstream vision-language models",
            ],
            pitfalls=[
                "Effective batch size dominates quality — smaller batches need gradient accumulation OR "
                "switch to SigLIP (per-pair sigmoid, batch-size agnostic).",
                "τ (logit scale) is learned but clamped — uncapped τ blows up gradients.",
                "Symmetric loss = (rows + cols) / 2 — applying only one direction halves the signal.",
                "Within-batch duplicates / near-duplicates poison the contrastive loss.",
            ],
            see_also=[
                "[CLIP (Radford et al. 2021)](https://arxiv.org/abs/2103.00020)",
                "[SigLIP (Zhai et al. 2023)](https://arxiv.org/abs/2303.15343)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "SimCLR, MoCo, MoCo-v3 — image SSL",
                "CPC (van den Oord et al. 2018) — sequential predictive coding",
                "DINO-style methods (with teacher target)",
            ],
            tasks=[
                "Self-supervised representation learning from augmented views",
                "Cross-modal alignment (when restricted to one direction)",
            ],
            pitfalls=[
                "Temperature τ ~0.07–0.2 is typical — far off and the loss saturates or vanishes.",
                "Need many negatives; without a memory bank (MoCo) or large batch, quality plateaus.",
                "Hard-negative mining helps in fine-grained settings (face recognition, retrieval).",
            ],
            see_also=[
                "[CPC / InfoNCE (van den Oord et al. 2018)](https://arxiv.org/abs/1807.03748)",
                "[SimCLR (Chen et al. 2020)](https://arxiv.org/abs/2002.05709)",
                "[MoCo (He et al. 2019)](https://arxiv.org/abs/1911.05722)",
            ],
        ),
    ),
}

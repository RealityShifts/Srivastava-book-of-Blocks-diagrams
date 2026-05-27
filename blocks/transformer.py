"""Transformer encoder / decoder, FFN variants, MoE."""

from typing import Dict

from dsl import (  # noqa: F401
    Spec, _io, _op, _norm, _act, _attn, _merge, _emb, _loss, _ref, _notes,
)

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
        [],
        None,
        _notes(
            used_in=[
                "Every Transformer block ever — the 'MLP' or 'FFN' sub-layer",
                "BERT, GPT, T5, ViT, Whisper — r is typically 4",
            ],
            tasks=[
                "Per-token non-linear transformation (the 'thinking' between attention layers)",
                "Where most parameters of a Transformer live (~2/3 of total)",
            ],
            pitfalls=[
                "Activation choice matters a lot at scale — GELU > ReLU; gated variants (SwiGLU/GEGLU) "
                "further improve language modelling.",
                "Inner ratio r=4 is convention, not a constant — Chinchilla-scaled models sometimes use r=3 "
                "(SwiGLU) or 8/3 (LLaMA) to keep param count comparable.",
                "FFN dominates compute on short sequences (T < D); fuse with the next attention LN where possible.",
            ],
            see_also=[
                "[Attention Is All You Need (Vaswani et al. 2017)](https://arxiv.org/abs/1706.03762)",
                "[GLU Variants (Shazeer 2020)](https://arxiv.org/abs/2002.05202)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "LLaMA / LLaMA-2 / LLaMA-3 — flagship open-weights LLMs",
                "PaLM, Gemma, Mistral, DeepSeek — most modern decoder-only LLMs",
            ],
            tasks=[
                "Replacing the FFN in any Transformer block to get a consistent perplexity win",
                "Architectures targeting language modelling and scaling",
            ],
            pitfalls=[
                "Three linear projections (gate, up, down) — to match a 4D-FFN's parameter count, "
                "use inner hidden dim ≈ 8/3·D, NOT 4·D.",
                "Custom CUDA kernels for fused SwiGLU exist (xFormers, Triton) — naïve impl is "
                "memory-bound and slower than necessary.",
                "Gate path needs the same dtype handling as activations — bf16 generally fine, "
                "fp16 can underflow at the silu(·).",
            ],
            see_also=[
                "[GLU Variants Improve Transformer (Shazeer 2020)](https://arxiv.org/abs/2002.05202)",
                "[PaLM (Chowdhery et al. 2022)](https://arxiv.org/abs/2204.02311)",
                "[LLaMA (Touvron et al. 2023)](https://arxiv.org/abs/2302.13971)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "T5 v1.1 and FLAN-T5",
                "Gemini-1 reported variants",
                "Stable Diffusion's text encoder branch",
            ],
            tasks=[
                "Drop-in FFN replacement when GELU is preferred over SiLU (e.g. matching legacy initialisations)",
            ],
            pitfalls=[
                "Same three-projection accounting trap as SwiGLU — adjust inner dim to keep params constant.",
                "Marginal difference vs SwiGLU in practice — choose based on tokenizer / init compatibility.",
            ],
            see_also=[
                "[GLU Variants (Shazeer 2020)](https://arxiv.org/abs/2002.05202)",
                "[T5 v1.1 release notes](https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md#t511)",
            ],
        ),
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
        None,
        _notes(
            used_in=[
                "BERT, RoBERTa, DeBERTa — masked-LM pre-training",
                "ViT, DeiT, BEiT — image classification",
                "Whisper encoder, wav2vec 2.0 — speech",
            ],
            tasks=[
                "Bidirectional representation learning",
                "Sequence-level / token-level classification, regression, retrieval",
            ],
            pitfalls=[
                "Post-norm (original Vaswani) is unstable at depth — Pre-norm is the modern default.",
                "Skip-connection scaling: at extreme depth (>100 layers) consider DeepNet or ReZero "
                "to control variance growth.",
                "Each block has TWO residual sums — implementations sometimes forget one and 'work' "
                "with degraded quality.",
            ],
            see_also=[
                "[Pre-norm Transformer (Xiong et al. 2020)](https://arxiv.org/abs/2002.04745)",
                "[BERT (Devlin et al. 2018)](https://arxiv.org/abs/1810.04805)",
                "[DeepNet (Wang et al. 2022)](https://arxiv.org/abs/2203.00555)",
            ],
        ),
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
        None,
        _notes(
            used_in=[
                "Original Transformer / Marian NMT",
                "T5, BART, Pegasus — sequence-to-sequence pre-training",
                "Whisper decoder, audio-to-text translation",
            ],
            tasks=[
                "Machine translation, summarisation, text generation conditioned on encoder context",
                "Cross-modal generation (image-to-text, audio-to-text)",
            ],
            pitfalls=[
                "Three residuals per block — order is causal-self / cross / FFN; reordering changes "
                "what the model can attend to.",
                "Cross-attention K/V come from a frozen encoder during inference — cache them once.",
                "Causal mask applies to self-attn only; cross-attn must remain unmasked over context.",
            ],
            see_also=[
                "[Attention Is All You Need (Vaswani et al. 2017)](https://arxiv.org/abs/1706.03762)",
                "[BART (Lewis et al. 2019)](https://arxiv.org/abs/1910.13461)",
                "[T5 (Raffel et al. 2020)](https://arxiv.org/abs/1910.10683)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "GShard, Switch Transformer — large-scale Google LMs",
                "Mixtral 8×7B, Mixtral 8×22B, DeepSeek-V2/V3 — open MoE LLMs",
                "GLaM, ST-MoE for translation and language modelling",
            ],
            tasks=[
                "Scaling parameter count without scaling per-token FLOPs",
                "Multi-domain / multi-task models where experts can specialise",
            ],
            pitfalls=[
                "Load imbalance — most tokens route to few experts unless an auxiliary load-balancing "
                "loss is added (Switch Transformer §3.2).",
                "Top-k > 1 doubles FLOPs but tames training instability vs top-1.",
                "Expert parallelism complicates training — needs all-to-all communication and careful "
                "pipeline overlap.",
                "Inference batching is harder — pads or drops tokens at expert capacity limits.",
            ],
            see_also=[
                "[Sparsely-Gated MoE (Shazeer et al. 2017)](https://arxiv.org/abs/1701.06538)",
                "[GShard (Lepikhin et al. 2020)](https://arxiv.org/abs/2006.16668)",
                "[ST-MoE (Zoph et al. 2022)](https://arxiv.org/abs/2202.08906)",
                "[Mixtral of Experts (Jiang et al. 2024)](https://arxiv.org/abs/2401.04088)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "Switch Transformer — the original top-1 routing design (Fedus et al. 2021)",
                "Many production MoE LLMs default to top-1 routing for inference simplicity",
            ],
            tasks=[
                "Maximum-throughput MoE training at trillion-parameter scale",
                "Sparse fine-tuning where compute budget per token is fixed",
            ],
            pitfalls=[
                "Routing collapse risk is HIGHER than top-2 — must enforce expert capacity "
                "and z-loss for stability.",
                "Top-1 routing is non-differentiable in the argmax — gradient flows via the router "
                "logits scaling the expert output (straight-through estimator).",
                "Dropped tokens (over capacity) silently become zeros — keep an eye on the drop rate.",
            ],
            see_also=[
                "[Switch Transformer (Fedus et al. 2021)](https://arxiv.org/abs/2101.03961)",
            ],
        ),
    ),
}

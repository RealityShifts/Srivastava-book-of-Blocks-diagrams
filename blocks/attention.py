"""Attention mechanisms."""

from typing import Dict

from dsl import (  # noqa: F401
    Spec, _io, _op, _norm, _act, _attn, _merge, _emb, _loss, _ref, _notes,
)

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
        [],
        None,
        _notes(
            used_in=[
                "Transformer encoder / decoder — the original 'Attention is All You Need'",
                "BERT, GPT, T5, ViT and every LLM and ViT derivative",
                "Cross-modal models (CLIP, BLIP-2, Flamingo) for vision–language alignment",
            ],
            tasks=[
                "Modelling long-range dependencies without convolutional locality bias",
                "Conditioning one sequence on another (cross-attention)",
                "Set / unordered-input processing where position is encoded explicitly",
            ],
            pitfalls=[
                "Quadratic memory and compute in sequence length — switch to FlashAttention or "
                "sliding-window variants for T > a few thousand.",
                "Forgetting `1/√d_h` scaling makes the softmax saturate at long heads (poor gradients).",
                "Mask shape / dtype mismatches silently break causal attention — assert with a unit test.",
            ],
            see_also=[
                "[Attention Is All You Need (Vaswani et al. 2017)](https://arxiv.org/abs/1706.03762)",
                "[FlashAttention (Dao et al. 2022)](https://arxiv.org/abs/2205.14135)",
                "[Multi-Query Attention (Shazeer 2019)](https://arxiv.org/abs/1911.02150)",
                "[Grouped-Query Attention (Ainslie et al. 2023)](https://arxiv.org/abs/2305.13245)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "BERT and other encoder-only transformer towers",
                "ViT / DeiT / DINO image transformers",
                "Speech encoders (Whisper, wav2vec 2.0)",
            ],
            tasks=[
                "Bidirectional context mixing — every token attends to every other token",
                "Backbone for non-autoregressive tasks (classification, segmentation, span extraction)",
            ],
            pitfalls=[
                "Identical Q = K = V projection weights are NOT used — three separate W_q, W_k, W_v are.",
                "Without positional encoding the operation is fully permutation-equivariant, which is "
                "almost never what you want.",
            ],
            see_also=[
                "[BERT (Devlin et al. 2018)](https://arxiv.org/abs/1810.04805)",
                "[ViT (Dosovitskiy et al. 2020)](https://arxiv.org/abs/2010.11929)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "GPT-1/2/3/4, LLaMA, Mistral, Gemma — all decoder-only LLMs",
                "Decision Transformer and Trajectory Transformer in RL",
                "Autoregressive image / audio models (ImageGPT, MusicLM)",
            ],
            tasks=[
                "Next-token prediction / language modelling",
                "Any task where outputs are generated left-to-right",
            ],
            pitfalls=[
                "Mask must be applied to logits BEFORE softmax (set to −∞), not after.",
                "Cached K/V at inference must respect the mask — slicing past the current position "
                "is a common bug that 'works' but leaks future tokens at evaluation.",
                "Use `is_causal=True` in PyTorch SDPA to enable FlashAttention's fast path rather than "
                "passing an explicit mask tensor (much faster, less memory).",
            ],
            see_also=[
                "[GPT-2 (Radford et al. 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)",
                "[GPT-3 (Brown et al. 2020)](https://arxiv.org/abs/2005.14165)",
                "[Decision Transformer (Chen et al. 2021)](https://arxiv.org/abs/2106.01345)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "Encoder-decoder Transformers (machine translation, T5, BART)",
                "Diffusion U-Nets: image latents attend to text embeddings (Stable Diffusion)",
                "Perceiver / Q-Former / Flamingo — latents attend to media features",
            ],
            tasks=[
                "Conditional generation (text-to-image, translation, captioning)",
                "Fusing two different modalities or sequence lengths",
            ],
            pitfalls=[
                "K and V must come from the SAME source (the context). A common bug is letting them "
                "diverge accidentally during refactors.",
                "Q-length and K-length differ — never confuse `(B, T_q, D)` and `(B, T_k, D)` in masks.",
                "Memory grows as T_q × T_k, not T².",
            ],
            see_also=[
                "[Attention Is All You Need (Vaswani et al. 2017)](https://arxiv.org/abs/1706.03762)",
                "[Stable Diffusion (Rombach et al. 2021)](https://arxiv.org/abs/2112.10752)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "Swin Transformer v1 / v2 — image classification, detection, segmentation backbones",
                "VideoSwin, Swin-Unet for medical segmentation",
            ],
            tasks=[
                "Hierarchical vision backbones where global self-attention would be infeasible",
                "Dense prediction tasks needing high-resolution feature maps",
            ],
            pitfalls=[
                "On its own, no information crosses windows — must alternate with a shifted variant.",
                "Image H, W must be divisible by the window size; pad and crop carefully.",
                "Relative position bias table grows as (2w−1)² — large windows blow up parameters fast.",
            ],
            see_also=[
                "[Swin Transformer (Liu et al. 2021)](https://arxiv.org/abs/2103.14030)",
                "[Swin v2 (Liu et al. 2021)](https://arxiv.org/abs/2111.09883)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "Performer / FAVOR+ — orthogonal random feature kernel",
                "Linformer, Linear Transformers (Katharopoulos et al.)",
                "RWKV — RNN/transformer hybrid using linear-attention recurrence",
            ],
            tasks=[
                "Long-context modelling (audio, DNA, code) where T ≫ 10⁴",
                "Streaming inference where state can be kept as a fixed-size matrix",
            ],
            pitfalls=[
                "Quality lags softmax attention on language modelling — closes only with careful "
                "kernel choice and longer training.",
                "Normalisation is numerically delicate — pick a kernel φ(·) that is strictly positive.",
                "Causal masking is non-trivial — requires a per-token running sum, not a single matmul.",
            ],
            see_also=[
                "[Linear Transformers (Katharopoulos et al. 2020)](https://arxiv.org/abs/2006.16236)",
                "[Performer (Choromanski et al. 2020)](https://arxiv.org/abs/2009.14794)",
                "[RWKV (Peng et al. 2023)](https://arxiv.org/abs/2305.13048)",
            ],
        ),
    ),
    "FlashAttention": (
        "Mathematically identical to MHA, but uses tiled IO-aware kernels.",
        "(B, T, D) → (B, T, D)",
        [
            [_io("Q, K, V  (B, T, D)")],
            [_attn("flash kernel  (tiled softmax, no materialised attn matrix)")],
            [_io("y  (B, T, D)")],
        ],
        [],
        None,
        _notes(
            used_in=[
                "PyTorch 2.x `scaled_dot_product_attention` fast path",
                "All modern LLM training stacks (Megatron-LM, vLLM, TGI, Triton)",
                "Mamba's selective-scan kernel and FlashAttention-2 for ViTs",
            ],
            tasks=[
                "Replacing standard MHA in any model with sequence length > ~512 to cut memory and time",
                "Long-context fine-tuning (32k–128k tokens) that would otherwise OOM",
            ],
            pitfalls=[
                "Custom attention biases (e.g. ALiBi, T5 relative bias) may NOT be supported on all "
                "FlashAttention versions — check the kernel signature.",
                "FP32 fallback is required for some bias shapes — surprise speed cliff.",
                "Backward pass recomputes attention — increases compute by ~2× for a memory-bound win.",
            ],
            see_also=[
                "[FlashAttention (Dao et al. 2022)](https://arxiv.org/abs/2205.14135)",
                "[FlashAttention-2 (Dao 2023)](https://arxiv.org/abs/2307.08691)",
                "[FlashAttention-3 (Shah et al. 2024)](https://arxiv.org/abs/2407.08608)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "LLaMA / LLaMA-2 / LLaMA-3, Mistral, Qwen, GPT-NeoX, PaLM",
                "Most open LLMs since 2022 — RoPE has displaced learned absolute embeddings",
                "Position-aware variants for image models (RoPE-ViT)",
            ],
            tasks=[
                "Length-extrapolatable position encoding for LLMs",
                "Continuous-position retrieval and matching",
            ],
            pitfalls=[
                "Extrapolation beyond training length still degrades — mitigated by YaRN / NTK / "
                "Position Interpolation, not eliminated.",
                "Half-precision needs care: cos/sin tables in fp32, then cast at use site.",
                "RoPE rotates Q and K but NOT V — a frequent re-implementation bug.",
            ],
            see_also=[
                "[RoFormer / RoPE (Su et al. 2021)](https://arxiv.org/abs/2104.09864)",
                "[YaRN (Peng et al. 2023)](https://arxiv.org/abs/2309.00071)",
                "[NTK-aware RoPE scaling discussion](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "T5 (bucketed log-spaced relative positions)",
                "Swin Transformer (2D relative position bias)",
                "DeBERTa, ALiBi (linear bias instead of learned table)",
            ],
            tasks=[
                "Position-aware attention without losing translation equivariance",
                "Extrapolating to lengths beyond training (especially with ALiBi)",
            ],
            pitfalls=[
                "Naïve table size is O(T²); use log-bucketing or relative-distance clipping for long T.",
                "Bias shape may be incompatible with FlashAttention — check before adopting.",
                "Sharing the table across heads vs per-head is an under-appreciated knob.",
            ],
            see_also=[
                "[Self-Attention with Relative Position (Shaw et al. 2018)](https://arxiv.org/abs/1803.02155)",
                "[T5 (Raffel et al. 2020)](https://arxiv.org/abs/1910.10683)",
                "[ALiBi (Press et al. 2021)](https://arxiv.org/abs/2108.12409)",
            ],
        ),
    ),
    "AttentionPooling": (
        "A learnable query attends over a sequence to produce a single pooled vector.",
        "(B, T, D) → (B, D)",
        [
            [_io("learnable query  (1, D)"), _io("x  (B, T, D)")],
            [_attn("MHA(query, x, x)")],
            [_io("pooled  (B, D)")],
        ],
        [],
        None,
        _notes(
            used_in=[
                "CLIP text/image projection head (last-token / attention pooling variants)",
                "Set Transformer / PMA — pooling-by-multihead-attention",
                "Speech / video classification heads",
            ],
            tasks=[
                "Reducing a variable-length sequence to a single (or k) summary vector",
                "Replacing mean / max pooling for set-structured inputs",
            ],
            pitfalls=[
                "Single learnable query is a narrow bottleneck — use k > 1 queries (Set Transformer "
                "PMA) for richer summaries.",
                "Causal masking is rarely needed here, but accidentally inheriting it from a parent "
                "model breaks pooling silently.",
            ],
            see_also=[
                "[Set Transformer (Lee et al. 2018)](https://arxiv.org/abs/1810.00825)",
                "[CLIP (Radford et al. 2021)](https://arxiv.org/abs/2103.00020)",
            ],
        ),
    ),
}

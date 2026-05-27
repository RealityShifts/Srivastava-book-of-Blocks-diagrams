"""Multimodal / agentic blocks."""

from typing import Dict

from dsl import (  # noqa: F401
    Spec, _io, _op, _norm, _act, _attn, _merge, _emb, _loss, _ref, _notes,
)

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
        [],
        None,
        _notes(
            used_in=[
                "CLIP / OpenCLIP / SigLIP — image-text contrastive pretraining",
                "Stable Diffusion's text encoder (frozen CLIP ViT-L)",
                "DALL·E 2, ImageBind, BLIP-2 vision branch",
            ],
            tasks=[
                "Zero-shot image classification, retrieval, captioning",
                "Text-conditioned generation (the text tower's outputs condition diffusion / generative LMs)",
                "Cross-modal alignment as a foundation for downstream multimodal tasks",
            ],
            pitfalls=[
                "Image and text towers have DIFFERENT pretraining recipes — naïve swapping breaks alignment.",
                "Tokenizer mismatch between training and inference is the #1 silent failure.",
                "Bias in pre-training data (e.g. ImageNet-style centred subjects) leaks into downstream tasks.",
                "CLIP loss requires huge batch (32k+) for top quality; SigLIP relaxes this.",
            ],
            see_also=[
                "[CLIP (Radford et al. 2021)](https://arxiv.org/abs/2103.00020)",
                "[OpenCLIP](https://github.com/mlfoundations/open_clip)",
                "[SigLIP (Zhai et al. 2023)](https://arxiv.org/abs/2303.15343)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "Perceiver / Perceiver IO — modality-agnostic encoder",
                "Flamingo's Perceiver Resampler (downsamples video frames to fixed token count)",
                "Idefics, OpenFlamingo — open-source Flamingo replicas",
            ],
            tasks=[
                "Compressing variable-length media (video, audio, point clouds) to fixed-length tokens",
                "Cross-modal pre-pooling before passing to a language model",
            ],
            pitfalls=[
                "Number of latents L is a hard hyperparameter — too small loses info, too large defeats "
                "the purpose.",
                "Cross-attention with M ≫ L still costs O(L·M) per layer — not free.",
                "Latents are PER MODEL, not per sample — same latents broadcast across the batch.",
            ],
            see_also=[
                "[Perceiver (Jaegle et al. 2021)](https://arxiv.org/abs/2103.03206)",
                "[Perceiver IO (Jaegle et al. 2021)](https://arxiv.org/abs/2107.14795)",
                "[Flamingo (Alayrac et al. 2022)](https://arxiv.org/abs/2204.14198)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "BLIP-2 — bridges frozen ViT and frozen LLM",
                "InstructBLIP, MiniGPT-4 (early versions)",
                "X-InstructBLIP for multimodal instruction tuning",
            ],
            tasks=[
                "Connecting a frozen vision encoder to a frozen LLM with minimal trainable parameters",
                "Cross-modal alignment via a learnable bottleneck of Q queries",
            ],
            pitfalls=[
                "Q ≈ 32 queries is the BLIP-2 default — fewer harms downstream tasks.",
                "Two-stage training (contrastive then generative) is necessary; skipping the contrastive "
                "stage degrades alignment.",
                "Q-Former is small (~188M) but the vision and LM stacks aren't — most inference cost is "
                "elsewhere.",
                "Mostly superseded by simpler linear-projector designs (LLaVA, Idefics-3) for new builds.",
            ],
            see_also=[
                "[BLIP-2 (Li et al. 2023)](https://arxiv.org/abs/2301.12597)",
                "[InstructBLIP (Dai et al. 2023)](https://arxiv.org/abs/2305.06500)",
                "[LLaVA (Liu et al. 2023)](https://arxiv.org/abs/2304.08485)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "Toolformer — LM learns when to call APIs via self-supervised tags",
                "Gorilla, ToolLLM — tool-use fine-tuning of open LLMs",
                "OpenAI function calling / Anthropic tool use APIs (in spirit)",
                "ReAct / MRKL agent patterns",
            ],
            tasks=[
                "Agentic LLMs that delegate sub-tasks to calculators, search, code interpreters",
                "Modular pipelines that compose deterministic tools with neural reasoning",
            ],
            pitfalls=[
                "Soft routing rarely beats hard selection in deployed systems — most production agents "
                "do hard calls with thresholding.",
                "Tool latency dominates end-to-end response time; pipeline parallelism helps.",
                "Error propagation: failed tool returns must be turned into tokens the LM can correct from.",
            ],
            see_also=[
                "[Toolformer (Schick et al. 2023)](https://arxiv.org/abs/2302.04761)",
                "[ReAct (Yao et al. 2022)](https://arxiv.org/abs/2210.03629)",
                "[Gorilla (Patil et al. 2023)](https://arxiv.org/abs/2305.15334)",
            ],
        ),
    ),
    "MemoryAttention": (
        "Cross-attention layer that reads from an external memory bank.",
        "x:(B, T, D), mem:(M, D) → (B, T, D)",
        [
            [_io("x  (B, T, D)"), _io("memory bank  (M, D)")],
            [_attn("cross-attention over memory")],
            [_io("y  (B, T, D)")],
        ],
        [],
        None,
        _notes(
            used_in=[
                "Memorizing Transformers (Wu et al. 2022) — kNN-augmented attention",
                "Retro / Retro-fitted models",
                "Agentic memory layers (long-term episodic memory)",
            ],
            tasks=[
                "Extending effective context via retrieval-as-attention",
                "Personalisation by storing user-specific tokens in memory",
            ],
            pitfalls=[
                "Memory bank size grows over time — needs eviction or summarisation.",
                "Approximate kNN (top-k) over memory is essential for scale; exhaustive dot-product is "
                "infeasible past ~10⁶ items.",
                "Training distribution and memory distribution can diverge — periodic refresh helps.",
            ],
            see_also=[
                "[Memorizing Transformers (Wu et al. 2022)](https://arxiv.org/abs/2203.08913)",
                "[kNN-LM (Khandelwal et al. 2019)](https://arxiv.org/abs/1911.00172)",
            ],
        ),
    ),
}

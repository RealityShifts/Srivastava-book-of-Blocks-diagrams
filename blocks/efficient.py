"""Sparsity, quantisation, parallelism, low-rank."""

from typing import Dict

from dsl import (  # noqa: F401
    Spec, _io, _op, _norm, _act, _attn, _merge, _emb, _loss, _ref, _notes,
)

CATEGORY = "efficient"
CATEGORY_DESC = "Sparsity, quantisation, parallelism, low-rank."

BLOCKS: Dict[str, Spec] = {
    "QuantizedLinearInt8": (
        "Linear with weights stored in int8 + per-output-channel scale.",
        "(B, in) → (B, out)",
        [
            [_io("x  (B, in)  fp")],
            [_op("W_int8  (per-channel scale s)")],
            [_op("matmul x · W_int8")],
            [_op("× s   (dequantise)")],
            [_op("+ bias")],
            [_io("y  (B, out)")],
        ],
        [],
        None,
        _notes(
            used_in=[
                "LLM.int8() / bitsandbytes — first widely-deployed int8 LLM inference",
                "SmoothQuant — outlier-aware activation+weight quantisation",
                "PyTorch quantisation, TensorRT, ONNX Runtime",
            ],
            tasks=[
                "2× inference speedup and ~2× memory reduction on int8-capable hardware",
                "Deploying large LLMs on commodity GPUs (e.g. LLaMA-65B → single A100)",
            ],
            pitfalls=[
                "Outlier channels can blow up the dynamic range — LLM.int8() detects them and runs fp16 "
                "for those rows; SmoothQuant rebalances activations into weights.",
                "Per-channel scales are essential; per-tensor scales lose accuracy on LLMs.",
                "Activations are NOT quantised here — that step (AQA) is a separate decision.",
            ],
            see_also=[
                "[LLM.int8() (Dettmers et al. 2022)](https://arxiv.org/abs/2208.07339)",
                "[SmoothQuant (Xiao et al. 2022)](https://arxiv.org/abs/2211.10438)",
                "[bitsandbytes](https://github.com/TimDettmers/bitsandbytes)",
            ],
        ),
    ),
    "QuantizedLinear4bit": (
        "Linear with weights packed to 4-bit + grouped scales / zeros (NF4 / GPTQ style).",
        "(B, in) → (B, out)",
        [
            [_io("x  (B, in)  fp")],
            [_op("unpack 4-bit → int")],
            [_op("dequantise per group  (s, z)")],
            [_op("matmul x · W")],
            [_op("+ bias")],
            [_io("y  (B, out)")],
        ],
        [],
        None,
        _notes(
            used_in=[
                "QLoRA — NF4 storage + LoRA fine-tuning on consumer GPUs",
                "GPTQ / AWQ — post-training quantisation methods",
                "llama.cpp Q4_0 / Q4_K — ggml community quantisations",
            ],
            tasks=[
                "4× memory reduction to fit 30B+ models in 24 GB VRAM",
                "Cheap LoRA fine-tuning on top of 4-bit base weights",
            ],
            pitfalls=[
                "Group size (32 / 64 / 128) trades accuracy for memory of scales+zeros — 64–128 typical.",
                "Activation outliers degrade 4-bit quality more than int8; AWQ / GPTQ post-tune the choice.",
                "Saving 4-bit weights as 8-bit packed bytes is essential; many bugs come from accidental "
                "fp16 inflation in the loader.",
                "Backward through a 4-bit weight requires bf16 / fp16 dequant on the fly — slower than int8.",
            ],
            see_also=[
                "[QLoRA / NF4 (Dettmers et al. 2023)](https://arxiv.org/abs/2305.14314)",
                "[GPTQ (Frantar et al. 2022)](https://arxiv.org/abs/2210.17323)",
                "[AWQ (Lin et al. 2023)](https://arxiv.org/abs/2306.00978)",
            ],
        ),
    ),
    "MagnitudePruner": (
        "Zero-out weights with |W| below a threshold τ.",
        "W → sparse W'",
        [
            [_io("weights W  (out, in)")],
            [_op("|W| < τ → mask = 0")],
            [_op("W' = W ⊙ mask")],
            [_io("sparse W'  (out, in)")],
        ],
        [],
        None,
        _notes(
            used_in=[
                "Deep Compression (Han et al. 2015) — pioneering work",
                "Lottery-Ticket Hypothesis fine-tunes after magnitude pruning",
                "NVIDIA 2:4 structured-sparsity recipes (A100 sparse tensor cores)",
            ],
            tasks=[
                "Model compression for embedded / mobile deployment",
                "Studying generalisation via sparse subnetworks",
            ],
            pitfalls=[
                "Unstructured sparsity is hard to accelerate on standard hardware — go 2:4 / N:M sparse "
                "for real wall-clock gains.",
                "One-shot pruning beyond ~80 % sparsity collapses accuracy — use iterative magnitude pruning.",
                "Threshold τ per-layer matters; global magnitude pruning often outperforms per-layer.",
                "Fine-tuning after pruning is essential to recover accuracy.",
            ],
            see_also=[
                "[Deep Compression (Han et al. 2015)](https://arxiv.org/abs/1510.00149)",
                "[Lottery Ticket (Frankle & Carbin 2018)](https://arxiv.org/abs/1803.03635)",
                "[N:M Structured Sparsity (Mishra et al. 2021)](https://arxiv.org/abs/2104.08378)",
            ],
        ),
    ),
    "TokenPruner": (
        "Drop low-importance tokens before further attention layers.",
        "(B, N, D) → (B, K, D),  K < N",
        [
            [_io("tokens  (B, N, D)")],
            [_op("importance score per token")],
            [_op("top-k keep")],
            [_io("reduced tokens  (B, K, D)")],
        ],
        [],
        None,
        _notes(
            used_in=[
                "DynamicViT — token pruning in Vision Transformers",
                "EViT, TokenLearner — adaptive token reduction",
                "ToMe (Token Merging) — merges similar tokens rather than dropping",
            ],
            tasks=[
                "Cutting attention cost on long-token inputs (high-res images, long sequences)",
                "Dynamic compute allocation per sample",
            ],
            pitfalls=[
                "Hard top-k is non-differentiable — typical impls use a soft mask during training with "
                "ST-gumbel or score-based ranking, then hard-prune at inference.",
                "Dropping CLS token is catastrophic; always keep it in the keep-set.",
                "Variable token count per batch breaks fixed-shape tensors — pad and mask.",
            ],
            see_also=[
                "[DynamicViT (Rao et al. 2021)](https://arxiv.org/abs/2106.02034)",
                "[Token Merging / ToMe (Bolya et al. 2022)](https://arxiv.org/abs/2210.09461)",
            ],
        ),
    ),
    "LowRankLinear": (
        "Two stacked linears whose product approximates a full matrix W ≈ U Vᵀ.",
        "(B, in) → (B, out)",
        [
            [_io("x  (B, in)")],
            [_op("down: linear  in → r")],
            [_op("up: linear   r → out")],
            [_io("y  (B, out)")],
        ],
        [],
        None,
        _notes(
            used_in=[
                "ALBERT factorised embeddings",
                "LoRA / DoRA / VeRA — PEFT methods built on low-rank residuals",
                "Compressed inference (post-training SVD of dense layers)",
            ],
            tasks=[
                "Parameter reduction when in × out ≫ r·(in + out)",
                "Building block for adapters and intrinsic-dimension fine-tuning",
            ],
            pitfalls=[
                "Initialisation: down with N(0, σ), up zero — gives identity start when used as residual; "
                "wrong order produces gradient surprises.",
                "Rank r should be < min(in, out) for any saving; too small loses expressivity.",
                "Two matmuls have a latency cost on small GPUs vs a single fused matmul.",
            ],
            see_also=[
                "[ALBERT (Lan et al. 2019)](https://arxiv.org/abs/1909.11942)",
                "[LoRA (Hu et al. 2021)](https://arxiv.org/abs/2106.09685)",
                "[Intrinsic Dimensionality (Aghajanyan et al. 2020)](https://arxiv.org/abs/2012.13255)",
            ],
        ),
    ),
    "ColumnParallelLinear": (
        "Linear whose output columns are sharded across tensor-parallel ranks.",
        "x → y  (gathered)",
        [
            [_io("x  (B, in)  replicated")],
            [_op("split W cols across ranks")],
            [_op("local matmul")],
            [_op("all-gather")],
            [_io("y  (B, out)  gathered")],
        ],
        [],
        None,
        _notes(
            used_in=[
                "Megatron-LM tensor parallelism (TP) for LLM training",
                "DeepSpeed, FasterTransformer, vLLM TP for inference",
            ],
            tasks=[
                "Sharding the Q/K/V/O linears of a Transformer across N GPUs",
                "Fitting layers that don't fit in a single GPU's memory",
            ],
            pitfalls=[
                "Pair with RowParallelLinear immediately after (Q→attn→O) so the all-gather and all-reduce "
                "cancel out — Megatron's clever design.",
                "Bias is duplicated across ranks by default — be careful with weight decay accounting.",
                "Sequence-parallel variant reduces activation memory by another factor of N.",
            ],
            see_also=[
                "[Megatron-LM (Shoeybi et al. 2019)](https://arxiv.org/abs/1909.08053)",
            ],
        ),
    ),
    "RowParallelLinear": (
        "Linear whose input is sharded across ranks; outputs all-reduced.",
        "x  (sharded) → y",
        [
            [_io("x  (B, in/r)  sharded")],
            [_op("local matmul")],
            [_op("all-reduce sum")],
            [_io("y  (B, out)")],
        ],
        [],
        None,
        _notes(
            used_in=[
                "Megatron-LM TP (the partner of ColumnParallelLinear)",
                "Inference engines (TensorRT-LLM, vLLM) with TP > 1",
            ],
            tasks=[
                "Completing the column-then-row sharding pattern within a single Transformer sublayer",
                "Reducing per-GPU memory and compute for very wide hidden states",
            ],
            pitfalls=[
                "All-reduce is bandwidth-bound — NVLink between GPUs is far better than PCIe.",
                "Don't all-reduce inside fp16 / fp8 — cast to fp32 for the reduction or use higher-precision "
                "all-reduce APIs.",
                "Backward also needs an all-reduce; total comm cost is 2× the forward.",
            ],
            see_also=[
                "[Megatron-LM (Shoeybi et al. 2019)](https://arxiv.org/abs/1909.08053)",
            ],
        ),
    ),
    "PipelineStage": (
        "One stage of pipeline parallelism — receives micro-batches, computes, sends forward.",
        "x_chunk → y_chunk",
        [
            [_io("x_chunk  (μb, …)")],
            [_op("stage_i forward")],
            [_op("send to stage_{i+1}")],
            [_io("y_chunk  (μb, …)")],
        ],
        [],
        None,
        _notes(
            used_in=[
                "GPipe — micro-batched pipeline",
                "PipeDream / Megatron-LM 1F1B — interleaved schedule",
                "Production LLM training on >100 GPUs",
            ],
            tasks=[
                "Scaling beyond a single tensor-parallel group",
                "Training models too large to fit on TP alone (combining PP × TP × DP × ZeRO)",
            ],
            pitfalls=[
                "Pipeline bubbles — first and last microbatches see idle GPUs; smaller microbatches help.",
                "Activation memory grows with number of in-flight microbatches; 1F1B keeps it bounded.",
                "Pipeline stages must be balanced in compute; an imbalance bottlenecks the whole pipe.",
                "Send/recv ordering bugs cause silent hangs that look like NCCL timeouts.",
            ],
            see_also=[
                "[GPipe (Huang et al. 2018)](https://arxiv.org/abs/1811.06965)",
                "[PipeDream (Narayanan et al. 2018)](https://arxiv.org/abs/1806.03377)",
                "[Megatron-LM scheduling (Narayanan et al. 2021)](https://arxiv.org/abs/2104.04473)",
            ],
        ),
    ),
}

"""Sparsity, quantisation, parallelism, low-rank."""

from typing import Dict

from dsl import Spec, _io, _op, _norm, _act, _attn, _merge, _emb, _loss, _ref  # noqa: F401

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
    ),
}

"""Optimisers, schedulers, EMA, mixed-precision, checkpointing."""

from typing import Dict

from dsl import Spec, _io, _op, _norm, _act, _attn, _merge, _emb, _loss, _ref  # noqa: F401

CATEGORY = "optimization"
CATEGORY_DESC = "Optimisers, schedulers, EMA, mixed-precision, checkpointing."

BLOCKS: Dict[str, Spec] = {
    "Lion": (
        "Sign-of-momentum optimiser (Chen et al. 2023).",
        "g, θ → θ'",
        [
            [_io("grad g  (params)")],
            [_op("m ← β₁·m + (1 − β₁)·g")],
            [_op("update = sign(β₂·m + (1 − β₂)·g)")],
            [_op("θ ← θ − lr · update")],
            [_io("θ'  (params)")],
        ],
    ),
    "Sophia": (
        "Hessian-clipped second-order optimiser.",
        "g, h, θ → θ'",
        [
            [_io("g, h  (params)  (Hessian estimate)")],
            [_op("m ← β₁·m + (1 − β₁)·g")],
            [_op("update = clip(m / max(h, ε), ρ)")],
            [_op("θ ← θ − lr · update")],
            [_io("θ'  (params)")],
        ],
    ),
    "EMA": (
        "Exponential moving average of model weights.",
        "θ_online → θ_ema",
        [
            [_io("θ_online  (params)")],
            [_op("θ_ema ← τ · θ_ema + (1 − τ) · θ_online")],
            [_io("θ_ema  (params)")],
        ],
    ),
    "MixedPrecisionTrainer": (
        "Loss-scaled fp16 forward, fp32 master weights, gradient unscale + clip + step.",
        "x → updated θ",
        [
            [_io("x  (B, …)")],
            [_op("fp16 forward")],
            [_op("loss × scale")],
            [_op("backward (fp32 master grads)")],
            [_op("unscale + clip")],
            [_op("optimizer step")],
            [_io("updated θ  (params)")],
        ],
    ),
    "CheckpointedSequential": (
        "Sequential whose forward is rematerialised on the backward pass to save memory.",
        "(*) → (*)",
        [
            [_io("x  (*)")],
            [_op("layer 1   (recompute on backward)")],
            [_op("layer 2   (recompute on backward)")],
            [_op("…  (k layers)")],
            [_io("y  (*)")],
        ],
    ),
}

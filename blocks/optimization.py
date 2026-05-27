"""Optimisers, schedulers, EMA, mixed-precision, checkpointing."""

from typing import Dict

from dsl import (  # noqa: F401
    Spec, _io, _op, _norm, _act, _attn, _merge, _emb, _loss, _ref, _notes,
)

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
        [],
        None,
        _notes(
            used_in=[
                "Google's internal ViT-22B, JFT-classification training (Chen et al. 2023)",
                "Open replications on ViT, diffusion, LLM pretraining",
                "Production CTR models at Google",
            ],
            tasks=[
                "Memory-efficient training — keeps only momentum, not Adam's second moment (≈ 1/3 less state)",
                "Large-batch pretraining where Adam's update magnitude is unstable",
            ],
            pitfalls=[
                "Learning rate must be ~3–10× SMALLER than Adam's because sign(·) has unit per-coord magnitude.",
                "Sensitive to weight decay — decoupled WD (AdamW-style) is essential.",
                "Gains shrink at small batch sizes; original paper reports best at large-batch regimes.",
            ],
            see_also=[
                "[Lion / Symbolic Discovery (Chen et al. 2023)](https://arxiv.org/abs/2302.06675)",
                "[Lion implementation](https://github.com/google/automl/tree/master/lion)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "GPT-2 125M..1.5B reproductions (claim ~2× speedup vs AdamW)",
                "Research code-bases experimenting with second-order methods at scale",
            ],
            tasks=[
                "Language-model pretraining where Hessian heterogeneity across dims is large",
                "Reducing the compute / wall-clock budget for matching a target perplexity",
            ],
            pitfalls=[
                "Hessian diagonal estimate via Hutchinson is noisy — must average across many steps "
                "(default: every k iterations).",
                "Clipping bound ρ matters — too tight blocks progress, too loose loses the second-order signal.",
                "Independent reproductions show smaller gains than the paper at very large scale.",
                "Per-step overhead is small only because Hessian is updated INTERMITTENTLY — implementations "
                "that update every step are much slower.",
            ],
            see_also=[
                "[Sophia (Liu et al. 2023)](https://arxiv.org/abs/2305.14342)",
                "[Sophia implementation](https://github.com/Liuhong99/Sophia)",
            ],
        ),
    ),
    "EMA": (
        "Exponential moving average of model weights.",
        "θ_online → θ_ema",
        [
            [_io("θ_online  (params)")],
            [_op("θ_ema ← τ · θ_ema + (1 − τ) · θ_online")],
            [_io("θ_ema  (params)")],
        ],
        [],
        None,
        _notes(
            used_in=[
                "BYOL / MoCo — momentum-encoder target",
                "Diffusion models — EMA of weights used for SAMPLING (essential for FID)",
                "Mean-teacher semi-supervised learning",
                "Reinforcement-learning target networks (effectively a hard-step EMA)",
            ],
            tasks=[
                "Stabilising self-supervised / generative training",
                "Improving generalisation by averaging across the training trajectory",
                "Decoupling sampling-time weights from training weights",
            ],
            pitfalls=[
                "EMA copy doubles the parameter memory — be aware on large models.",
                "Decay τ near 1 (e.g. 0.9999) is typical for diffusion; lower for SSL momentum encoders.",
                "Don't average buffers (batch-norm running stats, optimizer state) — only weights.",
                "Loading EMA weights into the training model at resume is a common bug.",
            ],
            see_also=[
                "[Polyak averaging (Polyak & Juditsky 1992)](https://epubs.siam.org/doi/10.1137/0330046)",
                "[BYOL (Grill et al. 2020)](https://arxiv.org/abs/2006.07733)",
                "[Improved DDPM (Nichol & Dhariwal 2021)](https://arxiv.org/abs/2102.09672)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "torch.cuda.amp / torch.amp",
                "DeepSpeed, Megatron-LM training stacks",
                "JAX with bfloat16 / float16 mixed precision",
            ],
            tasks=[
                "Doubling training throughput on tensor-core hardware (V100, A100, H100)",
                "Halving memory usage to fit larger batch / model sizes",
            ],
            pitfalls=[
                "fp16 underflow on small gradients — loss scaling required (or use bf16 with no loss scale).",
                "Overflow → dynamic scale halves; thrashing scales hurt convergence — log them.",
                "BatchNorm and softmax should stay fp32 — autocast handles this, manual impls don't.",
                "bf16 has wider range than fp16, narrower mantissa — usually preferred on A100+ and TPUs.",
            ],
            see_also=[
                "[Mixed Precision Training (Micikevicius et al. 2017)](https://arxiv.org/abs/1710.03740)",
                "[bfloat16 (Wang & Kanwar 2019)](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "Training of huge models (LLaMA, GPT-NeoX, Megatron-LM)",
                "Diffusion U-Nets and ViT-Huge fine-tuning",
                "PyTorch's `torch.utils.checkpoint`",
            ],
            tasks=[
                "Fitting deeper / wider models into a fixed memory budget",
                "Long-context training where activation memory is the binding constraint",
            ],
            pitfalls=[
                "~30 % slower per step due to the extra forward — only worth it if memory was the bottleneck.",
                "Doesn't compose well with non-determinism (dropout) unless you set `use_reentrant=False` "
                "or seed correctly.",
                "Checkpointing across optimizer step is wrong — only the FORWARD activations are rematerialised.",
                "Selective Activation Checkpointing (Megatron-LM) typically beats blind layer-wise CKPT.",
            ],
            see_also=[
                "[Gradient Checkpointing (Chen et al. 2016)](https://arxiv.org/abs/1604.06174)",
                "[Megatron Selective Activation Recompute (Korthikanti et al. 2022)](https://arxiv.org/abs/2205.05198)",
            ],
        ),
    ),
}

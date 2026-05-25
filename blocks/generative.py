"""VAE, autoregressive, normalising-flow, EBM, diffusion schedulers."""

from typing import Dict

from dsl import Spec, _io, _op, _norm, _act, _attn, _merge, _emb, _loss, _ref  # noqa: F401

CATEGORY = "generative"
CATEGORY_DESC = "VAE, autoregressive, normalising-flow, EBM, diffusion schedulers."

BLOCKS: Dict[str, Spec] = {
    "VAE": (
        "Encoder → posterior (μ, σ) → reparameterised z → decoder.",
        "(B, C, H, W) → (B, C, H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("encoder")],
            [_op("linear → (μ, log σ²)")],
            [_op("z = μ + ε ⊙ σ,   ε ∼ N(0, I)")],
            [_op("decoder")],
            [_io("x̂  (B, C, H, W)")],
        ],
    ),
    "MaskedConv2d": (
        "Conv2d with an autoregressive mask preventing each pixel from seeing future pixels (PixelCNN).",
        "(B, C, H, W) → (B, C', H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("Conv  (kernel multiplied by AR mask)")],
            [_io("y  (B, C', H, W)")],
        ],
    ),
    "AutoregressiveBlock": (
        "Stack of masked convs producing a per-pixel distribution over the next pixel.",
        "(B, C, H, W) → (B, K, H, W)  logits",
        [
            [_io("x  (B, C, H, W)")],
            [_op("MaskedConv  type A")],
            [_op("MaskedConv  type B  × N")],
            [_op("Conv 1×1 → logits")],
            [_io("logits  (B, K, H, W)")],
        ],
    ),
    "AffineCouplingLayer": (
        "Normalising-flow coupling: split, predict scale/shift from one half, transform the other.",
        "x = (a, b) → y = (a, b · exp(s) + t)",
        [
            [_io("x = (a, b)  (B, D)")],
            [_op("NN(a) → (s, t)")],
            [_op("a' = a,  b' = b ⊙ exp(s) + t")],
            [_io("y = (a', b')  (B, D)")],
        ],
    ),
    "EnergyBasedModel": (
        "A network that maps x to a scalar energy E(x); samples drawn via Langevin/MCMC.",
        "x → E(x):(B,)",
        [
            [_io("x  (B, …)")],
            [_op("network")],
            [_io("E(x)  (B,)")],
        ],
    ),
    "DDPMScheduler": (
        "Forward diffusion: x_t = √α̅_t · x_0 + √(1 − α̅_t) · ε.",
        "x_0, t → x_t",
        [
            [_io("x_0  (B, C, H, W)"), _io("t  (B,)")],
            [_op("β schedule → α̅_t = ∏(1 − β)")],
            [_op("x_t = √α̅_t · x_0 + √(1 − α̅_t) · ε")],
            [_io("x_t  (B, C, H, W)")],
        ],
    ),
    "DDIMScheduler": (
        "Deterministic non-Markovian sampler that reuses the predicted x̂_0.",
        "x_t, ε̂ → x_{t-1}",
        [
            [_io("x_t  (B, C, H, W)"), _io("ε̂  (B, C, H, W)")],
            [_op("x̂_0 = (x_t − √(1 − α̅_t) ε̂) / √α̅_t")],
            [_op("x_{t−1} = √α̅_{t−1} · x̂_0 + √(1 − α̅_{t−1}) · ε̂")],
            [_io("x_{t−1}  (B, C, H, W)")],
        ],
    ),
}

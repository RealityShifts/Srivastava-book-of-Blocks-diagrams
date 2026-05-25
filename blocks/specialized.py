"""Specialised research blocks (NeuralODE, FNO, KAN, capsules, slots)."""

from typing import Dict

from dsl import Spec, _io, _op, _norm, _act, _attn, _merge, _emb, _loss, _ref  # noqa: F401

CATEGORY = "specialized"
CATEGORY_DESC = "Specialised research blocks (NeuralODE, FNO, KAN, capsules, slots)."

BLOCKS: Dict[str, Spec] = {
    "NeuralODE": (
        "Treat depth as continuous time and integrate dx/dt = f_θ(x, t).",
        "x_0 → x_T",
        [
            [_io("x_0  (B, D)"), _io("f_θ(x, t)  (·)→(·)")],
            [_op("ODE solver  (Euler / RK4 / dopri5)")],
            [_op("x(T) = x_0 + ∫₀ᵀ f dt")],
            [_io("x_T  (B, D)")],
        ],
    ),
    "SpectralConv2d": (
        "Multiply low-frequency Fourier modes by learned weights (Fourier Neural Operator).",
        "(B, C, H, W) → (B, C', H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("FFT2")],
            [_op("× learned weights  (low modes only)")],
            [_op("IFFT2")],
            [_io("y  (B, C', H, W)")],
        ],
    ),
    "FNOBlock": (
        "FNO block: spectral conv + 1×1 conv, summed and activated.",
        "(B, C, H, W) → (B, C, H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_ref("SpectralConv2d"), _op("1×1 Conv")],
            [_merge("+")],
            [_act("GELU")],
            [_io("y  (B, C, H, W)")],
        ],
    ),
    "KANLayer": (
        "Kolmogorov–Arnold layer: every edge has a learned univariate spline; node = sum of edges.",
        "(B, in) → (B, out)",
        [
            [_io("x  (B, in)")],
            [_op("learned spline  φ_ij(x_i)  per edge")],
            [_op("sum over inputs at each output node")],
            [_io("y  (B, out)")],
        ],
    ),
    "CapsuleLayer": (
        "Capsule routing layer (Sabour et al.): predict votes, dynamic routing-by-agreement, squash.",
        "primary caps → output caps",
        [
            [_io("primary capsules u_i  (N₁, d₁)")],
            [_op("votes  û_{j|i} = W_ij · u_i")],
            [_op("dynamic routing × T")],
            [_op("squash")],
            [_io("output capsules v_j  (N₂, d₂)")],
        ],
    ),
    "SlotAttention": (
        "Iterated cross-attention from a small set of slots into per-pixel features (SlotAttention).",
        "feats:(B, N, D), slots:(B, K, D) → slots",
        [
            [_io("features  (B, N, D)"), _io("slots  (B, K, D)")],
            [_attn("cross-attn  (slots query)")],
            [_op("GRU update")],
            [_op("× T iterations")],
            [_io("updated slots  (B, K, D)")],
        ],
    ),
}

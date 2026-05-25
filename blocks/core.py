"""Core neural-network primitives."""

from typing import Dict

from dsl import Spec, _io, _op, _norm, _act, _attn, _merge, _emb, _loss, _ref  # noqa: F401

CATEGORY = "core"
CATEGORY_DESC = "Core neural-network primitives."

BLOCKS: Dict[str, Spec] = {
    "Linear": (
        "Affine projection y = x · Wᵀ + b.",
        "(B, in) → (B, out)",
        [
            [_io("x  (B, in)")],
            [_op("matmul  x · Wᵀ")],
            [_op("+ bias")],
            [_io("y  (B, out)")],
        ],
    ),
    "ConvBlock": (
        "Conv → Norm → Activation, the canonical CNN unit.",
        "(B, C, H, W) → (B, C', H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("Conv 2D")],
            [_norm("BatchNorm / GroupNorm / LayerNorm")],
            [_act("ReLU / GELU / SiLU / Mish")],
            [_io("y  (B, C', H, W)")],
        ],
    ),
    "DepthwiseSeparableConv2d": (
        "Depthwise spatial conv followed by 1×1 pointwise conv (MobileNet).",
        "(B, C, H, W) → (B, C', H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("Depthwise Conv K×K  (groups=C)")],
            [_op("Pointwise Conv 1×1  (C → C')")],
            [_io("y  (B, C', H, W)")],
        ],
    ),
    "DilatedConv2d": (
        "Standard conv with dilation > 1 (atrous), enlarges receptive field for free.",
        "(B, C, H, W) → (B, C', H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("Conv K×K  (dilation = d)")],
            [_io("y  (B, C', H, W)")],
        ],
    ),
    "GroupConv2d": (
        "Conv2d with channel groups (ResNeXt cardinality).",
        "(B, C, H, W) → (B, C', H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("Conv K×K  (groups = g)")],
            [_io("y  (B, C', H, W)")],
        ],
    ),
    "Conv1d": (
        "1-D convolution wrapper.",
        "(B, C, T) → (B, C', T)",
        [
            [_io("x  (B, C, T)")],
            [_op("Conv 1D K  (stride, padding)")],
            [_io("y  (B, C', T)")],
        ],
    ),
    "Conv3d": (
        "3-D convolution wrapper.",
        "(B, C, T, H, W) → (B, C', T, H, W)",
        [
            [_io("x  (B, C, T, H, W)")],
            [_op("Conv 3D K  (stride, padding)")],
            [_io("y  (B, C', T, H, W)")],
        ],
    ),
    "Mish": (
        "Self-gated activation: x · tanh(softplus(x)).",
        "(*) → (*)",
        [
            [_io("x  (*)")],
            [_op("softplus(x)")],
            [_op("tanh(·)")],
            [_merge("× x")],
            [_io("y  (*)")],
        ],
        [(0, 0, 3, 0)],
    ),
    "RMSNorm": (
        "Root-mean-square normalisation, scale-only (no mean subtraction).",
        "(*, D) → (*, D)",
        [
            [_io("x  (*, D)")],
            [_op("RMS = √mean(x²)")],
            [_op("x / (RMS + ε)")],
            [_op("× learned g")],
            [_io("y  (*, D)")],
        ],
    ),
    "AdaIN": (
        "Adaptive Instance Normalisation: replace x's per-channel stats with style stats.",
        "x:(B, C, H, W), s:(B, C) → y:(B, C, H, W)",
        [
            [_io("x  content  (B, C, H, W)"), _io("s  style  (B, C)")],
            [_op("μ_x, σ_x  per channel"), _op("γ, β = MLP(s)")],
            [_op("(x − μ_x) / σ_x  · γ + β")],
            [_io("y  (B, C, H, W)")],
        ],
    ),
    "SPADE": (
        "Spatially-adaptive denormalisation: γ, β come from a segmentation map.",
        "x:(B, C, H, W), seg:(B, K, H, W) → y:(B, C, H, W)",
        [
            [_io("x  (B, C, H, W)"), _io("seg map  (B, K, H, W)")],
            [_norm("BatchNorm(x)"), _op("Conv → γ(spatial), β(spatial)")],
            [_op("γ · x_norm + β")],
            [_io("y  (B, C, H, W)")],
        ],
    ),
    "ResidualBlock": (
        "Two conv-norm-act stack with identity skip and post-add activation.",
        "(B, C, H, W) → (B, C, H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("Conv 3×3")],
            [_norm("Norm")],
            [_act("ReLU")],
            [_op("Conv 3×3")],
            [_norm("Norm")],
            [_merge("+")],
            [_act("ReLU")],
            [_io("y  (B, C, H, W)")],
        ],
        [(0, 0, 6, 0)],
    ),
    "SkipConnection": (
        "Generic identity skip around any sub-module f.",
        "(*) → (*)",
        [
            [_io("x  (*)")],
            [_op("f(x)")],
            [_merge("+")],
            [_io("y  (*)")],
        ],
        [(0, 0, 2, 0)],
    ),
}

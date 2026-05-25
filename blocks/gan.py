"""GAN building blocks: StyleGAN, PGGAN, equalised LR."""

from typing import Dict

from dsl import Spec, _io, _op, _norm, _act, _attn, _merge, _emb, _loss, _ref  # noqa: F401

CATEGORY = "gan"
CATEGORY_DESC = "GAN building blocks: StyleGAN, PGGAN, equalised LR."

BLOCKS: Dict[str, Spec] = {
    "EqualLinear": (
        "Linear with equalized learning rate: weight scaled at runtime by gain/√fan_in.",
        "(B, in) → (B, out)",
        [
            [_io("x  (B, in)")],
            [_op("matmul  x · (W · s)")],
            [_op("+ bias · lr_mul")],
            [_io("y  (B, out)")],
        ],
    ),
    "EqualConv2d": (
        "Conv2d with equalized learning rate (StyleGAN family).",
        "(B, C, H, W) → (B, C', H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("Conv with W · (gain/√(k²·C))")],
            [_op("+ bias")],
            [_io("y  (B, C', H, W)")],
        ],
    ),
    "GeneratorBlock": (
        "Vanilla generator block: upsample → conv → norm → activation.",
        "(B, C, H, W) → (B, C', 2H, 2W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("Upsample 2×")],
            [_op("Conv 3×3")],
            [_norm("BatchNorm")],
            [_act("ReLU")],
            [_io("y  (B, C', 2H, 2W)")],
        ],
    ),
    "DiscriminatorBlock": (
        "Vanilla discriminator block: strided conv → norm → leaky ReLU.",
        "(B, C, H, W) → (B, C', H/2, W/2)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("Conv 3×3, stride 2")],
            [_norm("InstanceNorm")],
            [_act("LeakyReLU 0.2")],
            [_io("y  (B, C', H/2, W/2)")],
        ],
    ),
    "MappingNetwork": (
        "8-layer MLP with reduced LR that maps z → w (StyleGAN).",
        "z:(B, D) → w:(B, D)",
        [
            [_io("z  (B, D)")],
            [_op("PixelNorm")],
            [_op("EqualLinear × 8  (lr_mul = 0.01)")],
            [_io("w  (B, D)")],
        ],
    ),
    "StyleBlock": (
        "Conv + per-pixel noise + AdaIN modulated by style w (StyleGAN v1).",
        "x:(B, C, H, W), w:(B, D) → y",
        [
            [_io("x  (B, C, H, W)"), _io("w  (B, D)")],
            [_op("Conv 3×3"), _op("A: linear w → (γ, β)")],
            [_op("+ Gaussian noise · learned scale")],
            [_op("AdaIN(γ, β)")],
            [_act("LeakyReLU")],
            [_io("y  (B, C, H, W)")],
        ],
    ),
    "ModulatedConv2d": (
        "StyleGAN2 modulated conv: scale weights by style, demodulate, then convolve.",
        "x:(B, C, H, W), w:(B, D) → y",
        [
            [_io("x  (B, C, H, W)"), _io("w  (B, D)")],
            [_op("weights W"), _op("A: linear w → s")],
            [_op("W' = W · s")],
            [_op("demod: W'' = W' / ||W'||")],
            [_op("Conv with W''")],
            [_io("y  (B, C', H, W)")],
        ],
    ),
    "MinibatchStdDev": (
        "Append a per-batch standard-deviation map as an extra channel (PGGAN/StyleGAN).",
        "(B, C, H, W) → (B, C+1, H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("std over batch  (per channel, per pixel)")],
            [_op("average → scalar")],
            [_op("tile → (B, 1, H, W)")],
            [_merge("concat as new channel")],
            [_io("y  (B, C+1, H, W)")],
        ],
        [(0, 0, 4, 0)],
    ),
    "ProgressiveGrowing": (
        "Fade in a new high-res block via α-blend with the previous resolution (PGGAN).",
        "x → upsampled / new block → y",
        [
            [_io("x  (B, C, H, W)")],
            [_op("old layers (already trained)")],
            [_op("new high-res layer")],
            [_merge("(1−α) old  +  α new")],
            [_io("y  (B, C', 2H, 2W)")],
        ],
        [(1, 0, 3, 0)],
    ),
}

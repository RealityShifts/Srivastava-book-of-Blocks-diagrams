"""CNN and vision-specific blocks."""

from typing import Dict

from dsl import Spec, _io, _op, _norm, _act, _attn, _merge, _emb, _loss, _ref  # noqa: F401

CATEGORY = "cnn_vision"
CATEGORY_DESC = "CNN and vision-specific blocks."

BLOCKS: Dict[str, Spec] = {
    "InceptionBlock": (
        "Parallel multi-scale convs concatenated channel-wise (GoogLeNet).",
        "(B, C, H, W) → (B, C', H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("1×1 Conv"), _op("1×1 → 3×3"), _op("1×1 → 5×5"), _op("3×3 MaxPool → 1×1")],
            [_merge("concat (channel)")],
            [_io("y  (B, C', H, W)")],
        ],
    ),
    "DenseBlock": (
        "Each layer's input is the concatenation of all earlier layers' outputs (DenseNet).",
        "(B, C, H, W) → (B, C + k·L, H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("BN-ReLU-Conv  (layer 1)")],
            [_merge("concat with x")],
            [_op("BN-ReLU-Conv  (layer 2)")],
            [_merge("concat with all prev")],
            [_op("…  (L layers)")],
            [_io("y  (B, C + k·L, H, W)")],
        ],
    ),
    "SqueezeExcitation": (
        "Channel-wise gating via global pool → bottleneck MLP → sigmoid.",
        "(B, C, H, W) → (B, C, H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("Global AvgPool")],
            [_op("FC down  (C → C/r)")],
            [_act("ReLU")],
            [_op("FC up  (C/r → C)")],
            [_act("Sigmoid")],
            [_merge("· x  (channel scale)")],
            [_io("y  (B, C, H, W)")],
        ],
        [(0, 0, 6, 0)],
    ),
    "CBAM": (
        "Sequential channel attention then spatial attention.",
        "(B, C, H, W) → (B, C, H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("Channel Attention  (MLP on max+avg pool)")],
            [_merge("· x")],
            [_op("Spatial Attention  (Conv on max+avg over C)")],
            [_merge("· x")],
            [_io("y  (B, C, H, W)")],
        ],
        [(0, 0, 2, 0), (2, 0, 4, 0)],
    ),
    "SpatialPyramidPooling": (
        "Multi-scale fixed-output pooling, concatenated.",
        "(B, C, H, W) → (B, C·Σbins²,)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("AvgPool 1×1"), _op("AvgPool 2×2"), _op("AvgPool 4×4")],
            [_merge("flatten + concat")],
            [_io("y  (B, C·Σbins²,)")],
        ],
    ),
    "FeaturePyramidNetwork": (
        "Build multi-scale feature maps via top-down upsampling + lateral 1×1 connections.",
        "(C3, C4, C5) → (P3, P4, P5)",
        [
            [_io("C3, C4, C5  (B, Cᵢ, H/2ⁱ, W/2ⁱ)")],
            [_op("1×1 lateral on each level")],
            [_op("top-down: upsample + add")],
            [_op("3×3 smooth")],
            [_io("P3, P4, P5  (B, C, H/2ⁱ, W/2ⁱ)")],
        ],
    ),
    "ASPP": (
        "Atrous Spatial Pyramid Pooling: parallel atrous convs + image pool, concatenated.",
        "(B, C, H, W) → (B, C', H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("1×1 Conv"), _op("3×3 dil=6"), _op("3×3 dil=12"), _op("3×3 dil=18"), _op("Image pool")],
            [_merge("concat")],
            [_op("1×1 Conv")],
            [_io("y  (B, C', H, W)")],
        ],
    ),
    "PixelShuffleUpsample": (
        "Sub-pixel upsampling: rearrange r² channels into r×r spatial blocks.",
        "(B, C, H, W) → (B, C, r·H, r·W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("Conv → C·r² channels")],
            [_op("PixelShuffle r")],
            [_io("y  (B, C, r·H, r·W)")],
        ],
    ),
    "DeformableConv2d": (
        "Conv whose sample locations are shifted by a learned offset field.",
        "(B, C, H, W) → (B, C', H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("offset Conv → Δp")],
            [_op("bilinear sample at p + Δp")],
            [_op("Conv on sampled features")],
            [_io("y  (B, C', H, W)")],
        ],
    ),
    "DeformableAttention": (
        "Attention that samples a small set of keys at learned offsets per query (Deformable DETR).",
        "(B, N, C) → (B, N, C)",
        [
            [_io("x  (B, N, C)")],
            [_op("MLP → reference + sampling offsets")],
            [_op("bilinear sample features")],
            [_attn("attention weights (per head, per point)")],
            [_merge("weighted sum + output proj")],
            [_io("y  (B, N, C)")],
        ],
    ),
}

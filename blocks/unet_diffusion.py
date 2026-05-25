"""UNet, time conditioning, ControlNet, LoRA, hypernets."""

from typing import Dict

from dsl import Spec, _io, _op, _norm, _act, _attn, _merge, _emb, _loss, _ref  # noqa: F401

CATEGORY = "unet_diffusion"
CATEGORY_DESC = "UNet, time conditioning, ControlNet, LoRA, hypernets."

BLOCKS: Dict[str, Spec] = {
    "SinusoidalTimeEmbedding": (
        "Sin/cos positional embedding of the diffusion timestep t.",
        "t:(B,) → emb:(B, D)",
        [
            [_io("t  (B,)")],
            [_op("freqs = 10000^(−2i/D)")],
            [_op("[sin(t·f),  cos(t·f)]")],
            [_emb("emb  (B, D)")],
        ],
    ),
    "TimestepMLP": (
        "Two-layer MLP applied to the time embedding before injection.",
        "(B, D) → (B, D')",
        [
            [_io("t_emb  (B, D)")],
            [_op("linear")],
            [_act("SiLU")],
            [_op("linear")],
            [_io("y  (B, D')")],
        ],
    ),
    "DownsampleBlock": (
        "Spatial downsample by 2× via a strided 3×3 conv.",
        "(B, C, H, W) → (B, C, H/2, W/2)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("Conv 3×3, stride 2")],
            [_io("y  (B, C, H/2, W/2)")],
        ],
    ),
    "UpsampleBlock": (
        "Nearest-neighbour upsample by 2× followed by a 3×3 conv.",
        "(B, C, H, W) → (B, C, 2H, 2W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("nearest upsample 2×")],
            [_op("Conv 3×3")],
            [_io("y  (B, C, 2H, 2W)")],
        ],
    ),
    "UNetResBlock": (
        "Residual block conditioned on a time embedding (added between the two convs).",
        "x:(B, C, H, W), t:(B, D) → (B, C', H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_norm("GroupNorm")],
            [_act("SiLU")],
            [_op("Conv 3×3")],
            [_merge("+ MLP(t_emb)  (broadcast)")],
            [_norm("GroupNorm")],
            [_act("SiLU")],
            [_op("Conv 3×3")],
            [_merge("+")],
            [_io("y  (B, C', H, W)")],
        ],
        [(0, 0, 8, 0)],
    ),
    "UNet": (
        "Encoder-decoder with skip connections at matching resolutions; bottleneck attention.",
        "(B, C, H, W) → (B, C, H, W)",
        [
            [_io("x  noised  (B, C, H, W)")],
            [_op("Encoder Block 1")],
            [_op("Down 2×")],
            [_op("Encoder Block 2")],
            [_op("Down 2×")],
            [_op("Bottleneck (+ self-attn)")],
            [_op("Up 2×  + skip")],
            [_op("Decoder Block 2")],
            [_op("Up 2×  + skip")],
            [_op("Decoder Block 1")],
            [_op("Conv → ε̂")],
            [_io("y  noise pred  (B, C, H, W)")],
        ],
        [(1, 0, 9, 0), (3, 0, 7, 0)],
    ),
    "NoisePredictor": (
        "End-to-end ε-prediction network used by DDPM/DDIM samplers.",
        "x_t:(B, C, H, W), t:(B,) → ε̂:(B, C, H, W)",
        [
            [_io("x_t  (B, C, H, W)"), _io("t  (B,)")],
            [_op("image stem"), _ref("SinusoidalTimeEmbedding")],
            [_ref("UNet")],
            [_io("ε̂  (B, C, H, W)")],
        ],
    ),
    "ZeroConv2d": (
        "Conv2d initialised to zero — outputs zero at start so it can be added safely.",
        "(B, C, H, W) → (B, C', H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("Conv  (W = 0, b = 0 at init)")],
            [_io("y  (B, C', H, W)  (= 0 at init)")],
        ],
    ),
    "ControlNetBlock": (
        "Trainable copy of UNet encoder + ZeroConvs; outputs are added to the frozen UNet decoder.",
        "x, control → ΔUNet decoder features",
        [
            [_io("x  noised  (B, C, H, W)"), _io("control image  (B, 3, H, W)")],
            [_op("frozen UNet encoder"), _op("trainable encoder copy")],
            [_op("ZeroConv on each level")],
            [_merge("add into UNet decoder skips")],
            [_io("Δ feats  (multi-scale, matches UNet decoder)")],
        ],
    ),
    "LoRALinear": (
        "Frozen linear plus a low-rank residual B·A·x scaled by α/r.",
        "(B, in) → (B, out)",
        [
            [_io("x  (B, in)")],
            [_op("frozen W · x"), _op("A · x   (in → r)")],
            [_op("identity"), _op("B · (A · x)   (r → out)")],
            [_merge("+  α/r ·")],
            [_io("y  (B, out)")],
        ],
    ),
    "LoRAConv2d": (
        "Same low-rank residual idea applied to 2-D convolutions.",
        "(B, C, H, W) → (B, C', H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("frozen Conv"), _op("Conv A  (C → r, k×k)")],
            [_op("identity"), _op("Conv B  (r → C', 1×1)")],
            [_merge("+  α/r ·")],
            [_io("y  (B, C', H, W)")],
        ],
    ),
    "HyperNetwork": (
        "A meta-network that emits weights consumed by a target network.",
        "cond:(B, D_c) → params → y",
        [
            [_io("condition  (B, D_c)")],
            [_op("meta MLP")],
            [_op("generated weights θ")],
            [_op("target net  (uses θ on input x)")],
            [_io("y  (task-dependent)")],
        ],
    ),
    "IPAdapterCrossAttention": (
        "Two parallel cross-attentions (text and image) summed into the residual stream.",
        "x:(B, T, D), text/image features → (B, T, D)",
        [
            [_io("x  Q  (B, T, D)")],
            [_attn("Cross-Attn  (text K, V)"), _attn("Cross-Attn  (image K, V)")],
            [_merge("+")],
            [_io("y  (B, T, D)")],
        ],
    ),
}

"""Core neural-network primitives."""

from typing import Dict

from dsl import (  # noqa: F401
    Spec, _io, _op, _norm, _act, _attn, _merge, _emb, _loss, _ref, _notes,
)

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
        [],
        None,
        _notes(
            used_in=[
                "Every MLP / classifier / projection head ever",
                "Q/K/V/O projections inside attention",
                "Final logits head on language and vision models",
                "Patch / channel mixers in MLP-Mixer-style nets",
            ],
            tasks=[
                "Any time you need a learned `(B, in) → (B, out)` map",
                "Bottleneck / expansion in residual blocks",
                "Read-out heads for regression and classification",
            ],
            pitfalls=[
                "Initialisation matters — Kaiming for ReLU-family, Xavier for tanh/sigmoid; "
                "wrong init can stall training entirely.",
                "Huge final layers (e.g. softmax over 50k tokens) dominate parameter count — "
                "tie input/output embeddings or factorise.",
                "Forgetting `bias=False` before a BatchNorm is harmless but wasteful.",
            ],
            see_also=[
                "[Kaiming init (He et al. 2015)](https://arxiv.org/abs/1502.01852)",
                "[Xavier init (Glorot & Bengio 2010)](https://proceedings.mlr.press/v9/glorot10a.html)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "VGG, ResNet, EfficientNet — the canonical CNN stem & body",
                "U-Net encoder/decoder layers",
                "Generator / discriminator backbones in GANs and diffusion",
            ],
            tasks=[
                "Image classification / segmentation / detection backbones",
                "Feature extraction prior to global pooling or upsampling",
            ],
            pitfalls=[
                "Conv → BN → ReLU vs BN → ReLU → Conv (pre-act) matters at depth — "
                "pre-activation is more stable for very deep nets.",
                "Bias on the conv before a BatchNorm is redundant (BN absorbs it).",
                "Padding mismatches silently change output spatial dims — sanity-check with a forward pass.",
            ],
            see_also=[
                "[VGG (Simonyan & Zisserman 2014)](https://arxiv.org/abs/1409.1556)",
                "[BatchNorm (Ioffe & Szegedy 2015)](https://arxiv.org/abs/1502.03167)",
                "[Pre-activation ResNet (He et al. 2016)](https://arxiv.org/abs/1603.05027)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "MobileNet v1 / v2 / v3 — flagship mobile backbones",
                "Xception — replaces every Inception module with depthwise separable convs",
                "EfficientNet & EfficientNetV2 — search space built on inverted residuals + DW",
            ],
            tasks=[
                "Edge / mobile inference where FLOPs and parameters must be tiny",
                "Replacing standard 3×3 convs to cut ~8–9× compute at similar accuracy",
            ],
            pitfalls=[
                "Memory-bound on GPUs — wall-clock speedup is often less than the FLOP reduction suggests.",
                "Channel scaling matters: pair with an expansion ratio (inverted residual) or "
                "the depthwise stage will bottleneck capacity.",
                "Some autograd backends fuse standard convs but not depthwise — benchmark first.",
            ],
            see_also=[
                "[MobileNet v1 (Howard et al. 2017)](https://arxiv.org/abs/1704.04861)",
                "[MobileNetV2 (Sandler et al. 2018)](https://arxiv.org/abs/1801.04381)",
                "[Xception (Chollet 2017)](https://arxiv.org/abs/1610.02357)",
            ],
        ),
    ),
    "DilatedConv2d": (
        "Standard conv with dilation > 1 (atrous), enlarges receptive field for free.",
        "(B, C, H, W) → (B, C', H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("Conv K×K  (dilation = d)")],
            [_io("y  (B, C', H, W)")],
        ],
        [],
        None,
        _notes(
            used_in=[
                "DeepLab v1/v2/v3 semantic segmentation",
                "WaveNet — stacks of dilated 1-D convs for raw audio",
                "Dilated TCNs for long-range time-series",
            ],
            tasks=[
                "Dense prediction (segmentation, depth) where spatial resolution must be preserved",
                "Long-range temporal modelling without growing parameters",
            ],
            pitfalls=[
                "Naïve stacks of equal dilations cause gridding artefacts — vary dilations "
                "(1, 2, 4, ...) or use HDC schedules.",
                "Receptive field grows but resolution doesn't — pair with multi-scale fusion (ASPP) "
                "for best results.",
            ],
            see_also=[
                "[Multi-Scale Context (Yu & Koltun 2015)](https://arxiv.org/abs/1511.07122)",
                "[DeepLab v3 (Chen et al. 2017)](https://arxiv.org/abs/1706.05587)",
                "[WaveNet (van den Oord et al. 2016)](https://arxiv.org/abs/1609.03499)",
            ],
        ),
    ),
    "GroupConv2d": (
        "Conv2d with channel groups (ResNeXt cardinality).",
        "(B, C, H, W) → (B, C', H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("Conv K×K  (groups = g)")],
            [_io("y  (B, C', H, W)")],
        ],
        [],
        None,
        _notes(
            used_in=[
                "AlexNet — the original use, split across two GPUs",
                "ResNeXt — uses cardinality (group count) as a third capacity axis",
                "ShuffleNet — group conv + channel shuffle for cheap exchange",
            ],
            tasks=[
                "Trading per-channel mixing for compute / parameter savings",
                "Backbone surgery that needs intermediate cost between standard and depthwise conv",
            ],
            pitfalls=[
                "Information is partitioned across groups — without channel shuffle or a 1×1 mix-up "
                "after, capacity drops sharply.",
                "Groups must divide both `in_ch` and `out_ch`; otherwise it silently errors at build time.",
            ],
            see_also=[
                "[ResNeXt (Xie et al. 2016)](https://arxiv.org/abs/1611.05431)",
                "[ShuffleNet (Zhang et al. 2017)](https://arxiv.org/abs/1707.01083)",
            ],
        ),
    ),
    "Conv1d": (
        "1-D convolution wrapper.",
        "(B, C, T) → (B, C', T)",
        [
            [_io("x  (B, C, T)")],
            [_op("Conv 1D K  (stride, padding)")],
            [_io("y  (B, C', T)")],
        ],
        [],
        None,
        _notes(
            used_in=[
                "WaveNet, Tacotron, Jukebox — raw audio / speech",
                "Text-CNN (Kim 2014) — n-gram feature detectors over token embeddings",
                "Mamba / S4 input projections",
            ],
            tasks=[
                "Sequence modelling where temporal locality dominates (audio, EEG, sensor)",
                "Token-level n-gram features prior to pooling",
            ],
            pitfalls=[
                "Channel-first vs channel-last layout differs between PyTorch and JAX — "
                "transpose at the boundary, not inside.",
                "Causal padding for autoregressive streams must be left-only.",
            ],
            see_also=[
                "[Text-CNN (Kim 2014)](https://arxiv.org/abs/1408.5882)",
                "[WaveNet (van den Oord et al. 2016)](https://arxiv.org/abs/1609.03499)",
            ],
        ),
    ),
    "Conv3d": (
        "3-D convolution wrapper.",
        "(B, C, T, H, W) → (B, C', T, H, W)",
        [
            [_io("x  (B, C, T, H, W)")],
            [_op("Conv 3D K  (stride, padding)")],
            [_io("y  (B, C', T, H, W)")],
        ],
        [],
        None,
        _notes(
            used_in=[
                "C3D, I3D, SlowFast — video classification",
                "V-Net / 3D U-Net — volumetric medical segmentation",
                "Spatio-temporal anomaly detection",
            ],
            tasks=[
                "Video action recognition / temporal segmentation",
                "MRI / CT volumetric analysis where slices share structure",
            ],
            pitfalls=[
                "Memory explodes — a single conv with K=3 over (T=16, H=224, W=224) needs careful batching.",
                "Pretrained 2D weights inflate poorly into 3D — use I3D-style inflation or "
                "(2+1)D factorisation for transfer.",
            ],
            see_also=[
                "[C3D (Tran et al. 2014)](https://arxiv.org/abs/1412.0767)",
                "[I3D (Carreira & Zisserman 2017)](https://arxiv.org/abs/1705.07750)",
                "[V-Net (Milletari et al. 2016)](https://arxiv.org/abs/1606.04797)",
            ],
        ),
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
        None,
        _notes(
            used_in=[
                "YOLOv4 and successors — replaced LeakyReLU in the backbone",
                "CSPNet variants for image classification / detection",
            ],
            tasks=[
                "Drop-in replacement for ReLU/Swish in CNNs aiming for slight accuracy gains",
            ],
            pitfalls=[
                "More expensive than ReLU/SiLU — measure wall-clock impact, not just FLOPs.",
                "Smooth-but-nonmonotonic; gradients in the negative tail are small but not zero, "
                "which helps deep nets but is not always better than SiLU/GELU.",
            ],
            see_also=[
                "[Mish (Misra 2019)](https://arxiv.org/abs/1908.08681)",
                "[YOLOv4 (Bochkovskiy et al. 2020)](https://arxiv.org/abs/2004.10934)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "LLaMA / LLaMA-2 / LLaMA-3 — standard pre-norm",
                "T5 v1.1, Gemma, Mistral, Qwen — most modern LLMs",
                "Diffusion transformers (DiT) and audio models",
            ],
            tasks=[
                "Drop-in LayerNorm replacement with ~10–15 % faster forward at no quality cost",
                "Pre-normalisation in deep transformer stacks",
            ],
            pitfalls=[
                "Skips the mean-subtraction — won't help on inputs with strong DC bias.",
                "ε placement (inside vs outside the sqrt) differs across implementations; match the "
                "convention of the checkpoint you're loading or fine-tuning.",
                "No `bias` parameter — don't try to load a LayerNorm checkpoint into RMSNorm naïvely.",
            ],
            see_also=[
                "[RMSNorm (Zhang & Sennrich 2019)](https://arxiv.org/abs/1910.07467)",
                "[LLaMA (Touvron et al. 2023)](https://arxiv.org/abs/2302.13971)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "Arbitrary neural style transfer (Huang & Belongie 2017)",
                "StyleGAN v1 — every conv layer is AdaIN-modulated by a style code w",
                "Voice-conversion and image-to-image translation",
            ],
            tasks=[
                "Inject a per-sample style code into a content stream",
                "Single-network arbitrary style transfer (no per-style fine-tuning)",
            ],
            pitfalls=[
                "Removes spatial mean/std — fine for style but throws away content statistics; "
                "StyleGAN2 replaced it with weight-space modulation (ModulatedConv2d) for this reason.",
                "Style collapse if (γ, β) come from a low-rank MLP — keep the conditioning head wide.",
            ],
            see_also=[
                "[AdaIN (Huang & Belongie 2017)](https://arxiv.org/abs/1703.06868)",
                "[StyleGAN (Karras et al. 2019)](https://arxiv.org/abs/1812.04948)",
                "[StyleGAN2 critique of AdaIN (Karras et al. 2020)](https://arxiv.org/abs/1912.04958)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "GauGAN / SPADE — semantic image synthesis from segmentation masks",
                "Image editing pipelines that need spatial conditioning",
            ],
            tasks=[
                "Image generation conditioned on layout / semantic mask",
                "Sketch-to-image, label-to-image translation",
            ],
            pitfalls=[
                "γ and β are spatially dense — needs care to broadcast over batch but not space.",
                "Standard BatchNorm step washes away the conditioning if applied AFTER the modulation; "
                "BN goes first, modulation second.",
            ],
            see_also=[
                "[SPADE / GauGAN (Park et al. 2019)](https://arxiv.org/abs/1903.07291)",
            ],
        ),
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
        None,
        _notes(
            used_in=[
                "ResNet-18/34/50/101/152 — the most-cited deep-learning architecture",
                "U-Net diffusion noise prediction backbone",
                "AlphaGo / AlphaZero policy-value tower",
                "Almost every modern CNN backbone (ConvNeXt, RegNet)",
            ],
            tasks=[
                "Enabling depth > 30 layers without vanishing gradients",
                "Optimisation landscape smoothing — skips create much shorter back-prop paths",
            ],
            pitfalls=[
                "Pre-activation order (BN → ReLU → Conv) often trains deeper nets more stably than post-act.",
                "When channel count changes across the skip, you need a 1×1 projection or zero-pad — "
                "naïve add will dimension-error.",
                "ReLU AFTER the add caps activations to ≥ 0, which can hurt; SiLU/GELU is sometimes better.",
            ],
            see_also=[
                "[ResNet (He et al. 2015)](https://arxiv.org/abs/1512.03385)",
                "[Pre-activation ResNet (He et al. 2016)](https://arxiv.org/abs/1603.05027)",
                "[Identity Mappings paper analysis](https://arxiv.org/abs/1603.05027)",
            ],
        ),
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
        None,
        _notes(
            used_in=[
                "Transformer pre-norm / post-norm residuals (Attention is All You Need)",
                "U-Net long skips between matched-resolution encoder and decoder",
                "Highway networks and HighwayLSTM predecessors of ResNet",
            ],
            tasks=[
                "Gradient highway in deep networks — every skip shortens back-prop by a layer",
                "Preserving low-level information past depth (segmentation, super-resolution)",
            ],
            pitfalls=[
                "Variance grows with depth if skips aren't scaled — pre-norm or √-scaling addresses this.",
                "Skips across different shapes need a 1×1 projection (channel) or interpolation (spatial).",
                "Without an activation after the add, two consecutive residuals collapse into one linear map "
                "for the gradient.",
            ],
            see_also=[
                "[Highway Networks (Srivastava et al. 2015)](https://arxiv.org/abs/1505.00387)",
                "[ResNet (He et al. 2015)](https://arxiv.org/abs/1512.03385)",
                "[U-Net (Ronneberger et al. 2015)](https://arxiv.org/abs/1505.04597)",
            ],
        ),
    ),
}

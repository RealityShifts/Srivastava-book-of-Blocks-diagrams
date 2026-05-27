"""GAN building blocks: StyleGAN, PGGAN, equalised LR."""

from typing import Dict

from dsl import (  # noqa: F401
    Spec, _io, _op, _norm, _act, _attn, _merge, _emb, _loss, _ref, _notes,
)

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
        [],
        None,
        _notes(
            used_in=[
                "PGGAN, StyleGAN, StyleGAN2/3 — every fully-connected layer",
                "MSG-GAN, GANformer — high-quality image synthesis",
            ],
            tasks=[
                "Maintaining uniform per-parameter learning rate across layers of vastly different fan-in",
                "Stable training of progressively-growing or modulated generators",
            ],
            pitfalls=[
                "Naïve replacement of Linear with EqualLinear without adjusting LR ratios degrades quality.",
                "Scale factor is APPLIED at runtime, not at init — keeping weights N(0,1) and scaling on "
                "forward is the whole point.",
                "Mixing standard and equalised layers in the same net usually hurts; commit to one.",
            ],
            see_also=[
                "[PGGAN (Karras et al. 2017)](https://arxiv.org/abs/1710.10196)",
                "[StyleGAN (Karras et al. 2019)](https://arxiv.org/abs/1812.04948)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "StyleGAN generator and discriminator convolution layers",
                "PGGAN progressively-grown image stages",
            ],
            tasks=[
                "Stable training of CNN generators at high resolution",
                "Drop-in replacement for Conv2d in generative architectures",
            ],
            pitfalls=[
                "Scale factor is per-layer based on fan-in — wrong fan-in computation silently mis-scales.",
                "Bias initialisation should be zero — non-zero bias defeats the equalised LR effect.",
            ],
            see_also=[
                "[PGGAN (Karras et al. 2017)](https://arxiv.org/abs/1710.10196)",
                "[StyleGAN2 (Karras et al. 2020)](https://arxiv.org/abs/1912.04958)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "DCGAN — original deep convolutional GAN",
                "BigGAN, SAGAN — class-conditional image synthesis",
            ],
            tasks=[
                "Upsampling path in convolutional generators",
                "Image-to-image translation decoders (CycleGAN, Pix2Pix)",
            ],
            pitfalls=[
                "Transposed-conv upsample causes checkerboard artefacts — prefer nearest/bilinear + conv.",
                "BatchNorm in the generator depends on batch statistics that diverge from real data; "
                "consider InstanceNorm or no norm for unconditional GANs.",
                "ReLU dying-out problem in small-batch settings — LeakyReLU is safer.",
            ],
            see_also=[
                "[DCGAN (Radford et al. 2015)](https://arxiv.org/abs/1511.06434)",
                "[Checkerboard Artifacts (Odena et al. 2016)](https://distill.pub/2016/deconv-checkerboard/)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "DCGAN / WGAN / StyleGAN discriminators",
                "PatchGAN discriminator in Pix2Pix / CycleGAN",
            ],
            tasks=[
                "Downsampling and feature extraction in the critic / discriminator",
            ],
            pitfalls=[
                "Spectral Normalisation (SN-GAN) is often used instead of InstanceNorm for Lipschitz control.",
                "Stride-2 conv can lose information — combine with anti-aliased pooling (StyleGAN3).",
                "LeakyReLU slope 0.2 is convention; smaller slopes saturate gradients in the negative tail.",
            ],
            see_also=[
                "[Spectral Norm (Miyato et al. 2018)](https://arxiv.org/abs/1802.05957)",
                "[PatchGAN / Pix2Pix (Isola et al. 2016)](https://arxiv.org/abs/1611.07004)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "StyleGAN v1 / v2 / v3 — the f: z → w map",
                "StyleGAN-T text-to-image",
            ],
            tasks=[
                "Disentangling latent noise into a more linear style space W",
                "Enabling style mixing and W+ inversion / editing",
            ],
            pitfalls=[
                "Without the low LR multiplier (~0.01), the mapping net dominates training updates.",
                "Skipping PixelNorm on z makes early training unstable.",
                "Depth of 8 is empirical; smaller maps under-disentangle, deeper maps offer no gains.",
            ],
            see_also=[
                "[StyleGAN (Karras et al. 2019)](https://arxiv.org/abs/1812.04948)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "StyleGAN v1 generator (every resolution stage)",
                "Style-based generators for portraits, faces, cars",
            ],
            tasks=[
                "Spatially-uniform style injection at each resolution",
                "Stochastic detail through per-pixel noise",
            ],
            pitfalls=[
                "AdaIN-induced droplet artefacts (StyleGAN v1) — fixed in StyleGAN2's modulated conv.",
                "Noise scale per channel matters — fix or learn carefully; too large dominates the output.",
                "Style mixing regularisation (random truncation across resolutions) is essential for quality.",
            ],
            see_also=[
                "[StyleGAN (Karras et al. 2019)](https://arxiv.org/abs/1812.04948)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "StyleGAN2 / StyleGAN3 generator (replaces AdaIN style injection)",
                "Modulated decoders in modern image synthesis",
            ],
            tasks=[
                "Style injection through weight modulation rather than feature statistics",
                "Eliminating droplet artefacts caused by AdaIN",
            ],
            pitfalls=[
                "Demodulation factor is computed PER OUTPUT CHANNEL — bookkeeping error breaks training.",
                "Grouped convolution implementation per sample is needed (since W differs per batch element) — "
                "naïve loop is too slow; use grouped conv with B groups.",
                "Style vector s is BROADCAST over spatial — confusing AdaIN's spatial γ/β with this is wrong.",
            ],
            see_also=[
                "[StyleGAN2 (Karras et al. 2020)](https://arxiv.org/abs/1912.04958)",
                "[StyleGAN3 (Karras et al. 2021)](https://arxiv.org/abs/2106.12423)",
            ],
        ),
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
        None,
        _notes(
            used_in=[
                "PGGAN and StyleGAN discriminator's last block",
            ],
            tasks=[
                "Discriminator-side diversity signal — penalises mode collapse",
                "Cheap addition that helps GANs avoid producing nearly identical samples",
            ],
            pitfalls=[
                "Needs a meaningful batch size — std over batch ≤ 2 is noise.",
                "Multi-GPU training: std must be computed PER local batch or all-reduced for consistency.",
                "Placement matters — at the deepest discriminator layer, not the input.",
            ],
            see_also=[
                "[PGGAN (Karras et al. 2017)](https://arxiv.org/abs/1710.10196)",
            ],
        ),
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
        None,
        _notes(
            used_in=[
                "PGGAN — first method to scale GANs to 1024² faces",
                "Some MSG-GAN and progressive diffusion variants",
            ],
            tasks=[
                "Stable training of very high-resolution GANs",
                "Reducing training-divergence risk by gradually expanding capacity",
            ],
            pitfalls=[
                "α schedule must be slow enough — abrupt fade-in destabilises training.",
                "Discriminator and generator must grow in LOCK-STEP; mismatched resolutions diverge.",
                "Replaced in modern recipes by StyleGAN2/3 single-shot training, which is simpler.",
            ],
            see_also=[
                "[PGGAN (Karras et al. 2017)](https://arxiv.org/abs/1710.10196)",
            ],
        ),
    ),
}

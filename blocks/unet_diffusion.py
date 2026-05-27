"""UNet, time conditioning, ControlNet, LoRA, hypernets."""

from typing import Dict

from dsl import (  # noqa: F401
    Spec, _io, _op, _norm, _act, _attn, _merge, _emb, _loss, _ref, _notes,
)

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
        [],
        None,
        _notes(
            used_in=[
                "DDPM, DDIM, Stable Diffusion, Imagen — diffusion noise predictors",
                "Score-based generative models",
                "Continuous-time NeuralODE / flow-matching conditioning",
            ],
            tasks=[
                "Encoding a continuous scalar (timestep, noise level) into a vector for conditioning",
            ],
            pitfalls=[
                "Frequency base (10000 in original Transformer; sometimes scaled differently for diffusion) "
                "affects which timescales are resolved — match the convention of the reference impl.",
                "When t is in [0, 1] vs [0, T] (T~1000) the same sinusoidal table behaves very differently.",
                "Half-precision can underflow for very small t — keep the embedding in fp32.",
            ],
            see_also=[
                "[DDPM (Ho et al. 2020)](https://arxiv.org/abs/2006.11239)",
                "[Attention Is All You Need (Vaswani et al. 2017)](https://arxiv.org/abs/1706.03762)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "Stable Diffusion U-Net's time conditioning head",
                "ControlNet / IP-Adapter side branches",
            ],
            tasks=[
                "Mapping raw sinusoidal time encodings to a richer feature space matched to each ResBlock",
            ],
            pitfalls=[
                "Output dim D' should match the ResBlock channel — otherwise broadcasting is wrong.",
                "Don't fold this into a single layer — the non-linearity is what gives time embeddings "
                "their expressive power.",
            ],
            see_also=[
                "[DDPM (Ho et al. 2020)](https://arxiv.org/abs/2006.11239)",
            ],
        ),
    ),
    "DownsampleBlock": (
        "Spatial downsample by 2× via a strided 3×3 conv.",
        "(B, C, H, W) → (B, C, H/2, W/2)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("Conv 3×3, stride 2")],
            [_io("y  (B, C, H/2, W/2)")],
        ],
        [],
        None,
        _notes(
            used_in=[
                "U-Net encoders in DDPM, Stable Diffusion, video diffusion",
                "CNN backbones generally",
            ],
            tasks=[
                "Cutting spatial resolution while keeping (or growing) channel count",
            ],
            pitfalls=[
                "Stride-2 conv aliases — for low-frequency-preserving downsample, use blur-then-stride "
                "(StyleGAN3 anti-aliasing) or avg-pool.",
                "Padding choice affects boundary statistics — keep consistent with the upsample path.",
            ],
            see_also=[
                "[Anti-Aliased CNNs (Zhang 2019)](https://arxiv.org/abs/1904.11486)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "U-Net decoders in diffusion models, segmentation",
                "Super-resolution and image-to-image networks",
            ],
            tasks=[
                "Doubling spatial size while smoothing aliasing introduced by nearest-up",
            ],
            pitfalls=[
                "Transposed-conv alternatives cause checkerboard artefacts — nearest+conv is safer.",
                "Bilinear can be smoother than nearest but trades sharpness; pick per task.",
            ],
            see_also=[
                "[Checkerboard Artifacts (Odena et al. 2016)](https://distill.pub/2016/deconv-checkerboard/)",
            ],
        ),
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
        None,
        _notes(
            used_in=[
                "DDPM noise predictor",
                "Stable Diffusion U-Net (every resolution level)",
                "Latent video diffusion (SVD, AnimateDiff)",
            ],
            tasks=[
                "Time-conditioned feature extraction in a diffusion backbone",
            ],
            pitfalls=[
                "Time embedding is added BETWEEN the two convs (not at the input) — placement matters.",
                "GroupNorm groups must divide channel count; 32 is standard, mismatches silently break.",
                "Skip connection must match channel count — add a 1×1 projection when C' ≠ C.",
            ],
            see_also=[
                "[DDPM (Ho et al. 2020)](https://arxiv.org/abs/2006.11239)",
                "[GroupNorm (Wu & He 2018)](https://arxiv.org/abs/1803.08494)",
            ],
        ),
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
        None,
        _notes(
            used_in=[
                "Original U-Net for biomedical segmentation (Ronneberger 2015)",
                "DDPM / Stable Diffusion / Imagen noise prediction",
                "nnU-Net — semi-automated medical segmentation",
            ],
            tasks=[
                "Diffusion-model noise prediction",
                "Dense prediction (segmentation, depth, optical flow)",
            ],
            pitfalls=[
                "Skip channel counts must match decoder side — channel-list mismatch errors are common.",
                "Bottleneck attention costs O((H·W)²) — fine at 16×16 latents, infeasible at 64×64 pixels.",
                "Diffusion U-Nets are huge (~860M for SD 1.5) — memory-bound; gradient checkpointing helps.",
            ],
            see_also=[
                "[U-Net (Ronneberger et al. 2015)](https://arxiv.org/abs/1505.04597)",
                "[DDPM (Ho et al. 2020)](https://arxiv.org/abs/2006.11239)",
                "[Stable Diffusion (Rombach et al. 2021)](https://arxiv.org/abs/2112.10752)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "DDPM, DDIM, EDM samplers",
                "Stable Diffusion's denoising backbone",
            ],
            tasks=[
                "Predicting noise added at step t given the noisy latent / pixel",
                "Equivalently used in v-prediction or x0-prediction parametrisations",
            ],
            pitfalls=[
                "Training objective parametrisation (ε / v / x0) changes the loss weighting — match "
                "training and sampling exactly.",
                "Classifier-free guidance requires conditional + unconditional passes at sample time — "
                "double the forward cost.",
                "EMA of weights is usually used for sampling rather than the raw training weights.",
            ],
            see_also=[
                "[DDPM (Ho et al. 2020)](https://arxiv.org/abs/2006.11239)",
                "[v-prediction (Salimans & Ho 2022)](https://arxiv.org/abs/2202.00512)",
                "[EDM (Karras et al. 2022)](https://arxiv.org/abs/2206.00364)",
            ],
        ),
    ),
    "ZeroConv2d": (
        "Conv2d initialised to zero — outputs zero at start so it can be added safely.",
        "(B, C, H, W) → (B, C', H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("Conv  (W = 0, b = 0 at init)")],
            [_io("y  (B, C', H, W)  (= 0 at init)")],
        ],
        [],
        None,
        _notes(
            used_in=[
                "ControlNet — bridge between trainable copy and frozen U-Net",
                "AdaLN-Zero / DiT — adaptive layer norm initialised to identity",
            ],
            tasks=[
                "Safely adding a new branch to a pretrained model without breaking its outputs at init",
                "Smooth transfer learning where the new module starts as a no-op",
            ],
            pitfalls=[
                "Zero init kills gradients to the input — only the conv WEIGHTS get updated, not earlier params.",
                "Use only at the JUNCTION between trained and frozen networks; cascading zero-convs back-to-back "
                "breaks training.",
            ],
            see_also=[
                "[ControlNet (Zhang et al. 2023)](https://arxiv.org/abs/2302.05543)",
                "[DiT (Peebles & Xie 2022)](https://arxiv.org/abs/2212.09748)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "Stable Diffusion ControlNet (canny, depth, pose, scribble, etc.)",
                "ControlNet-XS — smaller variants for cheap inference",
            ],
            tasks=[
                "Adding structural conditioning (edges, depth, pose) to a pretrained diffusion model",
                "Fine-tuning T2I models on new control signals without forgetting the base distribution",
            ],
            pitfalls=[
                "Frozen base U-Net must be EXACTLY the model you'll run at inference — fine-tunes of "
                "the base break compatibility.",
                "Zero-conv at the junction is essential; without it, training breaks the base model immediately.",
                "Each ControlNet is condition-specific — stacking many at inference is possible but quality drops.",
            ],
            see_also=[
                "[ControlNet (Zhang et al. 2023)](https://arxiv.org/abs/2302.05543)",
                "[T2I-Adapter (Mou et al. 2023)](https://arxiv.org/abs/2302.08453)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "LoRA / QLoRA — LLM and diffusion fine-tuning",
                "PEFT library — Hugging Face standard",
                "SDXL LoRAs on civitai etc. for style/character training",
            ],
            tasks=[
                "Parameter-efficient fine-tuning (PEFT) — train < 1 % of parameters",
                "Composable / mergeable adapters that can be added on the fly",
            ],
            pitfalls=[
                "A is init Gaussian, B init zero — swapping breaks the 'starts as identity' invariant.",
                "Rank r vs α/r scaling: doubling α at fixed r is equivalent to doubling learning rate.",
                "Merging LoRA back into frozen W (`W ← W + B·A·α/r`) is needed for inference speed.",
                "Stacking LoRAs at inference is additive — careful with cumulative drift.",
            ],
            see_also=[
                "[LoRA (Hu et al. 2021)](https://arxiv.org/abs/2106.09685)",
                "[QLoRA (Dettmers et al. 2023)](https://arxiv.org/abs/2305.14314)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "LoRA for image diffusion models (Stable Diffusion, SDXL)",
                "Vision-model fine-tuning with limited compute",
            ],
            tasks=[
                "Adapting convolutional generators to new styles or domains cheaply",
            ],
            pitfalls=[
                "Conv A uses the full kernel size; Conv B uses 1×1 — reversing reduces expressivity.",
                "Memory of the auxiliary feature map (`r`-channel) can still be significant at full resolution.",
            ],
            see_also=[
                "[LoRA (Hu et al. 2021)](https://arxiv.org/abs/2106.09685)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "Stable Diffusion Hypernetworks (early SD fine-tuning trick)",
                "HyperNetworks (Ha et al. 2016) — original RNN-emitting-CNN-weights paper",
                "Meta-learning / few-shot adaptation",
            ],
            tasks=[
                "Conditioning a target network on a high-dimensional context by emitting its weights",
                "Few-shot personalisation in vision / text models",
            ],
            pitfalls=[
                "Output dimension is the FULL parameter count of the target — grows quickly; usually emit "
                "low-rank or per-layer scalars instead.",
                "Joint optimisation is delicate — meta-LR usually much smaller than target-LR.",
                "Inference cost includes meta-net forward + target-net forward.",
            ],
            see_also=[
                "[HyperNetworks (Ha et al. 2016)](https://arxiv.org/abs/1609.09106)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "IP-Adapter — image-prompt conditioning for Stable Diffusion",
                "IP-Adapter-FaceID, IP-Adapter-Plus variants",
            ],
            tasks=[
                "Conditioning a text-to-image diffusion model on a REFERENCE IMAGE prompt",
                "Image-prompted personalisation without per-subject fine-tuning",
            ],
            pitfalls=[
                "Image and text branches use SEPARATE projections — sharing weights collapses modality.",
                "Image-branch scale (λ) is critical; high values overpower the text prompt.",
                "Reference encoder choice (CLIP ViT-L vs OpenCLIP H) changes downstream behaviour.",
            ],
            see_also=[
                "[IP-Adapter (Ye et al. 2023)](https://arxiv.org/abs/2308.06721)",
            ],
        ),
    ),
}

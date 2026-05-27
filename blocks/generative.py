"""VAE, autoregressive, normalising-flow, EBM, diffusion schedulers."""

from typing import Dict

from dsl import (  # noqa: F401
    Spec, _io, _op, _norm, _act, _attn, _merge, _emb, _loss, _ref, _notes,
)

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
        [],
        None,
        _notes(
            used_in=[
                "VAE (Kingma & Welling 2013) — original deep generative latent-variable model",
                "VQ-VAE / VQ-VAE-2 — discrete-latent variants for images / audio",
                "Stable Diffusion's image autoencoder (compresses pixels → 8× smaller latents)",
                "Talking-head and motion synthesis (FaceVAE)",
            ],
            tasks=[
                "Unsupervised representation learning with explicit latent prior",
                "Low-dimensional latent space for downstream models (e.g. diffusion on latents)",
                "Anomaly detection (high reconstruction loss = unusual sample)",
            ],
            pitfalls=[
                "Posterior collapse — z carries no information when the decoder is too powerful. "
                "Mitigate with β-VAE (down-weight KL), free-bits, or KL warmup.",
                "Blurry reconstructions are a known limitation of pixel L2 / L1 — perceptual or adversarial "
                "loss helps (SD's AE uses both).",
                "Reparameterisation must use ε ~ N(0,I) during training and z = μ at inference for "
                "deterministic encoding.",
            ],
            see_also=[
                "[VAE (Kingma & Welling 2013)](https://arxiv.org/abs/1312.6114)",
                "[β-VAE (Higgins et al. 2017)](https://openreview.net/forum?id=Sy2fzU9gl)",
                "[VQ-VAE (van den Oord et al. 2017)](https://arxiv.org/abs/1711.00937)",
            ],
        ),
    ),
    "MaskedConv2d": (
        "Conv2d with an autoregressive mask preventing each pixel from seeing future pixels (PixelCNN).",
        "(B, C, H, W) → (B, C', H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("Conv  (kernel multiplied by AR mask)")],
            [_io("y  (B, C', H, W)")],
        ],
        [],
        None,
        _notes(
            used_in=[
                "PixelCNN / Gated PixelCNN / PixelCNN++ — autoregressive image models",
                "Image GPT (van den Oord -> Chen 2020) for unsupervised representation pre-training",
            ],
            tasks=[
                "Density modelling of images / discrete grids",
                "Lossless image compression baselines",
            ],
            pitfalls=[
                "Blind-spot bug: a naïve mask over a vertical stack creates a region the centre pixel "
                "never sees — fixed by Gated PixelCNN's vertical + horizontal stack design.",
                "Type-A (input layer) vs Type-B (subsequent layers) masks differ — confusing them leaks "
                "information through residual connections.",
                "Inherently sequential at sampling — O(H·W) forward passes per image.",
            ],
            see_also=[
                "[PixelCNN (van den Oord et al. 2016)](https://arxiv.org/abs/1606.05328)",
                "[PixelRNN / PixelCNN (van den Oord et al. 2016)](https://arxiv.org/abs/1601.06759)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "PixelCNN family for image density modelling",
                "WaveNet (1D analogue) for raw audio waveform modelling",
            ],
            tasks=[
                "Exact-likelihood density modelling",
                "Sample diversity at the cost of slow inference",
            ],
            pitfalls=[
                "Output K = #bins of pixel intensity (256 for 8-bit) — softmax memory grows fast.",
                "Mostly superseded by VQ + transformer or diffusion for image generation quality.",
                "Conditional variants need careful broadcasting of the conditioning signal under the mask.",
            ],
            see_also=[
                "[PixelCNN++ (Salimans et al. 2017)](https://arxiv.org/abs/1701.05517)",
                "[WaveNet (van den Oord et al. 2016)](https://arxiv.org/abs/1609.03499)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "RealNVP / NICE — original normalising flow architectures",
                "Glow — invertible image generation",
                "Neural Spline Flows, Flow++",
            ],
            tasks=[
                "Exact-likelihood generative modelling",
                "Density estimation, variational inference posteriors",
            ],
            pitfalls=[
                "Determinant of the Jacobian is just `exp(Σ s)` because the lower triangle is identity — "
                "lose this property and you lose tractability.",
                "Variable splitting (which half is transformed) must alternate across layers; static "
                "split leaves half the variables untransformed.",
                "Compute / memory both scale O(D) — far less expressive per parameter than autoregressive "
                "or diffusion models for images.",
            ],
            see_also=[
                "[RealNVP (Dinh et al. 2016)](https://arxiv.org/abs/1605.08803)",
                "[Glow (Kingma & Dhariwal 2018)](https://arxiv.org/abs/1807.03039)",
                "[NICE (Dinh et al. 2014)](https://arxiv.org/abs/1410.8516)",
            ],
        ),
    ),
    "EnergyBasedModel": (
        "A network that maps x to a scalar energy E(x); samples drawn via Langevin/MCMC.",
        "x → E(x):(B,)",
        [
            [_io("x  (B, …)")],
            [_op("network")],
            [_io("E(x)  (B,)")],
        ],
        [],
        None,
        _notes(
            used_in=[
                "Implicit Generation (Du & Mordatch 2019)",
                "JEM — joint generative-discriminative training",
                "Score-based models (gradient of log-density ≈ −∇E)",
            ],
            tasks=[
                "Density modelling without a tractable partition function",
                "Out-of-distribution detection (low energy = in-distribution)",
            ],
            pitfalls=[
                "Training is unstable — contrastive divergence and short-run MCMC are common workarounds.",
                "Sampling needs Langevin dynamics or HMC — slow compared to feed-forward generators.",
                "Mode coverage is poor without replay buffers and persistent chains.",
            ],
            see_also=[
                "[Implicit Generation (Du & Mordatch 2019)](https://arxiv.org/abs/1903.08689)",
                "[JEM (Grathwohl et al. 2019)](https://arxiv.org/abs/1912.03263)",
                "[A Tutorial on EBMs (LeCun et al. 2006)](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "DDPM / DDIM training-time noising",
                "Stable Diffusion 1.5 / 2.x default scheduler",
                "Improved-DDPM, EDM, Karras schedule",
            ],
            tasks=[
                "Adding noise during training to teach the denoiser",
                "Defining the SDE / discrete chain that the model inverts at sampling time",
            ],
            pitfalls=[
                "β schedule choice (linear, cosine, sigmoid) materially changes sample quality.",
                "Numerical stability for very large t — keep α̅_t in fp32.",
                "Sampling-time scheduler must match the noise schedule used at training.",
            ],
            see_also=[
                "[DDPM (Ho et al. 2020)](https://arxiv.org/abs/2006.11239)",
                "[Improved DDPM (Nichol & Dhariwal 2021)](https://arxiv.org/abs/2102.09672)",
                "[EDM (Karras et al. 2022)](https://arxiv.org/abs/2206.00364)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "DDIM sampler (Stable Diffusion default before DPM-Solver)",
                "Image editing methods (SDEdit, prompt-to-prompt) that rely on deterministic noise",
            ],
            tasks=[
                "Fast deterministic sampling with fewer steps (50 → ~20 typical)",
                "Reproducible editing through inversion-then-resample",
            ],
            pitfalls=[
                "Quality at very low step counts (<10) lags behind DPM-Solver++ / Heun-2.",
                "Stochasticity parameter η controls determinism — η = 0 is fully deterministic, η = 1 "
                "recovers DDPM.",
                "Inversion via DDIM is approximate; long edits accumulate drift.",
            ],
            see_also=[
                "[DDIM (Song et al. 2020)](https://arxiv.org/abs/2010.02502)",
                "[DPM-Solver (Lu et al. 2022)](https://arxiv.org/abs/2206.00927)",
            ],
        ),
    ),
}

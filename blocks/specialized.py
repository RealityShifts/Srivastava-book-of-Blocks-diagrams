"""Specialised research blocks (NeuralODE, FNO, KAN, capsules, slots)."""

from typing import Dict

from dsl import (  # noqa: F401
    Spec, _io, _op, _norm, _act, _attn, _merge, _emb, _loss, _ref, _notes,
)

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
        [],
        None,
        _notes(
            used_in=[
                "Neural ODE (Chen et al. 2018) — original paper, NeurIPS best paper",
                "FFJORD — continuous normalising flows",
                "Latent ODEs for irregular time series",
                "Flow-matching / rectified-flow diffusion (probability-flow ODE view)",
            ],
            tasks=[
                "Continuous-depth networks where memory scales with the adjoint, not the network depth",
                "Modelling irregular time-series (medical, finance) at arbitrary sample times",
                "Generative modelling via instantaneous change of variables",
            ],
            pitfalls=[
                "Adjoint method saves memory but is numerically unstable — checkpointing the forward "
                "trajectory is often more reliable.",
                "Adaptive solvers (dopri5) can stall on stiff dynamics — bound max_steps.",
                "Wall-clock per step is far higher than discrete networks at equal expressivity.",
            ],
            see_also=[
                "[Neural ODE (Chen et al. 2018)](https://arxiv.org/abs/1806.07366)",
                "[FFJORD (Grathwohl et al. 2018)](https://arxiv.org/abs/1810.01367)",
                "[Latent ODEs (Rubanova et al. 2019)](https://arxiv.org/abs/1907.03907)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "Fourier Neural Operator (FNO) — Navier-Stokes, Darcy flow",
                "GNOT / NeuralOperator library — generalised operators",
                "Weather forecasting (FourCastNet, GraphCast hybrids)",
            ],
            tasks=[
                "Learning solution operators for PDEs (input → solution map)",
                "Discretisation-invariant inference — train at one grid, evaluate at another",
            ],
            pitfalls=[
                "Truncating to `mode` low frequencies loses sharp features — too few modes, blurry outputs.",
                "Complex-valued weights are needed; real-valued FFT representations sometimes confuse "
                "naïve PyTorch users.",
                "Boundary conditions matter — periodic-only is a strong restriction for non-periodic PDEs.",
            ],
            see_also=[
                "[FNO (Li et al. 2020)](https://arxiv.org/abs/2010.08895)",
                "[FourCastNet (Pathak et al. 2022)](https://arxiv.org/abs/2202.11214)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "FNO scientific-ML stacks",
                "AFNO (Adaptive FNO) — used in FourCastNet for high-res weather",
            ],
            tasks=[
                "Each block of an FNO; stacked 4 times in the canonical architecture",
                "PDE surrogate models trained on simulator data",
            ],
            pitfalls=[
                "1×1 conv path captures high-frequency residuals that the spectral path truncates — "
                "removing it kills accuracy.",
                "Number of modes is a sensitive hyperparameter per axis.",
            ],
            see_also=[
                "[FNO (Li et al. 2020)](https://arxiv.org/abs/2010.08895)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "KAN, KAN 2.0 — symbolic discovery on small AI+Science tasks",
                "Hybrid KAN+MLP exploratory architectures",
            ],
            tasks=[
                "Symbolic regression and physics-equation discovery",
                "Interpretable function fitting with small, dense datasets",
            ],
            pitfalls=[
                "Far slower to train than MLPs at comparable expressivity — current limitation.",
                "Spline grid extension (refinement) is required for fine-grained accuracy; doing it on "
                "the fly is non-trivial.",
                "Original promises about scaling laws don't hold at large scale on standard benchmarks — "
                "use cautiously for production.",
                "Implementation correctness is subtle; pykan is the reference but slow.",
            ],
            see_also=[
                "[KAN (Liu et al. 2024)](https://arxiv.org/abs/2404.19756)",
                "[KAN 2.0 (Liu et al. 2024)](https://arxiv.org/abs/2408.10205)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "Dynamic Routing Between Capsules (Sabour, Frosst, Hinton 2017)",
                "Matrix Capsules with EM Routing (Hinton et al. 2018)",
                "Stacked Capsule Autoencoders",
            ],
            tasks=[
                "Part-whole hierarchy modelling — argued to be a more 'biologically plausible' alternative "
                "to CNNs",
                "Pose / orientation reasoning where vectors carry geometric info",
            ],
            pitfalls=[
                "Dynamic routing is expensive and hard to scale beyond small images (MNIST / CIFAR).",
                "Largely supplanted by self-attention in modern vision — capsules saw limited adoption.",
                "Routing iterations T must be small (3) to be tractable; too few and routing under-trains.",
            ],
            see_also=[
                "[Dynamic Routing Between Capsules (Sabour et al. 2017)](https://arxiv.org/abs/1710.09829)",
                "[Matrix Capsules with EM Routing (Hinton et al. 2018)](https://openreview.net/forum?id=HJWLfGWRb)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "Slot Attention (Locatello et al. 2020)",
                "OSRT / SAVi for video object discovery",
                "Object-centric world models in RL",
            ],
            tasks=[
                "Unsupervised object discovery / instance segmentation",
                "Compositional scene representations from images / videos",
            ],
            pitfalls=[
                "Number of slots K is a strong inductive bias — fewer than objects merges, more leaves "
                "slots empty.",
                "Slot symmetry-breaking is induced by random init each forward — beware deterministic "
                "fixes that collapse all slots.",
                "Softmax over slots (competition) is essential — without it, all slots converge to the "
                "same content.",
                "Works on small / synthetic scenes; transfer to realistic images is an active research area.",
            ],
            see_also=[
                "[Slot Attention (Locatello et al. 2020)](https://arxiv.org/abs/2006.15055)",
                "[SAVi (Kipf et al. 2021)](https://arxiv.org/abs/2111.12594)",
            ],
        ),
    ),
}

"""Recurrent and state-space sequence models."""

from typing import Dict

from dsl import (  # noqa: F401
    Spec, _io, _op, _norm, _act, _attn, _merge, _emb, _loss, _ref, _notes,
)

CATEGORY = "sequence"
CATEGORY_DESC = "Recurrent and state-space sequence models."

BLOCKS: Dict[str, Spec] = {
    "RNNCell": (
        "Vanilla recurrent cell: h_t = tanh(W_x x_t + W_h h_{t-1} + b).",
        "x_t:(B, D_x), h_{t-1}:(B, D_h) → h_t:(B, D_h)",
        [
            [_io("x_t  (B, D_x)"), _io("h_{t−1}  (B, D_h)")],
            [_op("W_x · x_t"), _op("W_h · h_{t−1}")],
            [_merge("+ b")],
            [_act("tanh")],
            [_io("h_t  (B, D_h)")],
        ],
        [],
        None,
        _notes(
            used_in=[
                "Early language models and sequence taggers (pre-LSTM era)",
                "Hopfield-style associative memories (modernised in Ramsauer et al. 2020)",
                "Toy / teaching examples illustrating vanishing-gradient",
            ],
            tasks=[
                "Sequence modelling when sequences are very short and overhead matters",
                "Foundational analysis of recurrence and dynamical systems",
            ],
            pitfalls=[
                "Vanishing / exploding gradients with sequences > ~20 steps — almost never used in modern code.",
                "Stacking deep RNNs without orthogonal init or gradient clipping is unstable.",
                "Replaced in nearly all practical use by LSTM, GRU, Transformer, or SSMs.",
            ],
            see_also=[
                "[Elman RNN (Elman 1990)](https://onlinelibrary.wiley.com/doi/10.1207/s15516709cog1402_1)",
                "[On the difficulty of training RNNs (Pascanu et al. 2013)](https://arxiv.org/abs/1211.5063)",
            ],
        ),
    ),
    "LSTMCell": (
        "Long short-term memory cell with input/forget/cell/output gates.",
        "x_t, h_{t-1}, c_{t-1} → h_t, c_t",
        [
            [_io("x_t  (B, D_x)"), _io("h_{t−1}  (B, D_h)"), _io("c_{t−1}  (B, D_h)")],
            [_op("linear → (i, f, g, o)")],
            [_op("c_t = f ⊙ c_{t−1} + i ⊙ g")],
            [_op("h_t = o ⊙ tanh(c_t)")],
            [_io("h_t, c_t  (B, D_h)")],
        ],
        [],
        None,
        _notes(
            used_in=[
                "seq2seq neural machine translation (pre-Transformer)",
                "Speech recognition (DeepSpeech, Listen-Attend-Spell)",
                "Time-series forecasting, on-device keyword spotting",
                "AlphaStar (StarCraft II) policy core",
            ],
            tasks=[
                "Sequence modelling when memory is constant in time (streaming inference)",
                "Settings where compute / latency budget excludes attention",
            ],
            pitfalls=[
                "Forget-gate bias should be initialised to 1 (Jozefowicz et al. 2015) — most defaults init to 0.",
                "Per-step latency limits throughput on GPUs; batched-time matmul helps but not by much.",
                "Vanishing gradients are TAMED, not removed — bidirectional or attention helps for very long T.",
            ],
            see_also=[
                "[LSTM (Hochreiter & Schmidhuber 1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)",
                "[Empirical Eval of Gated RNNs (Jozefowicz et al. 2015)](https://proceedings.mlr.press/v37/jozefowicz15.html)",
            ],
        ),
    ),
    "GRUCell": (
        "Gated recurrent unit: reset (r), update (z) and candidate (n) gates.",
        "x_t, h_{t-1} → h_t",
        [
            [_io("x_t  (B, D_x)"), _io("h_{t−1}  (B, D_h)")],
            [_op("linear → (r, z, n)")],
            [_op("h_t = (1 − z) ⊙ h_{t−1} + z ⊙ n")],
            [_io("h_t  (B, D_h)")],
        ],
        [],
        None,
        _notes(
            used_in=[
                "Speech enhancement / VAD (smaller than LSTM)",
                "Slot-Attention's per-iteration update (Locatello et al. 2020)",
                "Reinforcement learning recurrent policies (R2D2, MuZero-RNN)",
            ],
            tasks=[
                "Drop-in lighter alternative to LSTM with similar quality on most benchmarks",
                "Update head for iterative refinement (Slot Attention, RAFT)",
            ],
            pitfalls=[
                "Fewer gates than LSTM — slightly less expressive on tasks needing long forgetting.",
                "Same per-step latency issues as LSTM; not faster on GPU despite simpler equations.",
            ],
            see_also=[
                "[GRU (Cho et al. 2014)](https://arxiv.org/abs/1406.1078)",
                "[Empirical Eval of Gated RNNs (Chung et al. 2014)](https://arxiv.org/abs/1412.3555)",
            ],
        ),
    ),
    "StateSpaceModel": (
        "Linear SSM with discretized A, B and an output mapping y = C h + D u.",
        "u:(B, T, D) → y:(B, T, D)",
        [
            [_io("u_t  (B, T, D)")],
            [_op("discretize  Ā = exp(Δ·A),  B̄ = (Ā − I)·A⁻¹·B")],
            [_op("scan  h_t = Ā h_{t−1} + B̄ u_t")],
            [_op("y_t = C h_t + D u_t")],
            [_io("y_t  (B, T, D)")],
        ],
        [],
        None,
        _notes(
            used_in=[
                "S4 (Gu et al. 2021) — first SSM to beat Transformers on Long Range Arena",
                "S5, H3, Hyena — variants exploring kernels and gating",
                "Backbone for Mamba's selective SSM",
            ],
            tasks=[
                "Very long-range sequence modelling (Path-X, audio, DNA)",
                "Tasks where memory must remain constant w.r.t. sequence length at inference",
            ],
            pitfalls=[
                "HiPPO initialisation of A is critical — random init is far worse.",
                "Numerical stability of the discretisation (ZOH vs bilinear) affects training.",
                "Linear SSMs lack content-based gating — outperformed by Mamba on language tasks.",
                "FFT-based parallel scan is fast for inference but needs careful padding.",
            ],
            see_also=[
                "[S4 / HiPPO (Gu et al. 2021)](https://arxiv.org/abs/2111.00396)",
                "[S5 (Smith et al. 2022)](https://arxiv.org/abs/2208.04933)",
                "[H3 (Fu et al. 2022)](https://arxiv.org/abs/2212.14052)",
            ],
        ),
    ),
    "MambaBlock": (
        "Selective state-space block: gates, conv, input-dependent SSM, gated output.",
        "(B, T, D) → (B, T, D)",
        [
            [_io("x  (B, T, D)")],
            [_op("in_proj  (D → 2·D')")],
            [_op("Conv 1D"), _act("SiLU (gate path)")],
            [_op("selective SSM  (A, Δ, B, C from x)")],
            [_merge("⊙ SiLU(gate)")],
            [_op("out_proj  (D' → D)")],
            [_io("y  (B, T, D)")],
        ],
        [],
        None,
        _notes(
            used_in=[
                "Mamba / Mamba-2 — language modelling at 3B+ scale",
                "Vision Mamba (ViM, VMamba), MedMamba for medical imaging",
                "Audio / DNA / time-series with very long context",
            ],
            tasks=[
                "Linear-time alternative to Transformers for million-token contexts",
                "Streaming inference (constant-memory state)",
            ],
            pitfalls=[
                "Selective scan needs a custom CUDA kernel for any reasonable speed (built into mamba-ssm).",
                "Selective parameters Δ, B, C must be input-DEPENDENT for the language-modelling win — "
                "passing constants degrades to S4 performance.",
                "Bf16 training is fine; fp16 sometimes needs careful loss scaling.",
                "No proven scaling law parity with Transformers yet — verify on YOUR distribution.",
            ],
            see_also=[
                "[Mamba (Gu & Dao 2023)](https://arxiv.org/abs/2312.00752)",
                "[Mamba-2 (Dao & Gu 2024)](https://arxiv.org/abs/2405.21060)",
            ],
        ),
    ),
}

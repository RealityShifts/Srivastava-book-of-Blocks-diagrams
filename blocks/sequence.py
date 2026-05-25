"""Recurrent and state-space sequence models."""

from typing import Dict

from dsl import Spec, _io, _op, _norm, _act, _attn, _merge, _emb, _loss, _ref  # noqa: F401

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
    ),
}

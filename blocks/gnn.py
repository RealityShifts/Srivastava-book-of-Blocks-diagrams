"""Graph neural network layers."""

from typing import Dict

from dsl import Spec, _io, _op, _norm, _act, _attn, _merge, _emb, _loss, _ref  # noqa: F401

CATEGORY = "gnn"
CATEGORY_DESC = "Graph neural network layers."

BLOCKS: Dict[str, Spec] = {
    "MessagePassing": (
        "Generic message-passing skeleton: message → aggregate → update.",
        "X:(N, F), edges:(2, E) → X':(N, F')",
        [
            [_io("node feats X  (N, F)"), _io("edges (i, j)  (2, E)")],
            [_op("message  m_ij = φ(h_i, h_j, e_ij)")],
            [_op("aggregate  m_j = Σ_{i ∈ N(j)} m_ij")],
            [_op("update  h_j' = ψ(h_j, m_j)")],
            [_io("X'  (N, F')")],
        ],
    ),
    "GraphConv": (
        "GCN propagation: H' = Â · X · W with symmetric normalisation.",
        "X:(N, F), A:(N, N) → H':(N, F')",
        [
            [_io("X  (N, F)"), _io("A  (N, N)")],
            [_op("Â = D^(−1/2) (A + I) D^(−1/2)")],
            [_op("H' = Â · X · W")],
            [_io("H'  (N, F')")],
        ],
    ),
    "GraphAttention": (
        "GAT: per-edge attention coefficients followed by neighbour aggregation.",
        "X:(N, F), edges → H':(N, F')",
        [
            [_io("X  (N, F)"), _io("edges  (2, E)")],
            [_op("linear projection W·h")],
            [_attn("attention coef α_ij per edge")],
            [_act("softmax over neighbours")],
            [_op("aggregate  Σ α_ij · W·h_j")],
            [_io("H'  (N, F')")],
        ],
    ),
}

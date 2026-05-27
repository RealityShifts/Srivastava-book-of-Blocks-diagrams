"""Graph neural network layers."""

from typing import Dict

from dsl import (  # noqa: F401
    Spec, _io, _op, _norm, _act, _attn, _merge, _emb, _loss, _ref, _notes,
)

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
        [],
        None,
        _notes(
            used_in=[
                "PyTorch Geometric's `MessagePassing` base class",
                "DeepMind GraphNets (Battaglia et al. 2018) — the unifying formulation",
                "Almost every GNN variant — GCN, GAT, GraphSAGE, GIN, MPNN, EGNN",
            ],
            tasks=[
                "Molecular property prediction (QM9, ZINC)",
                "Recommendation (PinSAGE, GraphSAGE on social graphs)",
                "Combinatorial optimisation, knowledge-graph reasoning",
            ],
            pitfalls=[
                "Over-smoothing at depth — after ~3 hops nodes converge to similar vectors. Mitigate with "
                "PairNorm, skip connections, or DropEdge.",
                "Over-squashing — long-range info bottlenecks through bridge edges; rewiring helps.",
                "Aggregation choice (sum / mean / max) changes expressiveness — sum is strictly more "
                "expressive (GIN paper).",
                "Mini-batching irregular graphs is non-trivial — use neighbor sampling or cluster-GCN.",
            ],
            see_also=[
                "[Relational Inductive Biases (Battaglia et al. 2018)](https://arxiv.org/abs/1806.01261)",
                "[MPNN (Gilmer et al. 2017)](https://arxiv.org/abs/1704.01212)",
                "[GIN / How Powerful Are GNNs (Xu et al. 2018)](https://arxiv.org/abs/1810.00826)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "Kipf & Welling GCN (citation, classification on Cora/Citeseer/Pubmed)",
                "Baseline GNN in nearly every paper since 2017",
                "Semi-supervised node classification, link prediction",
            ],
            tasks=[
                "Transductive node classification with a single fixed graph",
                "Strong baseline before reaching for attention or sampling",
            ],
            pitfalls=[
                "Symmetric normalisation needs SELF-loops (A+I) — forgetting them silently halves performance.",
                "Full-batch propagation needs the whole graph in memory — doesn't scale to billions of nodes.",
                "Limited expressive power — cannot distinguish certain regular structures (see WL test).",
                "Over-smoothing kicks in by layer 3–4; very deep GCNs underperform 2-layer GCNs.",
            ],
            see_also=[
                "[GCN (Kipf & Welling 2016)](https://arxiv.org/abs/1609.02907)",
                "[How Powerful Are GNNs (Xu et al. 2018)](https://arxiv.org/abs/1810.00826)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "GAT, GATv2 — node classification, relation prediction",
                "Mesh / point-cloud learning (Point Transformer)",
                "Heterogeneous graph attention (HAN, HGT)",
            ],
            tasks=[
                "Tasks needing differential edge importance",
                "Heterogeneous graphs where edge types carry semantics",
            ],
            pitfalls=[
                "Original GAT (v1) has a static attention pattern that GATv2 fixes — use GATv2.",
                "Softmax over neighbours requires segment-softmax (scatter softmax); naïve dense impl OOMs.",
                "Attention quality degrades when nodes have very high degree — sample or sparsify.",
                "Multi-head attention helps but multiplies compute; concatenation vs averaging changes "
                "downstream dim.",
            ],
            see_also=[
                "[GAT (Veličković et al. 2017)](https://arxiv.org/abs/1710.10903)",
                "[GATv2 (Brody et al. 2021)](https://arxiv.org/abs/2105.14491)",
            ],
        ),
    ),
}

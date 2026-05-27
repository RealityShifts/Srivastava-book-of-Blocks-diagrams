# GraphAttention

> GAT: per-edge attention coefficients followed by neighbour aggregation.

**Shapes:** `X:(N, F), edges → H':(N, F')`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["X  (N, F)"]:::io
    n0_1["edges  (2, E)"]:::io
    n1_0["linear projection W·h"]:::op
    n2_0["attention coef α_ij per edge"]:::attn
    n3_0["softmax over neighbours"]:::act
    n4_0["aggregate  Σ α_ij · W·h_j"]:::op
    n5_0["H'  (N, F')"]:::io
    n0_0 --> n1_0
    n0_1 --> n1_0
    n1_0 --> n2_0
    n2_0 --> n3_0
    n3_0 --> n4_0
    n4_0 --> n5_0
    classDef io fill:#f1f5f9,stroke:#334155,stroke-width:1.4px,color:#0f172a
    classDef op fill:#dbeafe,stroke:#1d4ed8,stroke-width:1.4px,color:#1e3a8a
    classDef norm fill:#dcfce7,stroke:#15803d,stroke-width:1.4px,color:#14532d
    classDef act fill:#ffedd5,stroke:#c2410c,stroke-width:1.4px,color:#7c2d12
    classDef attn fill:#ede9fe,stroke:#6d28d9,stroke-width:1.4px,color:#4c1d95
    classDef merge fill:#fef3c7,stroke:#b45309,stroke-width:1.4px,color:#78350f
    classDef emb fill:#fef9c3,stroke:#a16207,stroke-width:1.4px,color:#713f12
    classDef loss fill:#fee2e2,stroke:#b91c1c,stroke-width:1.4px,color:#7f1d1d
    classDef ctrl fill:#f5f5f4,stroke:#52525b,stroke-width:1.4px,color:#27272a
    classDef ref fill:#e0f2fe,stroke:#0369a1,stroke-width:2px,color:#0c4a6e,stroke-dasharray: 4 2
```

**Used in**

- GAT, GATv2 — node classification, relation prediction
- Mesh / point-cloud learning (Point Transformer)
- Heterogeneous graph attention (HAN, HGT)

**Tasks**

- Tasks needing differential edge importance
- Heterogeneous graphs where edge types carry semantics

**Common pitfalls**

- Original GAT (v1) has a static attention pattern that GATv2 fixes — use GATv2.
- Softmax over neighbours requires segment-softmax (scatter softmax); naïve dense impl OOMs.
- Attention quality degrades when nodes have very high degree — sample or sparsify.
- Multi-head attention helps but multiplies compute; concatenation vs averaging changes downstream dim.

**See also**

- [GAT (Veličković et al. 2017)](https://arxiv.org/abs/1710.10903)
- [GATv2 (Brody et al. 2021)](https://arxiv.org/abs/2105.14491)

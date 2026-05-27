# GraphConv

> GCN propagation: H' = Â · X · W with symmetric normalisation.

**Shapes:** `X:(N, F), A:(N, N) → H':(N, F')`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["X  (N, F)"]:::io
    n0_1["A  (N, N)"]:::io
    n1_0["Â = D^(−1/2) (A + I) D^(−1/2)"]:::op
    n2_0["H' = Â · X · W"]:::op
    n3_0["H'  (N, F')"]:::io
    n0_0 --> n1_0
    n0_1 --> n1_0
    n1_0 --> n2_0
    n2_0 --> n3_0
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

- Kipf & Welling GCN (citation, classification on Cora/Citeseer/Pubmed)
- Baseline GNN in nearly every paper since 2017
- Semi-supervised node classification, link prediction

**Tasks**

- Transductive node classification with a single fixed graph
- Strong baseline before reaching for attention or sampling

**Common pitfalls**

- Symmetric normalisation needs SELF-loops (A+I) — forgetting them silently halves performance.
- Full-batch propagation needs the whole graph in memory — doesn't scale to billions of nodes.
- Limited expressive power — cannot distinguish certain regular structures (see WL test).
- Over-smoothing kicks in by layer 3–4; very deep GCNs underperform 2-layer GCNs.

**See also**

- [GCN (Kipf & Welling 2016)](https://arxiv.org/abs/1609.02907)
- [How Powerful Are GNNs (Xu et al. 2018)](https://arxiv.org/abs/1810.00826)

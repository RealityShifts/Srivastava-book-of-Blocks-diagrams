# MessagePassing

> Generic message-passing skeleton: message → aggregate → update.

**Shapes:** `X:(N, F), edges:(2, E) → X':(N, F')`

```mermaid
flowchart TD
    n0_0["node feats X  (N, F)"]:::io
    n0_1["edges (i, j)  (2, E)"]:::io
    n1_0["message  m_ij = φ(h_i, h_j, e_ij)"]:::op
    n2_0["aggregate  m_j = Σ_{i ∈ N(j)} m_ij"]:::op
    n3_0["update  h_j' = ψ(h_j, m_j)"]:::op
    n4_0["X'  (N, F')"]:::io
    n0_0 --> n1_0
    n0_1 --> n1_0
    n1_0 --> n2_0
    n2_0 --> n3_0
    n3_0 --> n4_0
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

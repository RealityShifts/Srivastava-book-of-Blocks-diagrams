# ShiftedWindowAttention

> Cyclic-shift variant that lets adjacent windows exchange information.

**Shapes:** `(B, N, C) → (B, N, C)  with implicit (H, W)`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["x  (B, H, W, C)"]:::io
    n1_0["cyclic shift (−w/2)"]:::op
    n2_0["window partition"]:::op
    n3_0["MHA + shifted-window mask"]:::attn
    n4_0["window reverse"]:::op
    n5_0["cyclic shift (+w/2)"]:::op
    n6_0["y  (B, H, W, C)"]:::io
    n0_0 --> n1_0
    n1_0 --> n2_0
    n2_0 --> n3_0
    n3_0 --> n4_0
    n4_0 --> n5_0
    n5_0 --> n6_0
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

- Swin Transformer's odd-numbered blocks (SW-MSA after W-MSA)
- Hybrid CNN-transformer detectors using Swin as backbone

**Tasks**

- Letting adjacent windows exchange tokens without paying global-attention cost

**Common pitfalls**

- The shift mask is non-trivial — getting the connectivity wrong silently halves quality.
- Cyclic shift must be exactly undone after attention; off-by-one rolls leak features.
- Combine carefully with relative position bias — shift changes which (i,j) pairs are valid.

**See also**

- [Swin Transformer (Liu et al. 2021)](https://arxiv.org/abs/2103.14030)

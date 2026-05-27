# Linear

> Affine projection y = x · Wᵀ + b.

**Shapes:** `(B, in) → (B, out)`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["x  (B, in)"]:::io
    n1_0["matmul  x · Wᵀ"]:::op
    n2_0["+ bias"]:::op
    n3_0["y  (B, out)"]:::io
    n0_0 --> n1_0
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

- Every MLP / classifier / projection head ever
- Q/K/V/O projections inside attention
- Final logits head on language and vision models
- Patch / channel mixers in MLP-Mixer-style nets

**Tasks**

- Any time you need a learned `(B, in) → (B, out)` map
- Bottleneck / expansion in residual blocks
- Read-out heads for regression and classification

**Common pitfalls**

- Initialisation matters — Kaiming for ReLU-family, Xavier for tanh/sigmoid; wrong init can stall training entirely.
- Huge final layers (e.g. softmax over 50k tokens) dominate parameter count — tie input/output embeddings or factorise.
- Forgetting `bias=False` before a BatchNorm is harmless but wasteful.

**See also**

- [Kaiming init (He et al. 2015)](https://arxiv.org/abs/1502.01852)
- [Xavier init (Glorot & Bengio 2010)](https://proceedings.mlr.press/v9/glorot10a.html)

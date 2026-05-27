# LowRankLinear

> Two stacked linears whose product approximates a full matrix W ≈ U Vᵀ.

**Shapes:** `(B, in) → (B, out)`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["x  (B, in)"]:::io
    n1_0["down: linear  in → r"]:::op
    n2_0["up: linear   r → out"]:::op
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

- ALBERT factorised embeddings
- LoRA / DoRA / VeRA — PEFT methods built on low-rank residuals
- Compressed inference (post-training SVD of dense layers)

**Tasks**

- Parameter reduction when in × out ≫ r·(in + out)
- Building block for adapters and intrinsic-dimension fine-tuning

**Common pitfalls**

- Initialisation: down with N(0, σ), up zero — gives identity start when used as residual; wrong order produces gradient surprises.
- Rank r should be < min(in, out) for any saving; too small loses expressivity.
- Two matmuls have a latency cost on small GPUs vs a single fused matmul.

**See also**

- [ALBERT (Lan et al. 2019)](https://arxiv.org/abs/1909.11942)
- [LoRA (Hu et al. 2021)](https://arxiv.org/abs/2106.09685)
- [Intrinsic Dimensionality (Aghajanyan et al. 2020)](https://arxiv.org/abs/2012.13255)

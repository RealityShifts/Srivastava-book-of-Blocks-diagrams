# EqualLinear

> Linear with equalized learning rate: weight scaled at runtime by gain/√fan_in.

**Shapes:** `(B, in) → (B, out)`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["x  (B, in)"]:::io
    n1_0["matmul  x · (W · s)"]:::op
    n2_0["+ bias · lr_mul"]:::op
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

- PGGAN, StyleGAN, StyleGAN2/3 — every fully-connected layer
- MSG-GAN, GANformer — high-quality image synthesis

**Tasks**

- Maintaining uniform per-parameter learning rate across layers of vastly different fan-in
- Stable training of progressively-growing or modulated generators

**Common pitfalls**

- Naïve replacement of Linear with EqualLinear without adjusting LR ratios degrades quality.
- Scale factor is APPLIED at runtime, not at init — keeping weights N(0,1) and scaling on forward is the whole point.
- Mixing standard and equalised layers in the same net usually hurts; commit to one.

**See also**

- [PGGAN (Karras et al. 2017)](https://arxiv.org/abs/1710.10196)
- [StyleGAN (Karras et al. 2019)](https://arxiv.org/abs/1812.04948)

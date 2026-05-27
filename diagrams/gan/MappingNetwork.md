# MappingNetwork

> 8-layer MLP with reduced LR that maps z → w (StyleGAN).

**Shapes:** `z:(B, D) → w:(B, D)`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["z  (B, D)"]:::io
    n1_0["PixelNorm"]:::op
    n2_0["EqualLinear × 8  (lr_mul = 0.01)"]:::op
    n3_0["w  (B, D)"]:::io
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

- StyleGAN v1 / v2 / v3 — the f: z → w map
- StyleGAN-T text-to-image

**Tasks**

- Disentangling latent noise into a more linear style space W
- Enabling style mixing and W+ inversion / editing

**Common pitfalls**

- Without the low LR multiplier (~0.01), the mapping net dominates training updates.
- Skipping PixelNorm on z makes early training unstable.
- Depth of 8 is empirical; smaller maps under-disentangle, deeper maps offer no gains.

**See also**

- [StyleGAN (Karras et al. 2019)](https://arxiv.org/abs/1812.04948)

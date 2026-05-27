# ResidualBlock

> Two conv-norm-act stack with identity skip and post-add activation.

**Shapes:** `(B, C, H, W) → (B, C, H, W)`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["x  (B, C, H, W)"]:::io
    n1_0["Conv 3×3"]:::op
    n2_0["Norm"]:::norm
    n3_0["ReLU"]:::act
    n4_0["Conv 3×3"]:::op
    n5_0["Norm"]:::norm
    n6_0["+"]:::merge
    n7_0["ReLU"]:::act
    n8_0["y  (B, C, H, W)"]:::io
    n0_0 --> n1_0
    n1_0 --> n2_0
    n2_0 --> n3_0
    n3_0 --> n4_0
    n4_0 --> n5_0
    n5_0 --> n6_0
    n6_0 --> n7_0
    n7_0 --> n8_0
    n0_0 -.->|"skip"| n6_0
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

- ResNet-18/34/50/101/152 — the most-cited deep-learning architecture
- U-Net diffusion noise prediction backbone
- AlphaGo / AlphaZero policy-value tower
- Almost every modern CNN backbone (ConvNeXt, RegNet)

**Tasks**

- Enabling depth > 30 layers without vanishing gradients
- Optimisation landscape smoothing — skips create much shorter back-prop paths

**Common pitfalls**

- Pre-activation order (BN → ReLU → Conv) often trains deeper nets more stably than post-act.
- When channel count changes across the skip, you need a 1×1 projection or zero-pad — naïve add will dimension-error.
- ReLU AFTER the add caps activations to ≥ 0, which can hurt; SiLU/GELU is sometimes better.

**See also**

- [ResNet (He et al. 2015)](https://arxiv.org/abs/1512.03385)
- [Pre-activation ResNet (He et al. 2016)](https://arxiv.org/abs/1603.05027)
- [Identity Mappings paper analysis](https://arxiv.org/abs/1603.05027)

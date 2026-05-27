# UNetResBlock

> Residual block conditioned on a time embedding (added between the two convs).

**Shapes:** `x:(B, C, H, W), t:(B, D) → (B, C', H, W)`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["x  (B, C, H, W)"]:::io
    n1_0["GroupNorm"]:::norm
    n2_0["SiLU"]:::act
    n3_0["Conv 3×3"]:::op
    n4_0["+ MLP(t_emb)  (broadcast)"]:::merge
    n5_0["GroupNorm"]:::norm
    n6_0["SiLU"]:::act
    n7_0["Conv 3×3"]:::op
    n8_0["+"]:::merge
    n9_0["y  (B, C', H, W)"]:::io
    n0_0 --> n1_0
    n1_0 --> n2_0
    n2_0 --> n3_0
    n3_0 --> n4_0
    n4_0 --> n5_0
    n5_0 --> n6_0
    n6_0 --> n7_0
    n7_0 --> n8_0
    n8_0 --> n9_0
    n0_0 -.->|"skip"| n8_0
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

- DDPM noise predictor
- Stable Diffusion U-Net (every resolution level)
- Latent video diffusion (SVD, AnimateDiff)

**Tasks**

- Time-conditioned feature extraction in a diffusion backbone

**Common pitfalls**

- Time embedding is added BETWEEN the two convs (not at the input) — placement matters.
- GroupNorm groups must divide channel count; 32 is standard, mismatches silently break.
- Skip connection must match channel count — add a 1×1 projection when C' ≠ C.

**See also**

- [DDPM (Ho et al. 2020)](https://arxiv.org/abs/2006.11239)
- [GroupNorm (Wu & He 2018)](https://arxiv.org/abs/1803.08494)

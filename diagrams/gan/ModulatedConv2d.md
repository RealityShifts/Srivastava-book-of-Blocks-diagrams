# ModulatedConv2d

> StyleGAN2 modulated conv: scale weights by style, demodulate, then convolve.

**Shapes:** `x:(B, C, H, W), w:(B, D) → y`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["x  (B, C, H, W)"]:::io
    n0_1["w  (B, D)"]:::io
    n1_0["weights W"]:::op
    n1_1["A: linear w → s"]:::op
    n2_0["W' = W · s"]:::op
    n3_0["demod: W'' = W' / ||W'||"]:::op
    n4_0["Conv with W''"]:::op
    n5_0["y  (B, C', H, W)"]:::io
    n0_0 --> n1_0
    n0_1 --> n1_1
    n1_0 --> n2_0
    n1_1 --> n2_0
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

- StyleGAN2 / StyleGAN3 generator (replaces AdaIN style injection)
- Modulated decoders in modern image synthesis

**Tasks**

- Style injection through weight modulation rather than feature statistics
- Eliminating droplet artefacts caused by AdaIN

**Common pitfalls**

- Demodulation factor is computed PER OUTPUT CHANNEL — bookkeeping error breaks training.
- Grouped convolution implementation per sample is needed (since W differs per batch element) — naïve loop is too slow; use grouped conv with B groups.
- Style vector s is BROADCAST over spatial — confusing AdaIN's spatial γ/β with this is wrong.

**See also**

- [StyleGAN2 (Karras et al. 2020)](https://arxiv.org/abs/1912.04958)
- [StyleGAN3 (Karras et al. 2021)](https://arxiv.org/abs/2106.12423)

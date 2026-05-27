# ProgressiveGrowing

> Fade in a new high-res block via α-blend with the previous resolution (PGGAN).

**Shapes:** `x → upsampled / new block → y`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["x  (B, C, H, W)"]:::io
    n1_0["old layers (already trained)"]:::op
    n2_0["new high-res layer"]:::op
    n3_0["(1−α) old  +  α new"]:::merge
    n4_0["y  (B, C', 2H, 2W)"]:::io
    n0_0 --> n1_0
    n1_0 --> n2_0
    n2_0 --> n3_0
    n3_0 --> n4_0
    n1_0 -.->|"skip"| n3_0
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

- PGGAN — first method to scale GANs to 1024² faces
- Some MSG-GAN and progressive diffusion variants

**Tasks**

- Stable training of very high-resolution GANs
- Reducing training-divergence risk by gradually expanding capacity

**Common pitfalls**

- α schedule must be slow enough — abrupt fade-in destabilises training.
- Discriminator and generator must grow in LOCK-STEP; mismatched resolutions diverge.
- Replaced in modern recipes by StyleGAN2/3 single-shot training, which is simpler.

**See also**

- [PGGAN (Karras et al. 2017)](https://arxiv.org/abs/1710.10196)

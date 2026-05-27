# ConvBlock

> Conv → Norm → Activation, the canonical CNN unit.

**Shapes:** `(B, C, H, W) → (B, C', H, W)`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["x  (B, C, H, W)"]:::io
    n1_0["Conv 2D"]:::op
    n2_0["BatchNorm / GroupNorm / LayerNorm"]:::norm
    n3_0["ReLU / GELU / SiLU / Mish"]:::act
    n4_0["y  (B, C', H, W)"]:::io
    n0_0 --> n1_0
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

**Used in**

- VGG, ResNet, EfficientNet — the canonical CNN stem & body
- U-Net encoder/decoder layers
- Generator / discriminator backbones in GANs and diffusion

**Tasks**

- Image classification / segmentation / detection backbones
- Feature extraction prior to global pooling or upsampling

**Common pitfalls**

- Conv → BN → ReLU vs BN → ReLU → Conv (pre-act) matters at depth — pre-activation is more stable for very deep nets.
- Bias on the conv before a BatchNorm is redundant (BN absorbs it).
- Padding mismatches silently change output spatial dims — sanity-check with a forward pass.

**See also**

- [VGG (Simonyan & Zisserman 2014)](https://arxiv.org/abs/1409.1556)
- [BatchNorm (Ioffe & Szegedy 2015)](https://arxiv.org/abs/1502.03167)
- [Pre-activation ResNet (He et al. 2016)](https://arxiv.org/abs/1603.05027)

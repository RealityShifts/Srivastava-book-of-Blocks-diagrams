# PixelShuffleUpsample

> Sub-pixel upsampling: rearrange r² channels into r×r spatial blocks.

**Shapes:** `(B, C, H, W) → (B, C, r·H, r·W)`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["x  (B, C, H, W)"]:::io
    n1_0["Conv → C·r² channels"]:::op
    n2_0["PixelShuffle r"]:::op
    n3_0["y  (B, C, r·H, r·W)"]:::io
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

- ESPCN — original sub-pixel CNN for super-resolution
- EDSR, RCAN, ESRGAN — modern SR architectures
- Diffusion-model decoders / VAE upsampling stages

**Tasks**

- Image super-resolution
- Upsampling within a decoder without introducing checkerboard artefacts

**Common pitfalls**

- Initialisation matters — sub-pixel layers benefit from ICNR init to avoid checkerboards.
- Channel count blows up before the shuffle (C·r²) — memory pressure on large feature maps.
- Equivalent to a learned transpose-conv but typically cheaper at the same quality.

**See also**

- [Sub-pixel CNN / ESPCN (Shi et al. 2016)](https://arxiv.org/abs/1609.05158)
- [Checkerboard Artifacts (Odena et al. 2016)](https://distill.pub/2016/deconv-checkerboard/)
- [ICNR init (Aitken et al. 2017)](https://arxiv.org/abs/1707.02937)

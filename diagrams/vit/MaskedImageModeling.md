# MaskedImageModeling

> MAE/BEiT objective: random-mask patches, encode visible tokens, reconstruct.

**Shapes:** `(B, N, D) → masked, (B, N, p²·C) reconstructed`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["image patches  (B, N, p²·C)"]:::io
    n1_0["random mask (≈ 75 %)"]:::op
    n2_0["encoder  (visible tokens only)"]:::op
    n3_0["insert [MASK] tokens"]:::op
    n4_0["decoder"]:::op
    n5_0["reconstructed patches  (B, N, p²·C)"]:::io
    n0_0 --> n1_0
    n1_0 --> n2_0
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

- MAE — Masked Autoencoders pre-training (He et al. 2021)
- BEiT v1 / v2 — predicts discrete tokens instead of pixels
- VideoMAE, SiT — video and audio extensions

**Tasks**

- Self-supervised pre-training of Vision Transformers
- Few-label fine-tuning where labeled data is scarce

**Common pitfalls**

- 75 % masking ratio is empirically best for images; lower ratios under-train the encoder.
- Decoder is intentionally tiny — making it deeper does NOT help downstream tasks.
- Pixel-reconstruction loss is unnormalised — pre-normalise patches for stable training.
- Heavy memory at sequence-level reordering; tensor-shuffle indexing is easy to get wrong.

**See also**

- [MAE (He et al. 2021)](https://arxiv.org/abs/2111.06377)
- [BEiT (Bao et al. 2021)](https://arxiv.org/abs/2106.08254)
- [VideoMAE (Tong et al. 2022)](https://arxiv.org/abs/2203.12602)

# DeformableAttention

> Attention that samples a small set of keys at learned offsets per query (Deformable DETR).

**Shapes:** `(B, N, C) → (B, N, C)`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["x  (B, N, C)"]:::io
    n1_0["MLP → reference + sampling offsets"]:::op
    n2_0["bilinear sample features"]:::op
    n3_0["attention weights (per head, per point)"]:::attn
    n4_0["weighted sum + output proj"]:::merge
    n5_0["y  (B, N, C)"]:::io
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

- Deformable DETR — convergence in ~10× fewer epochs than DETR
- DINO, DN-DETR, H-DETR — modern detection transformers
- 3D detection (Deformable 3D-DETR)

**Tasks**

- Detection / segmentation with transformer backbones
- Multi-scale feature aggregation in attention with sub-quadratic cost

**Common pitfalls**

- Each query samples K points per level — too small (K < 4) hurts recall, too large kills speed.
- Reference-point initialisation matters; query anchors are typically tied to encoder positions.
- Implementation needs custom CUDA op for full speed (ms-deformable-attention).

**See also**

- [Deformable DETR (Zhu et al. 2020)](https://arxiv.org/abs/2010.04159)
- [DINO (Zhang et al. 2022)](https://arxiv.org/abs/2203.03605)

# ProjectionHead

> Two-layer MLP + L2 normalisation, used for contrastive / representation learning.

**Shapes:** `(B, D) → (B, D')`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["x  (B, D)"]:::io
    n1_0["linear"]:::op
    n2_0["GELU"]:::act
    n3_0["linear"]:::op
    n4_0["L2 normalize"]:::op
    n5_0["z  (B, D')"]:::emb
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

- SimCLR, MoCo, BYOL — contrastive image SSL
- CLIP, BLIP — image-text alignment heads
- Sentence-BERT, GTE, BGE — text embedding models

**Tasks**

- Mapping features to a contrastive / retrieval space (often discarded after pretraining)
- Stable training of contrastive losses (the head absorbs feature distortion)

**Common pitfalls**

- L2 normalisation is essential before computing cosine similarity / temperature softmax.
- Projector is typically DISCARDED at inference — use the pre-projection features instead (SimCLR §6, MoCo v2).
- Width and depth of the projector matter more than people expect (BYOL has 4096-d hidden).

**See also**

- [SimCLR (Chen et al. 2020)](https://arxiv.org/abs/2002.05709)
- [MoCo v2 (Chen et al. 2020)](https://arxiv.org/abs/2003.04297)
- [BYOL (Grill et al. 2020)](https://arxiv.org/abs/2006.07733)

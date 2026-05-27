# CLSToken

> Prepend a learnable [CLS] token to each sequence (used as the global representation).

**Shapes:** `(B, N, D) → (B, 1+N, D)`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["tokens  (B, N, D)"]:::io
    n1_0["prepend [CLS]  (learnable, broadcast over batch)"]:::op
    n2_0["tokens'  (B, 1+N, D)"]:::io
    n0_0 --> n1_0
    n1_0 --> n2_0
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

- BERT — [CLS] for sentence classification
- ViT, DeiT — image classification head reads only the CLS token
- DINO — CLS distillation between teacher and student

**Tasks**

- Producing a single global representation from a Transformer encoder
- Acting as a per-input prompt position when fine-tuning

**Common pitfalls**

- CLS pooling can be sub-optimal vs mean-pooling of patch tokens — verify empirically.
- When using register tokens (Darcet et al. 2023), additional learnable tokens reduce attention artefacts and may replace CLS for retrieval.
- Position embedding must reserve slot 0 for CLS — off-by-one bugs corrupt all positions.

**See also**

- [BERT (Devlin et al. 2018)](https://arxiv.org/abs/1810.04805)
- [ViT (Dosovitskiy et al. 2020)](https://arxiv.org/abs/2010.11929)
- [Register Tokens (Darcet et al. 2023)](https://arxiv.org/abs/2309.16588)

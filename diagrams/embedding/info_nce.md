# info_nce

> Pairwise InfoNCE used by SimCLR / MoCo.

**Shapes:** `z₁, z₂:(B, D) → loss`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["z₁, z₂  (B, D)"]:::io
    n1_0["L2 normalize"]:::op
    n2_0["logits = z₁ · z₂ᵀ / τ"]:::op
    n3_0["cross-entropy with diagonal targets"]:::loss
    n4_0["loss  (1,)"]:::io
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

- SimCLR, MoCo, MoCo-v3 — image SSL
- CPC (van den Oord et al. 2018) — sequential predictive coding
- DINO-style methods (with teacher target)

**Tasks**

- Self-supervised representation learning from augmented views
- Cross-modal alignment (when restricted to one direction)

**Common pitfalls**

- Temperature τ ~0.07–0.2 is typical — far off and the loss saturates or vanishes.
- Need many negatives; without a memory bank (MoCo) or large batch, quality plateaus.
- Hard-negative mining helps in fine-grained settings (face recognition, retrieval).

**See also**

- [CPC / InfoNCE (van den Oord et al. 2018)](https://arxiv.org/abs/1807.03748)
- [SimCLR (Chen et al. 2020)](https://arxiv.org/abs/2002.05709)
- [MoCo (He et al. 2019)](https://arxiv.org/abs/1911.05722)

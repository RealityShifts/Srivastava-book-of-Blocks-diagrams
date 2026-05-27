# TokenPruner

> Drop low-importance tokens before further attention layers.

**Shapes:** `(B, N, D) → (B, K, D),  K < N`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["tokens  (B, N, D)"]:::io
    n1_0["importance score per token"]:::op
    n2_0["top-k keep"]:::op
    n3_0["reduced tokens  (B, K, D)"]:::io
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

- DynamicViT — token pruning in Vision Transformers
- EViT, TokenLearner — adaptive token reduction
- ToMe (Token Merging) — merges similar tokens rather than dropping

**Tasks**

- Cutting attention cost on long-token inputs (high-res images, long sequences)
- Dynamic compute allocation per sample

**Common pitfalls**

- Hard top-k is non-differentiable — typical impls use a soft mask during training with ST-gumbel or score-based ranking, then hard-prune at inference.
- Dropping CLS token is catastrophic; always keep it in the keep-set.
- Variable token count per batch breaks fixed-shape tensors — pad and mask.

**See also**

- [DynamicViT (Rao et al. 2021)](https://arxiv.org/abs/2106.02034)
- [Token Merging / ToMe (Bolya et al. 2022)](https://arxiv.org/abs/2210.09461)

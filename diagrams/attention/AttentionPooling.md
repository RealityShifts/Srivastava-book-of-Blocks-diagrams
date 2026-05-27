# AttentionPooling

> A learnable query attends over a sequence to produce a single pooled vector.

**Shapes:** `(B, T, D) → (B, D)`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["learnable query  (1, D)"]:::io
    n0_1["x  (B, T, D)"]:::io
    n1_0["MHA(query, x, x)"]:::attn
    n2_0["pooled  (B, D)"]:::io
    n0_0 --> n1_0
    n0_1 --> n1_0
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

- CLIP text/image projection head (last-token / attention pooling variants)
- Set Transformer / PMA — pooling-by-multihead-attention
- Speech / video classification heads

**Tasks**

- Reducing a variable-length sequence to a single (or k) summary vector
- Replacing mean / max pooling for set-structured inputs

**Common pitfalls**

- Single learnable query is a narrow bottleneck — use k > 1 queries (Set Transformer PMA) for richer summaries.
- Causal masking is rarely needed here, but accidentally inheriting it from a parent model breaks pooling silently.

**See also**

- [Set Transformer (Lee et al. 2018)](https://arxiv.org/abs/1810.00825)
- [CLIP (Radford et al. 2021)](https://arxiv.org/abs/2103.00020)

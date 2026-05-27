# LearnedPositionalEmbedding

> A separate learned vector per absolute position, added to the token embedding.

**Shapes:** `(B, T, D) → (B, T, D)`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["positions 0..T−1  (T,)"]:::io
    n1_0["lookup  P[pos]"]:::op
    n2_0["+ token emb"]:::merge
    n3_0["pos-encoded tokens  (B, T, D)"]:::emb
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

- BERT, RoBERTa, GPT-2 — original Transformer-era position encoding
- ViT — learned position embeddings per patch

**Tasks**

- Position injection for fixed-length sequences

**Common pitfalls**

- Cannot extrapolate beyond the training length without interpolation/extension hacks.
- Adds D × T_max parameters — large at long contexts.
- Largely replaced by RoPE / ALiBi in modern LLMs for length extrapolation reasons.

**See also**

- [BERT (Devlin et al. 2018)](https://arxiv.org/abs/1810.04805)
- [ViT (Dosovitskiy et al. 2020)](https://arxiv.org/abs/2010.11929)

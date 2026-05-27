# SlotAttention

> Iterated cross-attention from a small set of slots into per-pixel features (SlotAttention).

**Shapes:** `feats:(B, N, D), slots:(B, K, D) → slots`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["features  (B, N, D)"]:::io
    n0_1["slots  (B, K, D)"]:::io
    n1_0["cross-attn  (slots query)"]:::attn
    n2_0["GRU update"]:::op
    n3_0["× T iterations"]:::op
    n4_0["updated slots  (B, K, D)"]:::io
    n0_0 --> n1_0
    n0_1 --> n1_0
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

- Slot Attention (Locatello et al. 2020)
- OSRT / SAVi for video object discovery
- Object-centric world models in RL

**Tasks**

- Unsupervised object discovery / instance segmentation
- Compositional scene representations from images / videos

**Common pitfalls**

- Number of slots K is a strong inductive bias — fewer than objects merges, more leaves slots empty.
- Slot symmetry-breaking is induced by random init each forward — beware deterministic fixes that collapse all slots.
- Softmax over slots (competition) is essential — without it, all slots converge to the same content.
- Works on small / synthetic scenes; transfer to realistic images is an active research area.

**See also**

- [Slot Attention (Locatello et al. 2020)](https://arxiv.org/abs/2006.15055)
- [SAVi (Kipf et al. 2021)](https://arxiv.org/abs/2111.12594)

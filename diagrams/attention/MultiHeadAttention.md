# MultiHeadAttention

> Standard scaled dot-product attention with H heads and an output projection.

**Shapes:** `Q,K,V:(B, T, D) → (B, T, D)`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["Q  (B, T, D)"]:::io
    n0_1["K  (B, T, D)"]:::io
    n0_2["V  (B, T, D)"]:::io
    n1_0["linear Q"]:::op
    n1_1["linear K"]:::op
    n1_2["linear V"]:::op
    n2_0["split into H heads"]:::op
    n3_0["Q · Kᵀ / √d_h"]:::op
    n4_0["softmax (+ optional mask)"]:::act
    n5_0["· V"]:::op
    n6_0["concat heads"]:::op
    n7_0["output linear"]:::op
    n8_0["y  (B, T, D)"]:::io
    n0_0 --> n1_0
    n0_1 --> n1_1
    n0_2 --> n1_2
    n1_0 --> n2_0
    n1_1 --> n2_0
    n1_2 --> n2_0
    n2_0 --> n3_0
    n3_0 --> n4_0
    n4_0 --> n5_0
    n5_0 --> n6_0
    n6_0 --> n7_0
    n7_0 --> n8_0
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

- Transformer encoder / decoder — the original 'Attention is All You Need'
- BERT, GPT, T5, ViT and every LLM and ViT derivative
- Cross-modal models (CLIP, BLIP-2, Flamingo) for vision–language alignment

**Tasks**

- Modelling long-range dependencies without convolutional locality bias
- Conditioning one sequence on another (cross-attention)
- Set / unordered-input processing where position is encoded explicitly

**Common pitfalls**

- Quadratic memory and compute in sequence length — switch to FlashAttention or sliding-window variants for T > a few thousand.
- Forgetting `1/√d_h` scaling makes the softmax saturate at long heads (poor gradients).
- Mask shape / dtype mismatches silently break causal attention — assert with a unit test.

**See also**

- [Attention Is All You Need (Vaswani et al. 2017)](https://arxiv.org/abs/1706.03762)
- [FlashAttention (Dao et al. 2022)](https://arxiv.org/abs/2205.14135)
- [Multi-Query Attention (Shazeer 2019)](https://arxiv.org/abs/1911.02150)
- [Grouped-Query Attention (Ainslie et al. 2023)](https://arxiv.org/abs/2305.13245)

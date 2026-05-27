# RelativePositionBias

> Learnable bias added to attention logits as a function of (i − j).

**Shapes:** `logits:(*, T, T) → biased logits`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["position pairs (i, j)  (T, T)"]:::io
    n1_0["bias table lookup B[i − j]"]:::op
    n2_0["+ to attn logits"]:::merge
    n3_0["biased logits  (*, T, T)"]:::io
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

- T5 (bucketed log-spaced relative positions)
- Swin Transformer (2D relative position bias)
- DeBERTa, ALiBi (linear bias instead of learned table)

**Tasks**

- Position-aware attention without losing translation equivariance
- Extrapolating to lengths beyond training (especially with ALiBi)

**Common pitfalls**

- Naïve table size is O(T²); use log-bucketing or relative-distance clipping for long T.
- Bias shape may be incompatible with FlashAttention — check before adopting.
- Sharing the table across heads vs per-head is an under-appreciated knob.

**See also**

- [Self-Attention with Relative Position (Shaw et al. 2018)](https://arxiv.org/abs/1803.02155)
- [T5 (Raffel et al. 2020)](https://arxiv.org/abs/1910.10683)
- [ALiBi (Press et al. 2021)](https://arxiv.org/abs/2108.12409)

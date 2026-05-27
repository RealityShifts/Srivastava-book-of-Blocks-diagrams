# LinearAttention

> Kernelised attention with O(N) cost via Q · (Kᵀ V).

**Shapes:** `(B, T, D) → (B, T, D)`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["Q, K, V  (B, T, D)"]:::io
    n1_0["φ(Q),  φ(K)"]:::op
    n2_0["Kᵀ · V"]:::op
    n3_0["Q · (Kᵀ V)"]:::op
    n4_0["normalise by Q · Σ K"]:::op
    n5_0["y  (B, T, D)"]:::io
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

- Performer / FAVOR+ — orthogonal random feature kernel
- Linformer, Linear Transformers (Katharopoulos et al.)
- RWKV — RNN/transformer hybrid using linear-attention recurrence

**Tasks**

- Long-context modelling (audio, DNA, code) where T ≫ 10⁴
- Streaming inference where state can be kept as a fixed-size matrix

**Common pitfalls**

- Quality lags softmax attention on language modelling — closes only with careful kernel choice and longer training.
- Normalisation is numerically delicate — pick a kernel φ(·) that is strictly positive.
- Causal masking is non-trivial — requires a per-token running sum, not a single matmul.

**See also**

- [Linear Transformers (Katharopoulos et al. 2020)](https://arxiv.org/abs/2006.16236)
- [Performer (Choromanski et al. 2020)](https://arxiv.org/abs/2009.14794)
- [RWKV (Peng et al. 2023)](https://arxiv.org/abs/2305.13048)

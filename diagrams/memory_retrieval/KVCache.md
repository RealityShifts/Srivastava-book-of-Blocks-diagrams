# KVCache

> Cache K, V tensors per layer per token to skip re-encoding history.

**Shapes:** `x_t → K_{1..t}, V_{1..t}`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["x_t  (B, 1, D)"]:::io
    n1_0["compute K_t, V_t"]:::op
    n2_0["append to cache"]:::op
    n3_0["cached K, V  (L, B, T, D)"]:::io
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

- Every production LLM inference stack (vLLM, TGI, TensorRT-LLM, llama.cpp)
- PagedAttention / vLLM's blocked KV cache
- Speculative decoding pipelines

**Tasks**

- Reducing autoregressive inference from O(T²) to O(T) per token
- Batched serving across multiple concurrent sessions

**Common pitfalls**

- Memory is the bottleneck — `L × B × T × 2 × D` in fp16 is huge for long contexts.
- Multi-Query / Grouped-Query Attention shrink the cache (1 or g K-V heads instead of H).
- Beam search and continuous-batching need careful cache-block layout (see PagedAttention).
- Quantising the cache (FP8 / INT8) saves memory but can hurt long-context accuracy.

**See also**

- [Multi-Query Attention (Shazeer 2019)](https://arxiv.org/abs/1911.02150)
- [Grouped-Query Attention (Ainslie et al. 2023)](https://arxiv.org/abs/2305.13245)
- [PagedAttention / vLLM (Kwon et al. 2023)](https://arxiv.org/abs/2309.06180)

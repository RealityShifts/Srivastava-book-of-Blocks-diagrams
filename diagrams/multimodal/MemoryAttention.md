# MemoryAttention

> Cross-attention layer that reads from an external memory bank.

**Shapes:** `x:(B, T, D), mem:(M, D) → (B, T, D)`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["x  (B, T, D)"]:::io
    n0_1["memory bank  (M, D)"]:::io
    n1_0["cross-attention over memory"]:::attn
    n2_0["y  (B, T, D)"]:::io
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

- Memorizing Transformers (Wu et al. 2022) — kNN-augmented attention
- Retro / Retro-fitted models
- Agentic memory layers (long-term episodic memory)

**Tasks**

- Extending effective context via retrieval-as-attention
- Personalisation by storing user-specific tokens in memory

**Common pitfalls**

- Memory bank size grows over time — needs eviction or summarisation.
- Approximate kNN (top-k) over memory is essential for scale; exhaustive dot-product is infeasible past ~10⁶ items.
- Training distribution and memory distribution can diverge — periodic refresh helps.

**See also**

- [Memorizing Transformers (Wu et al. 2022)](https://arxiv.org/abs/2203.08913)
- [kNN-LM (Khandelwal et al. 2019)](https://arxiv.org/abs/1911.00172)

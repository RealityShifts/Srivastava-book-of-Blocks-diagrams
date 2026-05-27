# ExternalMemory

> Soft attention over an external memory bank acts as a read operation.

**Shapes:** `q:(B, D), mem:(M, D) → r:(B, D)`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["query  (B, D)"]:::io
    n0_1["memory bank  (M, D)"]:::io
    n1_0["attention(query, mem, mem)"]:::attn
    n2_0["retrieved  (B, D)"]:::io
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

- Neural Turing Machines / Differentiable Neural Computers
- Memory-Augmented Networks (MANN)
- Memorizing Transformers (Wu et al. 2022)

**Tasks**

- Few-shot learning where context is too large to keep in attention
- Long-term episodic memory in agents

**Common pitfalls**

- Memory bank grows unbounded — needs eviction policy or summarisation.
- Soft addressing is slow to learn — hard / sparse attention often does better in practice but is non-differentiable.
- Mostly superseded by retrieval (RAG / kNN-LM) for language tasks.

**See also**

- [Neural Turing Machines (Graves et al. 2014)](https://arxiv.org/abs/1410.5401)
- [DNC (Graves et al. 2016)](https://www.nature.com/articles/nature20101)
- [Memorizing Transformers (Wu et al. 2022)](https://arxiv.org/abs/2203.08913)

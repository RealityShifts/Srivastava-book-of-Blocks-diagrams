# ColumnParallelLinear

> Linear whose output columns are sharded across tensor-parallel ranks.

**Shapes:** `x → y  (gathered)`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["x  (B, in)  replicated"]:::io
    n1_0["split W cols across ranks"]:::op
    n2_0["local matmul"]:::op
    n3_0["all-gather"]:::op
    n4_0["y  (B, out)  gathered"]:::io
    n0_0 --> n1_0
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

- Megatron-LM tensor parallelism (TP) for LLM training
- DeepSpeed, FasterTransformer, vLLM TP for inference

**Tasks**

- Sharding the Q/K/V/O linears of a Transformer across N GPUs
- Fitting layers that don't fit in a single GPU's memory

**Common pitfalls**

- Pair with RowParallelLinear immediately after (Q→attn→O) so the all-gather and all-reduce cancel out — Megatron's clever design.
- Bias is duplicated across ranks by default — be careful with weight decay accounting.
- Sequence-parallel variant reduces activation memory by another factor of N.

**See also**

- [Megatron-LM (Shoeybi et al. 2019)](https://arxiv.org/abs/1909.08053)

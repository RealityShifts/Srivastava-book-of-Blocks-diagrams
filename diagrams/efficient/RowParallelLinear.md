# RowParallelLinear

> Linear whose input is sharded across ranks; outputs all-reduced.

**Shapes:** `x  (sharded) → y`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["x  (B, in/r)  sharded"]:::io
    n1_0["local matmul"]:::op
    n2_0["all-reduce sum"]:::op
    n3_0["y  (B, out)"]:::io
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

- Megatron-LM TP (the partner of ColumnParallelLinear)
- Inference engines (TensorRT-LLM, vLLM) with TP > 1

**Tasks**

- Completing the column-then-row sharding pattern within a single Transformer sublayer
- Reducing per-GPU memory and compute for very wide hidden states

**Common pitfalls**

- All-reduce is bandwidth-bound — NVLink between GPUs is far better than PCIe.
- Don't all-reduce inside fp16 / fp8 — cast to fp32 for the reduction or use higher-precision all-reduce APIs.
- Backward also needs an all-reduce; total comm cost is 2× the forward.

**See also**

- [Megatron-LM (Shoeybi et al. 2019)](https://arxiv.org/abs/1909.08053)

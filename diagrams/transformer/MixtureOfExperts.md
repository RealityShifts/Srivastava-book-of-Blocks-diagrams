# MixtureOfExperts

> Top-k gated mixture: router scores experts, top-k run, weighted sum.

**Shapes:** `(B, T, D) → (B, T, D)`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["x  (B, T, D)"]:::io
    n1_0["router (linear → softmax → top-k)"]:::op
    n2_0["Expert 1"]:::op
    n2_1["Expert 2"]:::op
    n2_2["…"]:::op
    n2_3["Expert E"]:::op
    n3_0["weighted sum (gates · experts)"]:::merge
    n4_0["y  (B, T, D)"]:::io
    n0_0 --> n1_0
    n1_0 --> n2_0
    n1_0 --> n2_1
    n1_0 --> n2_2
    n1_0 --> n2_3
    n2_0 --> n3_0
    n2_1 --> n3_0
    n2_2 --> n3_0
    n2_3 --> n3_0
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

- GShard, Switch Transformer — large-scale Google LMs
- Mixtral 8×7B, Mixtral 8×22B, DeepSeek-V2/V3 — open MoE LLMs
- GLaM, ST-MoE for translation and language modelling

**Tasks**

- Scaling parameter count without scaling per-token FLOPs
- Multi-domain / multi-task models where experts can specialise

**Common pitfalls**

- Load imbalance — most tokens route to few experts unless an auxiliary load-balancing loss is added (Switch Transformer §3.2).
- Top-k > 1 doubles FLOPs but tames training instability vs top-1.
- Expert parallelism complicates training — needs all-to-all communication and careful pipeline overlap.
- Inference batching is harder — pads or drops tokens at expert capacity limits.

**See also**

- [Sparsely-Gated MoE (Shazeer et al. 2017)](https://arxiv.org/abs/1701.06538)
- [GShard (Lepikhin et al. 2020)](https://arxiv.org/abs/2006.16668)
- [ST-MoE (Zoph et al. 2022)](https://arxiv.org/abs/2202.08906)
- [Mixtral of Experts (Jiang et al. 2024)](https://arxiv.org/abs/2401.04088)

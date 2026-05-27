# HyperNetwork

> A meta-network that emits weights consumed by a target network.

**Shapes:** `cond:(B, D_c) → params → y`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["condition  (B, D_c)"]:::io
    n1_0["meta MLP"]:::op
    n2_0["generated weights θ"]:::op
    n3_0["target net  (uses θ on input x)"]:::op
    n4_0["y  (task-dependent)"]:::io
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

- Stable Diffusion Hypernetworks (early SD fine-tuning trick)
- HyperNetworks (Ha et al. 2016) — original RNN-emitting-CNN-weights paper
- Meta-learning / few-shot adaptation

**Tasks**

- Conditioning a target network on a high-dimensional context by emitting its weights
- Few-shot personalisation in vision / text models

**Common pitfalls**

- Output dimension is the FULL parameter count of the target — grows quickly; usually emit low-rank or per-layer scalars instead.
- Joint optimisation is delicate — meta-LR usually much smaller than target-LR.
- Inference cost includes meta-net forward + target-net forward.

**See also**

- [HyperNetworks (Ha et al. 2016)](https://arxiv.org/abs/1609.09106)

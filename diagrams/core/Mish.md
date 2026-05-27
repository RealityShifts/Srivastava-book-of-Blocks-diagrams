# Mish

> Self-gated activation: x · tanh(softplus(x)).

**Shapes:** `(*) → (*)`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["x  (*)"]:::io
    n1_0["softplus(x)"]:::op
    n2_0["tanh(·)"]:::op
    n3_0["× x"]:::merge
    n4_0["y  (*)"]:::io
    n0_0 --> n1_0
    n1_0 --> n2_0
    n2_0 --> n3_0
    n3_0 --> n4_0
    n0_0 -.->|"skip"| n3_0
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

- YOLOv4 and successors — replaced LeakyReLU in the backbone
- CSPNet variants for image classification / detection

**Tasks**

- Drop-in replacement for ReLU/Swish in CNNs aiming for slight accuracy gains

**Common pitfalls**

- More expensive than ReLU/SiLU — measure wall-clock impact, not just FLOPs.
- Smooth-but-nonmonotonic; gradients in the negative tail are small but not zero, which helps deep nets but is not always better than SiLU/GELU.

**See also**

- [Mish (Misra 2019)](https://arxiv.org/abs/1908.08681)
- [YOLOv4 (Bochkovskiy et al. 2020)](https://arxiv.org/abs/2004.10934)

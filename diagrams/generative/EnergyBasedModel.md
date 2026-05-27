# EnergyBasedModel

> A network that maps x to a scalar energy E(x); samples drawn via Langevin/MCMC.

**Shapes:** `x → E(x):(B,)`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["x  (B, …)"]:::io
    n1_0["network"]:::op
    n2_0["E(x)  (B,)"]:::io
    n0_0 --> n1_0
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

- Implicit Generation (Du & Mordatch 2019)
- JEM — joint generative-discriminative training
- Score-based models (gradient of log-density ≈ −∇E)

**Tasks**

- Density modelling without a tractable partition function
- Out-of-distribution detection (low energy = in-distribution)

**Common pitfalls**

- Training is unstable — contrastive divergence and short-run MCMC are common workarounds.
- Sampling needs Langevin dynamics or HMC — slow compared to feed-forward generators.
- Mode coverage is poor without replay buffers and persistent chains.

**See also**

- [Implicit Generation (Du & Mordatch 2019)](https://arxiv.org/abs/1903.08689)
- [JEM (Grathwohl et al. 2019)](https://arxiv.org/abs/1912.03263)
- [A Tutorial on EBMs (LeCun et al. 2006)](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf)

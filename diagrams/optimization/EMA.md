# EMA

> Exponential moving average of model weights.

**Shapes:** `θ_online → θ_ema`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["θ_online  (params)"]:::io
    n1_0["θ_ema ← τ · θ_ema + (1 − τ) · θ_online"]:::op
    n2_0["θ_ema  (params)"]:::io
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

- BYOL / MoCo — momentum-encoder target
- Diffusion models — EMA of weights used for SAMPLING (essential for FID)
- Mean-teacher semi-supervised learning
- Reinforcement-learning target networks (effectively a hard-step EMA)

**Tasks**

- Stabilising self-supervised / generative training
- Improving generalisation by averaging across the training trajectory
- Decoupling sampling-time weights from training weights

**Common pitfalls**

- EMA copy doubles the parameter memory — be aware on large models.
- Decay τ near 1 (e.g. 0.9999) is typical for diffusion; lower for SSL momentum encoders.
- Don't average buffers (batch-norm running stats, optimizer state) — only weights.
- Loading EMA weights into the training model at resume is a common bug.

**See also**

- [Polyak averaging (Polyak & Juditsky 1992)](https://epubs.siam.org/doi/10.1137/0330046)
- [BYOL (Grill et al. 2020)](https://arxiv.org/abs/2006.07733)
- [Improved DDPM (Nichol & Dhariwal 2021)](https://arxiv.org/abs/2102.09672)

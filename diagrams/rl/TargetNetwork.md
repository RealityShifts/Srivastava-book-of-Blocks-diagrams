# TargetNetwork

> Slowly-tracking copy of the online network used to stabilise bootstrap targets.

**Shapes:** `θ_online → θ_target`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["θ_online  (params)"]:::io
    n1_0["θ_target ← τ · θ_target + (1 − τ) · θ_online"]:::op
    n2_0["θ_target  (params)"]:::io
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

- DQN — hard copy every C steps
- DDPG / TD3 / SAC — soft Polyak averaging (`τ ≈ 0.005`)
- MuZero — separate target net for the value prediction head
- Distillation pipelines — teacher network tracking a moving average of the student

**Tasks**

- Stabilising the bootstrap target in TD-learning
- Decoupling target evaluation from policy improvement to avoid moving-goalposts divergence

**Common pitfalls**

- τ too high → targets move fast → instability (oscillating Q).
- τ too low → slow learning, lagging targets.
- Forgetting to detach `θ_target` from the autograd graph leaks gradients into the target — always `with torch.no_grad():`.
- Hard copies every C steps cause periodic learning spikes; soft updates smooth them.

**See also**

- [DQN (Mnih et al. 2015, Nature)](https://www.nature.com/articles/nature14236)
- [DDPG (Lillicrap et al. 2016)](https://arxiv.org/abs/1509.02971)
- [TD3 (Fujimoto et al. 2018)](https://arxiv.org/abs/1802.09477)

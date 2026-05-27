# ValueNetwork

> Estimates V(s) — the expected return from a state under the current policy.

**Shapes:** `s:(B, D_s) → V:(B,)`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["state  (B, D_s)"]:::io
    n1_0["MLP"]:::op
    n2_0["head → V(s)"]:::op
    n3_0["value  (B,)"]:::io
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

- Critic in every actor-critic algorithm (A2C, A3C, PPO, SAC)
- Baseline for variance reduction in policy gradients
- Bootstrap target for n-step / TD(λ) returns
- Generalised Advantage Estimation (GAE)

**Tasks**

- Reducing gradient variance in REINFORCE / PG methods
- Computing advantages `A(s, a) = Q(s, a) − V(s)`
- On-policy bootstrapping where Q(s, a) is impractical (continuous action spaces)

**Common pitfalls**

- Bootstrapping with the same network used for updates causes instability — use a slowly-tracking TargetNetwork for the target.
- Tightly coupling actor and critic learning rates often destabilises both; tune them independently.
- MSE loss against high-variance returns can over-fit; clip the value loss (PPO) or use Huber loss (DQN family).

**See also**

- [Temporal-difference learning (Sutton 1988)](https://link.springer.com/article/10.1007/BF00115009)
- [GAE (Schulman et al. 2015)](https://arxiv.org/abs/1506.02438)
- [PPO value-function clipping](https://arxiv.org/abs/1707.06347)

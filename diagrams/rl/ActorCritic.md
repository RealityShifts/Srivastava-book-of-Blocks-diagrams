# ActorCritic

> Shared trunk that branches into a policy head and a value head.

**Shapes:** `s → π(a | s),  V(s)`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["state  (B, D_s)"]:::io
    n1_0["shared MLP trunk"]:::op
    n2_0["policy head"]:::op
    n2_1["value head"]:::op
    n3_0["π, V  (B, D_a), (B,)"]:::io
    n0_0 --> n1_0
    n1_0 --> n2_0
    n1_0 --> n2_1
    n2_0 --> n3_0
    n2_1 --> n3_0
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

- A2C / A3C — original sync / async actor-critic
- PPO — clipped policy ratio + value loss on the same trunk
- IMPALA — distributed actor-critic with V-trace corrections
- MuZero — model + policy + value heads on a shared latent

**Tasks**

- Sample-efficient on-policy learning where critic reduces policy-gradient variance
- Large-scale distributed RL where one trunk feeds many workers
- RLHF — actor outputs token distributions, critic scores partial responses for advantage computation

**Common pitfalls**

- Policy and value losses compete for trunk capacity — clip the value loss or weight it (`c_v ≈ 0.5`).
- Entropy bonus is essential for exploration in early training; anneal it as the policy sharpens.
- Gradient norm explodes on bad batches — global-norm clip (typically 0.5–1.0).

**See also**

- [A3C (Mnih et al. 2016)](https://arxiv.org/abs/1602.01783)
- [PPO (Schulman et al. 2017)](https://arxiv.org/abs/1707.06347)
- [IMPALA (Espeholt et al. 2018)](https://arxiv.org/abs/1802.01561)

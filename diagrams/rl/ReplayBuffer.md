# ReplayBuffer

> Ring buffer storing transitions; sampled in mini-batches for off-policy learning.

**Shapes:** `(s, a, r, s', d) → batch`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["transition (s, a, r, s', d)  (·,)"]:::io
    n1_0["ring buffer  (capacity N)"]:::op
    n2_0["uniform / prioritised sample"]:::op
    n3_0["mini-batch  (B, ·)"]:::io
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

- DQN — original uniform-replay experience buffer
- PER — sampling proportional to TD-error magnitude
- DDPG / SAC / TD3 — off-policy continuous control
- Ape-X / R2D2 — distributed prioritised replay at scale
- HER — Hindsight Experience Replay for sparse-reward goals

**Tasks**

- Breaking temporal correlation in updates
- Reusing rare / high-value transitions many times
- Off-policy correction when the data-generating policy differs from the current learner

**Common pitfalls**

- Buffer stays full of stale data when the policy drifts fast; tune capacity vs. update ratio.
- PER without importance-sampling weights introduces bias — always anneal β from ~0.4 → 1.0.
- Storing full image observations is RAM-heavy — use frame stacks + uint8 storage, decode in the worker.
- Multi-step returns need transition aggregation (n-step) before sampling, not after.

**See also**

- [Experience replay (Lin 1992)](https://link.springer.com/article/10.1007/BF00992699)
- [Prioritised Experience Replay (Schaul et al. 2015)](https://arxiv.org/abs/1511.05952)
- [Hindsight Experience Replay (Andrychowicz et al. 2017)](https://arxiv.org/abs/1707.01495)
- [Ape-X (Horgan et al. 2018)](https://arxiv.org/abs/1803.00933)

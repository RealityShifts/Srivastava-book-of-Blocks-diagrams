# PolicyNetwork

> Maps a state to an action distribution / Gaussian (μ, σ) / discrete logits.

**Shapes:** `s:(B, D_s) → π(a | s)`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["state  (B, D_s)"]:::io
    n1_0["MLP trunk"]:::op
    n2_0["head → π(a | s)"]:::op
    n3_0["action / dist  (B, D_a)"]:::io
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

- REINFORCE — vanilla policy gradient
- A2C / A3C — synchronous & asynchronous actor-critic
- TRPO / PPO — trust-region / clipped policy improvement
- SAC — entropy-regularised stochastic policy for continuous control
- Decision-Transformer policies for offline RL

**Tasks**

- Continuous control (MuJoCo, robotic manipulation, locomotion)
- Discrete-action games (Atari, board games, card games)
- Dialogue / tool-use policies in LLM agents (RLHF, RLAIF)
- Anywhere you need a *stochastic* mapping `state → action`

**Common pitfalls**

- High-variance gradients — pair with a value baseline / GAE.
- Action squashing (`tanh`) breaks log-probability — use the tanh-squashed-Gaussian correction.
- Stochastic policies can collapse to a deterministic mode; add an entropy bonus (SAC, max-ent RL).
- Discrete-action heads need numerically-stable softmax + log-softmax for the log-prob (don't compute `log(softmax(x))` naively).

**See also**

- [REINFORCE (Williams 1992)](https://link.springer.com/article/10.1007/BF00992696)
- [A3C (Mnih et al. 2016)](https://arxiv.org/abs/1602.01783)
- [PPO (Schulman et al. 2017)](https://arxiv.org/abs/1707.06347)
- [SAC (Haarnoja et al. 2018)](https://arxiv.org/abs/1801.01290)

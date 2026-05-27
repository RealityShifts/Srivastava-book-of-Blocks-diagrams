# NeuralODE

> Treat depth as continuous time and integrate dx/dt = f_θ(x, t).

**Shapes:** `x_0 → x_T`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["x_0  (B, D)"]:::io
    n0_1["f_θ(x, t)  (·)→(·)"]:::io
    n1_0["ODE solver  (Euler / RK4 / dopri5)"]:::op
    n2_0["x(T) = x_0 + ∫₀ᵀ f dt"]:::op
    n3_0["x_T  (B, D)"]:::io
    n0_0 --> n1_0
    n0_1 --> n1_0
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

- Neural ODE (Chen et al. 2018) — original paper, NeurIPS best paper
- FFJORD — continuous normalising flows
- Latent ODEs for irregular time series
- Flow-matching / rectified-flow diffusion (probability-flow ODE view)

**Tasks**

- Continuous-depth networks where memory scales with the adjoint, not the network depth
- Modelling irregular time-series (medical, finance) at arbitrary sample times
- Generative modelling via instantaneous change of variables

**Common pitfalls**

- Adjoint method saves memory but is numerically unstable — checkpointing the forward trajectory is often more reliable.
- Adaptive solvers (dopri5) can stall on stiff dynamics — bound max_steps.
- Wall-clock per step is far higher than discrete networks at equal expressivity.

**See also**

- [Neural ODE (Chen et al. 2018)](https://arxiv.org/abs/1806.07366)
- [FFJORD (Grathwohl et al. 2018)](https://arxiv.org/abs/1810.01367)
- [Latent ODEs (Rubanova et al. 2019)](https://arxiv.org/abs/1907.03907)

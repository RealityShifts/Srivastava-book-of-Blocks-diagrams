# Sophia

> Hessian-clipped second-order optimiser.

**Shapes:** `g, h, θ → θ'`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["g, h  (params)  (Hessian estimate)"]:::io
    n1_0["m ← β₁·m + (1 − β₁)·g"]:::op
    n2_0["update = clip(m / max(h, ε), ρ)"]:::op
    n3_0["θ ← θ − lr · update"]:::op
    n4_0["θ'  (params)"]:::io
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

- GPT-2 125M..1.5B reproductions (claim ~2× speedup vs AdamW)
- Research code-bases experimenting with second-order methods at scale

**Tasks**

- Language-model pretraining where Hessian heterogeneity across dims is large
- Reducing the compute / wall-clock budget for matching a target perplexity

**Common pitfalls**

- Hessian diagonal estimate via Hutchinson is noisy — must average across many steps (default: every k iterations).
- Clipping bound ρ matters — too tight blocks progress, too loose loses the second-order signal.
- Independent reproductions show smaller gains than the paper at very large scale.
- Per-step overhead is small only because Hessian is updated INTERMITTENTLY — implementations that update every step are much slower.

**See also**

- [Sophia (Liu et al. 2023)](https://arxiv.org/abs/2305.14342)
- [Sophia implementation](https://github.com/Liuhong99/Sophia)

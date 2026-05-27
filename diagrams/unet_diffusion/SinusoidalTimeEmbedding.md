# SinusoidalTimeEmbedding

> Sin/cos positional embedding of the diffusion timestep t.

**Shapes:** `t:(B,) → emb:(B, D)`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["t  (B,)"]:::io
    n1_0["freqs = 10000^(−2i/D)"]:::op
    n2_0["[sin(t·f),  cos(t·f)]"]:::op
    n3_0["emb  (B, D)"]:::emb
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

- DDPM, DDIM, Stable Diffusion, Imagen — diffusion noise predictors
- Score-based generative models
- Continuous-time NeuralODE / flow-matching conditioning

**Tasks**

- Encoding a continuous scalar (timestep, noise level) into a vector for conditioning

**Common pitfalls**

- Frequency base (10000 in original Transformer; sometimes scaled differently for diffusion) affects which timescales are resolved — match the convention of the reference impl.
- When t is in [0, 1] vs [0, T] (T~1000) the same sinusoidal table behaves very differently.
- Half-precision can underflow for very small t — keep the embedding in fp32.

**See also**

- [DDPM (Ho et al. 2020)](https://arxiv.org/abs/2006.11239)
- [Attention Is All You Need (Vaswani et al. 2017)](https://arxiv.org/abs/1706.03762)

# CapsuleLayer

> Capsule routing layer (Sabour et al.): predict votes, dynamic routing-by-agreement, squash.

**Shapes:** `primary caps → output caps`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["primary capsules u_i  (N₁, d₁)"]:::io
    n1_0["votes  û_{j|i} = W_ij · u_i"]:::op
    n2_0["dynamic routing × T"]:::op
    n3_0["squash"]:::op
    n4_0["output capsules v_j  (N₂, d₂)"]:::io
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

- Dynamic Routing Between Capsules (Sabour, Frosst, Hinton 2017)
- Matrix Capsules with EM Routing (Hinton et al. 2018)
- Stacked Capsule Autoencoders

**Tasks**

- Part-whole hierarchy modelling — argued to be a more 'biologically plausible' alternative to CNNs
- Pose / orientation reasoning where vectors carry geometric info

**Common pitfalls**

- Dynamic routing is expensive and hard to scale beyond small images (MNIST / CIFAR).
- Largely supplanted by self-attention in modern vision — capsules saw limited adoption.
- Routing iterations T must be small (3) to be tractable; too few and routing under-trains.

**See also**

- [Dynamic Routing Between Capsules (Sabour et al. 2017)](https://arxiv.org/abs/1710.09829)
- [Matrix Capsules with EM Routing (Hinton et al. 2018)](https://openreview.net/forum?id=HJWLfGWRb)

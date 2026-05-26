# LSTMCell

> Long short-term memory cell with input/forget/cell/output gates.

**Shapes:** `x_t, h_{t-1}, c_{t-1} → h_t, c_t`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["x_t  (B, D_x)"]:::io
    n0_1["h_{t−1}  (B, D_h)"]:::io
    n0_2["c_{t−1}  (B, D_h)"]:::io
    n1_0["linear → (i, f, g, o)"]:::op
    n2_0["c_t = f ⊙ c_{t−1} + i ⊙ g"]:::op
    n3_0["h_t = o ⊙ tanh(c_t)"]:::op
    n4_0["h_t, c_t  (B, D_h)"]:::io
    n0_0 --> n1_0
    n0_1 --> n1_0
    n0_2 --> n1_0
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

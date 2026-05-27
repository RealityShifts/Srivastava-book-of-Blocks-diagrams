# GroupConv2d

> Conv2d with channel groups (ResNeXt cardinality).

**Shapes:** `(B, C, H, W) → (B, C', H, W)`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["x  (B, C, H, W)"]:::io
    n1_0["Conv K×K  (groups = g)"]:::op
    n2_0["y  (B, C', H, W)"]:::io
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

- AlexNet — the original use, split across two GPUs
- ResNeXt — uses cardinality (group count) as a third capacity axis
- ShuffleNet — group conv + channel shuffle for cheap exchange

**Tasks**

- Trading per-channel mixing for compute / parameter savings
- Backbone surgery that needs intermediate cost between standard and depthwise conv

**Common pitfalls**

- Information is partitioned across groups — without channel shuffle or a 1×1 mix-up after, capacity drops sharply.
- Groups must divide both `in_ch` and `out_ch`; otherwise it silently errors at build time.

**See also**

- [ResNeXt (Xie et al. 2016)](https://arxiv.org/abs/1611.05431)
- [ShuffleNet (Zhang et al. 2017)](https://arxiv.org/abs/1707.01083)

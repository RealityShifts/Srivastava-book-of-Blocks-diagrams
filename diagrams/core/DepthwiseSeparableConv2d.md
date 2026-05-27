# DepthwiseSeparableConv2d

> Depthwise spatial conv followed by 1×1 pointwise conv (MobileNet).

**Shapes:** `(B, C, H, W) → (B, C', H, W)`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["x  (B, C, H, W)"]:::io
    n1_0["Depthwise Conv K×K  (groups=C)"]:::op
    n2_0["Pointwise Conv 1×1  (C → C')"]:::op
    n3_0["y  (B, C', H, W)"]:::io
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

- MobileNet v1 / v2 / v3 — flagship mobile backbones
- Xception — replaces every Inception module with depthwise separable convs
- EfficientNet & EfficientNetV2 — search space built on inverted residuals + DW

**Tasks**

- Edge / mobile inference where FLOPs and parameters must be tiny
- Replacing standard 3×3 convs to cut ~8–9× compute at similar accuracy

**Common pitfalls**

- Memory-bound on GPUs — wall-clock speedup is often less than the FLOP reduction suggests.
- Channel scaling matters: pair with an expansion ratio (inverted residual) or the depthwise stage will bottleneck capacity.
- Some autograd backends fuse standard convs but not depthwise — benchmark first.

**See also**

- [MobileNet v1 (Howard et al. 2017)](https://arxiv.org/abs/1704.04861)
- [MobileNetV2 (Sandler et al. 2018)](https://arxiv.org/abs/1801.04381)
- [Xception (Chollet 2017)](https://arxiv.org/abs/1610.02357)

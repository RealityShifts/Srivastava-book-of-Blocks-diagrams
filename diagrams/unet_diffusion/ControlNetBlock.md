# ControlNetBlock

> Trainable copy of UNet encoder + ZeroConvs; outputs are added to the frozen UNet decoder.

**Shapes:** `x, control → ΔUNet decoder features`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["x  noised  (B, C, H, W)"]:::io
    n0_1["control image  (B, 3, H, W)"]:::io
    n1_0["frozen UNet encoder"]:::op
    n1_1["trainable encoder copy"]:::op
    n2_0["ZeroConv on each level"]:::op
    n3_0["add into UNet decoder skips"]:::merge
    n4_0["Δ feats  (multi-scale, matches UNet decoder)"]:::io
    n0_0 --> n1_0
    n0_1 --> n1_1
    n1_0 --> n2_0
    n1_1 --> n2_0
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

- Stable Diffusion ControlNet (canny, depth, pose, scribble, etc.)
- ControlNet-XS — smaller variants for cheap inference

**Tasks**

- Adding structural conditioning (edges, depth, pose) to a pretrained diffusion model
- Fine-tuning T2I models on new control signals without forgetting the base distribution

**Common pitfalls**

- Frozen base U-Net must be EXACTLY the model you'll run at inference — fine-tunes of the base break compatibility.
- Zero-conv at the junction is essential; without it, training breaks the base model immediately.
- Each ControlNet is condition-specific — stacking many at inference is possible but quality drops.

**See also**

- [ControlNet (Zhang et al. 2023)](https://arxiv.org/abs/2302.05543)
- [T2I-Adapter (Mou et al. 2023)](https://arxiv.org/abs/2302.08453)

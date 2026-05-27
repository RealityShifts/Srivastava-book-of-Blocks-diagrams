# CheckpointedSequential

> Sequential whose forward is rematerialised on the backward pass to save memory.

**Shapes:** `(*) → (*)`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["x  (*)"]:::io
    n1_0["layer 1   (recompute on backward)"]:::op
    n2_0["layer 2   (recompute on backward)"]:::op
    n3_0["…  (k layers)"]:::op
    n4_0["y  (*)"]:::io
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

- Training of huge models (LLaMA, GPT-NeoX, Megatron-LM)
- Diffusion U-Nets and ViT-Huge fine-tuning
- PyTorch's `torch.utils.checkpoint`

**Tasks**

- Fitting deeper / wider models into a fixed memory budget
- Long-context training where activation memory is the binding constraint

**Common pitfalls**

- ~30 % slower per step due to the extra forward — only worth it if memory was the bottleneck.
- Doesn't compose well with non-determinism (dropout) unless you set `use_reentrant=False` or seed correctly.
- Checkpointing across optimizer step is wrong — only the FORWARD activations are rematerialised.
- Selective Activation Checkpointing (Megatron-LM) typically beats blind layer-wise CKPT.

**See also**

- [Gradient Checkpointing (Chen et al. 2016)](https://arxiv.org/abs/1604.06174)
- [Megatron Selective Activation Recompute (Korthikanti et al. 2022)](https://arxiv.org/abs/2205.05198)

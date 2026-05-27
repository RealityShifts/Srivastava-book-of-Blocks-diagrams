# MixedPrecisionTrainer

> Loss-scaled fp16 forward, fp32 master weights, gradient unscale + clip + step.

**Shapes:** `x → updated θ`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["x  (B, …)"]:::io
    n1_0["fp16 forward"]:::op
    n2_0["loss × scale"]:::op
    n3_0["backward (fp32 master grads)"]:::op
    n4_0["unscale + clip"]:::op
    n5_0["optimizer step"]:::op
    n6_0["updated θ  (params)"]:::io
    n0_0 --> n1_0
    n1_0 --> n2_0
    n2_0 --> n3_0
    n3_0 --> n4_0
    n4_0 --> n5_0
    n5_0 --> n6_0
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

- torch.cuda.amp / torch.amp
- DeepSpeed, Megatron-LM training stacks
- JAX with bfloat16 / float16 mixed precision

**Tasks**

- Doubling training throughput on tensor-core hardware (V100, A100, H100)
- Halving memory usage to fit larger batch / model sizes

**Common pitfalls**

- fp16 underflow on small gradients — loss scaling required (or use bf16 with no loss scale).
- Overflow → dynamic scale halves; thrashing scales hurt convergence — log them.
- BatchNorm and softmax should stay fp32 — autocast handles this, manual impls don't.
- bf16 has wider range than fp16, narrower mantissa — usually preferred on A100+ and TPUs.

**See also**

- [Mixed Precision Training (Micikevicius et al. 2017)](https://arxiv.org/abs/1710.03740)
- [bfloat16 (Wang & Kanwar 2019)](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)

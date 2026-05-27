# FlashAttention

> Mathematically identical to MHA, but uses tiled IO-aware kernels.

**Shapes:** `(B, T, D) → (B, T, D)`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["Q, K, V  (B, T, D)"]:::io
    n1_0["flash kernel  (tiled softmax, no materialised attn matrix)"]:::attn
    n2_0["y  (B, T, D)"]:::io
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

- PyTorch 2.x `scaled_dot_product_attention` fast path
- All modern LLM training stacks (Megatron-LM, vLLM, TGI, Triton)
- Mamba's selective-scan kernel and FlashAttention-2 for ViTs

**Tasks**

- Replacing standard MHA in any model with sequence length > ~512 to cut memory and time
- Long-context fine-tuning (32k–128k tokens) that would otherwise OOM

**Common pitfalls**

- Custom attention biases (e.g. ALiBi, T5 relative bias) may NOT be supported on all FlashAttention versions — check the kernel signature.
- FP32 fallback is required for some bias shapes — surprise speed cliff.
- Backward pass recomputes attention — increases compute by ~2× for a memory-bound win.

**See also**

- [FlashAttention (Dao et al. 2022)](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 (Dao 2023)](https://arxiv.org/abs/2307.08691)
- [FlashAttention-3 (Shah et al. 2024)](https://arxiv.org/abs/2407.08608)

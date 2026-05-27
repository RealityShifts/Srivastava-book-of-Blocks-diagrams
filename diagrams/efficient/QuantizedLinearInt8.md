# QuantizedLinearInt8

> Linear with weights stored in int8 + per-output-channel scale.

**Shapes:** `(B, in) → (B, out)`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["x  (B, in)  fp"]:::io
    n1_0["W_int8  (per-channel scale s)"]:::op
    n2_0["matmul x · W_int8"]:::op
    n3_0["× s   (dequantise)"]:::op
    n4_0["+ bias"]:::op
    n5_0["y  (B, out)"]:::io
    n0_0 --> n1_0
    n1_0 --> n2_0
    n2_0 --> n3_0
    n3_0 --> n4_0
    n4_0 --> n5_0
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

- LLM.int8() / bitsandbytes — first widely-deployed int8 LLM inference
- SmoothQuant — outlier-aware activation+weight quantisation
- PyTorch quantisation, TensorRT, ONNX Runtime

**Tasks**

- 2× inference speedup and ~2× memory reduction on int8-capable hardware
- Deploying large LLMs on commodity GPUs (e.g. LLaMA-65B → single A100)

**Common pitfalls**

- Outlier channels can blow up the dynamic range — LLM.int8() detects them and runs fp16 for those rows; SmoothQuant rebalances activations into weights.
- Per-channel scales are essential; per-tensor scales lose accuracy on LLMs.
- Activations are NOT quantised here — that step (AQA) is a separate decision.

**See also**

- [LLM.int8() (Dettmers et al. 2022)](https://arxiv.org/abs/2208.07339)
- [SmoothQuant (Xiao et al. 2022)](https://arxiv.org/abs/2211.10438)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)

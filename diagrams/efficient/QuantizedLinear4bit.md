# QuantizedLinear4bit

> Linear with weights packed to 4-bit + grouped scales / zeros (NF4 / GPTQ style).

**Shapes:** `(B, in) → (B, out)`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["x  (B, in)  fp"]:::io
    n1_0["unpack 4-bit → int"]:::op
    n2_0["dequantise per group  (s, z)"]:::op
    n3_0["matmul x · W"]:::op
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

- QLoRA — NF4 storage + LoRA fine-tuning on consumer GPUs
- GPTQ / AWQ — post-training quantisation methods
- llama.cpp Q4_0 / Q4_K — ggml community quantisations

**Tasks**

- 4× memory reduction to fit 30B+ models in 24 GB VRAM
- Cheap LoRA fine-tuning on top of 4-bit base weights

**Common pitfalls**

- Group size (32 / 64 / 128) trades accuracy for memory of scales+zeros — 64–128 typical.
- Activation outliers degrade 4-bit quality more than int8; AWQ / GPTQ post-tune the choice.
- Saving 4-bit weights as 8-bit packed bytes is essential; many bugs come from accidental fp16 inflation in the loader.
- Backward through a 4-bit weight requires bf16 / fp16 dequant on the fly — slower than int8.

**See also**

- [QLoRA / NF4 (Dettmers et al. 2023)](https://arxiv.org/abs/2305.14314)
- [GPTQ (Frantar et al. 2022)](https://arxiv.org/abs/2210.17323)
- [AWQ (Lin et al. 2023)](https://arxiv.org/abs/2306.00978)

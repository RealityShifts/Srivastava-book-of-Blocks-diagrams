# PerceiverResampler

> Fixed-length learnable latents cross-attend over arbitrary-length media features.

**Shapes:** `media:(B, M, D), latents:(L, D) → (B, L, D)`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["learnable latents  (L, D)"]:::io
    n0_1["media features  (B, M, D)"]:::io
    n1_0["cross-attn  (latents query)"]:::attn
    n2_0["FFN"]:::op
    n3_0["× N layers"]:::op
    n4_0["resampled latents  (B, L, D)"]:::io
    n0_0 --> n1_0
    n0_1 --> n1_0
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

- Perceiver / Perceiver IO — modality-agnostic encoder
- Flamingo's Perceiver Resampler (downsamples video frames to fixed token count)
- Idefics, OpenFlamingo — open-source Flamingo replicas

**Tasks**

- Compressing variable-length media (video, audio, point clouds) to fixed-length tokens
- Cross-modal pre-pooling before passing to a language model

**Common pitfalls**

- Number of latents L is a hard hyperparameter — too small loses info, too large defeats the purpose.
- Cross-attention with M ≫ L still costs O(L·M) per layer — not free.
- Latents are PER MODEL, not per sample — same latents broadcast across the batch.

**See also**

- [Perceiver (Jaegle et al. 2021)](https://arxiv.org/abs/2103.03206)
- [Perceiver IO (Jaegle et al. 2021)](https://arxiv.org/abs/2107.14795)
- [Flamingo (Alayrac et al. 2022)](https://arxiv.org/abs/2204.14198)

# RotaryEmbedding

> Inject absolute position into Q, K via per-pair rotations.

**Shapes:** `Q,K:(B, H, T, d) → rotated Q,K`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["Q, K  (B, H, T, d)"]:::io
    n1_0["freqs θ_pos = 10000^(−2i/d)"]:::op
    n2_0["rotate (even, odd) dim pairs"]:::op
    n3_0["Q', K'  (B, H, T, d)"]:::io
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

- LLaMA / LLaMA-2 / LLaMA-3, Mistral, Qwen, GPT-NeoX, PaLM
- Most open LLMs since 2022 — RoPE has displaced learned absolute embeddings
- Position-aware variants for image models (RoPE-ViT)

**Tasks**

- Length-extrapolatable position encoding for LLMs
- Continuous-position retrieval and matching

**Common pitfalls**

- Extrapolation beyond training length still degrades — mitigated by YaRN / NTK / Position Interpolation, not eliminated.
- Half-precision needs care: cos/sin tables in fp32, then cast at use site.
- RoPE rotates Q and K but NOT V — a frequent re-implementation bug.

**See also**

- [RoFormer / RoPE (Su et al. 2021)](https://arxiv.org/abs/2104.09864)
- [YaRN (Peng et al. 2023)](https://arxiv.org/abs/2309.00071)
- [NTK-aware RoPE scaling discussion](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/)

# CausalSelfAttention

> SelfAttention with an upper-triangular mask preventing future leakage.

**Shapes:** `(B, T, D) → (B, T, D)`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["x  (B, T, D)"]:::io
    n1_0["Q, K, V from x"]:::op
    n2_0["MultiHeadAttention + causal mask"]:::attn
    n3_0["y  (B, T, D)"]:::io
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

- GPT-1/2/3/4, LLaMA, Mistral, Gemma — all decoder-only LLMs
- Decision Transformer and Trajectory Transformer in RL
- Autoregressive image / audio models (ImageGPT, MusicLM)

**Tasks**

- Next-token prediction / language modelling
- Any task where outputs are generated left-to-right

**Common pitfalls**

- Mask must be applied to logits BEFORE softmax (set to −∞), not after.
- Cached K/V at inference must respect the mask — slicing past the current position is a common bug that 'works' but leaks future tokens at evaluation.
- Use `is_causal=True` in PyTorch SDPA to enable FlashAttention's fast path rather than passing an explicit mask tensor (much faster, less memory).

**See also**

- [GPT-2 (Radford et al. 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [GPT-3 (Brown et al. 2020)](https://arxiv.org/abs/2005.14165)
- [Decision Transformer (Chen et al. 2021)](https://arxiv.org/abs/2106.01345)

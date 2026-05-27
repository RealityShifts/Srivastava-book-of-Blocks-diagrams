# LoRALinear

> Frozen linear plus a low-rank residual B·A·x scaled by α/r.

**Shapes:** `(B, in) → (B, out)`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["x  (B, in)"]:::io
    n1_0["frozen W · x"]:::op
    n1_1["A · x   (in → r)"]:::op
    n2_0["identity"]:::op
    n2_1["B · (A · x)   (r → out)"]:::op
    n3_0["+  α/r ·"]:::merge
    n4_0["y  (B, out)"]:::io
    n0_0 --> n1_0
    n0_0 --> n1_1
    n1_0 --> n2_0
    n1_1 --> n2_1
    n2_0 --> n3_0
    n2_1 --> n3_0
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

- LoRA / QLoRA — LLM and diffusion fine-tuning
- PEFT library — Hugging Face standard
- SDXL LoRAs on civitai etc. for style/character training

**Tasks**

- Parameter-efficient fine-tuning (PEFT) — train < 1 % of parameters
- Composable / mergeable adapters that can be added on the fly

**Common pitfalls**

- A is init Gaussian, B init zero — swapping breaks the 'starts as identity' invariant.
- Rank r vs α/r scaling: doubling α at fixed r is equivalent to doubling learning rate.
- Merging LoRA back into frozen W (`W ← W + B·A·α/r`) is needed for inference speed.
- Stacking LoRAs at inference is additive — careful with cumulative drift.

**See also**

- [LoRA (Hu et al. 2021)](https://arxiv.org/abs/2106.09685)
- [QLoRA (Dettmers et al. 2023)](https://arxiv.org/abs/2305.14314)

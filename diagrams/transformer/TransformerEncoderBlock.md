# TransformerEncoderBlock

> Pre-norm transformer block: LN → MHA → +; LN → FFN → +.

**Shapes:** `(B, T, D) → (B, T, D)`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["x  (B, T, D)"]:::io
    n1_0["LayerNorm"]:::norm
    subgraph n2_0["MultiHeadAttention"]
        n2_0_0_0["Q  (B, T, D)"]:::io
        n2_0_0_1["K  (B, T, D)"]:::io
        n2_0_0_2["V  (B, T, D)"]:::io
        n2_0_1_0["linear Q"]:::op
        n2_0_1_1["linear K"]:::op
        n2_0_1_2["linear V"]:::op
        n2_0_2_0["split into H heads"]:::op
        n2_0_3_0["Q · Kᵀ / √d_h"]:::op
        n2_0_4_0["softmax (+ optional mask)"]:::act
        n2_0_5_0["· V"]:::op
        n2_0_6_0["concat heads"]:::op
        n2_0_7_0["output linear"]:::op
        n2_0_8_0["y  (B, T, D)"]:::io
    end
    n3_0["+"]:::merge
    n4_0["LayerNorm"]:::norm
    subgraph n5_0["FeedForward"]
        n5_0_0_0["x  (B, T, D)"]:::io
        n5_0_1_0["linear up  (D → r·D)"]:::op
        n5_0_2_0["GELU"]:::act
        n5_0_3_0["linear down  (r·D → D)"]:::op
        n5_0_4_0["y  (B, T, D)"]:::io
    end
    n6_0["+"]:::merge
    n7_0["y  (B, T, D)"]:::io
    n2_0_0_0 --> n2_0_1_0
    n2_0_0_1 --> n2_0_1_1
    n2_0_0_2 --> n2_0_1_2
    n2_0_1_0 --> n2_0_2_0
    n2_0_1_1 --> n2_0_2_0
    n2_0_1_2 --> n2_0_2_0
    n2_0_2_0 --> n2_0_3_0
    n2_0_3_0 --> n2_0_4_0
    n2_0_4_0 --> n2_0_5_0
    n2_0_5_0 --> n2_0_6_0
    n2_0_6_0 --> n2_0_7_0
    n2_0_7_0 --> n2_0_8_0
    n5_0_0_0 --> n5_0_1_0
    n5_0_1_0 --> n5_0_2_0
    n5_0_2_0 --> n5_0_3_0
    n5_0_3_0 --> n5_0_4_0
    n0_0 --> n1_0
    n1_0 --> n2_0_0_0
    n1_0 --> n2_0_0_1
    n1_0 --> n2_0_0_2
    n2_0_8_0 --> n3_0
    n3_0 --> n4_0
    n4_0 --> n5_0_0_0
    n5_0_4_0 --> n6_0
    n6_0 --> n7_0
    n0_0 -.->|"skip"| n3_0
    n3_0 -.->|"skip"| n6_0
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

- BERT, RoBERTa, DeBERTa — masked-LM pre-training
- ViT, DeiT, BEiT — image classification
- Whisper encoder, wav2vec 2.0 — speech

**Tasks**

- Bidirectional representation learning
- Sequence-level / token-level classification, regression, retrieval

**Common pitfalls**

- Post-norm (original Vaswani) is unstable at depth — Pre-norm is the modern default.
- Skip-connection scaling: at extreme depth (>100 layers) consider DeepNet or ReZero to control variance growth.
- Each block has TWO residual sums — implementations sometimes forget one and 'work' with degraded quality.

**See also**

- [Pre-norm Transformer (Xiong et al. 2020)](https://arxiv.org/abs/2002.04745)
- [BERT (Devlin et al. 2018)](https://arxiv.org/abs/1810.04805)
- [DeepNet (Wang et al. 2022)](https://arxiv.org/abs/2203.00555)

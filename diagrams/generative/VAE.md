# VAE

> Encoder → posterior (μ, σ) → reparameterised z → decoder.

**Shapes:** `(B, C, H, W) → (B, C, H, W)`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["x  (B, C, H, W)"]:::io
    n1_0["encoder"]:::op
    n2_0["linear → (μ, log σ²)"]:::op
    n3_0["z = μ + ε ⊙ σ,   ε ∼ N(0, I)"]:::op
    n4_0["decoder"]:::op
    n5_0["x̂  (B, C, H, W)"]:::io
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

- VAE (Kingma & Welling 2013) — original deep generative latent-variable model
- VQ-VAE / VQ-VAE-2 — discrete-latent variants for images / audio
- Stable Diffusion's image autoencoder (compresses pixels → 8× smaller latents)
- Talking-head and motion synthesis (FaceVAE)

**Tasks**

- Unsupervised representation learning with explicit latent prior
- Low-dimensional latent space for downstream models (e.g. diffusion on latents)
- Anomaly detection (high reconstruction loss = unusual sample)

**Common pitfalls**

- Posterior collapse — z carries no information when the decoder is too powerful. Mitigate with β-VAE (down-weight KL), free-bits, or KL warmup.
- Blurry reconstructions are a known limitation of pixel L2 / L1 — perceptual or adversarial loss helps (SD's AE uses both).
- Reparameterisation must use ε ~ N(0,I) during training and z = μ at inference for deterministic encoding.

**See also**

- [VAE (Kingma & Welling 2013)](https://arxiv.org/abs/1312.6114)
- [β-VAE (Higgins et al. 2017)](https://openreview.net/forum?id=Sy2fzU9gl)
- [VQ-VAE (van den Oord et al. 2017)](https://arxiv.org/abs/1711.00937)

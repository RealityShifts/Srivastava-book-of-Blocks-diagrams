# Architecture diagrams

## See [INDEX for charts](./INDEX.md)

One Mermaid `flowchart TD` per public block, organised by category. The
specs live in [`_generate.py`](./_generate.py); regenerate everything with

```bash
python _generate.py
```

## Where these render

| Tool | What to do |
| ---- | ---------- |
| **GitHub** | Renders the `.md` files inline — just open them in the browser. |
| **draw.io / diagrams.net** | *Arrange → Insert → Advanced → Mermaid*, paste the fenced block. |
| **Excalidraw** | *Library → Mermaid to Excalidraw*, paste the fenced block. |
| **Notion / Obsidian / GitLab / VS Code** | Native Mermaid in markdown preview. |
| **Standalone SVG** | `npx -y @mermaid-js/mermaid-cli -i path/to/Block.md -o Block.svg` |

## Recommended diagramming tools (alternatives to draw.io)

* **Excalidraw** — beautiful hand-drawn look, exports SVG, supports Mermaid import.
* **D2** (`d2lang.com`) — declarative, very clean output, good for hierarchies.
* **Mermaid Live Editor** (`mermaid.live`) — paste & download SVG/PNG.
* **TikZ / PGF** — gold standard for paper-quality figures (LaTeX).
* **PlotNeuralNet** — tex-based 3-D blocks for deep nets.
* **NN-SVG** — quick SVGs for classic CNN / FCN / LeNet shapes.
* **Penrose** — declarative diagram constraints if you need bespoke layouts.

## Color legend

| Class | Used for |
| --- | --- |
| `io`     | Inputs and outputs |
| `op`     | Generic differentiable op (matmul, conv, …) |
| `norm`   | Normalisation layers |
| `act`    | Activation functions |
| `attn`   | Attention operators |
| `merge`  | Sum / concat / element-wise combine |
| `emb`    | Embedding tables / encoded representations |
| `loss`   | Loss / objective |
| `ctrl`   | Control / non-differentiable flow |

Dashed arrows (`-. skip .->`) mark residual / skip connections.



# Index

## core

Core neural-network primitives.

* [Linear](./core/Linear.md)
* [ConvBlock](./core/ConvBlock.md)
* [DepthwiseSeparableConv2d](./core/DepthwiseSeparableConv2d.md)
* [DilatedConv2d](./core/DilatedConv2d.md)
* [GroupConv2d](./core/GroupConv2d.md)
* [Conv1d](./core/Conv1d.md)
* [Conv3d](./core/Conv3d.md)
* [Mish](./core/Mish.md)
* [RMSNorm](./core/RMSNorm.md)
* [AdaIN](./core/AdaIN.md)
* [SPADE](./core/SPADE.md)
* [ResidualBlock](./core/ResidualBlock.md)
* [SkipConnection](./core/SkipConnection.md)

## attention

Attention mechanisms.

* [MultiHeadAttention](./attention/MultiHeadAttention.md)
* [SelfAttention](./attention/SelfAttention.md)
* [CausalSelfAttention](./attention/CausalSelfAttention.md)
* [CrossAttention](./attention/CrossAttention.md)
* [WindowAttention](./attention/WindowAttention.md)
* [LinearAttention](./attention/LinearAttention.md)
* [FlashAttention](./attention/FlashAttention.md)
* [RotaryEmbedding](./attention/RotaryEmbedding.md)
* [RelativePositionBias](./attention/RelativePositionBias.md)
* [AttentionPooling](./attention/AttentionPooling.md)

## transformer

Transformer encoder / decoder, FFN variants, MoE.

* [FeedForward](./transformer/FeedForward.md)
* [SwiGLU](./transformer/SwiGLU.md)
* [GEGLU](./transformer/GEGLU.md)
* [TransformerEncoderBlock](./transformer/TransformerEncoderBlock.md)
* [TransformerDecoderBlock](./transformer/TransformerDecoderBlock.md)
* [MixtureOfExperts](./transformer/MixtureOfExperts.md)
* [SwitchMoE](./transformer/SwitchMoE.md)

## cnn_vision

CNN and vision-specific blocks.

* [InceptionBlock](./cnn_vision/InceptionBlock.md)
* [DenseBlock](./cnn_vision/DenseBlock.md)
* [SqueezeExcitation](./cnn_vision/SqueezeExcitation.md)
* [CBAM](./cnn_vision/CBAM.md)
* [SpatialPyramidPooling](./cnn_vision/SpatialPyramidPooling.md)
* [FeaturePyramidNetwork](./cnn_vision/FeaturePyramidNetwork.md)
* [ASPP](./cnn_vision/ASPP.md)
* [PixelShuffleUpsample](./cnn_vision/PixelShuffleUpsample.md)
* [DeformableConv2d](./cnn_vision/DeformableConv2d.md)
* [DeformableAttention](./cnn_vision/DeformableAttention.md)

## unet_diffusion

UNet, time conditioning, ControlNet, LoRA, hypernets.

* [SinusoidalTimeEmbedding](./unet_diffusion/SinusoidalTimeEmbedding.md)
* [TimestepMLP](./unet_diffusion/TimestepMLP.md)
* [DownsampleBlock](./unet_diffusion/DownsampleBlock.md)
* [UpsampleBlock](./unet_diffusion/UpsampleBlock.md)
* [UNetResBlock](./unet_diffusion/UNetResBlock.md)
* [UNet](./unet_diffusion/UNet.md)
* [NoisePredictor](./unet_diffusion/NoisePredictor.md)
* [ZeroConv2d](./unet_diffusion/ZeroConv2d.md)
* [ControlNetBlock](./unet_diffusion/ControlNetBlock.md)
* [LoRALinear](./unet_diffusion/LoRALinear.md)
* [LoRAConv2d](./unet_diffusion/LoRAConv2d.md)
* [HyperNetwork](./unet_diffusion/HyperNetwork.md)
* [IPAdapterCrossAttention](./unet_diffusion/IPAdapterCrossAttention.md)

## gan

GAN building blocks: StyleGAN, PGGAN, equalised LR.

* [EqualLinear](./gan/EqualLinear.md)
* [EqualConv2d](./gan/EqualConv2d.md)
* [GeneratorBlock](./gan/GeneratorBlock.md)
* [DiscriminatorBlock](./gan/DiscriminatorBlock.md)
* [MappingNetwork](./gan/MappingNetwork.md)
* [StyleBlock](./gan/StyleBlock.md)
* [ModulatedConv2d](./gan/ModulatedConv2d.md)
* [MinibatchStdDev](./gan/MinibatchStdDev.md)
* [ProgressiveGrowing](./gan/ProgressiveGrowing.md)

## vit

Vision Transformer blocks.

* [PatchEmbedding](./vit/PatchEmbedding.md)
* [CLSToken](./vit/CLSToken.md)
* [SwinWindowAttention](./vit/SwinWindowAttention.md)
* [ShiftedWindowAttention](./vit/ShiftedWindowAttention.md)
* [MaskedImageModeling](./vit/MaskedImageModeling.md)

## sequence

Recurrent and state-space sequence models.

* [RNNCell](./sequence/RNNCell.md)
* [LSTMCell](./sequence/LSTMCell.md)
* [GRUCell](./sequence/GRUCell.md)
* [StateSpaceModel](./sequence/StateSpaceModel.md)
* [MambaBlock](./sequence/MambaBlock.md)

## gnn

Graph neural network layers.

* [MessagePassing](./gnn/MessagePassing.md)
* [GraphConv](./gnn/GraphConv.md)
* [GraphAttention](./gnn/GraphAttention.md)

## generative

VAE, autoregressive, normalising-flow, EBM, diffusion schedulers.

* [VAE](./generative/VAE.md)
* [MaskedConv2d](./generative/MaskedConv2d.md)
* [AutoregressiveBlock](./generative/AutoregressiveBlock.md)
* [AffineCouplingLayer](./generative/AffineCouplingLayer.md)
* [EnergyBasedModel](./generative/EnergyBasedModel.md)
* [DDPMScheduler](./generative/DDPMScheduler.md)
* [DDIMScheduler](./generative/DDIMScheduler.md)

## rl

Reinforcement-learning building blocks.

* [PolicyNetwork](./rl/PolicyNetwork.md)
* [ValueNetwork](./rl/ValueNetwork.md)
* [QNetwork](./rl/QNetwork.md)
* [ActorCritic](./rl/ActorCritic.md)
* [ReplayBuffer](./rl/ReplayBuffer.md)
* [TargetNetwork](./rl/TargetNetwork.md)

## memory_retrieval

External memory, vector stores, RAG, KV caches.

* [ExternalMemory](./memory_retrieval/ExternalMemory.md)
* [VectorStore](./memory_retrieval/VectorStore.md)
* [RAGModule](./memory_retrieval/RAGModule.md)
* [KVCache](./memory_retrieval/KVCache.md)

## embedding

Token / positional / projection embeddings, contrastive losses.

* [TokenEmbedding](./embedding/TokenEmbedding.md)
* [LearnedPositionalEmbedding](./embedding/LearnedPositionalEmbedding.md)
* [SinusoidalPositionalEmbedding](./embedding/SinusoidalPositionalEmbedding.md)
* [ProjectionHead](./embedding/ProjectionHead.md)
* [CLIPLoss](./embedding/CLIPLoss.md)
* [info_nce](./embedding/info_nce.md)

## optimization

Optimisers, schedulers, EMA, mixed-precision, checkpointing.

* [Lion](./optimization/Lion.md)
* [Sophia](./optimization/Sophia.md)
* [EMA](./optimization/EMA.md)
* [MixedPrecisionTrainer](./optimization/MixedPrecisionTrainer.md)
* [CheckpointedSequential](./optimization/CheckpointedSequential.md)

## multimodal

Multimodal / agentic blocks.

* [CLIPEncoder](./multimodal/CLIPEncoder.md)
* [PerceiverResampler](./multimodal/PerceiverResampler.md)
* [QFormer](./multimodal/QFormer.md)
* [ToolUseBlock](./multimodal/ToolUseBlock.md)
* [MemoryAttention](./multimodal/MemoryAttention.md)

## efficient

Sparsity, quantisation, parallelism, low-rank.

* [QuantizedLinearInt8](./efficient/QuantizedLinearInt8.md)
* [QuantizedLinear4bit](./efficient/QuantizedLinear4bit.md)
* [MagnitudePruner](./efficient/MagnitudePruner.md)
* [TokenPruner](./efficient/TokenPruner.md)
* [LowRankLinear](./efficient/LowRankLinear.md)
* [ColumnParallelLinear](./efficient/ColumnParallelLinear.md)
* [RowParallelLinear](./efficient/RowParallelLinear.md)
* [PipelineStage](./efficient/PipelineStage.md)

## specialized

Specialised research blocks (NeuralODE, FNO, KAN, capsules, slots).

* [NeuralODE](./specialized/NeuralODE.md)
* [SpectralConv2d](./specialized/SpectralConv2d.md)
* [FNOBlock](./specialized/FNOBlock.md)
* [KANLayer](./specialized/KANLayer.md)
* [CapsuleLayer](./specialized/CapsuleLayer.md)
* [SlotAttention](./specialized/SlotAttention.md)


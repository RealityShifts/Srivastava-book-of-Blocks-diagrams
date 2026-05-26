# Architecture diagrams

One Mermaid `flowchart TD` per public block, organised by category. Specs
live in [`blocks/`](./blocks), the DSL helpers in [`dsl.py`](./dsl.py),
and the renderer in [`_generate.py`](./_generate.py). All generated
diagrams live under [`diagrams/`](./diagrams) — every block in the index
below links to its own `.md` file there. Regenerate everything with

```bash
python _generate.py
```

## Visual editor

A React Flow-based web editor lives in [`editor/`](./editor) — drag from
a palette of the 10 primitive kinds + all 122 built-in blocks, wire them
together, get live shape-checking (mismatched edges turn red), then export
to **Mermaid**, **DSL `.py`** (round-trips through `python _generate.py
--specs ...`), or **Graph JSON**. Use *Import Mermaid* to pull any `.md`
from [`diagrams/`](./diagrams) straight back onto the canvas.

```bash
cd editor
npm install
npm run extract-library   # snapshot blocks/ -> public/library.json
npm run dev               # http://localhost:5173
```

## Reverser: Mermaid → DSL `.py`

[`_reverse.py`](./_reverse.py) inverts the renderer. Point it at any
generated `.md` (or several) and it emits a drop-in spec that
`_generate.py --specs ...` re-renders byte-for-byte identically (122/122
built-ins round-trip exactly). Subgraphs collapse back to a single
`_ref("Name")`; skip arrows survive as positional skips; per-port shape
metadata is the only thing lost (it never reaches Mermaid in the first
place).

```bash
python _reverse.py diagrams/core/ResidualBlock.md            # writes ResidualBlock_spec.py next to it
python _reverse.py diagrams/attention/*.md -o attn_spec.py    # bundle a whole category
python _reverse.py diagrams/core/Linear.md -o -               # stdout
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
| `ref`    | Reference to another block (expanded as a subgraph) |

Dashed arrows (`-. skip .->`) mark residual / skip connections.

## Recursive expansion

Specs may use `_ref("BlockName")` to point at another registered block.
Run `python _generate.py --depth N` to inline references up to `N` levels
deep as nested Mermaid `subgraph` blocks (default: 1). At depth 0 each
ref renders as a single `ref`-styled box; cycles are detected and broken.

## Custom architectures (`--specs`)

The 122 built-in blocks act as a reusable library. To diagram your own
architecture, write a Python file that exposes either `BLOCKS` (single
category) or `CATEGORIES` (multi-category), reusing the DSL helpers and
referencing built-ins by name:

```python
# my_arch.py
from _generate import _io, _op, _ref

CATEGORY = "myarch"
CATEGORY_DESC = "A 12-layer transformer wired from built-in blocks."

BLOCKS = {
    "MyTransformer": (
        "Stacked TransformerEncoderBlocks fed by a token embedding.",
        "(B, T) → (B, T, D)",
        [
            [_io("ids  (B, T)")],
            [_ref("TokenEmbedding")],
            [_ref("TransformerEncoderBlock")],
            [_ref("TransformerEncoderBlock")],
            [_io("y  (B, T, D)")],
        ],
    ),
}
```

Then generate:

```bash
# writes into ./diagrams/<your category>/<BlockName>.md by default
python _generate.py --specs my_arch.py --depth 1
# only your blocks, library kept as registered references:
python _generate.py --specs my_arch.py --no-builtins
# or send the output anywhere else
python _generate.py --specs my_arch.py --out ./out
```

Pass `--specs` multiple times to merge several files. User block names
shadow built-ins of the same name.


## Index

122 blocks across 17 categories. Click a section to expand.

<details><summary><b>core</b> &middot; Core neural-network primitives. &middot; 13 blocks</summary>

- [Linear](diagrams/core/Linear.md)
- [ConvBlock](diagrams/core/ConvBlock.md)
- [DepthwiseSeparableConv2d](diagrams/core/DepthwiseSeparableConv2d.md)
- [DilatedConv2d](diagrams/core/DilatedConv2d.md)
- [GroupConv2d](diagrams/core/GroupConv2d.md)
- [Conv1d](diagrams/core/Conv1d.md)
- [Conv3d](diagrams/core/Conv3d.md)
- [Mish](diagrams/core/Mish.md)
- [RMSNorm](diagrams/core/RMSNorm.md)
- [AdaIN](diagrams/core/AdaIN.md)
- [SPADE](diagrams/core/SPADE.md)
- [ResidualBlock](diagrams/core/ResidualBlock.md)
- [SkipConnection](diagrams/core/SkipConnection.md)

</details>

<details><summary><b>attention</b> &middot; Attention mechanisms. &middot; 10 blocks</summary>

- [MultiHeadAttention](diagrams/attention/MultiHeadAttention.md)
- [SelfAttention](diagrams/attention/SelfAttention.md)
- [CausalSelfAttention](diagrams/attention/CausalSelfAttention.md)
- [CrossAttention](diagrams/attention/CrossAttention.md)
- [WindowAttention](diagrams/attention/WindowAttention.md)
- [LinearAttention](diagrams/attention/LinearAttention.md)
- [FlashAttention](diagrams/attention/FlashAttention.md)
- [RotaryEmbedding](diagrams/attention/RotaryEmbedding.md)
- [RelativePositionBias](diagrams/attention/RelativePositionBias.md)
- [AttentionPooling](diagrams/attention/AttentionPooling.md)

</details>

<details><summary><b>transformer</b> &middot; Transformer encoder / decoder, FFN variants, MoE. &middot; 7 blocks</summary>

- [FeedForward](diagrams/transformer/FeedForward.md)
- [SwiGLU](diagrams/transformer/SwiGLU.md)
- [GEGLU](diagrams/transformer/GEGLU.md)
- [TransformerEncoderBlock](diagrams/transformer/TransformerEncoderBlock.md)
- [TransformerDecoderBlock](diagrams/transformer/TransformerDecoderBlock.md)
- [MixtureOfExperts](diagrams/transformer/MixtureOfExperts.md)
- [SwitchMoE](diagrams/transformer/SwitchMoE.md)

</details>

<details><summary><b>cnn_vision</b> &middot; CNN and vision-specific blocks. &middot; 10 blocks</summary>

- [InceptionBlock](diagrams/cnn_vision/InceptionBlock.md)
- [DenseBlock](diagrams/cnn_vision/DenseBlock.md)
- [SqueezeExcitation](diagrams/cnn_vision/SqueezeExcitation.md)
- [CBAM](diagrams/cnn_vision/CBAM.md)
- [SpatialPyramidPooling](diagrams/cnn_vision/SpatialPyramidPooling.md)
- [FeaturePyramidNetwork](diagrams/cnn_vision/FeaturePyramidNetwork.md)
- [ASPP](diagrams/cnn_vision/ASPP.md)
- [PixelShuffleUpsample](diagrams/cnn_vision/PixelShuffleUpsample.md)
- [DeformableConv2d](diagrams/cnn_vision/DeformableConv2d.md)
- [DeformableAttention](diagrams/cnn_vision/DeformableAttention.md)

</details>

<details><summary><b>unet_diffusion</b> &middot; UNet, time conditioning, ControlNet, LoRA, hypernets. &middot; 13 blocks</summary>

- [SinusoidalTimeEmbedding](diagrams/unet_diffusion/SinusoidalTimeEmbedding.md)
- [TimestepMLP](diagrams/unet_diffusion/TimestepMLP.md)
- [DownsampleBlock](diagrams/unet_diffusion/DownsampleBlock.md)
- [UpsampleBlock](diagrams/unet_diffusion/UpsampleBlock.md)
- [UNetResBlock](diagrams/unet_diffusion/UNetResBlock.md)
- [UNet](diagrams/unet_diffusion/UNet.md)
- [NoisePredictor](diagrams/unet_diffusion/NoisePredictor.md)
- [ZeroConv2d](diagrams/unet_diffusion/ZeroConv2d.md)
- [ControlNetBlock](diagrams/unet_diffusion/ControlNetBlock.md)
- [LoRALinear](diagrams/unet_diffusion/LoRALinear.md)
- [LoRAConv2d](diagrams/unet_diffusion/LoRAConv2d.md)
- [HyperNetwork](diagrams/unet_diffusion/HyperNetwork.md)
- [IPAdapterCrossAttention](diagrams/unet_diffusion/IPAdapterCrossAttention.md)

</details>

<details><summary><b>gan</b> &middot; GAN building blocks: StyleGAN, PGGAN, equalised LR. &middot; 9 blocks</summary>

- [EqualLinear](diagrams/gan/EqualLinear.md)
- [EqualConv2d](diagrams/gan/EqualConv2d.md)
- [GeneratorBlock](diagrams/gan/GeneratorBlock.md)
- [DiscriminatorBlock](diagrams/gan/DiscriminatorBlock.md)
- [MappingNetwork](diagrams/gan/MappingNetwork.md)
- [StyleBlock](diagrams/gan/StyleBlock.md)
- [ModulatedConv2d](diagrams/gan/ModulatedConv2d.md)
- [MinibatchStdDev](diagrams/gan/MinibatchStdDev.md)
- [ProgressiveGrowing](diagrams/gan/ProgressiveGrowing.md)

</details>

<details><summary><b>vit</b> &middot; Vision Transformer blocks. &middot; 5 blocks</summary>

- [PatchEmbedding](diagrams/vit/PatchEmbedding.md)
- [CLSToken](diagrams/vit/CLSToken.md)
- [SwinWindowAttention](diagrams/vit/SwinWindowAttention.md)
- [ShiftedWindowAttention](diagrams/vit/ShiftedWindowAttention.md)
- [MaskedImageModeling](diagrams/vit/MaskedImageModeling.md)

</details>

<details><summary><b>sequence</b> &middot; Recurrent and state-space sequence models. &middot; 5 blocks</summary>

- [RNNCell](diagrams/sequence/RNNCell.md)
- [LSTMCell](diagrams/sequence/LSTMCell.md)
- [GRUCell](diagrams/sequence/GRUCell.md)
- [StateSpaceModel](diagrams/sequence/StateSpaceModel.md)
- [MambaBlock](diagrams/sequence/MambaBlock.md)

</details>

<details><summary><b>gnn</b> &middot; Graph neural network layers. &middot; 3 blocks</summary>

- [MessagePassing](diagrams/gnn/MessagePassing.md)
- [GraphConv](diagrams/gnn/GraphConv.md)
- [GraphAttention](diagrams/gnn/GraphAttention.md)

</details>

<details><summary><b>generative</b> &middot; VAE, autoregressive, normalising-flow, EBM, diffusion schedulers. &middot; 7 blocks</summary>

- [VAE](diagrams/generative/VAE.md)
- [MaskedConv2d](diagrams/generative/MaskedConv2d.md)
- [AutoregressiveBlock](diagrams/generative/AutoregressiveBlock.md)
- [AffineCouplingLayer](diagrams/generative/AffineCouplingLayer.md)
- [EnergyBasedModel](diagrams/generative/EnergyBasedModel.md)
- [DDPMScheduler](diagrams/generative/DDPMScheduler.md)
- [DDIMScheduler](diagrams/generative/DDIMScheduler.md)

</details>

<details><summary><b>rl</b> &middot; Reinforcement-learning building blocks. &middot; 6 blocks</summary>

- [PolicyNetwork](diagrams/rl/PolicyNetwork.md)
- [ValueNetwork](diagrams/rl/ValueNetwork.md)
- [QNetwork](diagrams/rl/QNetwork.md)
- [ActorCritic](diagrams/rl/ActorCritic.md)
- [ReplayBuffer](diagrams/rl/ReplayBuffer.md)
- [TargetNetwork](diagrams/rl/TargetNetwork.md)

</details>

<details><summary><b>memory_retrieval</b> &middot; External memory, vector stores, RAG, KV caches. &middot; 4 blocks</summary>

- [ExternalMemory](diagrams/memory_retrieval/ExternalMemory.md)
- [VectorStore](diagrams/memory_retrieval/VectorStore.md)
- [RAGModule](diagrams/memory_retrieval/RAGModule.md)
- [KVCache](diagrams/memory_retrieval/KVCache.md)

</details>

<details><summary><b>embedding</b> &middot; Token / positional / projection embeddings, contrastive losses. &middot; 6 blocks</summary>

- [TokenEmbedding](diagrams/embedding/TokenEmbedding.md)
- [LearnedPositionalEmbedding](diagrams/embedding/LearnedPositionalEmbedding.md)
- [SinusoidalPositionalEmbedding](diagrams/embedding/SinusoidalPositionalEmbedding.md)
- [ProjectionHead](diagrams/embedding/ProjectionHead.md)
- [CLIPLoss](diagrams/embedding/CLIPLoss.md)
- [info_nce](diagrams/embedding/info_nce.md)

</details>

<details><summary><b>optimization</b> &middot; Optimisers, schedulers, EMA, mixed-precision, checkpointing. &middot; 5 blocks</summary>

- [Lion](diagrams/optimization/Lion.md)
- [Sophia](diagrams/optimization/Sophia.md)
- [EMA](diagrams/optimization/EMA.md)
- [MixedPrecisionTrainer](diagrams/optimization/MixedPrecisionTrainer.md)
- [CheckpointedSequential](diagrams/optimization/CheckpointedSequential.md)

</details>

<details><summary><b>multimodal</b> &middot; Multimodal / agentic blocks. &middot; 5 blocks</summary>

- [CLIPEncoder](diagrams/multimodal/CLIPEncoder.md)
- [PerceiverResampler](diagrams/multimodal/PerceiverResampler.md)
- [QFormer](diagrams/multimodal/QFormer.md)
- [ToolUseBlock](diagrams/multimodal/ToolUseBlock.md)
- [MemoryAttention](diagrams/multimodal/MemoryAttention.md)

</details>

<details><summary><b>efficient</b> &middot; Sparsity, quantisation, parallelism, low-rank. &middot; 8 blocks</summary>

- [QuantizedLinearInt8](diagrams/efficient/QuantizedLinearInt8.md)
- [QuantizedLinear4bit](diagrams/efficient/QuantizedLinear4bit.md)
- [MagnitudePruner](diagrams/efficient/MagnitudePruner.md)
- [TokenPruner](diagrams/efficient/TokenPruner.md)
- [LowRankLinear](diagrams/efficient/LowRankLinear.md)
- [ColumnParallelLinear](diagrams/efficient/ColumnParallelLinear.md)
- [RowParallelLinear](diagrams/efficient/RowParallelLinear.md)
- [PipelineStage](diagrams/efficient/PipelineStage.md)

</details>

<details><summary><b>specialized</b> &middot; Specialised research blocks (NeuralODE, FNO, KAN, capsules, slots). &middot; 6 blocks</summary>

- [NeuralODE](diagrams/specialized/NeuralODE.md)
- [SpectralConv2d](diagrams/specialized/SpectralConv2d.md)
- [FNOBlock](diagrams/specialized/FNOBlock.md)
- [KANLayer](diagrams/specialized/KANLayer.md)
- [CapsuleLayer](diagrams/specialized/CapsuleLayer.md)
- [SlotAttention](diagrams/specialized/SlotAttention.md)

</details>

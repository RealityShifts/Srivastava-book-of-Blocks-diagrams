# Architecture diagrams
## See [Index](./INDEX.md)


One Mermaid `flowchart TD` per public block, organised by category. The
specs live in `[_generate.py](./_generate.py)`; regenerate everything with

```bash
python _generate.py
```

## Where these render


| Tool                                     | What to do                                                        |
| ---------------------------------------- | ----------------------------------------------------------------- |
| **GitHub**                               | Renders the `.md` files inline — just open them in the browser.   |
| **draw.io / diagrams.net**               | *Arrange → Insert → Advanced → Mermaid*, paste the fenced block.  |
| **Excalidraw**                           | *Library → Mermaid to Excalidraw*, paste the fenced block.        |
| **Notion / Obsidian / GitLab / VS Code** | Native Mermaid in markdown preview.                               |
| **Standalone SVG**                       | `npx -y @mermaid-js/mermaid-cli -i path/to/Block.md -o Block.svg` |


## Recommended diagramming tools (alternatives to draw.io)

- **Excalidraw** — beautiful hand-drawn look, exports SVG, supports Mermaid import.
- **D2** (`d2lang.com`) — declarative, very clean output, good for hierarchies.
- **Mermaid Live Editor** (`mermaid.live`) — paste & download SVG/PNG.
- **TikZ / PGF** — gold standard for paper-quality figures (LaTeX).
- **PlotNeuralNet** — tex-based 3-D blocks for deep nets.
- **NN-SVG** — quick SVGs for classic CNN / FCN / LeNet shapes.
- **Penrose** — declarative diagram constraints if you need bespoke layouts.

## Color legend


| Class   | Used for                                            |
| ------- | --------------------------------------------------- |
| `io`    | Inputs and outputs                                  |
| `op`    | Generic differentiable op (matmul, conv, …)         |
| `norm`  | Normalisation layers                                |
| `act`   | Activation functions                                |
| `attn`  | Attention operators                                 |
| `merge` | Sum / concat / element-wise combine                 |
| `emb`   | Embedding tables / encoded representations          |
| `loss`  | Loss / objective                                    |
| `ctrl`  | Control / non-differentiable flow                   |
| `ref`   | Reference to another block (expanded as a subgraph) |


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
python _generate.py --specs my_arch.py --out ./diagrams --depth 1
# only your blocks, library kept as registered references:
python _generate.py --specs my_arch.py --out ./diagrams --no-builtins
```

Pass `--specs` multiple times to merge several files. User block names
shadow built-ins of the same name.
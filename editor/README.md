# Block Diagrams Editor

Visual editor for authoring the same block-diagram specs that live in
[`../blocks/`](../blocks/). Built on **React Flow** (`@xyflow/react`) +
**Vite** + **TypeScript** + **TailwindCSS** + **Zustand**.

Exports to three formats:

| Format | Use for |
| --- | --- |
| **Mermaid (`.md`)** | Paste into GitHub / draw.io / Excalidraw |
| **DSL spec (`.py`)** | Drop into `python ../_generate.py --specs <file>` to regenerate diagrams from the parent toolchain |
| **Graph JSON** | Editor's native format — round-trippable save/load |

## Quickstart

```bash
cd editor
npm install
# (one-off) snapshot the 122 built-in blocks into public/library.json
npm run extract-library
npm run dev          # http://localhost:5173
npm run build        # production bundle in dist/
```

## What it ships with

- **Canvas** — pan / zoom / multiselect / minimap / controls.
- **Palette** — left sidebar:
  - 9 primitive kinds (`io`, `op`, `norm`, `act`, `attn`, `merge`, `emb`, `loss`, `ctrl`).
  - All 122 built-in library blocks, searchable, drag-to-add as a `_ref` node.
  - **Shift while dragging** a library item to inline its sub-graph instead of dropping a single ref.
  - **Right-click a `ref` node** to expand it into its sub-graph in place.
- **Inspector** — right sidebar:
  - Edit block metadata (name / category / description / shapes-string).
  - Edit selected node (kind / label / id / in / out shape).
  - Edit selected edge (label / solid vs skip).
- **Shape checking** — when a node declares `out` and the next declares `in`
  and they disagree (whitespace-normalised), the edge turns **red** with a
  `(out) ≠ (in)` label. Exactly mirrors the renderer in
  [`../_generate.py`](../_generate.py).
- **Toolbar** — Import JSON / **Import Mermaid** / Export JSON / Export Mermaid / Export DSL .py / Clear.
  *Import Mermaid* parses any `.md` produced by `../_generate.py` (or by *Export Mermaid* itself) and drops it back onto the canvas. Subgraphs collapse to a single `_ref` node; per-port shapes don't survive the round-trip since they aren't encoded in Mermaid.

## Authoring loop

1. Draw your graph in the editor.
2. Click **Export DSL .py** → `myblock_spec.py`.
3. Run the existing generator:

   ```bash
   # writes into ../diagrams/<your category>/<BlockName>.md by default
   python ../_generate.py --specs myblock_spec.py
   # or pick your own output dir:
   python ../_generate.py --specs myblock_spec.py --out ./out
   ```

4. Open `../diagrams/<category>/<BlockName>.md` for the Mermaid render, or
   open the `.md` exported directly from **Export Mermaid** if you just want
   the diagram.

## Future: PyTorch / Flax codegen

The data model (kind + id + shapes + explicit edges) is already enough to
emit PyTorch / Flax instantiation code by mapping `_ref(blockName)` to the
class implementations in
[`RealityShifts/Srivastava-book-of-Blocks`](https://github.com/RealityShifts/Srivastava-book-of-Blocks).
A `src/generators/pytorch.ts` (and Flax counterpart) is the natural place
to land that — it consumes the same `(meta, nodes, edges)` triple as the
existing exporters.

## Re-snapshotting the library

`public/library.json` is committed and used at runtime. Regenerate when you
touch `../blocks/`:

```bash
npm run extract-library
```

## Project layout

```
editor/
├── public/library.json           # 122 blocks, extracted from ../blocks/*.py
├── scripts/extract_library.py    # the extractor
└── src/
    ├── App.tsx                   # 3-pane shell
    ├── store.ts                  # Zustand store (RF state is the source of truth)
    ├── types.ts                  # NodeData / EdgeData / GraphJSON / LibSpec
    ├── components/
    │   ├── Editor.tsx            # canvas + drag-drop + ref expansion
    │   ├── Palette.tsx           # primitives + searchable library
    │   ├── Inspector.tsx         # metadata + selected node/edge editor
    │   ├── Toolbar.tsx           # import / export buttons
    │   ├── BlockNode.tsx         # custom React Flow node
    │   └── ShapeEdge.tsx         # custom edge with live shape-check colouring
    ├── generators/
    │   ├── mermaid.ts            # → flowchart TD (same output as _generate.py)
    │   ├── reverseMermaid.ts     # ← flowchart TD (mirrors ../_reverse.py)
    │   ├── dsl.ts                # → user spec.py with explicit edges
    │   ├── json.ts               # → / ← GraphJSON
    │   └── topo.ts               # longest-path layering used by dsl.ts
    └── library/
        └── shapeCheck.ts         # canonical shape-mismatch check
```

import type { BlockEdgeT, BlockMeta, BlockNodeT } from "../types";
import { checkShapes } from "../library/shapeCheck";

/**
 * Default Mermaid layout knobs — kept in sync with ../../../_generate.py's
 * DEFAULT_RANK_SPACING / DEFAULT_NODE_SPACING so editor exports and Python
 * regenerations render at the same density.
 */
export const DEFAULT_RANK_SPACING = 10;
export const DEFAULT_NODE_SPACING = 30;

const STYLE = `    classDef io fill:#f1f5f9,stroke:#334155,stroke-width:1.4px,color:#0f172a
    classDef op fill:#dbeafe,stroke:#1d4ed8,stroke-width:1.4px,color:#1e3a8a
    classDef norm fill:#dcfce7,stroke:#15803d,stroke-width:1.4px,color:#14532d
    classDef act fill:#ffedd5,stroke:#c2410c,stroke-width:1.4px,color:#7c2d12
    classDef attn fill:#ede9fe,stroke:#6d28d9,stroke-width:1.4px,color:#4c1d95
    classDef merge fill:#fef3c7,stroke:#b45309,stroke-width:1.4px,color:#78350f
    classDef emb fill:#fef9c3,stroke:#a16207,stroke-width:1.4px,color:#713f12
    classDef loss fill:#fee2e2,stroke:#b91c1c,stroke-width:1.4px,color:#7f1d1d
    classDef ctrl fill:#f5f5f4,stroke:#52525b,stroke-width:1.4px,color:#27272a
    classDef ref fill:#e0f2fe,stroke:#0369a1,stroke-width:2px,color:#0c4a6e,stroke-dasharray: 4 2`;

const MISMATCH_STYLE = "stroke:#dc2626,stroke-width:2.2px,color:#991b1b";

const esc = (s: string) => s.replace(/"/g, "&quot;").replace(/`/g, "&#96;");

const nodeLabel = (n: BlockNodeT) =>
  n.data.kind === "ref" && n.data.refName ? n.data.refName : n.data.label;

/**
 * Build the same flat Mermaid `flowchart TD` body that ../../_generate.py
 * emits: per-node declarations, then per-edge declarations, then any
 * `linkStyle` lines for shape-mismatch edges, then the shared STYLE block.
 *
 * Node ids in the output mirror React Flow's internal ids — that's fine for
 * Mermaid; the DSL exporter is the one that re-derives positional ids.
 */
export type LayoutOpts = {
  rankSpacing?: number;
  nodeSpacing?: number;
};

const initDirective = (rank: number, node: number) =>
  `%%{init: {'flowchart': {'rankSpacing': ${rank}, 'nodeSpacing': ${node}}}}%%`;

export const toMermaid = (
  meta: BlockMeta,
  nodes: BlockNodeT[],
  edges: BlockEdgeT[],
  layout: LayoutOpts = {},
): string => {
  const rank = layout.rankSpacing ?? DEFAULT_RANK_SPACING;
  const node = layout.nodeSpacing ?? DEFAULT_NODE_SPACING;
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const lines: string[] = [];

  for (const n of nodes) {
    lines.push(`${n.id}["${esc(nodeLabel(n))}"]:::${n.data.kind}`);
  }

  const mismatchIndices: number[] = [];
  let edgeIdx = 0;
  for (const e of edges) {
    const src = byId.get(e.source);
    const tgt = byId.get(e.target);
    const isSkip = e.data?.style === "skip";
    const { mismatch, label: mlabel } = isSkip
      ? { mismatch: false, label: undefined as string | undefined }
      : checkShapes(src, tgt);

    if (mismatch) {
      lines.push(`${e.source} ==>|"${esc(mlabel ?? "shape mismatch")}"| ${e.target}`);
      mismatchIndices.push(edgeIdx);
    } else if (isSkip) {
      lines.push(`${e.source} -.->|"${esc(e.data?.label ?? "skip")}"| ${e.target}`);
    } else if (e.data?.label) {
      lines.push(`${e.source} -->|"${esc(e.data.label)}"| ${e.target}`);
    } else {
      lines.push(`${e.source} --> ${e.target}`);
    }
    edgeIdx++;
  }

  if (mismatchIndices.length) {
    lines.push(`linkStyle ${mismatchIndices.join(",")} ${MISMATCH_STYLE}`);
  }

  const body = [
    initDirective(rank, node),
    "flowchart TD",
    ...lines.map((l) => "    " + l),
    STYLE,
  ].join("\n");

  return [
    `# ${meta.name}`,
    "",
    `> ${meta.desc}`,
    "",
    `**Shapes:** \`${meta.shapesStr}\``,
    "",
    "```mermaid",
    body,
    "```",
    "",
  ].join("\n");
};

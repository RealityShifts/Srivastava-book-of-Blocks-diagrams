import type { Edge, Node } from "@xyflow/react";

export const NODE_KINDS = [
  "io", "op", "norm", "act", "attn", "merge", "emb", "loss", "ctrl", "ref",
] as const;

export type NodeKind = (typeof NODE_KINDS)[number];

/** Payload stored on every React Flow node. */
export type NodeData = {
  kind: NodeKind;
  /** Visible label inside the node box. */
  label: string;
  /** Optional user-facing id, emitted into the DSL spec for explicit edges. */
  uid?: string;
  /** When kind === "ref", the registered library block name to inline. */
  refName?: string;
  /** Expected input shape (e.g. "(B, T, D)"). */
  in?: string;
  /** Produced output shape. */
  out?: string;
};

export type EdgeData = {
  label?: string;
  /** "skip" renders as a dashed residual; mismatches are recomputed on render. */
  style?: "solid" | "skip";
};

export type BlockNodeT = Node<NodeData>;
export type BlockEdgeT = Edge<EdgeData>;

export type BlockMeta = {
  name: string;
  category: string;
  desc: string;
  shapesStr: string;
};

/** What the JSON exporter writes / importer reads. */
export type GraphJSON = {
  meta: BlockMeta;
  nodes: BlockNodeT[];
  edges: BlockEdgeT[];
};

/** Shape of a single block as extracted from blocks/*.py into public/library.json. */
export type LibNode =
  | { kind: NodeKind; label: string }
  | { kind: NodeKind; label: string; meta: { id?: string; in?: string; out?: string } };

export type LibSpec = {
  desc: string;
  shapes: string;
  rows: LibNode[][];
  skips: [number, number, number, number][];
  edges: ([string, string] | [string, string, string])[] | null;
};

export type LibraryManifest = {
  categories: Record<string, { desc: string; blocks: Record<string, LibSpec> }>;
};

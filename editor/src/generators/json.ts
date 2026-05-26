import type { BlockEdgeT, BlockMeta, BlockNodeT, GraphJSON } from "../types";

export const toJSON = (
  meta: BlockMeta,
  nodes: BlockNodeT[],
  edges: BlockEdgeT[],
): string => JSON.stringify({ meta, nodes, edges } satisfies GraphJSON, null, 2);

export const fromJSON = (text: string): GraphJSON => {
  const obj = JSON.parse(text);
  if (!obj || typeof obj !== "object" || !obj.meta || !Array.isArray(obj.nodes) || !Array.isArray(obj.edges)) {
    throw new Error("Invalid graph JSON: expected { meta, nodes, edges }.");
  }
  return obj as GraphJSON;
};

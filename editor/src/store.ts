import { create } from "zustand";
import {
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  type Connection,
  type EdgeChange,
  type NodeChange,
} from "@xyflow/react";
import type { BlockEdgeT, BlockMeta, BlockNodeT, GraphJSON, NodeData } from "./types";

let _seq = 1;
export const nextId = (prefix = "n") => `${prefix}${_seq++}`;
/** Reset id seed (after import) so we don't collide with imported ids. */
export const seedFrom = (nodes: BlockNodeT[], edges: BlockEdgeT[]) => {
  const max = (arr: { id: string }[]) =>
    arr
      .map((x) => parseInt(x.id.replace(/\D+/g, ""), 10))
      .filter((n) => Number.isFinite(n))
      .reduce((a, b) => Math.max(a, b), 0);
  _seq = Math.max(max(nodes), max(edges), 0) + 1;
};

type Store = {
  meta: BlockMeta;
  nodes: BlockNodeT[];
  edges: BlockEdgeT[];
  selectedNodeId: string | null;
  selectedEdgeId: string | null;

  setMeta: (patch: Partial<BlockMeta>) => void;
  onNodesChange: (changes: NodeChange<BlockNodeT>[]) => void;
  onEdgesChange: (changes: EdgeChange<BlockEdgeT>[]) => void;
  onConnect: (conn: Connection) => void;

  addNode: (data: NodeData, position?: { x: number; y: number }) => string;
  updateNode: (id: string, patch: Partial<NodeData>) => void;
  updateEdge: (id: string, patch: Partial<BlockEdgeT["data"]>) => void;
  select: (kind: "node" | "edge" | null, id: string | null) => void;

  /** Replace the whole graph (used by JSON import). */
  loadGraph: (g: GraphJSON) => void;
  clearGraph: () => void;
};

const initialMeta: BlockMeta = {
  name: "MyArch",
  category: "myarch",
  desc: "Custom architecture authored in the editor.",
  shapesStr: "(*) → (*)",
};

export const useStore = create<Store>((set, get) => ({
  meta: initialMeta,
  nodes: [],
  edges: [],
  selectedNodeId: null,
  selectedEdgeId: null,

  setMeta: (patch) => set({ meta: { ...get().meta, ...patch } }),

  onNodesChange: (changes) => set({ nodes: applyNodeChanges(changes, get().nodes) }),
  onEdgesChange: (changes) => set({ edges: applyEdgeChanges(changes, get().edges) }),
  onConnect: (conn) =>
    set({
      edges: addEdge(
        { ...conn, type: "shape", data: { style: "solid" } } as BlockEdgeT,
        get().edges,
      ),
    }),

  addNode: (data, position) => {
    const id = nextId();
    const node: BlockNodeT = {
      id,
      type: "block",
      position: position ?? { x: 120 + Math.random() * 200, y: 120 + Math.random() * 200 },
      data,
    };
    set({ nodes: [...get().nodes, node], selectedNodeId: id, selectedEdgeId: null });
    return id;
  },

  updateNode: (id, patch) =>
    set({
      nodes: get().nodes.map((n) =>
        n.id === id ? { ...n, data: { ...n.data, ...patch } } : n,
      ),
    }),

  updateEdge: (id, patch) =>
    set({
      edges: get().edges.map((e) =>
        e.id === id ? { ...e, data: { ...(e.data ?? {}), ...patch } } : e,
      ),
    }),

  select: (kind, id) =>
    set({
      selectedNodeId: kind === "node" ? id : null,
      selectedEdgeId: kind === "edge" ? id : null,
    }),

  loadGraph: (g) => {
    seedFrom(g.nodes, g.edges);
    set({
      meta: g.meta,
      nodes: g.nodes.map((n) => ({ ...n, type: n.type ?? "block" })),
      edges: g.edges.map((e) => ({ ...e, type: e.type ?? "shape" })),
      selectedNodeId: null,
      selectedEdgeId: null,
    });
  },

  clearGraph: () =>
    set({ nodes: [], edges: [], selectedNodeId: null, selectedEdgeId: null }),
}));

/** Read-only helpers consumed by the exporters & shape-checker. */
export const findNode = (id: string) => useStore.getState().nodes.find((n) => n.id === id);

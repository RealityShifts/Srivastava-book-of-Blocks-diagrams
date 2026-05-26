import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Background,
  Controls,
  MiniMap,
  ReactFlow,
  ReactFlowProvider,
  type EdgeTypes,
  type NodeTypes,
  type ReactFlowInstance,
} from "@xyflow/react";
import { useStore, nextId } from "../store";
import type { BlockEdgeT, BlockNodeT, LibraryManifest, LibSpec, NodeData } from "../types";
import BlockNode from "./BlockNode";
import ShapeEdge from "./ShapeEdge";

const nodeTypes: NodeTypes = { block: BlockNode };
const edgeTypes: EdgeTypes = { shape: ShapeEdge };

type Drop =
  | { kind: "primitive"; nodeKind: NodeData["kind"] }
  | { kind: "library"; nodeKind: "ref"; category: string; blockName: string; desc: string; shapes: string };

/** Strip meta off library nodes for easy access. */
type FlatLibNode = { kind: NodeData["kind"]; label: string; id?: string; in?: string; out?: string };
const flattenLibNode = (n: unknown): FlatLibNode => {
  const o = n as { kind: NodeData["kind"]; label: string; meta?: { id?: string; in?: string; out?: string } };
  return { kind: o.kind, label: o.label, ...(o.meta ?? {}) };
};

const EditorCanvas = () => {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const [rf, setRf] = useState<ReactFlowInstance<BlockNodeT, BlockEdgeT> | null>(null);

  const {
    nodes, edges, onNodesChange, onEdgesChange, onConnect,
    addNode, loadGraph, select,
  } = useStore();

  const [library, setLibrary] = useState<LibraryManifest | null>(null);
  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}library.json`)
      .then((r) => (r.ok ? r.json() : null))
      .then(setLibrary)
      .catch(() => setLibrary(null));
  }, []);

  /** Resolve a library block by name across all categories. */
  const findLibBlock = useCallback(
    (blockName: string): { spec: LibSpec; category: string } | null => {
      if (!library) return null;
      for (const [cat, { blocks }] of Object.entries(library.categories)) {
        if (blocks[blockName]) return { spec: blocks[blockName], category: cat };
      }
      return null;
    },
    [library],
  );

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      if (!wrapperRef.current || !rf) return;
      const raw = event.dataTransfer.getData("application/json");
      if (!raw) return;
      const drop: Drop = JSON.parse(raw);
      const position = rf.screenToFlowPosition({ x: event.clientX, y: event.clientY });

      if (drop.kind === "primitive") {
        addNode({ kind: drop.nodeKind, label: drop.nodeKind }, position);
        return;
      }

      // Library drop -> either a ref node (default) or expand inline (shift).
      const shift = event.shiftKey;
      const block = findLibBlock(drop.blockName);

      if (!shift || !block) {
        addNode(
          { kind: "ref", label: drop.blockName, refName: drop.blockName },
          position,
        );
        return;
      }

      expandIntoCanvas(drop.blockName, block.spec, position);
    },
    [addNode, findLibBlock, rf],
  );

  /**
   * Inline a library block onto the canvas as actual nodes + edges, instead
   * of a single _ref placeholder. Uses the spec's own row layout for
   * positioning so the result lands in a tidy column near the drop point.
   */
  const expandIntoCanvas = useCallback(
    (blockName: string, spec: LibSpec, origin: { x: number; y: number }) => {
      const ROW_H = 90;
      const COL_W = 200;
      const newNodes: BlockNodeT[] = [];
      const idGrid: string[][] = [];
      spec.rows.forEach((row, r) => {
        const ids: string[] = [];
        row.forEach((rawNode, c) => {
          const n = flattenLibNode(rawNode);
          const id = nextId();
          ids.push(id);
          const xOffset = (c - (row.length - 1) / 2) * COL_W;
          newNodes.push({
            id,
            type: "block",
            position: { x: origin.x + xOffset, y: origin.y + r * ROW_H },
            data: {
              kind: n.kind,
              label: n.kind === "ref" ? n.label : n.label,
              uid: n.id,
              refName: n.kind === "ref" ? n.label : undefined,
              in: n.in,
              out: n.out,
            },
          });
        });
        idGrid.push(ids);
      });

      const newEdges: BlockEdgeT[] = [];
      const pushEdge = (s: string, t: string, label?: string, style?: "skip" | "solid") => {
        newEdges.push({
          id: nextId("e"),
          source: s,
          target: t,
          type: "shape",
          data: { label, style: style ?? "solid" },
        });
      };

      // Replay the renderer's wiring rule: equal-width rows = column-aligned;
      // otherwise full bipartite.
      if (spec.edges == null) {
        for (let r = 0; r < idGrid.length - 1; r++) {
          const a = idGrid[r];
          const b = idGrid[r + 1];
          if (a.length === b.length && a.length > 1) {
            for (let c = 0; c < a.length; c++) pushEdge(a[c], b[c]);
          } else {
            for (const s of a) for (const t of b) pushEdge(s, t);
          }
        }
      } else {
        // Explicit edges in the spec are keyed by user-supplied ids inside
        // the spec; we map them to the freshly-minted node ids via uid.
        const uidToId = new Map<string, string>();
        spec.rows.forEach((row, r) =>
          row.forEach((rawNode, c) => {
            const n = flattenLibNode(rawNode);
            if (n.id) uidToId.set(n.id, idGrid[r][c]);
          }),
        );
        for (const e of spec.edges) {
          const [s, t, label] = e;
          const sid = uidToId.get(s);
          const tid = uidToId.get(t);
          if (sid && tid) pushEdge(sid, tid, label as string | undefined);
        }
      }

      // Positional skips.
      for (const [r1, c1, r2, c2] of spec.skips) {
        const s = idGrid[r1]?.[c1];
        const t = idGrid[r2]?.[c2];
        if (s && t) pushEdge(s, t, "skip", "skip");
      }

      const all = [...useStore.getState().nodes, ...newNodes];
      const allE = [...useStore.getState().edges, ...newEdges];
      loadGraph({
        meta: { ...useStore.getState().meta, name: useStore.getState().meta.name },
        nodes: all,
        edges: allE,
      });

      // Brief visual cue: log so the user knows what happened.
      console.info(`Expanded ${blockName}: +${newNodes.length} nodes, +${newEdges.length} edges`);
    },
    [loadGraph],
  );

  const onNodeContextMenu = useCallback(
    (event: React.MouseEvent, node: BlockNodeT) => {
      if (node.data.kind !== "ref" || !node.data.refName) return;
      event.preventDefault();
      const block = findLibBlock(node.data.refName);
      if (!block) {
        alert(`No library block named ${node.data.refName!}`);
        return;
      }
      // Replace the ref node with its expansion.
      const state = useStore.getState();
      const remainingNodes = state.nodes.filter((n) => n.id !== node.id);
      const remainingEdges = state.edges.filter(
        (e) => e.source !== node.id && e.target !== node.id,
      );
      loadGraph({ meta: state.meta, nodes: remainingNodes, edges: remainingEdges });
      expandIntoCanvas(node.data.refName, block.spec, node.position);
    },
    [findLibBlock, expandIntoCanvas, loadGraph],
  );

  const defaultEdgeOptions = useMemo(
    () => ({ type: "shape", data: { style: "solid" as const } }),
    [],
  );

  return (
    <div
      ref={wrapperRef}
      className="h-full w-full bg-slate-50"
      onDragOver={(e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = "copy";
      }}
      onDrop={onDrop}
    >
      <ReactFlow<BlockNodeT, BlockEdgeT>
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onInit={setRf}
        onNodeClick={(_, n) => select("node", n.id)}
        onEdgeClick={(_, e) => select("edge", e.id)}
        onPaneClick={() => select(null, null)}
        onNodeContextMenu={onNodeContextMenu}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        defaultEdgeOptions={defaultEdgeOptions}
        fitView
        proOptions={{ hideAttribution: true }}
      >
        <Background gap={20} />
        <Controls />
        <MiniMap pannable zoomable />
      </ReactFlow>
      <div className="pointer-events-none absolute bottom-2 left-72 z-10 text-[11px] text-slate-500">
        Drag from palette · Drag between port dots to connect · Right-click a{" "}
        <span className="font-medium">ref</span> node to inline it · Shift-drag a library item to
        drop it expanded
      </div>
    </div>
  );
};

export default function Editor() {
  return (
    <ReactFlowProvider>
      <EditorCanvas />
    </ReactFlowProvider>
  );
}

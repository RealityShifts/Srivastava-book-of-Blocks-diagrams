/**
 * Parses a Mermaid `.md` (the exact shape `_generate.py` / `toMermaid` emits)
 * back into a `GraphJSON` that drops straight into `loadGraph()`.
 *
 * Mirrors `_reverse.py` line-for-line:
 *   - `nID["label"]:::kind`           -> node
 *   - `subgraph nID["RefName"] ... end` -> a single `_ref` node (the inner
 *     expansion is collapsed; all inner edges are dropped, and any external
 *     edges that pointed at inner ports are remapped to the ref's id)
 *   - `A --> B` / `A -->|"label"| B`   -> solid edge
 *   - `A -.->|"skip"| B`               -> skip (dashed)
 *   - `A ==>|"..."| B`                 -> solid (mismatch decoration is a
 *     render-time artifact and is dropped — the regenerator recomputes it)
 *
 * Per-port shapes (in/out) live only in the source DSL and are NOT recovered.
 * Node positions are recomputed via longest-path layering on a uniform grid.
 */

import { nextId } from "../store";
import type {
  BlockEdgeT,
  BlockMeta,
  BlockNodeT,
  GraphJSON,
  NodeKind,
} from "../types";

const KINDS: ReadonlySet<string> = new Set([
  "io", "op", "norm", "act", "attn", "merge", "emb", "loss", "ctrl", "ref",
]);

const RE_NODE = /^(\S+?)\["(.+)"\]:::(\w+)\s*$/;
const RE_SUB_START = /^subgraph\s+(\S+?)\["(.+)"\]\s*$/;
const RE_SUB_END = /^end\s*$/;
const RE_EDGE = /^(\S+)\s+(-->|-\.->|==>)(?:\|"([^"]*)"\|)?\s+(\S+)\s*$/;
const RE_INIT = /^%%\{.*\}%%\s*$/;
const RE_HDR = /^flowchart\s+\w+\s*$/;

const RE_FENCE = /```mermaid\s*\n([\s\S]*?)```/;
const RE_TITLE = /^#\s+(.+?)\s*$/m;
const RE_DESC = /^>\s+(.+?)\s*$/m;
const RE_SHAPES = /^\*\*Shapes:\*\*\s+`(.+?)`\s*$/m;

type ParsedNode = { kind: NodeKind; label: string };
type ParsedEdge = { src: string; dst: string; style: "solid" | "skip"; label?: string };

function parseBody(body: string): { nodes: Map<string, ParsedNode>; edges: ParsedEdge[] } {
  const nodes = new Map<string, ParsedNode>();
  const edgeKey = new Set<string>(); // dedup across subgraph collapse
  const edges: ParsedEdge[] = [];
  const subStack: string[] = [];
  const remap = new Map<string, string>();

  for (const raw of body.split("\n")) {
    const line = raw.trim();
    if (!line) continue;
    if (RE_INIT.test(line) || RE_HDR.test(line)) continue;
    if (line.startsWith("classDef ") || line.startsWith("linkStyle ") || line.startsWith("%%")) continue;

    const ms = RE_SUB_START.exec(line);
    if (ms) {
      const [, sid, refName] = ms;
      if (subStack.length > 0) {
        remap.set(sid, remap.get(subStack[0]) ?? subStack[0]);
      } else {
        nodes.set(sid, { kind: "ref", label: refName });
      }
      subStack.push(sid);
      continue;
    }

    if (RE_SUB_END.test(line)) {
      subStack.pop();
      continue;
    }

    const mn = RE_NODE.exec(line);
    if (mn) {
      const [, nid, label, kindRaw] = mn;
      if (subStack.length > 0) {
        remap.set(nid, remap.get(subStack[0]) ?? subStack[0]);
      } else {
        const kind = (KINDS.has(kindRaw) ? kindRaw : "op") as NodeKind;
        nodes.set(nid, { kind, label });
      }
      continue;
    }

    const me = RE_EDGE.exec(line);
    if (me) {
      const [, srcRaw, op, label, dstRaw] = me;
      const src = remap.get(srcRaw) ?? srcRaw;
      const dst = remap.get(dstRaw) ?? dstRaw;
      if (src === dst) continue;
      let style: "solid" | "skip";
      let useLabel: string | undefined = label || undefined;
      if (op === "-.->") {
        style = "skip";
      } else if (op === "==>") {
        // mismatch decoration; collapse to solid and drop auto-label
        style = "solid";
        useLabel = undefined;
      } else {
        style = "solid";
      }
      const key = `${src}->${dst}:${style}`;
      if (edgeKey.has(key)) continue;
      edgeKey.add(key);
      edges.push({ src, dst, style, label: useLabel });
    }
  }

  return { nodes, edges };
}

/** Longest-path layering: each node's depth = max(depth(preds)) + 1. */
function layer(nodes: Map<string, ParsedNode>, edges: ParsedEdge[]): string[][] {
  const ids = Array.from(nodes.keys());
  const fwd = new Map<string, string[]>();
  const indeg = new Map<string, number>();
  ids.forEach((i) => indeg.set(i, 0));
  for (const e of edges) {
    if (!nodes.has(e.src) || !nodes.has(e.dst)) continue;
    if (!fwd.has(e.src)) fwd.set(e.src, []);
    fwd.get(e.src)!.push(e.dst);
    indeg.set(e.dst, (indeg.get(e.dst) ?? 0) + 1);
  }

  const level = new Map<string, number>();
  const queue: string[] = [];
  for (const i of ids) {
    if ((indeg.get(i) ?? 0) === 0) {
      level.set(i, 0);
      queue.push(i);
    }
  }

  for (let head = 0; head < queue.length; head++) {
    const u = queue[head];
    const lu = level.get(u) ?? 0;
    for (const v of fwd.get(u) ?? []) {
      level.set(v, Math.max(level.get(v) ?? 0, lu + 1));
      const nd = (indeg.get(v) ?? 0) - 1;
      indeg.set(v, nd);
      if (nd === 0) queue.push(v);
    }
  }

  let maxLvl = 0;
  level.forEach((l) => { if (l > maxLvl) maxLvl = l; });
  const rows: string[][] = Array.from({ length: maxLvl + 1 }, () => []);
  for (const i of ids) {
    if (level.has(i)) rows[level.get(i)!].push(i);
  }
  // Cycle stragglers — never silently drop nodes
  const stragglers = ids.filter((i) => !level.has(i));
  if (stragglers.length) rows.push(stragglers);
  return rows;
}

/** Parse a Mermaid `.md` text into a GraphJSON ready for `loadGraph()`. */
export function fromMermaid(
  text: string,
  origin: { x: number; y: number } = { x: 240, y: 120 },
): GraphJSON {
  const fence = RE_FENCE.exec(text);
  if (!fence) throw new Error("No ```mermaid ... ``` block found in input.");
  const body = fence[1];

  const { nodes: pNodes, edges: pEdges } = parseBody(body);
  if (pNodes.size === 0) throw new Error("No nodes recognised in Mermaid body.");

  const rows = layer(pNodes, pEdges);

  const ROW_H = 90;
  const COL_W = 200;
  const oldToNew = new Map<string, string>();
  const uidOf = new Map<string, string>();

  const newNodes: BlockNodeT[] = [];
  rows.forEach((row, r) => {
    row.forEach((oid, c) => {
      const node = pNodes.get(oid)!;
      const newId = nextId();
      const uid = `r${r}c${c}`;
      oldToNew.set(oid, newId);
      uidOf.set(oid, uid);
      const xOffset = (c - (row.length - 1) / 2) * COL_W;
      newNodes.push({
        id: newId,
        type: "block",
        position: { x: origin.x + xOffset, y: origin.y + r * ROW_H },
        data: {
          kind: node.kind,
          label: node.label,
          uid,
          refName: node.kind === "ref" ? node.label : undefined,
        },
      });
    });
  });

  const newEdges: BlockEdgeT[] = [];
  for (const e of pEdges) {
    const s = oldToNew.get(e.src);
    const t = oldToNew.get(e.dst);
    if (!s || !t) continue;
    newEdges.push({
      id: nextId("e"),
      source: s,
      target: t,
      type: "shape",
      data: { label: e.label, style: e.style },
    });
  }

  const titleM = RE_TITLE.exec(text);
  const descM = RE_DESC.exec(text);
  const shapesM = RE_SHAPES.exec(text);
  const name = titleM?.[1] ?? "ReversedBlock";
  const meta: BlockMeta = {
    name,
    category: "reversed",
    desc: descM?.[1] ?? "Reversed from a Mermaid diagram.",
    shapesStr: shapesM?.[1] ?? "(*) → (*)",
  };

  return { meta, nodes: newNodes, edges: newEdges };
}

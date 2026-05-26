import type { BlockEdgeT, BlockNodeT } from "../types";

/**
 * Assign every node a row index = longest-path depth from any source.
 * Returns { rows, levelOf } where rows[r] is the list of node ids at depth r.
 *
 * Cycles fall back to insertion order: nodes that haven't been levelled by the
 * Kahn pass get appended to a final extra row to keep export from blowing up.
 */
export type Layered = {
  rows: string[][];
  levelOf: Map<string, number>;
};

export const layerByLongestPath = (nodes: BlockNodeT[], edges: BlockEdgeT[]): Layered => {
  const idSet = new Set(nodes.map((n) => n.id));
  const inDeg = new Map<string, number>();
  const fwd = new Map<string, string[]>();
  nodes.forEach((n) => {
    inDeg.set(n.id, 0);
    fwd.set(n.id, []);
  });
  edges.forEach((e) => {
    if (!idSet.has(e.source) || !idSet.has(e.target)) return;
    inDeg.set(e.target, (inDeg.get(e.target) ?? 0) + 1);
    fwd.get(e.source)!.push(e.target);
  });

  const level = new Map<string, number>();
  const queue: string[] = [];
  inDeg.forEach((d, id) => {
    if (d === 0) {
      level.set(id, 0);
      queue.push(id);
    }
  });

  while (queue.length) {
    const u = queue.shift()!;
    const lu = level.get(u) ?? 0;
    for (const v of fwd.get(u) ?? []) {
      level.set(v, Math.max(level.get(v) ?? 0, lu + 1));
      inDeg.set(v, (inDeg.get(v) ?? 0) - 1);
      if (inDeg.get(v) === 0) queue.push(v);
    }
  }

  let maxLvl = 0;
  level.forEach((l) => {
    if (l > maxLvl) maxLvl = l;
  });
  const rows: string[][] = Array.from({ length: maxLvl + 1 }, () => []);
  nodes.forEach((n) => {
    const l = level.get(n.id);
    if (l != null) rows[l].push(n.id);
  });

  // Stragglers (cycles) get a tail row so we never silently drop nodes.
  const stragglers = nodes.filter((n) => !level.has(n.id)).map((n) => n.id);
  if (stragglers.length) rows.push(stragglers);

  // Stable sort each row by the node's x position so left-to-right order
  // roughly matches the user's canvas layout.
  const xOf = new Map(nodes.map((n) => [n.id, n.position.x]));
  rows.forEach((row) => row.sort((a, b) => (xOf.get(a) ?? 0) - (xOf.get(b) ?? 0)));

  return { rows, levelOf: level };
};

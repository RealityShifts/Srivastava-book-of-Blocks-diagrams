/**
 * Compile the editor canvas into a runnable PyTorch ``nn.Module``, binding
 * every ``_ref`` node to a class in
 * `RealityShifts/Srivastava-book-of-Blocks/pytorch_blocks`.
 *
 * 1:1 mirror of `../_codegen.py` — same inline-op table, same kwargs
 * placeholder strategy, same topo-ordered forward emission.
 *
 * Registry is loaded once from `public/pytorch_blocks.json` (snapshotted by
 * `scripts/extract_block_registry.py`).
 */

import type { BlockEdgeT, BlockMeta, BlockNodeT } from "../types";

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

type InitParam = {
  name: string;
  annotation: string | null;
  default: string | null;
  kind: "positional" | "keyword";
};

type ClassEntry = {
  doc: string | null;
  init: InitParam[] | null;
  forward_arity: number | null;
};

type Registry = {
  framework: string;
  modules: Record<string, Record<string, ClassEntry>>;
  index: Record<string, string>; // ClassName -> module stem
};

let _registry: Registry | null = null;
export async function loadRegistry(): Promise<Registry> {
  if (_registry) return _registry;
  const url = `${import.meta.env.BASE_URL}pytorch_blocks.json`;
  const r = await fetch(url);
  if (!r.ok) {
    throw new Error(
      `Could not load pytorch_blocks.json from ${url}. ` +
        `Run editor/scripts/extract_block_registry.py first.`,
    );
  }
  _registry = (await r.json()) as Registry;
  return _registry;
}

// ---------------------------------------------------------------------------
// Inline primitive mapping (must stay in sync with _codegen.py)
// ---------------------------------------------------------------------------

const ACTIVATIONS: Record<string, (a: string) => string> = {
  relu: (a) => `F.relu(${a})`,
  gelu: (a) => `F.gelu(${a})`,
  silu: (a) => `F.silu(${a})`,
  swish: (a) => `F.silu(${a})`,
  sigmoid: (a) => `torch.sigmoid(${a})`,
  tanh: (a) => `torch.tanh(${a})`,
  softplus: (a) => `F.softplus(${a})`,
  softmax: (a) => `F.softmax(${a}, dim=-1)`,
  logsoftmax: (a) => `F.log_softmax(${a}, dim=-1)`,
  mish: (a) => `(${a} * torch.tanh(F.softplus(${a})))`,
  identity: (a) => a,
  leakyrelu: (a) => `F.leaky_relu(${a}, 0.2)`,
  elu: (a) => `F.elu(${a})`,
};

const MERGES: Record<string, (ins: string[]) => string> = {
  "+": (ins) => `(${ins.join(" + ")})`,
  add: (ins) => `(${ins.join(" + ")})`,
  sum: (ins) => `(${ins.join(" + ")})`,
  "*": (ins) => `(${ins.join(" * ")})`,
  mul: (ins) => `(${ins.join(" * ")})`,
  concat: (ins) => `torch.cat([${ins.join(", ")}], dim=-1)`,
  cat: (ins) => `torch.cat([${ins.join(", ")}], dim=-1)`,
  stack: (ins) => `torch.stack([${ins.join(", ")}], dim=1)`,
};

const normLabel = (l: string): string => l.toLowerCase().replace(/[^a-z0-9+*]/g, "");

function inlineOp(label: string, inputs: string[]): string | null {
  const key = normLabel(label);
  if (!inputs.length) return null;
  if (key in ACTIVATIONS && inputs.length === 1) return ACTIVATIONS[key](inputs[0]);
  if (key in MERGES) return MERGES[key](inputs);
  return null;
}

// ---------------------------------------------------------------------------
// Name helpers
// ---------------------------------------------------------------------------

function safeAttr(s: string): string {
  let out = s.replace(/[^A-Za-z0-9]+/g, "_").replace(/^_+|_+$/g, "").toLowerCase();
  if (!out || /^[0-9]/.test(out)) out = "x_" + out;
  return out;
}

function safeClass(s: string): string {
  const parts = s.split(/[^A-Za-z0-9]+/).filter(Boolean);
  let name = parts.map((p) => p[0].toUpperCase() + p.slice(1)).join("") || "Block";
  if (/^[0-9]/.test(name)) name = "B" + name;
  return name;
}

const varOf = (id: string): string => "t_" + id.replace(/[^A-Za-z0-9]+/g, "_");

function placeholderKwargs(init: InitParam[] | null): { call: string; required: string[] } {
  if (!init || init.length === 0) return { call: "", required: [] };
  const parts: string[] = [];
  const required: string[] = [];
  for (const p of init) {
    if (p.default !== null) parts.push(`${p.name}=${p.default}`);
    else {
      parts.push(`${p.name}=...`);
      required.push(p.name);
    }
  }
  return { call: parts.join(", "), required };
}

// ---------------------------------------------------------------------------
// Graph view of the editor state
// ---------------------------------------------------------------------------

type G = {
  ids: string[];                                  // declaration order
  kind: Map<string, string>;
  label: Map<string, string>;
  refName: Map<string, string>;                   // id -> ref block name (kind=ref only)
  preds: Map<string, string[]>;
  succs: Map<string, string[]>;
};

function buildGraph(nodes: BlockNodeT[], edges: BlockEdgeT[]): G {
  const g: G = {
    ids: [],
    kind: new Map(),
    label: new Map(),
    refName: new Map(),
    preds: new Map(),
    succs: new Map(),
  };
  for (const n of nodes) {
    g.ids.push(n.id);
    g.kind.set(n.id, n.data.kind);
    g.label.set(n.id, n.data.label);
    if (n.data.kind === "ref" && n.data.refName) g.refName.set(n.id, n.data.refName);
    g.preds.set(n.id, []);
    g.succs.set(n.id, []);
  }
  for (const e of edges) {
    const s = e.source, t = e.target;
    if (!g.preds.has(t) || !g.succs.has(s)) continue;
    g.preds.get(t)!.push(s);
    g.succs.get(s)!.push(t);
  }
  return g;
}

function topo(g: G): string[] {
  const indeg = new Map<string, number>();
  for (const id of g.ids) indeg.set(id, g.preds.get(id)!.length);
  const queue: string[] = g.ids.filter((id) => indeg.get(id) === 0);
  const out: string[] = [];
  for (let head = 0; head < queue.length; head++) {
    const u = queue[head];
    out.push(u);
    for (const v of g.succs.get(u) ?? []) {
      indeg.set(v, (indeg.get(v) ?? 0) - 1);
      if (indeg.get(v) === 0) queue.push(v);
    }
  }
  for (const id of g.ids) if (!out.includes(id)) out.push(id);
  return out;
}

// ---------------------------------------------------------------------------
// Emit
// ---------------------------------------------------------------------------

function emitBlock(
  className: string,
  meta: BlockMeta,
  g: G,
  registry: Registry,
): string {
  const order = topo(g);
  const sources = g.ids.filter((id) => g.kind.get(id) === "io" && (g.preds.get(id) ?? []).length === 0);
  const sinks = g.ids.filter((id) => g.kind.get(id) === "io" && (g.succs.get(id) ?? []).length === 0);
  const finalSources = sources.length ? sources : g.ids.slice(0, 1);
  const finalSinks = sinks.length ? sinks : g.ids.slice(-1);

  // __init__: one attribute per ref node
  const initLines: string[] = [];
  const attrOf = new Map<string, string>();
  const usedCount = new Map<string, number>();

  for (const id of g.ids) {
    if (g.kind.get(id) !== "ref") continue;
    const refName = g.refName.get(id);
    if (!refName) continue;
    const base = safeAttr(refName);
    const n = (usedCount.get(base) ?? 0) + 1;
    usedCount.set(base, n);
    const attr = n === 1 ? base : `${base}_${n}`;
    attrOf.set(id, attr);

    const modName = registry.index[refName];
    if (!modName) {
      initLines.push(
        `        # TODO: '${refName}' is not in pytorch_blocks; ` +
          `replace with a local nn.Module.`,
      );
      initLines.push(`        self.${attr} = nn.Identity()`);
      continue;
    }
    const entry = registry.modules[modName][refName];
    const { call, required } = placeholderKwargs(entry.init);
    const alias = modName.replace(/_blocks$/, "");
    if (required.length) {
      initLines.push(`        # TODO: set required kwargs ${JSON.stringify(required)}`);
    }
    initLines.push(`        self.${attr} = ${alias}.${refName}(${call})`);
  }
  if (initLines.length === 0) {
    initLines.push("        pass  # no _ref sub-modules in this block");
  }

  // forward(...)
  const argList = finalSources.map(varOf).join(", ") || "x";
  const fwdLines: string[] = [];

  for (const id of order) {
    if (finalSources.includes(id)) continue;
    const v = varOf(id);
    const ins = (g.preds.get(id) ?? []).map(varOf);
    const kind = g.kind.get(id)!;
    const label = g.label.get(id)!;

    if (kind === "ref" && attrOf.has(id)) {
      const attr = attrOf.get(id)!;
      const call = ins.length ? ins.join(", ") : "...";
      const note = ins.length ? "" : `  # TODO: '${label}' has no wired inputs`;
      fwdLines.push(`        ${v} = self.${attr}(${call})${note}`);
      continue;
    }

    if (kind === "io") {
      if (ins.length) fwdLines.push(`        ${v} = ${ins[0]}`);
      else fwdLines.push(`        ${v} = ...  # TODO: unconnected IO '${label}'`);
      continue;
    }

    const expr = inlineOp(label, ins);
    if (expr !== null) {
      fwdLines.push(`        ${v} = ${expr}`);
      continue;
    }

    if (!ins.length) {
      fwdLines.push(`        ${v} = ...  # TODO: implement '${label}' (no inputs wired)`);
    } else {
      const more = ins.length > 1 ? `  # NOTE: other inputs ${JSON.stringify(ins.slice(1))} unused` : "";
      fwdLines.push(`        ${v} = ${ins[0]}  # TODO: implement '${label}' (kind=${kind})${more}`);
    }
  }
  const ret = finalSinks.map(varOf).join(", ") || "x";
  fwdLines.push(`        return ${ret}`);

  const doc = meta.desc
    ? `    """${meta.desc}\n\n    Shapes: \`\`${meta.shapesStr}\`\`\n    """`
    : "";

  return [
    `class ${className}(nn.Module):`,
    doc,
    "    def __init__(self):",
    "        super().__init__()",
    ...initLines,
    "",
    `    def forward(self, ${argList}):`,
    ...fwdLines,
  ]
    .filter((l) => l !== "")
    .join("\n");
}

/** Public entry point — what the toolbar button calls. */
export async function toPyTorch(
  meta: BlockMeta,
  nodes: BlockNodeT[],
  edges: BlockEdgeT[],
): Promise<string> {
  const registry = await loadRegistry();
  const g = buildGraph(nodes, edges);

  // Collect every submodule the refs actually touch.
  const needed = new Set<string>();
  for (const id of g.ids) {
    if (g.kind.get(id) !== "ref") continue;
    const refName = g.refName.get(id);
    if (!refName) continue;
    const mod = registry.index[refName];
    if (mod) needed.add(mod);
  }
  const imports = Array.from(needed)
    .sort()
    .map((m) => `from pytorch_blocks import ${m} as ${m.replace(/_blocks$/, "")}`)
    .join("\n");

  const className = safeClass(meta.name);
  const blockSrc = emitBlock(className, meta, g, registry);

  return (
    `"""Auto-generated by the Block Diagrams editor.\n\n` +
    `Built against RealityShifts/Srivastava-book-of-Blocks (pytorch_blocks).\n\n` +
    `Constructor kwargs marked \`\`...\`\` and forward lines marked\n` +
    `\`\`# TODO: implement '<label>'\`\` need real values before training —\n` +
    `they're the things the editor's free-form labels couldn't translate\n` +
    `automatically.\n"""\n\n` +
    `from __future__ import annotations\n\n` +
    `import torch\n` +
    `import torch.nn.functional as F\n` +
    `import torch.nn as nn\n` +
    (imports ? imports + "\n" : "") +
    `\n\n` +
    blockSrc +
    "\n"
  );
}

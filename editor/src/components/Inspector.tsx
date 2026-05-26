import { useStore } from "../store";
import { NODE_KINDS, type NodeKind } from "../types";

const LabelInput = ({
  label, value, onChange, placeholder, mono,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  mono?: boolean;
}) => (
  <label className="flex flex-col gap-0.5">
    <span className="text-[10px] font-medium uppercase tracking-wide text-slate-500">
      {label}
    </span>
    <input
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      className={`rounded border border-slate-300 px-2 py-1 text-xs outline-none focus:border-blue-500 ${
        mono ? "font-mono" : ""
      }`}
    />
  </label>
);

export default function Inspector() {
  const { meta, setMeta, nodes, edges, selectedNodeId, selectedEdgeId, updateNode, updateEdge } =
    useStore();

  const node = nodes.find((n) => n.id === selectedNodeId) ?? null;
  const edge = edges.find((e) => e.id === selectedEdgeId) ?? null;

  return (
    <aside className="flex h-full w-72 flex-col border-l border-slate-200 bg-white">
      <div className="border-b border-slate-200 px-3 py-2 text-sm font-semibold text-slate-700">
        Inspector
      </div>

      <div className="flex flex-col gap-2 border-b border-slate-200 px-3 py-3">
        <div className="text-[11px] font-medium uppercase tracking-wide text-slate-500">
          Block metadata
        </div>
        <LabelInput label="Name" value={meta.name} onChange={(v) => setMeta({ name: v })} />
        <LabelInput
          label="Category"
          value={meta.category}
          onChange={(v) => setMeta({ category: v })}
        />
        <LabelInput
          label="Shapes"
          value={meta.shapesStr}
          onChange={(v) => setMeta({ shapesStr: v })}
          mono
        />
        <label className="flex flex-col gap-0.5">
          <span className="text-[10px] font-medium uppercase tracking-wide text-slate-500">
            Description
          </span>
          <textarea
            value={meta.desc}
            onChange={(e) => setMeta({ desc: e.target.value })}
            rows={2}
            className="rounded border border-slate-300 px-2 py-1 text-xs outline-none focus:border-blue-500"
          />
        </label>
      </div>

      <div className="flex-1 overflow-y-auto px-3 py-3">
        {!node && !edge && (
          <div className="text-xs text-slate-500">
            Select a node or edge on the canvas to edit its properties.
          </div>
        )}

        {node && (
          <div className="flex flex-col gap-2">
            <div className="text-[11px] font-medium uppercase tracking-wide text-slate-500">
              Node — <span className="font-mono">{node.id}</span>
            </div>
            <label className="flex flex-col gap-0.5">
              <span className="text-[10px] font-medium uppercase tracking-wide text-slate-500">
                Kind
              </span>
              <select
                value={node.data.kind}
                onChange={(e) => updateNode(node.id, { kind: e.target.value as NodeKind })}
                disabled={node.data.kind === "ref"}
                className="rounded border border-slate-300 px-2 py-1 text-xs outline-none focus:border-blue-500"
              >
                {NODE_KINDS.map((k) => (
                  <option key={k} value={k}>
                    {k}
                  </option>
                ))}
              </select>
            </label>
            <LabelInput
              label={node.data.kind === "ref" ? "Ref name" : "Label"}
              value={node.data.kind === "ref" ? (node.data.refName ?? "") : node.data.label}
              onChange={(v) =>
                node.data.kind === "ref"
                  ? updateNode(node.id, { refName: v, label: v })
                  : updateNode(node.id, { label: v })
              }
              mono={node.data.kind === "ref"}
            />
            <LabelInput
              label="ID (optional)"
              value={node.data.uid ?? ""}
              onChange={(v) => updateNode(node.id, { uid: v || undefined })}
              placeholder="e.g. lin1"
              mono
            />
            <LabelInput
              label="Input shape (optional)"
              value={node.data.in ?? ""}
              onChange={(v) => updateNode(node.id, { in: v || undefined })}
              placeholder="e.g. (B, T, D)"
              mono
            />
            <LabelInput
              label="Output shape (optional)"
              value={node.data.out ?? ""}
              onChange={(v) => updateNode(node.id, { out: v || undefined })}
              placeholder="e.g. (B, T, D)"
              mono
            />
          </div>
        )}

        {edge && (
          <div className="flex flex-col gap-2">
            <div className="text-[11px] font-medium uppercase tracking-wide text-slate-500">
              Edge — <span className="font-mono">{edge.id}</span>
            </div>
            <LabelInput
              label="Label"
              value={edge.data?.label ?? ""}
              onChange={(v) => updateEdge(edge.id, { label: v || undefined })}
              placeholder="e.g. residual"
            />
            <label className="flex flex-col gap-0.5">
              <span className="text-[10px] font-medium uppercase tracking-wide text-slate-500">
                Style
              </span>
              <select
                value={edge.data?.style ?? "solid"}
                onChange={(e) =>
                  updateEdge(edge.id, { style: e.target.value as "solid" | "skip" })
                }
                className="rounded border border-slate-300 px-2 py-1 text-xs outline-none focus:border-blue-500"
              >
                <option value="solid">solid</option>
                <option value="skip">skip (dashed)</option>
              </select>
            </label>
            <p className="mt-1 text-[10px] text-slate-500 leading-snug">
              Shape mismatches are computed live from the connected nodes' in / out fields and
              cannot be overridden.
            </p>
          </div>
        )}
      </div>
    </aside>
  );
}

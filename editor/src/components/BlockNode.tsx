import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import type { BlockNodeT } from "../types";

/** Tiny rounded handle that picks up the node's stroke colour. */
const dot: React.CSSProperties = {
  width: 9,
  height: 9,
  background: "var(--stroke)",
  border: "1.5px solid white",
};

const BlockNode = ({ data, selected }: NodeProps<BlockNodeT>) => {
  const isRef = data.kind === "ref";
  return (
    <div
      data-kind={data.kind}
      className="node-shell"
      style={{ outline: selected ? "2px solid #2563eb" : "none", outlineOffset: 2 }}
    >
      <Handle type="target" position={Position.Top} style={dot} />
      <div className="flex flex-col gap-0.5">
        {data.uid && (
          <span className="text-[10px] font-mono uppercase tracking-wide opacity-60">
            #{data.uid}
          </span>
        )}
        <span className="leading-tight">
          {isRef && data.refName ? `↳ ${data.refName}` : data.label}
        </span>
        {(data.in || data.out) && (
          <div className="flex gap-1 flex-wrap">
            {data.in && <span className="node-shape">in: {data.in}</span>}
            {data.out && <span className="node-shape">out: {data.out}</span>}
          </div>
        )}
      </div>
      <Handle type="source" position={Position.Bottom} style={dot} />
    </div>
  );
};

export default memo(BlockNode);

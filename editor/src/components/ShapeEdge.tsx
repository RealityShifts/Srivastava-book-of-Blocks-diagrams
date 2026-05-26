import { memo } from "react";
import {
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  type EdgeProps,
} from "@xyflow/react";
import type { BlockEdgeT } from "../types";
import { useStore } from "../store";
import { checkShapes } from "../library/shapeCheck";

const ShapeEdge = ({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  source,
  target,
  data,
  selected,
}: EdgeProps<BlockEdgeT>) => {
  // Subscribe to nodes so the edge re-renders when shapes change.
  const nodes = useStore((s) => s.nodes);
  const src = nodes.find((n) => n.id === source);
  const tgt = nodes.find((n) => n.id === target);

  const isSkip = data?.style === "skip";
  const { mismatch, label } = isSkip
    ? { mismatch: false, label: undefined as string | undefined }
    : checkShapes(src, tgt);

  const [path, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  });

  const stroke = mismatch ? "var(--mismatch)" : selected ? "#2563eb" : "#475569";
  const strokeWidth = mismatch ? 2.2 : 1.4;
  const dasharray = isSkip ? "6 4" : undefined;

  const display = mismatch ? label : data?.label || (isSkip ? "skip" : undefined);

  return (
    <>
      <BaseEdge
        id={id}
        path={path}
        style={{ stroke, strokeWidth, strokeDasharray: dasharray }}
        className={mismatch ? "shape-mismatch" : undefined}
      />
      {display && (
        <EdgeLabelRenderer>
          <div
            style={{
              position: "absolute",
              transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
              pointerEvents: "all",
              fontSize: 11,
              fontWeight: mismatch ? 600 : 500,
              color: mismatch ? "var(--mismatch)" : "#334155",
              background: "white",
              padding: "1px 5px",
              borderRadius: 4,
              border: `1px solid ${mismatch ? "var(--mismatch)" : "#cbd5e1"}`,
            }}
            className="nodrag nopan"
          >
            {display}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
};

export default memo(ShapeEdge);

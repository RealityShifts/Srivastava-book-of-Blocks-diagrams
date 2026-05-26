import type { BlockNodeT } from "../types";

/** Strip all whitespace; used so "(B, T, D)" == "(B,T,D)". */
export const normShape = (s?: string): string | undefined =>
  s == null ? undefined : s.replace(/\s+/g, "");

export type Mismatch = { mismatch: boolean; label?: string };

export const checkShapes = (src?: BlockNodeT, tgt?: BlockNodeT): Mismatch => {
  if (!src || !tgt) return { mismatch: false };
  const a = normShape(src.data.out);
  const b = normShape(tgt.data.in);
  if (a == null || b == null) return { mismatch: false };
  if (a === b) return { mismatch: false };
  return { mismatch: true, label: `${src.data.out} ≠ ${tgt.data.in}` };
};

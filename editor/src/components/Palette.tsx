import { useEffect, useMemo, useState, type DragEvent } from "react";
import { NODE_KINDS, type LibraryManifest, type NodeKind } from "../types";

type PaletteItem =
  | { kind: "primitive"; nodeKind: NodeKind }
  | { kind: "library"; nodeKind: "ref"; category: string; blockName: string; desc: string; shapes: string };

const KIND_LABEL: Record<NodeKind, string> = {
  io: "IO", op: "Op", norm: "Norm", act: "Act", attn: "Attn",
  merge: "Merge", emb: "Emb", loss: "Loss", ctrl: "Ctrl", ref: "Ref",
};

/**
 * Left sidebar: drag primitives or any of the 122 library blocks onto canvas.
 * Drag payload is JSON encoded so the canvas drop handler can hydrate it.
 */
export default function Palette() {
  const [library, setLibrary] = useState<LibraryManifest | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}library.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setLibrary)
      .catch((e) => setError(String(e)));
  }, []);

  const libraryItems = useMemo<PaletteItem[]>(() => {
    if (!library) return [];
    const out: PaletteItem[] = [];
    for (const [cat, { blocks }] of Object.entries(library.categories)) {
      for (const [blockName, spec] of Object.entries(blocks)) {
        out.push({
          kind: "library",
          nodeKind: "ref",
          category: cat,
          blockName,
          desc: spec.desc,
          shapes: spec.shapes,
        });
      }
    }
    return out;
  }, [library]);

  const filteredLib = useMemo(() => {
    if (!query.trim()) return libraryItems;
    const q = query.toLowerCase();
    return libraryItems.filter(
      (it) =>
        it.kind === "library" &&
        (it.blockName.toLowerCase().includes(q) ||
          it.category.toLowerCase().includes(q) ||
          it.desc.toLowerCase().includes(q)),
    );
  }, [libraryItems, query]);

  const onDragStart = (e: DragEvent<HTMLDivElement>, item: PaletteItem) => {
    e.dataTransfer.setData("application/json", JSON.stringify(item));
    e.dataTransfer.effectAllowed = "copy";
  };

  return (
    <aside className="flex h-full w-64 flex-col border-r border-slate-200 bg-white">
      <div className="border-b border-slate-200 px-3 py-2 text-sm font-semibold text-slate-700">
        Palette
      </div>

      <div className="px-3 py-2">
        <div className="mb-1 text-[11px] font-medium uppercase tracking-wide text-slate-500">
          Primitives
        </div>
        <div className="grid grid-cols-3 gap-1.5">
          {NODE_KINDS.filter((k) => k !== "ref").map((k) => (
            <div
              key={k}
              draggable
              onDragStart={(e) => onDragStart(e, { kind: "primitive", nodeKind: k })}
              data-kind={k}
              className="palette-pill text-center"
              title={`Drag a ${KIND_LABEL[k]} node onto the canvas`}
            >
              {KIND_LABEL[k]}
            </div>
          ))}
        </div>
      </div>

      <div className="border-t border-slate-200 px-3 py-2">
        <div className="mb-1 flex items-center justify-between">
          <span className="text-[11px] font-medium uppercase tracking-wide text-slate-500">
            Library
          </span>
          <span className="text-[10px] text-slate-400">
            {library ? `${libraryItems.length} blocks` : error ? "error" : "loading…"}
          </span>
        </div>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search blocks…"
          className="w-full rounded border border-slate-300 px-2 py-1 text-xs outline-none focus:border-blue-500"
        />
      </div>

      <div className="flex-1 overflow-y-auto px-2 py-1">
        {error && (
          <div className="px-2 py-1 text-[11px] text-red-600">
            Could not load library.json — run <code>npm run extract-library</code>.
          </div>
        )}
        {filteredLib.map((it) => {
          if (it.kind !== "library") return null;
          return (
            <div
              key={`${it.category}/${it.blockName}`}
              draggable
              onDragStart={(e) => onDragStart(e, it)}
              data-kind="ref"
              className="palette-pill mb-1 block w-full text-left"
              title={`${it.desc}\n${it.shapes}\n\nDrag → adds as _ref; Shift while dragging → inlines the sub-graph.`}
            >
              <div className="text-[11px] font-semibold leading-tight">{it.blockName}</div>
              <div className="text-[9px] opacity-70">{it.category}</div>
            </div>
          );
        })}
      </div>
    </aside>
  );
}

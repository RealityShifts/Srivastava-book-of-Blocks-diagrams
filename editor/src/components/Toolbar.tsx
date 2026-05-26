import { useRef } from "react";
import { useStore } from "../store";
import { toMermaid } from "../generators/mermaid";
import { toDSL } from "../generators/dsl";
import { fromJSON, toJSON } from "../generators/json";
import { fromMermaid } from "../generators/reverseMermaid";
import { toPyTorch } from "../generators/pytorchCodegen";

const download = (filename: string, contents: string, mime = "text/plain") => {
  const blob = new Blob([contents], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
};

const Btn = ({
  children, onClick, title, kind = "primary",
}: {
  children: React.ReactNode;
  onClick: () => void;
  title?: string;
  kind?: "primary" | "ghost" | "danger";
}) => {
  const base = "rounded px-2.5 py-1 text-xs font-medium transition";
  const styles = {
    primary: "bg-blue-600 text-white hover:bg-blue-700",
    ghost: "bg-white text-slate-700 hover:bg-slate-100 border border-slate-300",
    danger: "bg-white text-red-600 hover:bg-red-50 border border-red-300",
  } as const;
  return (
    <button onClick={onClick} title={title} className={`${base} ${styles[kind]}`}>
      {children}
    </button>
  );
};

export default function Toolbar() {
  const { meta, nodes, edges, loadGraph, clearGraph } = useStore();
  const fileRef = useRef<HTMLInputElement>(null);
  const mdRef = useRef<HTMLInputElement>(null);

  const stem = meta.name.replace(/[^A-Za-z0-9_-]/g, "_") || "Block";

  const onImportJSON = (file: File) => {
    file.text().then((txt) => {
      try {
        loadGraph(fromJSON(txt));
      } catch (e) {
        alert(`Import failed: ${e}`);
      }
    });
  };

  const onImportMermaid = (file: File) => {
    file.text().then((txt) => {
      try {
        loadGraph(fromMermaid(txt));
      } catch (e) {
        alert(`Mermaid import failed: ${e}`);
      }
    });
  };

  return (
    <header className="flex items-center justify-between border-b border-slate-200 bg-white px-3 py-2">
      <div className="flex items-baseline gap-3">
        <span className="text-sm font-semibold text-slate-800">Block Diagrams Editor</span>
        <span className="text-[11px] text-slate-500">
          {nodes.length} nodes · {edges.length} edges
        </span>
      </div>
      <div className="flex gap-1.5">
        <Btn
          kind="ghost"
          onClick={() => fileRef.current?.click()}
          title="Load a previously exported Graph JSON"
        >
          Import JSON
        </Btn>
        <input
          ref={fileRef}
          type="file"
          accept="application/json,.json"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) onImportJSON(f);
            e.target.value = "";
          }}
        />
        <Btn
          kind="ghost"
          onClick={() => mdRef.current?.click()}
          title="Reverse a Mermaid .md (any from ../diagrams) back onto the canvas. Subgraphs collapse to _ref nodes."
        >
          Import Mermaid
        </Btn>
        <input
          ref={mdRef}
          type="file"
          accept=".md,text/markdown,text/plain"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) onImportMermaid(f);
            e.target.value = "";
          }}
        />
        <Btn onClick={() => download(`${stem}.json`, toJSON(meta, nodes, edges), "application/json")}>
          Export JSON
        </Btn>
        <Btn
          onClick={() => download(`${stem}.md`, toMermaid(meta, nodes, edges), "text/markdown")}
          title="Mermaid markdown — pastes into GitHub / draw.io / Excalidraw"
        >
          Export Mermaid
        </Btn>
        <Btn
          onClick={() =>
            download(
              `${stem.toLowerCase()}_spec.py`,
              toDSL(meta, nodes, edges),
              "text/x-python",
            )
          }
          title="DSL .py — feed back into ../_generate.py --specs"
        >
          Export DSL .py
        </Btn>
        <Btn
          onClick={() => {
            toPyTorch(meta, nodes, edges)
              .then((src) =>
                download(`${stem.toLowerCase()}_model.py`, src, "text/x-python"),
              )
              .catch((e) => alert(`PyTorch codegen failed: ${e}`));
          }}
          title="Runnable nn.Module backed by RealityShifts/Srivastava-book-of-Blocks"
        >
          Export PyTorch
        </Btn>
        <Btn
          kind="danger"
          onClick={() => {
            if (confirm("Clear the entire canvas?")) clearGraph();
          }}
        >
          Clear
        </Btn>
      </div>
    </header>
  );
}

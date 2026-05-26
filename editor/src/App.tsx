import Editor from "./components/Editor";
import Inspector from "./components/Inspector";
import Palette from "./components/Palette";
import Toolbar from "./components/Toolbar";

export default function App() {
  return (
    <div className="flex h-screen flex-col">
      <Toolbar />
      <div className="flex flex-1 overflow-hidden">
        <Palette />
        <main className="relative flex-1">
          <Editor />
        </main>
        <Inspector />
      </div>
    </div>
  );
}

import { lazy, Suspense, useEffect } from "react";
import { NavLink, Route, Routes, useLocation } from "react-router-dom";
import { Bot, Boxes, BrainCircuit, Database, FlaskConical, Gauge, GraduationCap, Library, ListChecks, MessagesSquare, Plug, Settings, Workflow } from "lucide-react";
import { ModeToggle, Spinner, TaskCenter, useData, VramPanel } from "./components/ui";
import { ErrorBoundary } from "./components/ErrorBoundary";
const Dashboard = lazy(() => import("./pages/Dashboard"));
const Datasets = lazy(() => import("./pages/Datasets"));
const Training = lazy(() => import("./pages/Training"));
const JobDetail = lazy(() => import("./pages/JobDetail"));
const CompareRuns = lazy(() => import("./pages/CompareRuns"));
const Models = lazy(() => import("./pages/Models"));
const Playground = lazy(() => import("./pages/Playground"));
const Rag = lazy(() => import("./pages/Rag"));
const Evaluation = lazy(() => import("./pages/Evaluation"));
const System = lazy(() => import("./pages/System"));
const Agents = lazy(() => import("./pages/Agents"));
const Flows = lazy(() => import("./pages/Flows"));
const Jobs = lazy(() => import("./pages/Jobs"));
const Connections = lazy(() => import("./pages/Connections"));

const NAV = [
  { to: "/", label: "Dashboard", icon: Gauge },
  { to: "/datasets", label: "Datasets", icon: Database },
  { to: "/training", label: "Training", icon: GraduationCap },
  { to: "/models", label: "Models", icon: Boxes },
  { to: "/jobs", label: "Jobs", icon: ListChecks },
  { to: "/playground", label: "Playground", icon: MessagesSquare },
  { to: "/rag", label: "RAG", icon: Library },
  { to: "/agents", label: "Agents", icon: Bot },
  { to: "/flows", label: "Flows", icon: Workflow },
  { to: "/eval", label: "Evaluation", icon: FlaskConical },
  { to: "/connections", label: "Connections", icon: Plug },
  { to: "/system", label: "System", icon: Settings },
];

export default function App() {
  const [sys] = useData<any>("/api/system", 3000);
  const gpu = sys?.gpus?.[0];
  const loc = useLocation();
  useEffect(() => {
    const item = [...NAV].reverse().find((n) => (n.to === "/" ? loc.pathname === "/" : loc.pathname.startsWith(n.to)));
    document.title = item && item.to !== "/" ? `${item.label} · LLM Trainer` : "LLM Trainer";
  }, [loc.pathname]);
  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand"><BrainCircuit size={20} /> <span className="nav-label">LLM Trainer</span></div>
        {NAV.map(({ to, label, icon: Icon }) => (
          <NavLink key={to} to={to} end={to === "/"} title={label} className={({ isActive }) => `nav ${isActive ? "active" : ""}`}>
            <Icon size={16} /> <span className="nav-label">{label}</span>
          </NavLink>
        ))}
        <div style={{ marginTop: "auto" }}><ModeToggle /><TaskCenter /></div>
        <div className="sidebar-foot" style={{ marginTop: 8 }}>
          <VramPanel gpu={gpu} />
          <div className="truncate" title={sys?.inference?.loaded?.name}>
            Model: {sys?.inference?.loading ? `loading ${sys.inference.loading}…` : sys?.inference?.loaded?.name ?? "none"}
          </div>
          {sys?.gpu_job_running && <span className="badge accent">Training…</span>}
        </div>
      </aside>
      <main className="main">
        <ErrorBoundary where={loc.pathname}>
          <Suspense fallback={<div className="card empty"><Spinner /> Loading…</div>}>
            <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="/training" element={<Training />} />
          <Route path="/training/compare" element={<CompareRuns />} />
          <Route path="/training/:id" element={<JobDetail />} />
          <Route path="/models" element={<Models />} />
          <Route path="/playground" element={<Playground />} />
          <Route path="/rag" element={<Rag />} />
          <Route path="/eval" element={<Evaluation />} />
          <Route path="/agents" element={<Agents />} />
          <Route path="/jobs" element={<Jobs />} />
              <Route path="/connections" element={<Connections />} />
          <Route path="/flows" element={<Flows />} />
          <Route path="/system" element={<System />} />
              <Route path="*" element={<div className="empty">Page not found</div>} />
            </Routes>
          </Suspense>
        </ErrorBoundary>
      </main>
    </div>
  );
}

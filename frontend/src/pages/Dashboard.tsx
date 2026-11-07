import { Link } from "react-router-dom";
import { type Collection, type Dataset, duration, type EvalRun, type Job, type Model } from "../api";
import { CheckCircle2, Circle } from "lucide-react";
import { Progress, StatusBadge, useData } from "../components/ui";

export default function Dashboard() {
  const [sys] = useData<any>("/api/system", 2000);
  const [jobs] = useData<Job[]>("/api/jobs", 5000);
  const [datasets] = useData<Dataset[]>("/api/datasets");
  const [models] = useData<Model[]>("/api/models");
  const [cols] = useData<Collection[]>("/api/rag/collections");
  const [evals] = useData<EvalRun[]>("/api/evals");
  const [convs] = useData<any[]>("/api/conversations");
  const gpu = sys?.gpus?.[0];
  const FLOW = [
    { to: "/datasets", title: "Datasets", desc: "Collect the examples the model will learn from: upload files, import from Hugging Face, write them, or generate them from your documents." },
    { to: "/training", title: "Training", desc: "Fine-tune a base model on those examples (LoRA/QLoRA). Also sweeps, embeddings and benchmarks." },
    { to: "/models", title: "Models", desc: "Everything you trained or downloaded. Load it into memory, merge it or export it to GGUF." },
    { to: "/rag", title: "RAG", desc: "Upload your documents so the model answers from them and cites its sources, without retraining." },
    { to: "/playground", title: "Playground and agents", desc: "Chat, compare models side by side and build agents that use tools." },
    { to: "/eval", title: "Evaluation", desc: "Measure which combination answers best: base or fine-tuned, with or without RAG." },
  ];
  const steps = [
    { done: !!datasets?.length, label: datasets?.length ? `${datasets.length} dataset(s)` : "No datasets yet" },
    { done: !!jobs?.some((j) => j.type === "train" && j.status === "completed"), label: `${jobs?.filter((j) => j.status === "completed").length ?? 0} completed job(s)` },
    { done: !!models?.length, label: `${models?.length ?? 0} registered model(s)` },
    { done: !!cols?.some((c) => c.num_chunks > 0), label: cols?.length ? `${cols.length} collection(s)` : "No collections" },
    { done: !!convs?.length, label: convs?.length ? `${convs.length} conversation(s)` : "No conversations yet" },
    { done: !!evals?.some((e) => e.status === "completed"), label: evals?.length ? `${evals.length} evaluation(s)` : "No evaluations" },
  ];
  const pending = steps.filter((s) => !s.done).length;

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Panel</h1>
          <div className="sub">System status and recent activity</div>
        </div>
      </div>

      <div className="grid g4">
        <div className="card stat">
          <div className="label">GPU</div>
          <div className="value" style={{ fontSize: 15 }}>{gpu?.name ?? "Not detected"}</div>
          {gpu && <div className="sub">{gpu.util}% load · {gpu.temp}°C · {gpu.power_w} W</div>}
        </div>
        <div className="card stat">
          <div className="label">VRAM</div>
          <div className="value">{gpu ? `${gpu.mem_used_gb.toFixed(1)} GB` : "—"}</div>
          {gpu && <div className="bar"><div style={{ width: `${(gpu.mem_used_gb / gpu.mem_total_gb) * 100}%` }} /></div>}
        </div>
        <div className="card stat">
          <div className="label">RAM</div>
          <div className="value">{sys ? `${sys.ram_used_gb} / ${sys.ram_total_gb} GB` : "—"}</div>
          <div className="sub">CPU {sys?.cpu ?? 0}%</div>
        </div>
        <div className="card stat">
          <div className="label">Free disk</div>
          <div className="value">{sys ? `${sys.disk_free_gb} GB` : "—"}</div>
          <div className="sub truncate" title={sys?.data_dir}>{sys?.data_dir}</div>
        </div>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <div className="row between"><h2 style={{ margin: 0 }}>How the suite fits together</h2>
          <span className="sub">{steps.length - pending}/{steps.length} steps done</span></div>
        <div style={{ margin: "10px 0 14px" }}><Progress value={(steps.length - pending) / steps.length} /></div>
        <div className="grid g3">
          {FLOW.map((f, i) => {
            const st = steps[i];
            return (
              <Link key={f.to} to={f.to} className="source" style={{ display: "block", borderColor: st.done ? "color-mix(in srgb, var(--ok) 40%, transparent)" : undefined }}>
                <div className="row" style={{ gap: 6 }}>
                  {st.done ? <CheckCircle2 size={15} color="var(--ok)" /> : <Circle size={15} color="var(--muted)" />}
                  <b>{i + 1}. {f.title}</b>
                </div>
                <div className="sub" style={{ fontSize: 12.5, marginTop: 4 }}>{f.desc}</div>
                <div className="sub" style={{ fontSize: 12, marginTop: 4, color: st.done ? "var(--ok)" : undefined }}>{st.label}</div>
              </Link>
            );
          })}
        </div>
      </div>

      <div className="card">
        <div className="row between"><h2 style={{ margin: 0 }}>Engines and tools</h2><Link to="/system">Install / configure →</Link></div>
        <div className="row" style={{ gap: 8, marginTop: 10 }}>
          {[["unsloth", "Unsloth (fast training)"], ["vllm", "vLLM (fast generation)"], ["llama_cpp", "llama.cpp (GGUF)"],
            ["tesseract", "OCR de PDFs"], ["hf_token", "Token de Hugging Face"]].map(([k, label]) => {
            const ok = k === "vllm" ? sys?.vllm : sys?.tools?.[k];
            return <span key={k} className={`badge ${ok ? "ok" : ""}`}>{ok ? "✓" : "○"} {label}</span>;
          })}
        </div>
        <div className="sub" style={{ fontSize: 12, marginTop: 8 }}>
          Loaded model: <b>{sys?.inference?.loaded ? `${sys.inference.loaded.name} (${sys.inference.loaded.backend})` : "none"}</b>
          {sys?.gpu_job_running && " · a GPU job is running"}
        </div>
      </div>

      <div className="grid g3" style={{ marginTop: 16 }}>
        <Link to="/datasets" className="card stat"><div className="label">Datasets</div><div className="value">{datasets?.length ?? "—"}</div></Link>
        <Link to="/models" className="card stat"><div className="label">Models</div><div className="value">{models?.length ?? "—"}</div></Link>
        <Link to="/rag" className="card stat"><div className="label">RAG collections</div><div className="value">{cols?.length ?? "—"}</div></Link>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <div className="row between"><h2>Recent jobs</h2><Link to="/training">New training run →</Link></div>
        {!jobs?.length ? (
          <div className="empty">No jobs yet. Upload a dataset and launch your first fine-tune.</div>
        ) : (
          <table>
            <thead><tr><th>#</th><th>Name</th><th>Type</th><th>Status</th><th>Duration</th></tr></thead>
            <tbody>
              {jobs.slice(0, 8).map((j) => (
                <tr key={j.id}>
                  <td>{j.id}</td>
                  <td><Link to={`/training/${j.id}`}>{j.name}</Link></td>
                  <td>{j.type === "train" ? `${j.config.method?.toUpperCase()} · ${j.config.use_lora ? (j.config.quantization === "4bit" ? "QLoRA" : "LoRA") : "Full"}${j.config.backend === "unsloth" ? " · 🦥 Unsloth" : ""}` : "Export"}</td>
                  <td><StatusBadge status={j.status} />{j.progress?.max_steps ? <div style={{ width: 90, marginTop: 4 }}><Progress value={j.progress.step / j.progress.max_steps} /></div> : null}</td>
                  <td>{duration(j.started_at, j.finished_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

    </>
  );
}

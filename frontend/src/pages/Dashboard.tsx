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
    { to: "/datasets", title: "Datasets", desc: "Reúne los ejemplos de los que aprenderá el modelo: sube archivos, importa de Hugging Face, escríbelos o genéralos desde tus documentos." },
    { to: "/training", title: "Entrenamiento", desc: "Ajusta un modelo base con esos ejemplos (LoRA/QLoRA). También barridos, embeddings y benchmarks." },
    { to: "/models", title: "Modelos", desc: "Todo lo entrenado y descargado. Cárgalo en memoria, fusiónalo o expórtalo a GGUF." },
    { to: "/rag", title: "RAG", desc: "Sube tus documentos para que el modelo responda con ellos y cite las fuentes, sin reentrenar." },
    { to: "/playground", title: "Playground y agentes", desc: "Chatea, compara modelos A/B y crea agentes que usen herramientas." },
    { to: "/eval", title: "Evaluación", desc: "Mide qué combinación responde mejor: base o finetuneado, con o sin RAG." },
  ];
  const steps = [
    { done: !!datasets?.length, label: datasets?.length ? `${datasets.length} dataset(s)` : "Aún no hay datasets" },
    { done: !!jobs?.some((j) => j.type === "train" && j.status === "completed"), label: `${jobs?.filter((j) => j.status === "completed").length ?? 0} trabajo(s) completado(s)` },
    { done: !!models?.length, label: `${models?.length ?? 0} modelo(s) registrados` },
    { done: !!cols?.some((c) => c.num_chunks > 0), label: cols?.length ? `${cols.length} colección(es)` : "Sin colecciones" },
    { done: !!convs?.length, label: convs?.length ? `${convs.length} conversación(es)` : "Sin conversaciones todavía" },
    { done: !!evals?.some((e) => e.status === "completed"), label: evals?.length ? `${evals.length} evaluación(es)` : "Sin evaluaciones" },
  ];
  const pending = steps.filter((s) => !s.done).length;

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Panel</h1>
          <div className="sub">Estado del sistema y actividad reciente</div>
        </div>
      </div>

      <div className="grid g4">
        <div className="card stat">
          <div className="label">GPU</div>
          <div className="value" style={{ fontSize: 15 }}>{gpu?.name ?? "No detectada"}</div>
          {gpu && <div className="sub">{gpu.util}% uso · {gpu.temp}°C · {gpu.power_w} W</div>}
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
          <div className="label">Disco libre</div>
          <div className="value">{sys ? `${sys.disk_free_gb} GB` : "—"}</div>
          <div className="sub truncate" title={sys?.data_dir}>{sys?.data_dir}</div>
        </div>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <div className="row between"><h2 style={{ margin: 0 }}>Cómo se usa la suite</h2>
          <span className="sub">{steps.length - pending}/{steps.length} pasos hechos</span></div>
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
        <div className="row between"><h2 style={{ margin: 0 }}>Motores y herramientas</h2><Link to="/system">Instalar / configurar →</Link></div>
        <div className="row" style={{ gap: 8, marginTop: 10 }}>
          {[["unsloth", "Unsloth (entrenar rápido)"], ["vllm", "vLLM (generar rápido)"], ["llama_cpp", "llama.cpp (GGUF)"],
            ["tesseract", "OCR de PDFs"], ["hf_token", "Token de Hugging Face"]].map(([k, label]) => {
            const ok = k === "vllm" ? sys?.vllm : sys?.tools?.[k];
            return <span key={k} className={`badge ${ok ? "ok" : ""}`}>{ok ? "✓" : "○"} {label}</span>;
          })}
        </div>
        <div className="sub" style={{ fontSize: 12, marginTop: 8 }}>
          Modelo cargado: <b>{sys?.inference?.loaded ? `${sys.inference.loaded.name} (${sys.inference.loaded.backend})` : "ninguno"}</b>
          {sys?.gpu_job_running && " · hay un trabajo de GPU en marcha"}
        </div>
      </div>

      <div className="grid g3" style={{ marginTop: 16 }}>
        <Link to="/datasets" className="card stat"><div className="label">Datasets</div><div className="value">{datasets?.length ?? "—"}</div></Link>
        <Link to="/models" className="card stat"><div className="label">Modelos</div><div className="value">{models?.length ?? "—"}</div></Link>
        <Link to="/rag" className="card stat"><div className="label">Colecciones RAG</div><div className="value">{cols?.length ?? "—"}</div></Link>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <div className="row between"><h2>Trabajos recientes</h2><Link to="/training">Nuevo entrenamiento →</Link></div>
        {!jobs?.length ? (
          <div className="empty">Aún no hay trabajos. Sube un dataset y lanza tu primer fine-tuning.</div>
        ) : (
          <table>
            <thead><tr><th>#</th><th>Nombre</th><th>Tipo</th><th>Estado</th><th>Duración</th></tr></thead>
            <tbody>
              {jobs.slice(0, 8).map((j) => (
                <tr key={j.id}>
                  <td>{j.id}</td>
                  <td><Link to={`/training/${j.id}`}>{j.name}</Link></td>
                  <td>{j.type === "train" ? `${j.config.method?.toUpperCase()} · ${j.config.use_lora ? (j.config.quantization === "4bit" ? "QLoRA" : "LoRA") : "Full"}${j.config.backend === "unsloth" ? " · 🦥 Unsloth" : ""}` : "Exportación"}</td>
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

import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ArrowLeft, Square, Trash2 } from "lucide-react";
import { api, duration, fmtDate, type Job } from "../api";
import { Markdown } from "../components/Markdownish";
import { Progress, StatusBadge, useData, useDialog, useToast } from "../components/ui";

type Ev = { type: string; t: number; [k: string]: any };

export default function JobDetail() {
  const { id } = useParams();
  const [job, reload, loadErr] = useData<Job>(`/api/jobs/${id}`);
  const [events, setEvents] = useState<Ev[]>([]);
  const [log, setLog] = useState("");
  const [status, setStatus] = useState<string>();
  const logRef = useRef<HTMLDivElement>(null);
  const { notify } = useToast();
  const { confirm } = useDialog();

  useEffect(() => {
    setEvents([]); setLog("");
    const es = new EventSource(`/api/jobs/${id}/stream`);
    es.addEventListener("events", (e) => setEvents((p) => [...p, ...JSON.parse((e as MessageEvent).data)]));
    es.addEventListener("log", (e) => setLog((p) => (p + JSON.parse((e as MessageEvent).data)).slice(-60000)));
    es.addEventListener("status", (e) => {
      const s = JSON.parse((e as MessageEvent).data);
      setStatus((prev) => { if (prev && prev !== s) reload(); return s; });
      if (["completed", "failed", "cancelled", "missing"].includes(s)) { es.close(); reload(); }
    });
    return () => es.close();
  }, [id, reload]);

  // Autoscroll: al cargar el log y mientras el usuario esté al final
  const stuck = useRef(true);
  useEffect(() => {
    const el = logRef.current;
    if (el && stuck.current) el.scrollTop = el.scrollHeight;
  }, [log, !!job]); // el log puede llegar antes que el trabajo (y el div aún no existe)

  const logs = useMemo(() => events.filter((e) => e.type === "log"), [events]);
  const trainPts = logs.filter((e) => e.loss !== undefined).map((e) => ({ step: e.step, loss: e.loss, lr: e.learning_rate, grad_norm: e.grad_norm, acc: e.mean_token_accuracy }));
  const evalPts = logs.filter((e) => e.eval_loss !== undefined).map((e) => ({ step: e.step, eval_loss: e.eval_loss }));
  const lossData = useMemo(() => {
    const m = new Map<number, any>();
    for (const p of trainPts) m.set(p.step, { ...m.get(p.step), ...p });
    for (const p of evalPts) m.set(p.step, { ...m.get(p.step), ...p });
    return [...m.values()].sort((a, b) => a.step - b.step);
  }, [trainPts.length, evalPts.length]); // eslint-disable-line
  const rewardKeys = useMemo(() => {
    const ks = new Set<string>();
    logs.forEach((e) => Object.keys(e).forEach((k) => (k === "reward" || (k.startsWith("rewards/") && k.endsWith("/mean"))) && ks.add(k)));
    return [...ks].sort((a) => (a === "reward" ? -1 : 1));
  }, [logs]);
  const progress = [...events].reverse().find((e) => e.type === "progress");
  const lastStatus = [...events].reverse().find((e) => e.type === "status");
  const params = events.find((e) => e.type === "params");
  const gpu = [...events].reverse().find((e) => e.type === "gpu");
  const errorEv = events.find((e) => e.type === "error");

  const eta = useMemo(() => {
    const prog = events.filter((e) => e.type === "progress" && e.step > 0);
    if (prog.length < 2 || !progress?.max_steps) return null;
    const a = prog[0], b = prog[prog.length - 1];
    const rate = (b.step - a.step) / (b.t - a.t);
    if (!rate) return null;
    const s = Math.round((progress.max_steps - b.step) / rate);
    return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${s % 60}s`;
  }, [events, progress]);

  if (!job) return <div className="empty">{loadErr ? <>Trabajo no encontrado. <Link to="/training">Volver a Entrenamiento</Link></> : "Cargando…"}</div>;
  const st = status ?? job.status;
  const cfg = job.config;
  const pct = progress?.max_steps ? progress.step / progress.max_steps : 0;

  return (
    <>
      <div className="page-head">
        <div>
          <Link to="/training" className="sub row" style={{ gap: 4 }}><ArrowLeft size={14} /> Entrenamiento</Link>
          <h1 className="row">{job.name} <StatusBadge status={st} /></h1>
          <div className="sub">#{job.id} · creado {fmtDate(job.created_at)} · duración {duration(job.started_at, job.finished_at)}</div>
        </div>
        <div className="row">
          {st === "running" && <button onClick={() => api.post(`/api/jobs/${id}/cancel`).then(() => notify("Deteniendo; se guardará el progreso…"))}><Square size={14} /> Detener y guardar</button>}
          {st === "running" && <button className="danger" onClick={async () => (await confirm("¿Matar el proceso?", "Se detiene inmediatamente SIN guardar. Usa «Detener y guardar» para conservar el progreso.", { label: "Matar" })) && api.post(`/api/jobs/${id}/kill`)}>Matar</button>}
          {st === "queued" && <button onClick={() => api.post(`/api/jobs/${id}/cancel`).then(reload)}>Cancelar</button>}
          {!["running", "queued"].includes(st) && <button className="danger" onClick={async () => (await confirm("¿Borrar el registro del trabajo?", "Los archivos (checkpoints, adaptadores) se conservan en disco.", { label: "Borrar" })) && api.del(`/api/jobs/${id}`).then(() => history.back())}><Trash2 size={14} /></button>}
        </div>
      </div>

      {job.type !== "train" ? (
        <div className="card stat"><div className="label">Progreso</div>
          <div className="value" style={{ fontSize: 16 }}>{lastStatus?.message ?? (st === "queued" ? "En cola" : "—")}</div><Progress value={pct} /></div>
      ) : (
      <div className="grid g4">
        <div className="card stat"><div className="label">Progreso</div><div className="value">{progress ? `${progress.step} / ${progress.max_steps}` : "—"}</div><Progress value={pct} /></div>
        <div className="card stat"><div className="label">Loss actual</div><div className="value">{trainPts.at(-1)?.loss?.toFixed(4) ?? "—"}</div><div className="sub">{evalPts.length ? `eval ${evalPts.at(-1)!.eval_loss.toFixed(4)}` : " "}</div></div>
        <div className="card stat"><div className="label">Tiempo restante</div><div className="value">{st === "running" ? eta ?? "…" : "—"}</div><div className="sub">época {progress?.epoch?.toFixed?.(2) ?? "—"}</div></div>
        <div className="card stat"><div className="label">VRAM pico (proceso)</div><div className="value">{gpu ? `${gpu.mem_gb} GB` : "—"}</div><div className="sub">{params ? `${(params.trainable / 1e6).toFixed(1)}M entrenables (${((100 * params.trainable) / params.total).toFixed(2)}%)` : " "}</div></div>
      </div>
      )}

      {st === "running" && lastStatus && job.type === "train" && <div className="card sub">{lastStatus.message}</div>}
      {(job.error || errorEv) && <div className="card"><div className="error-box">{errorEv?.message ?? job.error}</div></div>}
      {job.type === "embed" && (events.some((e) => e.type === "embed_eval")) && (
        <div className="card">
          <h3>Recuperación en el conjunto reservado ({job.config.n_pairs} pares en total)</h3>
          <table>
            <thead><tr><th></th>{Object.keys(events.find((e) => e.type === "embed_eval")!).filter((k) => !["type", "t", "stage"].includes(k)).map((k) => <th key={k}>{k}</th>)}</tr></thead>
            <tbody>{events.filter((e) => e.type === "embed_eval").map((e) => (
              <tr key={e.stage}><td><b>{e.stage === "antes" ? "Modelo base" : "Ajustado"}</b></td>
                {Object.entries(e).filter(([k]) => !["type", "t", "stage"].includes(k)).map(([k, v]: any) => {
                  const b = events.find((x) => x.type === "embed_eval" && x.stage === "antes")?.[k];
                  const d = e.stage !== "antes" && b != null ? v - b : null;
                  return <td key={k}>{(v * 100).toFixed(1)}%{d != null && <span style={{ color: d >= 0 ? "var(--ok)" : "var(--err)", marginLeft: 6 }}>{d >= 0 ? "+" : ""}{(d * 100).toFixed(1)}</span>}</td>;
                })}</tr>
            ))}</tbody>
          </table>
          {st === "completed" && <div className="sub" style={{ marginTop: 8 }}>Para usarlo: RAG → colección → Ajustes → «Re-indexar con otro modelo de embeddings».</div>}
        </div>
      )}
      {job.type === "bench" && events.some((e) => e.type === "bench_task") && (
        <div className="card">
          <h3>Resultados · {job.config.model_name}</h3>
          <table>
            <thead><tr><th>Tarea</th><th>Métrica</th><th>Valor</th><th>± error</th></tr></thead>
            <tbody>{events.filter((e) => e.type === "bench_task").map((e) => (
              <tr key={e.task}><td><b>{e.task}</b>{e.subtasks > 1 && <span className="sub"> ({e.subtasks} subtareas)</span>}</td><td className="mono">{e.metric}</td>
                <td><b>{e.value != null ? `${(e.value * 100).toFixed(1)}%` : "—"}</b></td><td className="sub">{e.stderr != null ? `±${(e.stderr * 100).toFixed(1)}` : ""}</td></tr>
            ))}</tbody>
          </table>
          <div className="sub" style={{ marginTop: 6, fontSize: 12 }}>Límite: {job.config.limit || "todos"} ejemplos por tarea. Compara con el modelo base marcando ambos trabajos en la lista.</div>
        </div>
      )}
      {st === "completed" && job.result?.gguf && (
        <div className="card stack">
          <div>✓ GGUF listo: <code>{job.result.gguf}</code></div>
          <div className="sub">Úsalo en el Playground (ya está registrado en Modelos), en LM Studio abriendo el archivo, o en Ollama:</div>
          <Markdown text={"```bash\ncd \"" + job.result.gguf.replace(/\/[^/]+$/, "") + "\"\nollama create " + job.name.replace(/-export$/, "") + " -f Modelfile\nollama run " + job.name.replace(/-export$/, "") + "\n```"} />
          <div className="row"><Link to="/playground"><button className="primary">Probar en Playground</button></Link><Link to="/system"><button>Usar vía API OpenAI</button></Link></div>
        </div>
      )}
      {st === "completed" && job.result?.model && (
        <div className="card row between">
          <div>✓ Modelo guardado en <code>{job.result.model}</code>{job.result.merged && <> · fusionado en <code>{job.result.merged}</code></>}</div>
          <div className="row"><Link to="/playground"><button className="primary">Probar en Playground</button></Link><Link to="/models"><button>Ver en Modelos</button></Link></div>
        </div>
      )}

      {job.type === "train" && rewardKeys.length > 0 && (
        <div className="card" style={{ marginTop: 16 }}>
          <h3>Recompensas (GRPO)</h3>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={logs.filter((e) => e.reward !== undefined)}>
              <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
              <XAxis dataKey="step" stroke="var(--muted)" fontSize={12} />
              <YAxis stroke="var(--muted)" fontSize={12} domain={["auto", "auto"]} />
              <Tooltip contentStyle={{ background: "var(--panel)", border: "1px solid var(--border)" }} />
              <Legend />
              {rewardKeys.map((k, i) => <Line key={k} type="monotone" dataKey={k} name={k.replace(/^rewards\//, "").replace(/\/mean$/, "")} stroke={["#3fb950", "#7c8cff", "#d29922", "#f85149", "#a371f7"][i % 5]} dot={false} strokeWidth={k === "reward" ? 3 : 1.5} isAnimationActive={false} />)}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
      {(job.type === "train" || job.type === "embed") && (
        <div className="grid g2" style={{ marginTop: 16 }}>
          <div className="card">
            <h3>Loss</h3>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={lossData}>
                <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                <XAxis dataKey="step" stroke="var(--muted)" fontSize={12} />
                <YAxis stroke="var(--muted)" fontSize={12} domain={["auto", "auto"]} />
                <Tooltip contentStyle={{ background: "var(--panel)", border: "1px solid var(--border)" }} />
                <Legend />
                <Line type="monotone" dataKey="loss" name="train" stroke="#7c8cff" dot={false} strokeWidth={2} isAnimationActive={false} connectNulls />
                <Line type="monotone" dataKey="eval_loss" name="eval" stroke="#3fb950" dot strokeWidth={2} isAnimationActive={false} connectNulls />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="card">
            <h3>Learning rate · grad norm</h3>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={trainPts}>
                <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                <XAxis dataKey="step" stroke="var(--muted)" fontSize={12} />
                <YAxis yAxisId="lr" stroke="var(--muted)" fontSize={12} tickFormatter={(v) => v.toExponential(0)} />
                <YAxis yAxisId="gn" orientation="right" stroke="var(--muted)" fontSize={12} />
                <Tooltip contentStyle={{ background: "var(--panel)", border: "1px solid var(--border)" }} />
                <Legend />
                <Line yAxisId="lr" type="monotone" dataKey="lr" name="lr" stroke="#d29922" dot={false} isAnimationActive={false} />
                <Line yAxisId="gn" type="monotone" dataKey="grad_norm" name="grad norm" stroke="#f85149" dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      <div className="grid g2" style={{ marginTop: 16, gridTemplateColumns: "minmax(0,2fr) minmax(0,1fr)" }}>
        <div className="card">
          <h3>Log</h3>
          <div className="log" ref={logRef} onScroll={(e) => { const el = e.currentTarget; stuck.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80; }}>{log.replace(/\r(?!\n)/g, "\n").split("\n").filter((l, i, a) => !(l.includes("it/s]") && a[i + 1]?.includes("it/s]"))).join("\n") || "Esperando salida…"}</div>
        </div>
        <div className="card">
          <h3>Configuración</h3>
          <div className="kv">
            {Object.entries(cfg).filter(([k]) => !["dataset_path", "output_dir"].includes(k)).map(([k, v]) => (
              <Fragment key={k}><span>{k}</span><span className="mono truncate" title={String(v)}>{typeof v === "object" ? JSON.stringify(v) : String(v)}</span></Fragment>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}

import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { BarChart3, Boxes, Eraser, GitCompare, GraduationCap, Lock, Ruler, Trash2 } from "lucide-react";
import { api, duration, type Job } from "../api";
import { Check, Field, Help, Modal, Progress, Spinner, StatusBadge, useData, useDialog, useToast } from "../components/ui";

/** Qué es cada tipo de trabajo, en cristiano. */
const KINDS: Record<string, { label: string; icon: any; desc: string }> = {
  train: { label: "Entrenamiento", icon: GraduationCap, desc: "Fine-tuning de un modelo de lenguaje" },
  embed: { label: "Embeddings", icon: Ruler, desc: "Ajuste del modelo de búsqueda del RAG" },
  bench: { label: "Benchmark", icon: BarChart3, desc: "Prueba estándar para medir capacidades generales" },
  export: { label: "Exportación", icon: Boxes, desc: "Fusión de LoRA o conversión a GGUF" },
};

/** Resumen de una línea con lo que define a ese trabajo. */
function detail(j: Job): string {
  if (j.type === "train") {
    const m = (j.config.method ?? "").toUpperCase();
    return [m, j.config.backend === "unsloth" ? "Unsloth" : null, j.config.base_model?.split("/").pop()].filter(Boolean).join(" · ");
  }
  if (j.type === "export") return j.config.gguf ? `GGUF ${j.config.gguf_quant}` : "fusión de LoRA";
  return KINDS[j.type]?.label ?? j.type;
}

export default function Jobs() {
  const [jobs, reload] = useData<Job[]>("/api/jobs", 4000);
  const [cmp, setCmp] = useState<number[]>([]);
  const [kind, setKind] = useState("");
  const [status, setStatus] = useState("");
  const [clean, setClean] = useState(false);
  const nav = useNavigate();
  const dialog = useDialog();
  const { notify } = useToast();

  const borrar = async (j: Job) => {
    const info = await api.get<any>(`/api/jobs/${j.id}/output`);
    const protegido = info.models?.length;
    const msg = !info.exists ? "Este trabajo no dejó archivos: sólo se borra la fila del historial."
      : protegido ? `Ocupa ${info.size_mb} MB, pero esos archivos son de ${info.models.join(", ")}, que está registrado en Modelos. Se borra la fila y los archivos se conservan.`
      : `Se borrará la fila y sus ${info.size_mb} MB de archivos, que no usa ningún modelo registrado.`;
    if (!(await dialog.confirm(`Borrar «${j.name}»`, msg))) return;
    const r = await api.del<any>(`/api/jobs/${j.id}?delete_output=${!protegido}`);
    notify(r.deleted_output ? `Borrado · ${r.freed_mb || "<0,1"} MB liberados` : "Borrado del historial");
    reload();
  };

  const shown = useMemo(() => (jobs ?? []).filter((j) => (!kind || j.type === kind) && (!status || j.status === status)), [jobs, kind, status]);
  const running = (jobs ?? []).filter((j) => j.status === "running" || j.status === "queued");
  // Sólo tiene sentido enfrentar curvas de entrenamientos y resultados de benchmarks
  const comparable = (j: Job) => j.type === "train" || j.type === "bench";

  return (
    <>
      <div className="page-head">
        <div><h1>Trabajos</h1><div className="sub">Todo lo que se ha lanzado: entrenamientos, ajustes de embeddings, benchmarks y exportaciones</div></div>
        <div className="row" style={{ gap: 8 }}>
          <button onClick={() => setClean(true)}><Eraser size={14} /> Limpiar historial</button>
          <button className="primary" disabled={cmp.length < 2} onClick={() => nav(`/training/compare?ids=${cmp.join(",")}`)}>
            <GitCompare size={14} /> Comparar ({cmp.length})
          </button>
        </div>
      </div>

      <Help title="¿Qué es esta pantalla y para qué sirve «Comparar»?">
        <p>Es el <b>historial</b>: cada vez que lanzas un entrenamiento, un benchmark, un ajuste de embeddings o una exportación, queda aquí una fila. No se configura nada en esta pantalla, sólo se consulta. Haz clic en el nombre para abrir el detalle con sus curvas, su configuración exacta y su registro completo.</p>
        <p style={{ marginTop: 8 }}><b>Comparar</b> marca la casilla de <b>dos o más</b> entrenamientos y los pone uno al lado del otro: las curvas de pérdida superpuestas, los hiperparámetros que los diferencian y las métricas finales. Es la forma de responder «¿me sirvió subir el learning rate?» o «¿el rango 32 mejoró algo frente al 16?» sin ir abriendo pestañas.</p>
        <ul>
          <li>Sólo se pueden comparar <b>entrenamientos entre sí y benchmarks entre sí</b>; las exportaciones no tienen métricas que enfrentar, por eso no llevan casilla.</li>
          <li>Para que la comparación diga algo, los entrenamientos deben haber usado <b>el mismo dataset</b>; si no, estarás comparando datos, no configuraciones.</li>
          <li>Si lo que quieres es probar muchas configuraciones de golpe, usa <b>Entrenamiento → Barrido</b>: lanza una por combinación y te las ordena solo.</li>
        </ul>
      </Help>

      {!!running.length && (
        <div className="card" style={{ borderColor: "var(--accent)" }}>
          <b>En curso ahora mismo</b>
          {running.map((j) => (
            <div key={j.id} style={{ marginTop: 6 }}>
              <Link to={`/training/${j.id}`}>{j.name}</Link> <span className="sub">· {detail(j)}</span>
              {j.progress?.max_steps ? (
                <><Progress value={j.progress.step / j.progress.max_steps} />
                  <div className="sub" style={{ fontSize: 11 }}>{j.progress.step}/{j.progress.max_steps} pasos</div></>
              ) : <div className="sub" style={{ fontSize: 12 }}>en cola</div>}
            </div>
          ))}
        </div>
      )}

      <div className="card">
        <div className="row" style={{ gap: 10, alignItems: "flex-end" }}>
          <Field label="Tipo">
            <select value={kind} onChange={(e) => setKind(e.target.value)}>
              <option value="">Todos</option>
              {Object.entries(KINDS).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}
            </select>
          </Field>
          <Field label="Estado">
            <select value={status} onChange={(e) => setStatus(e.target.value)}>
              <option value="">Todos</option>
              <option value="completed">Completados</option>
              <option value="failed">Fallidos</option>
              <option value="running">En curso</option>
              <option value="cancelled">Cancelados</option>
            </select>
          </Field>
          <div className="spacer" />
          <span className="sub">{shown.length} de {jobs?.length ?? 0}</span>
        </div>

        {!shown.length ? <div className="empty">{jobs?.length ? "Ningún trabajo con ese filtro" : "Todavía no has lanzado nada"}</div> : (
          <table style={{ marginTop: 10 }}>
            <thead><tr><th style={{ width: 28 }} title="Marca dos o más para compararlos" /><th>Nombre</th><th>Tipo</th><th>Duración</th><th>Estado</th><th /></tr></thead>
            <tbody>
              {shown.map((j) => {
                const K = KINDS[j.type];
                const Icon = K?.icon ?? GraduationCap;
                return (
                  <tr key={j.id}>
                    <td>{comparable(j) && (
                      <input type="checkbox" checked={cmp.includes(j.id)} title="Comparar con otro"
                        onChange={(e) => setCmp(e.target.checked ? [...cmp, j.id] : cmp.filter((x) => x !== j.id))} />
                    )}</td>
                    <td>
                      <Link to={`/training/${j.id}`}><b>{j.name}</b></Link>
                      <div className="sub" style={{ fontSize: 12 }}>#{j.id} · {detail(j)}</div>
                      {j.progress?.max_steps && j.status === "running" ? (
                        <div style={{ marginTop: 4 }}><Progress value={j.progress.step / j.progress.max_steps} /></div>
                      ) : null}
                      {j.error && <div className="sub ellipsis" style={{ fontSize: 11, maxWidth: 420 }} title={j.error}>{j.error}</div>}
                    </td>
                    <td className="sub" title={K?.desc}><Icon size={13} /> {K?.label ?? j.type}</td>
                    <td className="sub">{duration(j.started_at, j.finished_at)}</td>
                    <td><StatusBadge status={j.status} /></td>
                    <td><button className="icon" title="Borrar este trabajo" onClick={() => borrar(j)}><Trash2 size={13} /></button></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
        {cmp.length === 1 && <div className="sub" style={{ marginTop: 8 }}>Marca al menos otro para poder compararlos.</div>}
      </div>

      {clean && <Cleanup onClose={() => { setClean(false); reload(); }} />}
    </>
  );
}

/** Limpieza en bloque. Siempre enseña antes qué se va a borrar y cuánto disco liberaría de verdad. */
function Cleanup({ onClose }: { onClose: () => void }) {
  const [statuses, setStatuses] = useState<string[]>(["failed", "cancelled"]);
  const [days, setDays] = useState(0);
  const [outputs, setOutputs] = useState(false);
  const [preview, setPreview] = useState<any>(null);
  const [purga, setPurga] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const { notify } = useToast();

  const body = () => ({ statuses: statuses.length ? statuses : null, older_than_days: days || null,
                        delete_outputs: outputs, dry_run: true });

  const ver = async () => {
    setBusy(true);
    try { setPreview(await api.post<any>("/api/jobs/cleanup", body())); }
    catch (e: any) { notify(e.message, "err"); } finally { setBusy(false); }
  };

  const borrar = async () => {
    setBusy(true);
    try {
      const r = await api.post<any>("/api/jobs/cleanup", { ...body(), dry_run: false });
      notify(`${r.count} trabajo(s) borrado(s)${r.deleted_outputs ? ` · ${r.deleted_outputs} carpeta(s), ${r.freed_mb} MB liberados` : ""}`);
      onClose();
    } catch (e: any) { notify(e.message, "err"); } finally { setBusy(false); }
  };

  const verPurga = async () => {
    setBusy(true);
    try { setPurga(await api.post<any>("/api/jobs/purge-outputs", { dry_run: true })); }
    catch (e: any) { notify(e.message, "err"); } finally { setBusy(false); }
  };

  const purgar = async () => {
    setBusy(true);
    try {
      const r = await api.post<any>("/api/jobs/purge-outputs", { dry_run: false });
      notify(`${(r.freed_mb / 1000).toFixed(2)} GB liberados · el historial se conserva`);
      setPurga(null);
      onClose();
    } catch (e: any) { notify(e.message, "err"); } finally { setBusy(false); }
  };

  const toggle = (s: string) => setStatuses((p) => (p.includes(s) ? p.filter((x) => x !== s) : [...p, s]));

  return (
    <Modal title="Limpiar historial" onClose={onClose} wide>
      <p className="sub">
        Elige qué trabajos quitar del historial. Los que estén <b>en curso o en cola</b> nunca se tocan, y
        los archivos de un trabajo cuyo modelo esté registrado en <b>Modelos</b> se conservan siempre,
        para no dejar ese modelo apuntando a la nada.
      </p>
      <Field label="Estados a borrar">
        <div className="row" style={{ gap: 14, flexWrap: "wrap" }}>
          {[["failed", "Fallidos"], ["cancelled", "Cancelados"], ["completed", "Completados"]].map(([k, l]) => (
            <Check key={k} label={l} checked={statuses.includes(k)} onChange={() => toggle(k)} />
          ))}
        </div>
      </Field>
      <Field label="Sólo los más antiguos que…" hint="0 = sin límite de antigüedad">
        <input type="number" min={0} value={days} onChange={(e) => { setDays(+e.target.value); setPreview(null); }} />
      </Field>
      <Check label="Borrar también sus archivos del disco (adaptadores, checkpoints)" checked={outputs}
        onChange={(v) => { setOutputs(v); setPreview(null); }} />

      <div className="row" style={{ marginTop: 10 }}>
        <button onClick={ver} disabled={busy || !statuses.length}>{busy ? <Spinner /> : <Eraser size={13} />} Ver qué se borraría</button>
      </div>

      <div className="card" style={{ marginTop: 12 }}>
        <b>Sólo liberar espacio</b>
        <div className="sub" style={{ marginTop: 4 }}>
          Borra los pesos y checkpoints de todos los trabajos pero <b>conserva sus filas</b>, con sus curvas y
          sus métricas. Es lo que quieres casi siempre: los pesos ocupan gigas y rara vez se vuelven a ejecutar,
          mientras que lo que midió cada entrenamiento no pesa nada y es la razón de haberlo hecho.
        </div>
        <div className="row" style={{ marginTop: 8, gap: 8 }}>
          <button onClick={verPurga} disabled={busy}>{busy ? <Spinner /> : null} Ver cuánto se liberaría</button>
          {purga != null && purga.count > 0 && (
            <button className="danger" onClick={purgar}>
              <Trash2 size={13} /> Liberar {(purga.freed_mb / 1000).toFixed(2)} GB de {purga.count} trabajo(s)
            </button>
          )}
        </div>
        {purga != null && <div className="sub" style={{ marginTop: 6 }}>
          {purga.count ? `${purga.count} trabajo(s) con pesos en disco · ${(purga.freed_mb / 1000).toFixed(2)} GB` : "No queda ningún peso que borrar."}
        </div>}
      </div>

      {preview && (
        <div className="card" style={{ marginTop: 10 }}>
          {!preview.count ? <div className="empty">Ningún trabajo cumple esos criterios</div> : (<>
            <b>{preview.count} trabajo(s)</b>
            <div className="sub">
              {outputs
                ? preview.freeable_mb ? `Se liberarían ${preview.freeable_mb} MB de disco.` : "No se liberaría disco: todos sus archivos están protegidos."
                : "Sólo se quitan del historial; los archivos se quedan en disco."}
            </div>
            {!!preview.protected?.length && (
              <div className="sub" style={{ marginTop: 6 }}>
                <Lock size={12} /> Archivos conservados por tener un modelo registrado: {preview.protected.join(", ")}
              </div>
            )}
            <table style={{ marginTop: 8 }}>
              <tbody>
                {preview.jobs.slice(0, 40).map((j: any) => (
                  <tr key={j.id}>
                    <td>{j.name} <span className="sub">#{j.id} · {j.status}</span></td>
                    <td className="sub" style={{ textAlign: "right" }}>
                      {j.size_mb ? `${j.size_mb} MB` : "—"} {j.models?.length ? <Lock size={11} /> : null}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {preview.jobs.length > 40 && <div className="sub">…y {preview.jobs.length - 40} más</div>}
          </>)}
        </div>
      )}

      <div className="row end" style={{ marginTop: 10 }}>
        <button onClick={onClose}>Cancelar</button>
        <button className="danger" disabled={busy || !preview?.count} onClick={borrar}>
          <Trash2 size={13} /> Borrar {preview?.count ?? 0} trabajo(s)
        </button>
      </div>
    </Modal>
  );
}

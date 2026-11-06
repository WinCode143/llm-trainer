import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Play } from "lucide-react";
import { Link } from "react-router-dom";
import { api, type Collection, type Dataset, type Job, type Model } from "../api";
import { Adv, Field, SimpleNote, StatusBadge, useData, useMode, useToast } from "../components/ui";

/** Fine-tuning del modelo de embeddings del RAG. */
export function EmbedTrain() {
  const [cols] = useData<Collection[]>("/api/rag/collections");
  const [datasets] = useData<Dataset[]>("/api/datasets");
  const [embedModels] = useData<any[]>("/api/rag/embed-models");
  const [f, setF] = useState({ name: "", base_model: "intfloat/multilingual-e5-small", collection_id: "" as number | "", dataset_id: "" as number | "", epochs: 1, batch_size: 64, learning_rate: 1e-5 });
  const { notify } = useToast();
  const nav = useNavigate();
  const go = async () => {
    try {
      const j = await api.post<Job>("/api/jobs/embed", { ...f, name: f.name || `embed-${f.base_model.split("/").pop()}`, collection_id: f.collection_id || null, dataset_id: f.dataset_id || null });
      nav(`/training/${j.id}`);
    } catch (e: any) { notify(e.message, "err"); }
  };
  return (
    <div className="card stack" style={{ gap: 14 }}>
      <h2 style={{ margin: 0 }}>Ajustar embeddings del RAG</h2>
      <div className="sub">Enseña al modelo de embeddings el vocabulario de tus documentos para que la búsqueda encuentre mejor. Usa como pares de entrenamiento las <b>preguntas generadas por fragmento</b> de una colección (RAG → Documentos → «Generar preguntas») o las conversaciones de un dataset. Se mide recall@k y MRR antes y después sobre un 15 % reservado.</div>
      <div className="source sub" style={{ fontSize: 12.5, padding: "8px 10px" }}>⚠ Ese 15 % reservado es del mismo estilo que los pares de entrenamiento: la mejora ahí no garantiza mejora con preguntas reales. Tras re-indexar, compruébalo en RAG → Benchmark con preguntas escritas a mano. Con preguntas sintéticas de un modelo pequeño conviene un ajuste suave (1 época, lr 1e-5).</div>
      <div className="grid g2">
        <Field label="Modelo base"><select value={f.base_model} onChange={(e) => setF({ ...f, base_model: e.target.value })}>{embedModels?.filter((m) => !m.tuned).map((m) => <option key={m.name} value={m.name}>{m.name}</option>)}</select></Field>
        <Adv><Field label="Nombre"><input value={f.name} onChange={(e) => setF({ ...f, name: e.target.value })} placeholder="auto" /></Field></Adv>
        <Field label="Pares desde una colección enriquecida">
          <select value={f.collection_id} onChange={(e) => setF({ ...f, collection_id: e.target.value ? +e.target.value : "" })}>
            <option value="">—</option>{cols?.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
          </select>
        </Field>
        <Field label="…y/o desde un dataset de chat">
          <select value={f.dataset_id} onChange={(e) => setF({ ...f, dataset_id: e.target.value ? +e.target.value : "" })}>
            <option value="">—</option>{datasets?.filter((d) => d.format === "chat").map((d) => <option key={d.id} value={d.id}>{d.name} ({d.num_rows})</option>)}
          </select>
        </Field>
        <Adv>
          <Field label="Épocas"><input type="number" value={f.epochs} onChange={(e) => setF({ ...f, epochs: +e.target.value })} /></Field>
          <Field label="Learning rate"><input type="number" step="0.000001" value={f.learning_rate} onChange={(e) => setF({ ...f, learning_rate: +e.target.value })} /></Field>
          <Field label="Batch" hint="Más grande = más negativos por lote (mejor), más VRAM"><input type="number" value={f.batch_size} onChange={(e) => setF({ ...f, batch_size: +e.target.value })} /></Field>
        </Adv>
      </div>
      <SimpleNote>Ajuste suave: {f.epochs} época y learning rate {f.learning_rate}. Medimos que entrenar más mejora la métrica reservada pero empeora las preguntas reales.</SimpleNote>
      <div className="row"><div className="spacer" /><button className="primary" disabled={!f.collection_id && !f.dataset_id} onClick={go}><Play size={14} /> Entrenar embeddings</button></div>
    </div>
  );
}

/** Benchmarks estándar con lm-evaluation-harness. */
export function Bench() {
  const [models] = useData<Model[]>("/api/models");
  const [tasksList] = useData<any[]>("/api/jobs/bench-tasks");
  const [modelId, setModelId] = useState<number | "">("");
  const [sel, setSel] = useState<string[]>(["arc_easy", "arc_es", "hellaswag_es"]);
  const [limit, setLimit] = useState(200);
  const { notify } = useToast();
  const nav = useNavigate();
  const go = async () => {
    try { const j = await api.post<Job>("/api/jobs/bench", { model_id: modelId, tasks: sel, limit: limit || null }); nav(`/training/${j.id}`); } catch (e: any) { notify(e.message, "err"); }
  };
  return (
    <div className="card stack" style={{ gap: 14 }}>
      <h2 style={{ margin: 0 }}>Benchmarks estándar</h2>
      <div className="sub">Mide capacidades generales con lm-evaluation-harness. Pásalo al modelo base y al ajustado para detectar <b>olvido catastrófico</b> (el fine-tuning no debería bajar mucho estas cifras). Luego compáralos marcando ambos en la lista de trabajos.</div>
      <div className="grid g2">
        <Field label="Modelo"><select value={modelId} onChange={(e) => setModelId(e.target.value ? +e.target.value : "")}>
          <option value="">— elige —</option>{models?.filter((m) => ["hf", "merged", "lora"].includes(m.kind)).map((m) => <option key={m.id} value={m.id}>{m.name} [{m.kind}]</option>)}
        </select></Field>
        <Field label="Ejemplos por tarea" hint="200 da una estimación rápida (±3 %); 0 = todos (lento)"><input type="number" value={limit} onChange={(e) => setLimit(+e.target.value)} /></Field>
      </div>
      <div className="grid g3">
        {tasksList?.map((t) => (
          <label key={t.task} className="check source" style={{ padding: "6px 10px" }}>
            <input type="checkbox" checked={sel.includes(t.task)} onChange={(e) => setSel(e.target.checked ? [...sel, t.task] : sel.filter((x) => x !== t.task))} />
            <span><b>{t.label}</b><div className="sub" style={{ fontSize: 12 }}>{t.desc}</div></span>
          </label>
        ))}
      </div>
      <div className="row"><div className="spacer" /><button className="primary" disabled={!modelId || !sel.length} onClick={go}><Play size={14} /> Ejecutar benchmarks</button></div>
    </div>
  );
}


/** Los parámetros que se pueden barrer, ordenados por lo que de verdad cambian el resultado.
 *  «impacto» decide cuáles se ven en modo simple: barrer diez a la vez no da respuestas, da ruido. */
const SWEEP_PARAMS: { key: string; label: string; example: string; impacto: "alto" | "medio" | "bajo";
                      que: string; reco: string }[] = [
  { key: "learning_rate", label: "Learning rate", example: "1e-4, 2e-4, 5e-4", impacto: "alto",
    que: "Cuánto se mueve el modelo en cada paso. Demasiado alto y se desestabiliza (la pérdida sube o salta); demasiado bajo y apenas aprende. Es, con diferencia, el que más cambia el resultado.",
    reco: "Con LoRA, entre 1e-4 y 5e-4. Con DPO/ORPO baja mucho: 5e-6. Si sólo vas a barrer uno, que sea éste." },
  { key: "lora_r", label: "Rango LoRA (r)", example: "8, 16, 32", impacto: "alto",
    que: "El tamaño del adaptador: cuántos parámetros nuevos se entrenan. Más rango = más capacidad de aprender matices, pero más riesgo de memorizar en vez de aprender, y un archivo más grande.",
    reco: "8 para datasets pequeños (menos de 500 ejemplos), 16 para lo normal, 32–64 si tienes miles de ejemplos." },
  { key: "epochs", label: "Épocas", example: "1, 2, 3", impacto: "alto",
    que: "Cuántas veces se recorre el dataset entero. Es la causa número uno de sobreajuste: a partir de cierto punto el modelo deja de aprender y empieza a memorizar tus ejemplos.",
    reco: "1–3. Con datasets pequeños, 1 o 2. Sólo tiene efecto si «Pasos máx.» está en 0." },
  { key: "lora_alpha", label: "LoRA alpha", example: "16, 32, 64", impacto: "medio",
    que: "Multiplica la fuerza del adaptador. Lo que importa de verdad es la proporción alpha/rango, no el número suelto.",
    reco: "La convención es alpha = 2 × rango (rango 16 → alpha 32). Barrerlo aparte del rango multiplica combinaciones sin aportar mucho." },
  { key: "batch_size", label: "Batch", example: "2, 4", impacto: "medio",
    que: "Cuántos ejemplos se procesan a la vez. Más grande estabiliza el entrenamiento, pero es lo primero que te deja sin VRAM.",
    reco: "2 con modelos de 7B o más en 24 GB; 4–8 con modelos pequeños. Si da error de memoria, bájalo y sube la acumulación." },
  { key: "grad_accum", label: "Acumulación", example: "2, 4, 8", impacto: "medio",
    que: "Acumula varios lotes antes de actualizar. Sirve para simular un batch grande sin gastar más VRAM: el batch efectivo es batch × acumulación.",
    reco: "4 por defecto. Lo que importa es el producto: batch 2 × acum 4 equivale a batch 8." },
  { key: "max_seq_len", label: "Longitud máx.", example: "1024, 2048", impacto: "medio",
    que: "Cuántos tokens de cada ejemplo se usan. Si tus textos son más largos, se recortan y pierdes esa parte.",
    reco: "2048 sirve para casi todo. Súbelo sólo si tus ejemplos se cortan; gasta VRAM rápido." },
  { key: "lora_dropout", label: "LoRA dropout", example: "0, 0.05, 0.1", impacto: "bajo",
    que: "Apaga al azar parte del adaptador durante el entrenamiento para que no memorice.",
    reco: "0 normalmente. Súbelo a 0.05–0.1 sólo si ves sobreajuste. Con Unsloth conviene dejarlo en 0 (va más rápido)." },
  { key: "warmup_ratio", label: "Warmup", example: "0, 0.03, 0.1", impacto: "bajo",
    que: "Arranca con un learning rate bajo y lo sube poco a poco durante los primeros pasos, para no desestabilizar el modelo al principio.",
    reco: "0.03 y olvídate. Rara vez cambia nada." },
  { key: "weight_decay", label: "Weight decay", example: "0, 0.01", impacto: "bajo",
    que: "Penaliza que los pesos crezcan mucho; es otra forma de evitar el sobreajuste.",
    reco: "0 o 0.01. Con LoRA apenas se nota." },
  { key: "neftune_alpha", label: "NEFTune α", example: "0, 5", impacto: "bajo",
    que: "Añade un poco de ruido a los embeddings durante el entrenamiento. A veces mejora la calidad de las respuestas largas.",
    reco: "0 (desactivado) o 5. Es un truco, no un ajuste fundamental." },
];

/** Barrido de hiperparámetros: una combinación por trabajo, en cola. */
export function Sweep() {
  const [datasets] = useData<Dataset[]>("/api/datasets");
  const [models] = useData<Model[]>("/api/models");
  const [sys] = useData<any>("/api/system");
  const [sweeps, reloadSweeps] = useData<any[]>("/api/jobs/sweeps", 5000);
  const [base, setBase] = useState({ name: "sweep", base_model: "Qwen/Qwen3-0.6B", dataset_id: 0, method: "sft",
    backend: "hf", quantization: "4bit", max_steps: 60, eval_ratio: 0.15, batch_size: 2, grad_accum: 4 });
  const [grid, setGrid] = useState<Record<string, string>>({ learning_rate: "1e-4, 2e-4, 5e-4" });
  const [strategy, setStrategy] = useState("grid");
  const [maxRuns, setMaxRuns] = useState(6);
  const [abierto, setAbierto] = useState<string | null>(null);
  const { simple } = useMode();
  const { notify } = useToast();

  useEffect(() => { if (sys?.unsloth) setBase((b) => ({ ...b, backend: "unsloth" })); }, [sys?.unsloth]);
  const parsed = Object.fromEntries(Object.entries(grid).map(([k, v]) => [k, v.split(",").map((x) => parseFloat(x.trim())).filter((x) => !isNaN(x))]));
  const activos = Object.values(parsed).filter((v) => v.length);
  // Sin ningún parámetro marcado no hay barrido que lanzar (antes contaba 1 y el servidor lo rechazaba)
  const combos = activos.length ? activos.reduce((a, v) => a * v.length, 1) : 0;
  const runs = Math.min(combos, maxRuns);

  const go = async () => {
    try {
      const r = await api.post<any>("/api/jobs/sweep", { base: { ...base, base_model_id: null }, grid: parsed, strategy, max_runs: maxRuns });
      notify(`Barrido lanzado: ${r.runs} entrenamientos en cola`);
      reloadSweeps();
    } catch (e: any) { notify(e.message, "err"); }
  };

  return (
    <div className="stack" style={{ gap: 16 }}>
      <div className="card stack" style={{ gap: 14 }}>
        <h2 style={{ margin: 0 }}>Barrido de hiperparámetros</h2>
        <div className="sub">Lanza una combinación por entrenamiento, en cola. Se comparan por <b>eval_loss</b>, así que deja una parte para evaluación. Empieza con entrenamientos cortos (pocos pasos) para explorar y luego repite el ganador entero.</div>
        <div className="grid g3">
          <Field label="Nombre base"><input value={base.name} onChange={(e) => setBase({ ...base, name: e.target.value })} /></Field>
          <Field label="Modelo base"><input value={base.base_model} onChange={(e) => setBase({ ...base, base_model: e.target.value })} /></Field>
          <Field label="Dataset">
            <select value={base.dataset_id} onChange={(e) => setBase({ ...base, dataset_id: +e.target.value })}>
              <option value={0}>— elige —</option>{datasets?.map((d) => <option key={d.id} value={d.id}>{d.name} ({d.num_rows})</option>)}
            </select>
          </Field>
          <Field label="Método">
            <select value={base.method} onChange={(e) => setBase({ ...base, method: e.target.value })}>
              {["sft", "dpo", "orpo", "kto", "grpo"].map((m) => <option key={m}>{m}</option>)}
            </select>
          </Field>
          <Field label="Motor">
            <select value={base.backend} onChange={(e) => setBase({ ...base, backend: e.target.value })}>
              <option value="unsloth" disabled={!sys?.unsloth}>unsloth</option><option value="hf">hf</option>
              <option value="vllm" disabled={!sys?.vllm || base.method !== "grpo"}>vllm</option>
            </select>
          </Field>
          <Field label="Pasos por entrenamiento" hint="Cortos para explorar"><input type="number" value={base.max_steps} onChange={(e) => setBase({ ...base, max_steps: +e.target.value })} /></Field>
          <Field label={`Proporción de evaluación: ${base.eval_ratio}`}><input type="range" min={0} max={0.4} step={0.05} value={base.eval_ratio} onChange={(e) => setBase({ ...base, eval_ratio: +e.target.value })} /></Field>
          <Field label="Estrategia" hint="Aleatoria = muestrea combinaciones si hay muchas">
            <select value={strategy} onChange={(e) => setStrategy(e.target.value)}><option value="grid">rejilla completa</option><option value="random">aleatoria</option></select>
          </Field>
          <Field label="Máx. entrenamientos"><input type="number" value={maxRuns} onChange={(e) => setMaxRuns(+e.target.value)} /></Field>
        </div>
        <Field label="Parámetros a barrer"
          hint="Pulsa el nombre de cualquiera para ver qué hace y qué valores se recomiendan.">
          <div className="stack" style={{ gap: 2 }}>
            {SWEEP_PARAMS.filter((sp) => !simple || sp.impacto === "alto").map((sp) => (
              <div key={sp.key}>
                <div className="row" style={{ flexWrap: "nowrap", gap: 4 }}>
                  <input type="checkbox" checked={sp.key in grid} onChange={(e) => {
                    const g = { ...grid }; if (e.target.checked) g[sp.key] = sp.example; else delete g[sp.key]; setGrid(g);
                  }} />
                  <button className="param-nombre" title="Ver qué hace"
                    onClick={() => setAbierto(abierto === sp.key ? null : sp.key)}>
                    {sp.label} <span className={`impacto ${sp.impacto}`}>{sp.impacto}</span>
                  </button>
                  <input disabled={!(sp.key in grid)} value={grid[sp.key] ?? ""} placeholder={sp.example}
                    onChange={(e) => setGrid({ ...grid, [sp.key]: e.target.value })} />
                </div>
                {abierto === sp.key && (
                  <div className="param-info">
                    <p>{sp.que}</p>
                    <p><b>Recomendado:</b> {sp.reco}</p>
                  </div>
                )}
              </div>
            ))}
          </div>
        </Field>
        <SimpleNote>
          Se ocultan los ocho parámetros de impacto medio y bajo (alpha, batch, acumulación, longitud,
          dropout, warmup, weight decay y NEFTune). Barrer muchos a la vez no da respuestas: da
          combinaciones que luego no sabes a qué atribuir.
        </SimpleNote>
        <div className="row between">
          <span className="sub">{combos} combinación(es) → <b>{runs}</b> entrenamiento(s) en cola{models ? "" : ""}</span>
          <button className="primary" disabled={!base.dataset_id || !runs} onClick={go}><Play size={14} /> Lanzar barrido</button>
        </div>
      </div>

      {sweeps?.map((sw) => <SweepResults key={sw.sweep_id} sw={sw} />)}
    </div>
  );
}

function SweepResults({ sw }: { sw: any }) {
  const metric = sw.runs.some((r: any) => r.eval_loss != null) ? "eval_loss" : sw.runs.some((r: any) => r.reward != null) ? "reward" : "train_loss";
  const better = (a: number, b: number) => (metric === "reward" ? a > b : a < b);
  const done = sw.runs.filter((r: any) => r[metric] != null);
  const best = done.length ? done.reduce((a: any, b: any) => (better(b[metric], a[metric]) ? b : a)) : null;
  return (
    <div className="card">
      <div className="row between">
        <h3 style={{ margin: 0 }}>Barrido {sw.sweep_id}</h3>
        <span className="sub">{sw.runs.filter((r: any) => r.status === "completed").length}/{sw.runs.length} completados · mejor por <b>{metric}</b></span>
      </div>
      <div style={{ overflowX: "auto" }}>
        <table>
          <thead><tr><th>Run</th>{sw.params.map((p: string) => <th key={p} className="mono">{p}</th>)}<th>eval_loss</th><th>loss</th><th>reward</th><th>Estado</th></tr></thead>
          <tbody>
            {sw.runs.map((r: any) => (
              <tr key={r.job_id} className={best && r.job_id === best.job_id ? "selected" : ""}>
                <td><Link to={`/training/${r.job_id}`}>{r.name}</Link>{best && r.job_id === best.job_id && " ★"}</td>
                {sw.params.map((p: string) => <td key={p} className="mono">{String(r.params[p] ?? "—")}</td>)}
                <td>{r.eval_loss?.toFixed(4) ?? "—"}</td><td>{(r.train_loss ?? r.loss)?.toFixed(4) ?? "—"}</td>
                <td>{r.reward?.toFixed(3) ?? "—"}</td>
                <td><StatusBadge status={r.status} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {sw.runs.length > 1 && (
        <div className="row" style={{ marginTop: 8 }}>
          <Link to={`/training/compare?ids=${sw.runs.map((r: any) => r.job_id).join(",")}`}><button className="sm">Comparar curvas</button></Link>
        </div>
      )}
    </div>
  );
}

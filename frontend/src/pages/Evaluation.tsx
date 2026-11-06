import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Play, Star, Trash2 } from "lucide-react";
import { api, type Collection, type Dataset, type EvalRun, fmtDate, type Model } from "../api";
import { AssistantText } from "../components/Markdownish";
import { Adv, Field, Help, SimpleNote, StatusBadge, useData, useDialog, useToast } from "../components/ui";
import { HelperPicker } from "../components/Helper";

type Q = { question: string; expected?: string; keywords?: string[] };

export default function Evaluation() {
  const [runs, reload] = useData<EvalRun[]>("/api/evals", 4000);
  const [sel, setSel] = useState<number | "new">("new");
  return (
    <>
      <div className="page-head">
        <div><h1>Evaluación</h1><div className="sub">Mide cómo responden tus modelos a una batería de preguntas</div></div>
        <button className="primary" onClick={() => setSel("new")}>Nueva evaluación</button>
      </div>
      <Help title="¿Cómo se evalúa y qué significan las métricas?">
        <p>Defines una batería de preguntas (a mano, importadas de un dataset o <b>generadas desde tus documentos</b>) y la app las pasa por los modelos que elijas, <b>con y sin RAG</b>, para responder a la pregunta clave: ¿qué combinación responde mejor?</p>
        <ul>
          <li><b>Corrección (juez)</b>: otro modelo puntúa de 1 a 10 comparando con tu respuesta de referencia.</li>
          <li><b>Similitud</b>: parecido semántico con la referencia · <b>Palabras clave</b>: si aparecen los datos que pediste.</li>
          <li><b>Fidelidad</b>: si lo que afirma está respaldado por el contexto recuperado (detecta alucinaciones).</li>
          <li><b>Relevancia y recall del contexto</b>: si la búsqueda trajo lo pertinente.</li>
          <li>Puntúa tú con estrellas y la app te dirá <b>cuánto coincide el juez contigo</b>.</li>
        </ul>
        <p style={{ marginTop: 8 }}>Con el motor vLLM cargado, las respuestas se generan por lotes y la evaluación va ~6× más rápido.</p>
      </Help>
      <div className="split" style={{ gridTemplateColumns: "260px minmax(0,1fr)" }}>
        <div className="card" style={{ padding: 8 }}>
          {!runs?.length && <div className="empty">Sin evaluaciones</div>}
          <table><tbody>
            {runs?.map((r) => (
              <tr key={r.id} className={`clickable ${sel === r.id ? "selected" : ""}`} onClick={() => setSel(r.id)}>
                <td><b>{r.name}</b><div className="sub" style={{ fontSize: 12 }}>{fmtDate(r.created_at)}</div></td>
                <td style={{ textAlign: "right" }}><StatusBadge status={r.status} /></td>
              </tr>
            ))}
          </tbody></table>
        </div>
        {sel === "new" ? <NewEval onCreated={(id) => { reload(); setSel(id); }} /> : <RunDetail key={sel} id={sel} onDelete={() => { reload(); setSel("new"); }} />}
      </div>
    </>
  );
}

function parseQuestions(text: string): Q[] {
  const t = text.trim();
  if (!t) return [];
  if (t.startsWith("[")) return JSON.parse(t);
  if (t.startsWith("{")) return t.split("\n").filter(Boolean).map((l) => JSON.parse(l));
  // Formato simple: pregunta | respuesta esperada | palabra1, palabra2
  return t.split("\n").filter((l) => l.trim()).map((l) => {
    const [question, expected, kw] = l.split("|").map((s) => s.trim());
    return { question, expected: expected || undefined, keywords: kw ? kw.split(",").map((s) => s.trim()).filter(Boolean) : [] };
  });
}

function NewEval({ onCreated }: { onCreated: (id: number) => void }) {
  const [models] = useData<Model[]>("/api/models");
  const [cols] = useData<Collection[]>("/api/rag/collections");
  const [datasets] = useData<Dataset[]>("/api/datasets");
  const [name, setName] = useState("eval-" + new Date().toISOString().slice(5, 16).replace("T", "-"));
  const [sel, setSel] = useState<number[]>([]);
  const [text, setText] = useState("");
  const [system, setSystem] = useState("");
  const [rag, setRag] = useState<number | "">("");
  const [ragModes, setRagModes] = useState<string[]>(["off", "on"]);
  const [ragK, setRagK] = useState(0); // 0 = top_k de la colección
  const [judge, setJudge] = useState<number | "">("");
  const [judgeHelper, setJudgeHelper] = useState<number | null>(null);
  const [nombreSet, setNombreSet] = useState("");
  const [sets, recargarSets] = useData<any[]>("/api/question-sets");
  const [temp, setTemp] = useState(0.2);
  const [maxTok, setMaxTok] = useState(512);
  const [noThink, setNoThink] = useState(true);
  const { notify, track } = useToast();

  let questions: Q[] = [];
  let parseErr = "";
  try { questions = parseQuestions(text); } catch (e: any) { parseErr = e.message; }

  const [genColl, setGenColl] = useState<number | "">("");
  const [genBusy, setGenBusy] = useState(false);
  const fromCollection = async () => {
    setGenBusy(true);
    try {
      const t = await api.post<any>("/api/evals/generate-questions", { collection_id: genColl, n_chunks: 10, per_chunk: 2 });
      track(t, (done) => {
        setText((done.result as any[]).map((q) => JSON.stringify({ question: q.question, expected: q.expected, keywords: q.keywords })).join("\n"));
        if (!rag) setRag(genColl);
        setGenBusy(false);
      });
    } catch (e: any) { notify(e.message, "err"); setGenBusy(false); }
  };

  const fromDataset = async (id: number) => {
    const d = datasets!.find((x) => x.id === id)!;
    const r = await api.get<any>(`/api/datasets/${id}/rows?limit=200`);
    const qs: Q[] = r.rows.map((row: any) => {
      if (d.format === "chat") {
        const u = [...row.messages].reverse().find((m: any) => m.role === "user");
        const a = [...row.messages].reverse().find((m: any) => m.role === "assistant");
        return { question: u?.content, expected: a?.content };
      }
      return { question: row.prompt, expected: row.chosen };
    });
    setText(qs.map((q) => JSON.stringify(q)).join("\n"));
  };

  const run = async () => {
    try {
      const r = await api.post<any>("/api/evals", {
        name, model_ids: sel, questions, system_prompt: system || null, rag_collection_id: rag || null,
        rag_modes: ragModes, rag_k: ragK || null,
        judge_model_id: judgeHelper ? null : (judge || null), judge_helper: judgeHelper, params: { temperature: temp, max_new_tokens: maxTok, enable_thinking: noThink ? false : null },
      });
      track(r.task);
      onCreated(r.run.id);
    } catch (e: any) { notify(e.message, "err"); }
  };

  return (
    <div className="card stack" style={{ gap: 14 }}>
      <h2 style={{ margin: 0 }}>Nueva evaluación</h2>
      <div className="grid g2">
        <Field label="Nombre"><input value={name} onChange={(e) => setName(e.target.value)} /></Field>
        <Field label="Importar preguntas de un dataset" hint="Usa el último turno de usuario como pregunta y la respuesta como referencia">
          <select value="" onChange={(e) => e.target.value && fromDataset(+e.target.value)}>
            <option value="">— elige —</option>
            {datasets?.filter((d) => d.format !== "text").map((d) => <option key={d.id} value={d.id}>{d.name} ({d.num_rows})</option>)}
          </select>
        </Field>
      </div>
      <div className="row source" style={{ padding: "8px 10px" }}>
        <span className="sub">¿Sin preguntas? Genéralas desde tus documentos con el modelo cargado:</span>
        <select value={genColl} onChange={(e) => setGenColl(e.target.value ? +e.target.value : "")} style={{ width: "auto" }}>
          <option value="">— colección —</option>{cols?.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
        </select>
        <button className="sm" disabled={!genColl || genBusy} onClick={fromCollection}>{genBusy ? "Generando…" : "Generar ~20 preguntas"}</button>
      </div>
      <Field label="Modelos">
        <div className="row" style={{ gap: 14 }}>
          {models?.filter((m) => m.kind !== "embedding").map((m) => (
            <label key={m.id} className="check"><input type="checkbox" checked={sel.includes(m.id)} onChange={(e) => setSel(e.target.checked ? [...sel, m.id] : sel.filter((x) => x !== m.id))} />{m.name}</label>
          ))}
        </div>
      </Field>
      <div className="card" style={{ padding: "8px 10px", marginBottom: 8 }}>
        <div className="row between">
          <b style={{ fontSize: 13 }}>Conjuntos de preguntas guardados</b>
          <button className="sm" disabled={!questions.length || !nombreSet.trim()}
            onClick={async () => {
              try {
                await api.post("/api/question-sets", { name: nombreSet.trim(), questions, source: "manual" });
                setNombreSet(""); recargarSets(); notify("Conjunto guardado");
              } catch (e: any) { notify(e.message, "err"); }
            }}>Guardar estas {questions.length}</button>
        </div>
        <div className="sub" style={{ marginTop: 4 }}>
          Escribe tus preguntas una vez y reutilízalas: es lo que convierte las comparaciones en
          decisiones, porque todos los modelos se miden con la misma vara.
        </div>
        <div className="row" style={{ gap: 6, marginTop: 6 }}>
          <select value="" onChange={async (e) => {
            if (!e.target.value) return;
            const qs = await api.get<any>(`/api/question-sets/${e.target.value}`);
            setText(qs.questions.map((q: any) => JSON.stringify(q)).join("\n"));
            notify(`${qs.questions.length} preguntas cargadas de «${qs.name}»`);
          }}>
            <option value="">Cargar un conjunto…</option>
            {sets?.map((q) => <option key={q.id} value={q.id}>{q.name} ({q.n}, {q.con_referencia} con referencia)</option>)}
          </select>
          <input placeholder="nombre para guardar las de abajo" value={nombreSet} onChange={(e) => setNombreSet(e.target.value)} />
        </div>
      </div>

      <Field label={`Preguntas (${questions.length})`} hint="Una por línea: pregunta | respuesta esperada | palabra clave 1, palabra clave 2 — o JSONL/JSON con {question, expected, keywords}">
        <textarea rows={8} className="mono" value={text} onChange={(e) => setText(e.target.value)}
          placeholder={"¿Cuál es la capital de Francia? | París es la capital de Francia. | París\n¿Cuánto es 7×8? | 56 | 56"} />
      </Field>
      {parseErr && <div className="error-box">{parseErr}</div>}
      <div className="grid g2">
        <Field label="System prompt (opcional)"><textarea rows={2} value={system} onChange={(e) => setSystem(e.target.value)} /></Field>
        <div className="stack">
          <Field label="Con RAG (opcional)">
            <select value={rag} onChange={(e) => setRag(e.target.value ? +e.target.value : "")}>
              <option value="">Sin RAG</option>{cols?.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
            </select>
          </Field>
          {rag && (
            <div className="row" style={{ gap: 14 }}>
              {[["off", "Evaluar sin RAG"], ["on", "Evaluar con RAG"]].map(([v, l]) => (
                <label key={v} className="check"><input type="checkbox" checked={ragModes.includes(v)}
                  onChange={(e) => setRagModes(e.target.checked ? [...ragModes, v] : ragModes.filter((x) => x !== v))} />{l}</label>
              ))}
              <label className="field" style={{ flex: 1, minWidth: 120 }}>Fragmentos: {ragK || "los de la colección"}
                <input type="range" min={0} max={10} value={ragK} onChange={(e) => setRagK(+e.target.value)} /></label>
            </div>
          )}
          <Field label="Modelo juez (LLM-as-judge, opcional)" hint="Puntúa 1–10 cada respuesta frente a la referencia. Usa un modelo capaz.">
            <select value={judge} disabled={!!judgeHelper} onChange={(e) => setJudge(e.target.value ? +e.target.value : "")}>
              <option value="">Sin juez</option>{models?.filter((m) => m.kind !== "embedding").map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
            </select>
          </Field>
          <HelperPicker value={judgeHelper} onChange={setJudgeHelper} label="…o un juez externo"
            hint="No ocupa VRAM ni obliga a descargar el modelo evaluado, así que la evaluación va más rápida" />
        </div>
      </div>
      <Adv>
        <div className="grid g2">
          <Field label={`Temperatura: ${temp}`}><input type="range" min={0} max={1.5} step={0.05} value={temp} onChange={(e) => setTemp(+e.target.value)} /></Field>
          <Field label={`Máx. tokens: ${maxTok}`}><input type="range" min={32} max={2048} step={32} value={maxTok} onChange={(e) => setMaxTok(+e.target.value)} /></Field>
        </div>
      </Adv>
      <SimpleNote>Se generan las respuestas con temperatura {temp} y hasta {maxTok} tokens.</SimpleNote>
      <div className="row"><label className="check" title="Evita que modelos como Qwen3 gasten tokens en <think>; más rápido y comparable"><input type="checkbox" checked={noThink} onChange={(e) => setNoThink(e.target.checked)} />Desactivar razonamiento</label><div className="spacer" />{rag && ragModes.length === 2 && <span className="sub">Matriz: {sel.length} modelo(s) × 2 modos = {sel.length * 2} variantes · {sel.length * 2 * questions.length} respuestas</span>}
        <button className="primary" disabled={!sel.length || !questions.length || !!parseErr || (!!rag && !ragModes.length)} onClick={run}><Play size={14} /> Ejecutar</button></div>
    </div>
  );
}

type Metric = { key: string; label: string; help: string; scale: number; rag?: boolean };
const METRICS: Metric[] = [
  { key: "judge_score", label: "Corrección (juez)", help: "El modelo juez puntúa 1–10 la respuesta frente a la referencia", scale: 10 },
  { key: "similarity", label: "Similitud", help: "Similitud semántica (embeddings) entre respuesta y referencia", scale: 1 },
  { key: "keyword_score", label: "Palabras clave", help: "Fracción de palabras clave presentes en la respuesta", scale: 1 },
  { key: "answer_relevance", label: "Relevancia", help: "Similitud entre pregunta y respuesta: ¿responde a lo que se preguntó?", scale: 1 },
  { key: "faithfulness", label: "Fidelidad", help: "El juez puntúa 1–10 si lo afirmado está respaldado por el contexto (bajo = alucina)", scale: 10, rag: true },
  { key: "context_relevance", label: "Relevancia contexto", help: "Score medio de los fragmentos recuperados: ¿el retriever trajo algo pertinente?", scale: 1, rag: true },
  { key: "context_recall", label: "Recall contexto", help: "¿Algún fragmento recuperado contiene la respuesta de referencia?", scale: 1, rag: true },
  { key: "rating", label: "Tu nota", help: "Tu valoración media (1–5 estrellas)", scale: 5 },
];
const COLORS = ["#7c8cff", "#3fb950", "#d29922", "#f85149", "#a371f7", "#39c5cf"];
const fmt = (v: number | null | undefined, m: Metric) => (v == null ? "—" : m.scale === 1 ? (m.key === "keyword_score" ? `${Math.round(v * 100)}%` : v.toFixed(3)) : v.toFixed(1));
/** Color de celda: rojo (0) → ámbar → verde (1). */
const heat = (x: number) => `color-mix(in srgb, ${x >= 0.5 ? "var(--ok)" : "var(--err)"} ${Math.round(Math.abs(x - 0.5) * 2 * 45 + 8)}%, transparent)`;

function Matrix({ summary, metric }: { summary: Record<string, any>; metric: Metric }) {
  const models = [...new Set(Object.values(summary).map((s: any) => s.model_name))];
  const cell = (model: string, rag: boolean) => Object.values(summary).find((s: any) => s.model_name === model && !!s.rag === rag) as any;
  const vals = models.flatMap((m) => [cell(m, false), cell(m, true)]).map((c) => c?.[metric.key]).filter((v) => v != null);
  const best = vals.length ? Math.max(...vals) : null;
  return (
    <table className="matrix">
      <thead><tr><th>Modelo</th><th style={{ textAlign: "center" }}>Sin RAG</th><th style={{ textAlign: "center" }}>Con RAG</th><th style={{ textAlign: "center" }}>Efecto del RAG</th></tr></thead>
      <tbody>
        {models.map((m) => {
          const a = cell(m, false)?.[metric.key], b = cell(m, true)?.[metric.key];
          const delta = a != null && b != null ? b - a : null;
          return (
            <tr key={m}>
              <td><b>{m}</b></td>
              {[a, b].map((v, i) => (
                <td key={i} style={{ textAlign: "center", background: v != null ? heat(v / metric.scale) : undefined, fontWeight: v === best ? 700 : 400 }}>
                  {fmt(v, metric)}{v === best && vals.length > 1 ? " ★" : ""}
                </td>
              ))}
              <td style={{ textAlign: "center", color: delta == null ? undefined : delta > 0 ? "var(--ok)" : delta < 0 ? "var(--err)" : undefined }}>
                {delta == null ? "—" : `${delta > 0 ? "+" : ""}${metric.scale === 1 ? delta.toFixed(3) : delta.toFixed(1)}`}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function RunDetail({ id, onDelete }: { id: number; onDelete: () => void }) {
  const [run, reload] = useData<EvalRun & { judge_agreement?: { n: number; spearman: number | null; mae: number | null; judge_bias?: number } }>(`/api/evals/${id}`, 3000);
  const { confirm } = useDialog();
  const [view, setView] = useState<"side" | "list">("side");
  const [metricKey, setMetricKey] = useState<string | null>(null);
  if (!run) return <div className="empty">Cargando…</div>;
  const summary = run.summary ?? {};
  const variants = Object.keys(summary);
  const present = METRICS.filter((m) => variants.some((v) => summary[v][m.key] != null));
  const isMatrix = variants.some((v) => summary[v].rag) && variants.some((v) => !summary[v].rag);
  const metric = present.find((m) => m.key === metricKey) ?? present.find((m) => !m.rag && m.key !== "rating") ?? present[0];
  const chart = present.filter((m) => m.key !== "rating" || variants.some((v) => summary[v].rating != null)).map((m) => ({
    metric: m.label, ...Object.fromEntries(variants.map((v) => [v, summary[v][m.key] == null ? null : summary[v][m.key] / m.scale])),
  }));
  const questions = [...new Set(run.results.map((r) => r.question))];
  const rate = async (index: number, rating: number) => { await api.post(`/api/evals/${id}/rate`, { index, rating }); reload(); };

  return (
    <div className="stack" style={{ gap: 16 }}>
      <div className="card">
        <div className="row between">
          <div><h2 style={{ marginBottom: 2 }}>{run.name} <StatusBadge status={run.status} /></h2>
            <div className="sub">{run.config.questions.length} preguntas · {run.config.model_ids.length} modelo(s){run.config.rag_collection_id ? ` · RAG: ${(run.config.rag_modes ?? ["on"]).map((m: string) => (m === "on" ? "con" : "sin")).join(" y ")}` : ""}{run.config.judge_model_id ? " · con juez" : ""}</div></div>
          <button className="danger" onClick={async () => (await confirm("¿Borrar esta evaluación?", "Se perderán las respuestas y tus valoraciones.", { label: "Borrar" })) && api.del(`/api/evals/${id}`).then(onDelete)}><Trash2 size={14} /></button>
        </div>
        {run.error && <div className="error-box" style={{ marginTop: 10 }}>{run.error}</div>}
      </div>

      {isMatrix && metric && (
        <div className="card">
          <div className="row between" style={{ marginBottom: 10 }}>
            <div><h3 style={{ margin: 0 }}>Matriz fine-tuning × RAG</h3><div className="sub">{metric.help}</div></div>
            <select value={metric.key} onChange={(e) => setMetricKey(e.target.value)} style={{ width: "auto" }}>
              {present.filter((m) => !m.rag).map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}
            </select>
          </div>
          <Matrix summary={summary} metric={metric} />
        </div>
      )}

      {variants.length > 0 && (
        <div className="card">
          <h3>Métricas por variante</h3>
          <div style={{ overflowX: "auto" }}>
            <table>
              <thead><tr><th>Variante</th>{present.map((m) => <th key={m.key} title={m.help} style={{ cursor: "help" }}>{m.label}{m.rag ? " ⓡ" : ""}</th>)}<th>Latencia</th></tr></thead>
              <tbody>{variants.map((v) => (
                <tr key={v}><td><b>{summary[v].model_name}</b>{summary[v].rag && <span className="badge accent" style={{ marginLeft: 6 }}>RAG</span>}</td>
                  {present.map((m) => <td key={m.key}>{fmt(summary[v][m.key], m)}</td>)}<td>{summary[v].latency_s}s</td></tr>
              ))}</tbody>
            </table>
          </div>
          {run.judge_agreement && run.config.judge_model_id && (
            <div className="source" style={{ marginTop: 10, padding: "8px 12px" }}>
              <b>¿Coincide el juez contigo?</b>{" "}
              {run.judge_agreement.spearman == null
                ? <span className="sub">Valora con estrellas al menos 3 respuestas para calibrar el juez ({run.judge_agreement.n}/3).</span>
                : <span>correlación <b>{run.judge_agreement.spearman}</b> · error medio <b>{run.judge_agreement.mae}</b> estrellas
                  {run.judge_agreement.judge_bias ? ` · el juez puntúa ${run.judge_agreement.judge_bias > 0 ? "más alto" : "más bajo"} que tú (${run.judge_agreement.judge_bias > 0 ? "+" : ""}${run.judge_agreement.judge_bias})` : ""}
                  <span className="sub"> · {run.judge_agreement.spearman >= 0.7 ? "fiable" : run.judge_agreement.spearman >= 0.4 ? "orientativo" : "poco fiable: usa un juez más grande"} ({run.judge_agreement.n} valoraciones)</span></span>}
            </div>
          )}
          <div className="sub" style={{ marginTop: 6, fontSize: 12 }}>ⓡ métricas específicas de RAG · pasa el ratón por cada columna para ver qué mide · métricas del juez sólo si se eligió modelo juez</div>
          {chart.length > 0 && (
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={chart} margin={{ top: 16 }}>
                <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                <XAxis dataKey="metric" stroke="var(--muted)" fontSize={11} interval={0} />
                <YAxis domain={[0, 1]} stroke="var(--muted)" fontSize={12} />
                <Tooltip contentStyle={{ background: "var(--panel)", border: "1px solid var(--border)" }} formatter={(v: any) => (v == null ? "—" : Number(v).toFixed(3))} />
                <Legend />
                {variants.map((v, i) => <Bar key={v} dataKey={v} fill={COLORS[i % COLORS.length]} radius={[4, 4, 0, 0]} />)}
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      )}

      {run.results.length > 0 && (
        <div className="card">
          <div className="row between" style={{ marginBottom: 10 }}>
            <h3 style={{ margin: 0 }}>Respuestas</h3>
            <div className="tabs" style={{ margin: 0, border: 0 }}>
              <button className={`tab ${view === "side" ? "active" : ""}`} onClick={() => setView("side")}>Lado a lado</button>
              <button className={`tab ${view === "list" ? "active" : ""}`} onClick={() => setView("list")}>Lista</button>
            </div>
          </div>
          <div className="stack" style={{ gap: 18 }}>
            {questions.map((q) => {
              const rs = run.results.map((r, i) => ({ ...r, i })).filter((r) => r.question === q);
              return (
                <div key={q} style={{ borderBottom: "1px solid var(--border)", paddingBottom: 14 }}>
                  <div><b>{q}</b></div>
                  {rs[0].expected && <div className="sub" style={{ margin: "4px 0 8px" }}>Referencia: {rs[0].expected}</div>}
                  <div className={view === "side" ? "grid" : "stack"} style={{ gridTemplateColumns: view === "side" ? `repeat(${Math.min(rs.length, 4)}, minmax(0,1fr))` : undefined }}>
                    {rs.map((r) => (
                      <div key={r.i} className="source" style={{ padding: 10 }}>
                        <div className="row between" style={{ alignItems: "flex-start" }}>
                          <b>{r.model_name}{r.rag && <span className="badge accent" style={{ marginLeft: 6 }}>RAG</span>}
                            {r.rag_mode && r.rag_mode !== "rag" && <span className="badge warn" style={{ marginLeft: 4 }} title="Ningún fragmento superó el umbral: respondió sin contexto">sin contexto</span>}</b>
                          <div className="row" style={{ gap: 4, justifyContent: "flex-end" }}>
                            {r.keyword_score != null && <span className={`badge ${r.keyword_score === 1 ? "ok" : r.keyword_score === 0 ? "err" : "warn"}`} title="Palabras clave">{Math.round(r.keyword_score * 100)}%</span>}
                            {r.similarity != null && <span className="badge accent" title="Similitud con la referencia">{r.similarity.toFixed(2)}</span>}
                            {r.judge_score != null && <span className="badge warn" title={r.judge_reason}>juez {r.judge_score}</span>}
                            {r.faithfulness != null && <span className={`badge ${r.faithfulness >= 7 ? "ok" : r.faithfulness >= 4 ? "warn" : "err"}`} title={r.faithfulness_reason}>fiel {r.faithfulness}</span>}
                          </div></div>
                        <div style={{ whiteSpace: "pre-wrap", margin: "6px 0", fontSize: 13 }}><AssistantText text={r.raw_answer ?? r.answer} /></div>
                        {r.sources?.length > 0 && (
                          <details className="sub" style={{ fontSize: 12 }}>
                            <summary style={{ cursor: "pointer" }}>{r.sources.length} fragmentos · recall {r.context_recall?.toFixed(2) ?? "—"}</summary>
                            {r.sources.map((s: any, j: number) => (
                              <div key={j} style={{ marginTop: 6, whiteSpace: "pre-wrap" }}><b>[{j + 1}] {s.source} · {s.score}</b><br />{s.text}</div>
                            ))}
                          </details>
                        )}
                        <div className="row" style={{ gap: 0 }}>
                          {[1, 2, 3, 4, 5].map((n) => (
                            <button key={n} className="ghost sm" onClick={() => rate(r.i, n)} title={`${n}/5`}>
                              <Star size={13} fill={r.rating >= n ? "var(--warn)" : "none"} color={r.rating >= n ? "var(--warn)" : "var(--muted)"} />
                            </button>
                          ))}
                          <span className="spacer" /><span className="sub" style={{ fontSize: 12 }}>{r.latency_s}s</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

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
        <div><h1>Evaluation</h1><div className="sub">Measure how your models answer a set of questions</div></div>
        <button className="primary" onClick={() => setSel("new")}>New evaluation</button>
      </div>
      <Help title="How does evaluation work, and what do the metrics mean?">
        <p>You define a set of questions (by hand, imported from a dataset or <b>generated from your documents</b>) and the app runs them through the models you choose, <b>with and without RAG</b>, to answer the key question: which combination answers best?</p>
        <ul>
          <li><b>Correctness (judge)</b>: another model scores 1 to 10 against your reference answer.</li>
          <li><b>Similarity</b>: semantic closeness to the reference · <b>Keywords</b>: whether the facts you asked for appear.</li>
          <li><b>Faithfulness</b>: whether what it claims is backed by the retrieved context (catches hallucinations).</li>
          <li><b>Context relevance and recall</b>: whether retrieval brought back what was needed.</li>
          <li>Rate answers yourself with stars and the app will tell you <b>how much the judge agrees with you</b>.</li>
        </ul>
        <p style={{ marginTop: 8 }}>With the vLLM engine loaded, answers are generated in batches and evaluation runs ~6× faster.</p>
      </Help>
      <div className="split" style={{ gridTemplateColumns: "260px minmax(0,1fr)" }}>
        <div className="card" style={{ padding: 8 }}>
          {!runs?.length && <div className="empty">No evaluations</div>}
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
  // Simple format: question | expected answer | keyword1, keyword2
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
  const [ragK, setRagK] = useState(0); // 0 = the collection's top_k
  const [judge, setJudge] = useState<number | "">("");
  const [judgeHelper, setJudgeHelper] = useState<number | null>(null);
  const [saveAs, setSaveAs] = useState("");
  const [sets, reloadSets] = useData<any[]>("/api/question-sets");
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
      <h2 style={{ margin: 0 }}>New evaluation</h2>
      <div className="grid g2">
        <Field label="Name"><input value={name} onChange={(e) => setName(e.target.value)} /></Field>
        <Field label="Import questions from a dataset" hint="Uses the last user turn as the question and the answer as the reference">
          <select value="" onChange={(e) => e.target.value && fromDataset(+e.target.value)}>
            <option value="">— choose —</option>
            {datasets?.filter((d) => d.format !== "text").map((d) => <option key={d.id} value={d.id}>{d.name} ({d.num_rows})</option>)}
          </select>
        </Field>
      </div>
      <div className="row source" style={{ padding: "8px 10px" }}>
        <span className="sub">No questions? Generate them from your documents with the loaded model:</span>
        <select value={genColl} onChange={(e) => setGenColl(e.target.value ? +e.target.value : "")} style={{ width: "auto" }}>
          <option value="">— collection —</option>{cols?.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
        </select>
        <button className="sm" disabled={!genColl || genBusy} onClick={fromCollection}>{genBusy ? "Generating…" : "Generate ~20 questions"}</button>
      </div>
      <Field label="Models">
        <div className="row" style={{ gap: 14 }}>
          {models?.filter((m) => m.kind !== "embedding").map((m) => (
            <label key={m.id} className="check"><input type="checkbox" checked={sel.includes(m.id)} onChange={(e) => setSel(e.target.checked ? [...sel, m.id] : sel.filter((x) => x !== m.id))} />{m.name}</label>
          ))}
        </div>
      </Field>
      <div className="card" style={{ padding: "8px 10px", marginBottom: 8 }}>
        <div className="row between">
          <b style={{ fontSize: 13 }}>Saved question sets</b>
          <button className="sm" disabled={!questions.length || !saveAs.trim()}
            onClick={async () => {
              try {
                await api.post("/api/question-sets", { name: saveAs.trim(), questions, source: "manual" });
                setSaveAs(""); reloadSets(); notify("Set saved");
              } catch (e: any) { notify(e.message, "err"); }
            }}>Save these {questions.length}</button>
        </div>
        <div className="sub" style={{ marginTop: 4 }}>
          Write your questions once and reuse them: that's what turns comparisons into
          decisions, because every model is measured against the same yardstick.
        </div>
        <div className="row" style={{ gap: 6, marginTop: 6 }}>
          <select value="" onChange={async (e) => {
            if (!e.target.value) return;
            const qs = await api.get<any>(`/api/question-sets/${e.target.value}`);
            setText(qs.questions.map((q: any) => JSON.stringify(q)).join("\n"));
            notify(`${qs.questions.length} questions loaded from “${qs.name}”`);
          }}>
            <option value="">Load a set…</option>
            {sets?.map((q) => <option key={q.id} value={q.id}>{q.name} ({q.n}, {q.con_referencia} with a reference)</option>)}
          </select>
          <input placeholder="name to save the ones below" value={saveAs} onChange={(e) => setSaveAs(e.target.value)} />
        </div>
      </div>

      <Field label={`Questions (${questions.length})`} hint="One per line: question | expected answer | keyword 1, keyword 2 — or JSONL/JSON with {question, expected, keywords}">
        <textarea rows={8} className="mono" value={text} onChange={(e) => setText(e.target.value)}
          placeholder={"What is the capital of France? | Paris is the capital of France. | Paris\nWhat is 7×8? | 56 | 56"} />
      </Field>
      {parseErr && <div className="error-box">{parseErr}</div>}
      <div className="grid g2">
        <Field label="System prompt (optional)"><textarea rows={2} value={system} onChange={(e) => setSystem(e.target.value)} /></Field>
        <div className="stack">
          <Field label="With RAG (optional)">
            <select value={rag} onChange={(e) => setRag(e.target.value ? +e.target.value : "")}>
              <option value="">Sin RAG</option>{cols?.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
            </select>
          </Field>
          {rag && (
            <div className="row" style={{ gap: 14 }}>
              {[["off", "Evaluate without RAG"], ["on", "Evaluate with RAG"]].map(([v, l]) => (
                <label key={v} className="check"><input type="checkbox" checked={ragModes.includes(v)}
                  onChange={(e) => setRagModes(e.target.checked ? [...ragModes, v] : ragModes.filter((x) => x !== v))} />{l}</label>
              ))}
              <label className="field" style={{ flex: 1, minWidth: 120 }}>Chunks: {ragK || "the collection's default"}
                <input type="range" min={0} max={10} value={ragK} onChange={(e) => setRagK(+e.target.value)} /></label>
            </div>
          )}
          <Field label="Judge model (LLM-as-judge, optional)" hint="Scores each answer 1–10 against the reference. Use a capable model.">
            <select value={judge} disabled={!!judgeHelper} onChange={(e) => setJudge(e.target.value ? +e.target.value : "")}>
              <option value="">No judge</option>{models?.filter((m) => m.kind !== "embedding").map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
            </select>
          </Field>
          <HelperPicker value={judgeHelper} onChange={setJudgeHelper} label="…or an external judge"
            hint="It uses no VRAM and doesn't force the evaluated model to be unloaded, so evaluation runs faster" />
        </div>
      </div>
      <Adv>
        <div className="grid g2">
          <Field label={`Temperature: ${temp}`}><input type="range" min={0} max={1.5} step={0.05} value={temp} onChange={(e) => setTemp(+e.target.value)} /></Field>
          <Field label={`Max tokens: ${maxTok}`}><input type="range" min={32} max={2048} step={32} value={maxTok} onChange={(e) => setMaxTok(+e.target.value)} /></Field>
        </div>
      </Adv>
      <SimpleNote>Answers are generated at temperature {temp} with up to {maxTok} tokens.</SimpleNote>
      <div className="row"><label className="check" title="Keeps models like Qwen3 from spending tokens on <think>; faster and comparable"><input type="checkbox" checked={noThink} onChange={(e) => setNoThink(e.target.checked)} />Turn off reasoning</label><div className="spacer" />{rag && ragModes.length === 2 && <span className="sub">Matrix: {sel.length} model(s) × 2 modes = {sel.length * 2} variants · {sel.length * 2 * questions.length} answers</span>}
        <button className="primary" disabled={!sel.length || !questions.length || !!parseErr || (!!rag && !ragModes.length)} onClick={run}><Play size={14} /> Run</button></div>
    </div>
  );
}

type Metric = { key: string; label: string; help: string; scale: number; rag?: boolean };
const METRICS: Metric[] = [
  { key: "judge_score", label: "Correctness (judge)", help: "The judge model scores the answer 1–10 against the reference", scale: 10 },
  { key: "similarity", label: "Similarity", help: "Semantic similarity (embeddings) between the answer and the reference", scale: 1 },
  { key: "keyword_score", label: "Keywords", help: "Share of keywords present in the answer", scale: 1 },
  { key: "answer_relevance", label: "Relevance", help: "Similarity between question and answer: does it answer what was asked?", scale: 1 },
  { key: "faithfulness", label: "Faithfulness", help: "The judge scores 1–10 whether the claims are backed by the context (low = hallucinates)", scale: 10, rag: true },
  { key: "context_relevance", label: "Context relevance", help: "Mean score of the retrieved chunks: did the retriever bring back something relevant?", scale: 1, rag: true },
  { key: "context_recall", label: "Context recall", help: "Does any retrieved chunk contain the reference answer?", scale: 1, rag: true },
  { key: "rating", label: "Your rating", help: "Your average rating (1–5 stars)", scale: 5 },
];
const COLORS = ["#7c8cff", "#3fb950", "#d29922", "#f85149", "#a371f7", "#39c5cf"];
const fmt = (v: number | null | undefined, m: Metric) => (v == null ? "—" : m.scale === 1 ? (m.key === "keyword_score" ? `${Math.round(v * 100)}%` : v.toFixed(3)) : v.toFixed(1));
/** Cell colour: red (0) → amber → green (1). */
const heat = (x: number) => `color-mix(in srgb, ${x >= 0.5 ? "var(--ok)" : "var(--err)"} ${Math.round(Math.abs(x - 0.5) * 2 * 45 + 8)}%, transparent)`;

function Matrix({ summary, metric }: { summary: Record<string, any>; metric: Metric }) {
  const models = [...new Set(Object.values(summary).map((s: any) => s.model_name))];
  const cell = (model: string, rag: boolean) => Object.values(summary).find((s: any) => s.model_name === model && !!s.rag === rag) as any;
  const vals = models.flatMap((m) => [cell(m, false), cell(m, true)]).map((c) => c?.[metric.key]).filter((v) => v != null);
  const best = vals.length ? Math.max(...vals) : null;
  return (
    <table className="matrix">
      <thead><tr><th>Model</th><th style={{ textAlign: "center" }}>No RAG</th><th style={{ textAlign: "center" }}>With RAG</th><th style={{ textAlign: "center" }}>RAG effect</th></tr></thead>
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
  if (!run) return <div className="empty">Loading…</div>;
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
            <div className="sub">{run.config.questions.length} questions · {run.config.model_ids.length} model(s){run.config.rag_collection_id ? ` · RAG: ${(run.config.rag_modes ?? ["on"]).map((m: string) => (m === "on" ? "with" : "without")).join(" and ")}` : ""}{run.config.judge_model_id ? " · with a judge" : ""}</div></div>
          <button className="danger" onClick={async () => (await confirm("Delete this evaluation?", "The answers and your ratings will be lost.", { label: "Delete" })) && api.del(`/api/evals/${id}`).then(onDelete)}><Trash2 size={14} /></button>
        </div>
        {run.error && <div className="error-box" style={{ marginTop: 10 }}>{run.error}</div>}
      </div>

      {isMatrix && metric && (
        <div className="card">
          <div className="row between" style={{ marginBottom: 10 }}>
            <div><h3 style={{ margin: 0 }}>Fine-tuning × RAG matrix</h3><div className="sub">{metric.help}</div></div>
            <select value={metric.key} onChange={(e) => setMetricKey(e.target.value)} style={{ width: "auto" }}>
              {present.filter((m) => !m.rag).map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}
            </select>
          </div>
          <Matrix summary={summary} metric={metric} />
        </div>
      )}

      {variants.length > 0 && (
        <div className="card">
          <h3>Metrics per variant</h3>
          <div style={{ overflowX: "auto" }}>
            <table>
              <thead><tr><th>Variant</th>{present.map((m) => <th key={m.key} title={m.help} style={{ cursor: "help" }}>{m.label}{m.rag ? " ⓡ" : ""}</th>)}<th>Latency</th></tr></thead>
              <tbody>{variants.map((v) => (
                <tr key={v}><td><b>{summary[v].model_name}</b>{summary[v].rag && <span className="badge accent" style={{ marginLeft: 6 }}>RAG</span>}</td>
                  {present.map((m) => <td key={m.key}>{fmt(summary[v][m.key], m)}</td>)}<td>{summary[v].latency_s}s</td></tr>
              ))}</tbody>
            </table>
          </div>
          {run.judge_agreement && run.config.judge_model_id && (
            <div className="source" style={{ marginTop: 10, padding: "8px 12px" }}>
              <b>Does the judge agree with you?</b>{" "}
              {run.judge_agreement.spearman == null
                ? <span className="sub">Rate at least 3 answers with stars to calibrate the judge ({run.judge_agreement.n}/3).</span>
                : <span>correlation <b>{run.judge_agreement.spearman}</b> · mean error <b>{run.judge_agreement.mae}</b> stars
                  {run.judge_agreement.judge_bias ? ` · the judge scores ${run.judge_agreement.judge_bias > 0 ? "higher" : "lower"} than you (${run.judge_agreement.judge_bias > 0 ? "+" : ""}${run.judge_agreement.judge_bias})` : ""}
                  <span className="sub"> · {run.judge_agreement.spearman >= 0.7 ? "reliable" : run.judge_agreement.spearman >= 0.4 ? "indicative" : "unreliable: use a bigger judge"} ({run.judge_agreement.n} ratings)</span></span>}
            </div>
          )}
          <div className="sub" style={{ marginTop: 6, fontSize: 12 }}>ⓡ RAG-specific metrics · hover over a column to see what it measures · judge metrics only if a judge model was chosen</div>
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
            <h3 style={{ margin: 0 }}>Answers</h3>
            <div className="tabs" style={{ margin: 0, border: 0 }}>
              <button className={`tab ${view === "side" ? "active" : ""}`} onClick={() => setView("side")}>Side by side</button>
              <button className={`tab ${view === "list" ? "active" : ""}`} onClick={() => setView("list")}>List</button>
            </div>
          </div>
          <div className="stack" style={{ gap: 18 }}>
            {questions.map((q) => {
              const rs = run.results.map((r, i) => ({ ...r, i })).filter((r) => r.question === q);
              return (
                <div key={q} style={{ borderBottom: "1px solid var(--border)", paddingBottom: 14 }}>
                  <div><b>{q}</b></div>
                  {rs[0].expected && <div className="sub" style={{ margin: "4px 0 8px" }}>Reference: {rs[0].expected}</div>}
                  <div className={view === "side" ? "grid" : "stack"} style={{ gridTemplateColumns: view === "side" ? `repeat(${Math.min(rs.length, 4)}, minmax(0,1fr))` : undefined }}>
                    {rs.map((r) => (
                      <div key={r.i} className="source" style={{ padding: 10 }}>
                        <div className="row between" style={{ alignItems: "flex-start" }}>
                          <b>{r.model_name}{r.rag && <span className="badge accent" style={{ marginLeft: 6 }}>RAG</span>}
                            {r.rag_mode && r.rag_mode !== "rag" && <span className="badge warn" style={{ marginLeft: 4 }} title="No chunk passed the threshold: answered without context">no context</span>}</b>
                          <div className="row" style={{ gap: 4, justifyContent: "flex-end" }}>
                            {r.keyword_score != null && <span className={`badge ${r.keyword_score === 1 ? "ok" : r.keyword_score === 0 ? "err" : "warn"}`} title="Keywords">{Math.round(r.keyword_score * 100)}%</span>}
                            {r.similarity != null && <span className="badge accent" title="Similarity to the reference">{r.similarity.toFixed(2)}</span>}
                            {r.judge_score != null && <span className="badge warn" title={r.judge_reason}>judge {r.judge_score}</span>}
                            {r.faithfulness != null && <span className={`badge ${r.faithfulness >= 7 ? "ok" : r.faithfulness >= 4 ? "warn" : "err"}`} title={r.faithfulness_reason}>faithful {r.faithfulness}</span>}
                          </div></div>
                        <div style={{ whiteSpace: "pre-wrap", margin: "6px 0", fontSize: 13 }}><AssistantText text={r.raw_answer ?? r.answer} /></div>
                        {r.sources?.length > 0 && (
                          <details className="sub" style={{ fontSize: 12 }}>
                            <summary style={{ cursor: "pointer" }}>{r.sources.length} chunks · recall {r.context_recall?.toFixed(2) ?? "—"}</summary>
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

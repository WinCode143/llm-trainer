import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Play } from "lucide-react";
import { Link } from "react-router-dom";
import { api, type Collection, type Dataset, type Job, type Model } from "../api";
import { Adv, Field, SimpleNote, StatusBadge, useData, useMode, useToast } from "../components/ui";

/** Fine-tuning the RAG embedding model. */
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
      <h2 style={{ margin: 0 }}>Fine-tune the RAG embeddings</h2>
      <div className="sub">Teaches the embedding model your documents' vocabulary so search finds things better. It trains on the <b>questions generated per chunk</b> of a collection (RAG → Documents → “Generate questions”) or on a dataset's conversations. Recall@k and MRR are measured before and after on a held-out 15%.</div>
      <div className="source sub" style={{ fontSize: 12.5, padding: "8px 10px" }}>⚠ That held-out 15% has the same style as the training pairs: improving there doesn't guarantee improving on real questions. After re-indexing, check it in RAG → Benchmark with hand-written questions. With synthetic questions from a small model, a gentle fine-tune is best (1 epoch, lr 1e-5).</div>
      <div className="grid g2">
        <Field label="Base model"><select value={f.base_model} onChange={(e) => setF({ ...f, base_model: e.target.value })}>{embedModels?.filter((m) => !m.tuned).map((m) => <option key={m.name} value={m.name}>{m.name}</option>)}</select></Field>
        <Adv><Field label="Name"><input value={f.name} onChange={(e) => setF({ ...f, name: e.target.value })} placeholder="auto" /></Field></Adv>
        <Field label="Pairs from an enriched collection">
          <select value={f.collection_id} onChange={(e) => setF({ ...f, collection_id: e.target.value ? +e.target.value : "" })}>
            <option value="">—</option>{cols?.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
          </select>
        </Field>
        <Field label="…and/or from a chat dataset">
          <select value={f.dataset_id} onChange={(e) => setF({ ...f, dataset_id: e.target.value ? +e.target.value : "" })}>
            <option value="">—</option>{datasets?.filter((d) => d.format === "chat").map((d) => <option key={d.id} value={d.id}>{d.name} ({d.num_rows})</option>)}
          </select>
        </Field>
        <Adv>
          <Field label="Epochs"><input type="number" value={f.epochs} onChange={(e) => setF({ ...f, epochs: +e.target.value })} /></Field>
          <Field label="Learning rate"><input type="number" step="0.000001" value={f.learning_rate} onChange={(e) => setF({ ...f, learning_rate: +e.target.value })} /></Field>
          <Field label="Batch" hint="Bigger = more negatives per batch (better), more VRAM"><input type="number" value={f.batch_size} onChange={(e) => setF({ ...f, batch_size: +e.target.value })} /></Field>
        </Adv>
      </div>
      <SimpleNote>Gentle fine-tune: {f.epochs} epoch at learning rate {f.learning_rate}. We measured that training longer improves the held-out metric but makes real questions worse.</SimpleNote>
      <div className="row"><div className="spacer" /><button className="primary" disabled={!f.collection_id && !f.dataset_id} onClick={go}><Play size={14} /> Train embeddings</button></div>
    </div>
  );
}

/** Standard benchmarks with lm-evaluation-harness. */
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
      <h2 style={{ margin: 0 }}>Standard benchmarks</h2>
      <div className="sub">Measures general skills with lm-evaluation-harness. Run it on the base model and on the fine-tuned one to catch <b>catastrophic forgetting</b> (fine-tuning shouldn't lower these numbers much). Then compare them by ticking both in the jobs list.</div>
      <div className="grid g2">
        <Field label="Model"><select value={modelId} onChange={(e) => setModelId(e.target.value ? +e.target.value : "")}>
          <option value="">— choose —</option>{models?.filter((m) => ["hf", "merged", "lora"].includes(m.kind)).map((m) => <option key={m.id} value={m.id}>{m.name} [{m.kind}]</option>)}
        </select></Field>
        <Field label="Examples per task" hint="200 gives a quick estimate (±3%); 0 = all (slow)"><input type="number" value={limit} onChange={(e) => setLimit(+e.target.value)} /></Field>
      </div>
      <div className="grid g3">
        {tasksList?.map((t) => (
          <label key={t.task} className="check source" style={{ padding: "6px 10px" }}>
            <input type="checkbox" checked={sel.includes(t.task)} onChange={(e) => setSel(e.target.checked ? [...sel, t.task] : sel.filter((x) => x !== t.task))} />
            <span><b>{t.label}</b><div className="sub" style={{ fontSize: 12 }}>{t.desc}</div></span>
          </label>
        ))}
      </div>
      <div className="row"><div className="spacer" /><button className="primary" disabled={!modelId || !sel.length} onClick={go}><Play size={14} /> Run benchmarks</button></div>
    </div>
  );
}


/** The parameters you can sweep, ordered by how much they really change the result.
  *  “impact” decides which ones show in simple mode: sweeping ten at once gives noise, not answers. */
const SWEEP_PARAMS: { key: string; label: string; example: string; impact: "high" | "medium" | "low";
                      what: string; reco: string }[] = [
  { key: "learning_rate", label: "Learning rate", example: "1e-4, 2e-4, 5e-4", impact: "high",
    what: "How far the model moves at each step. Too high and it becomes unstable (the loss rises or jumps); too low and it barely learns. It's by far the one that changes the result the most.",
    reco: "With LoRA, between 1e-4 and 5e-4. With DPO/ORPO, much lower: 5e-6. If you sweep only one, make it this one." },
  { key: "lora_r", label: "LoRA rank (r)", example: "8, 16, 32", impact: "high",
    what: "The adapter's size: how many new parameters are trained. A higher rank = more capacity to learn nuances, but more risk of memorizing instead of learning, and a bigger file.",
    reco: "8 for small datasets (under 500 examples), 16 for the usual case, 32–64 if you have thousands of examples." },
  { key: "epochs", label: "Epochs", example: "1, 2, 3", impact: "high",
    what: "How many times the whole dataset is seen. It's the number one cause of overfitting: past a certain point the model stops learning and starts memorizing your examples.",
    reco: "1–3. With small datasets, 1 or 2. Only has an effect if “Max steps” is 0." },
  { key: "lora_alpha", label: "LoRA alpha", example: "16, 32, 64", impact: "medium",
    what: "Scales the adapter's strength. What really matters is the alpha/rank ratio, not the number on its own.",
    reco: "The convention is alpha = 2 × rank (rank 16 → alpha 32). Sweeping it separately from the rank multiplies combinations without adding much." },
  { key: "batch_size", label: "Batch", example: "2, 4", impact: "medium",
    what: "How many examples are processed at once. Bigger stabilizes training, but it's the first thing that runs you out of VRAM.",
    reco: "2 with 7B+ models on 24 GB; 4–8 with small models. If you get a memory error, lower it and raise accumulation." },
  { key: "grad_accum", label: "Accumulation", example: "2, 4, 8", impact: "medium",
    what: "Accumulates several batches before updating. It simulates a big batch without using more VRAM: the effective batch is batch × accumulation.",
    reco: "4 by default. What matters is the product: batch 2 × accumulation 4 equals batch 8." },
  { key: "max_seq_len", label: "Max length", example: "1024, 2048", impact: "medium",
    what: "How many tokens of each example are used. If your texts are longer, they're cut and you lose that part.",
    reco: "2048 works for almost everything. Only raise it if your examples get cut; it eats VRAM fast." },
  { key: "lora_dropout", label: "LoRA dropout", example: "0, 0.05, 0.1", impact: "low",
    what: "Randomly switches off part of the adapter during training so it doesn't memorize.",
    reco: "Usually 0. Raise it to 0.05–0.1 only if you see overfitting. With Unsloth, leave it at 0 (it's faster)." },
  { key: "warmup_ratio", label: "Warmup", example: "0, 0.03, 0.1", impact: "low",
    what: "Starts with a low learning rate and raises it gradually over the first steps, so the model isn't destabilized at the start.",
    reco: "0.03 and forget about it. It rarely changes anything." },
  { key: "weight_decay", label: "Weight decay", example: "0, 0.01", impact: "low",
    what: "Penalizes weights for growing large; another way to avoid overfitting.",
    reco: "0 or 0.01. With LoRA you barely notice it." },
  { key: "neftune_alpha", label: "NEFTune α", example: "0, 5", impact: "low",
    what: "Adds a little noise to the embeddings during training. Sometimes improves the quality of long answers.",
    reco: "0 (off) or 5. It's a trick, not a fundamental setting." },
];

/** Hyperparameter sweep: one combination per job, queued. */
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
  const [expanded, setExpanded] = useState<string | null>(null);
  const { simple } = useMode();
  const { notify } = useToast();

  useEffect(() => { if (sys?.unsloth) setBase((b) => ({ ...b, backend: "unsloth" })); }, [sys?.unsloth]);
  const parsed = Object.fromEntries(Object.entries(grid).map(([k, v]) => [k, v.split(",").map((x) => parseFloat(x.trim())).filter((x) => !isNaN(x))]));
  const active = Object.values(parsed).filter((v) => v.length);
  // With no parameter ticked there's no sweep to launch (it used to count 1 and the server rejected it)
  const combos = active.length ? active.reduce((a, v) => a * v.length, 1) : 0;
  const runs = Math.min(combos, maxRuns);

  const go = async () => {
    try {
      const r = await api.post<any>("/api/jobs/sweep", { base: { ...base, base_model_id: null }, grid: parsed, strategy, max_runs: maxRuns });
      notify(`Sweep launched: ${r.runs} training runs queued`);
      reloadSweeps();
    } catch (e: any) { notify(e.message, "err"); }
  };

  return (
    <div className="stack" style={{ gap: 16 }}>
      <div className="card stack" style={{ gap: 14 }}>
        <h2 style={{ margin: 0 }}>Hyperparameter sweep</h2>
        <div className="sub">Launches one combination per training run, queued. They're ranked by <b>eval_loss</b>, so hold out part of the data for evaluation. Start with short runs (few steps) to explore, then repeat the winner in full.</div>
        <div className="grid g3">
          <Field label="Base name"><input value={base.name} onChange={(e) => setBase({ ...base, name: e.target.value })} /></Field>
          <Field label="Base model"><input value={base.base_model} onChange={(e) => setBase({ ...base, base_model: e.target.value })} /></Field>
          <Field label="Dataset">
            <select value={base.dataset_id} onChange={(e) => setBase({ ...base, dataset_id: +e.target.value })}>
              <option value={0}>— choose —</option>{datasets?.map((d) => <option key={d.id} value={d.id}>{d.name} ({d.num_rows})</option>)}
            </select>
          </Field>
          <Field label="Method">
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
          <Field label="Steps per run" hint="Short, to explore"><input type="number" value={base.max_steps} onChange={(e) => setBase({ ...base, max_steps: +e.target.value })} /></Field>
          <Field label={`Eval split: ${base.eval_ratio}`}><input type="range" min={0} max={0.4} step={0.05} value={base.eval_ratio} onChange={(e) => setBase({ ...base, eval_ratio: +e.target.value })} /></Field>
          <Field label="Strategy" hint="Random = samples combinations when there are many">
            <select value={strategy} onChange={(e) => setStrategy(e.target.value)}><option value="grid">full grid</option><option value="random">random</option></select>
          </Field>
          <Field label="Max runs"><input type="number" value={maxRuns} onChange={(e) => setMaxRuns(+e.target.value)} /></Field>
        </div>
        <Field label="Parameters to sweep"
          hint="Click any name to see what it does and which values are recommended.">
          <div className="stack" style={{ gap: 2 }}>
            {SWEEP_PARAMS.filter((sp) => !simple || sp.impact === "high").map((sp) => (
              <div key={sp.key}>
                <div className="row" style={{ flexWrap: "nowrap", gap: 4 }}>
                  <input type="checkbox" checked={sp.key in grid} onChange={(e) => {
                    const g = { ...grid }; if (e.target.checked) g[sp.key] = sp.example; else delete g[sp.key]; setGrid(g);
                  }} />
                  <button className="param-name" title="What it does"
                    onClick={() => setExpanded(expanded === sp.key ? null : sp.key)}>
                    {sp.label} <span className={`impact ${sp.impact}`}>{sp.impact}</span>
                  </button>
                  <input disabled={!(sp.key in grid)} value={grid[sp.key] ?? ""} placeholder={sp.example}
                    onChange={(e) => setGrid({ ...grid, [sp.key]: e.target.value })} />
                </div>
                {expanded === sp.key && (
                  <div className="param-info">
                    <p>{sp.what}</p>
                    <p><b>Recommended:</b> {sp.reco}</p>
                  </div>
                )}
              </div>
            ))}
          </div>
        </Field>
        <SimpleNote>
          The eight medium- and low-impact parameters are hidden (alpha, batch, accumulation, length,
          dropout, warmup, weight decay and NEFTune). Sweeping many at once doesn't give answers: it gives
          combinations you then can't attribute.
        </SimpleNote>
        <div className="row between">
          <span className="sub">{combos} combination(s) → <b>{runs}</b> training run(s) queued{models ? "" : ""}</span>
          <button className="primary" disabled={!base.dataset_id || !runs} onClick={go}><Play size={14} /> Launch sweep</button>
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
        <h3 style={{ margin: 0 }}>Sweep {sw.sweep_id}</h3>
        <span className="sub">{sw.runs.filter((r: any) => r.status === "completed").length}/{sw.runs.length} completed · best by <b>{metric}</b></span>
      </div>
      <div style={{ overflowX: "auto" }}>
        <table>
          <thead><tr><th>Run</th>{sw.params.map((p: string) => <th key={p} className="mono">{p}</th>)}<th>eval_loss</th><th>loss</th><th>reward</th><th>Status</th></tr></thead>
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
          <Link to={`/training/compare?ids=${sw.runs.map((r: any) => r.job_id).join(",")}`}><button className="sm">Compare curves</button></Link>
        </div>
      )}
    </div>
  );
}

import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ArrowLeft } from "lucide-react";
import { api, duration, type Job } from "../api";
import { StatusBadge } from "../components/ui";
import { verdictForTraining, VerdictPanel } from "../components/Verdict";

const COLORS = ["#7c8cff", "#3fb950", "#d29922", "#f85149", "#a371f7", "#39c5cf"];
const HIDDEN = new Set(["dataset_path", "output_dir", "name"]);

type Run = { job: Job; events: any[] };

export default function CompareRuns() {
  const [params] = useSearchParams();
  const ids = (params.get("ids") ?? "").split(",").map(Number).filter(Boolean);
  const [runs, setRuns] = useState<Run[]>([]);
  const [metric, setMetric] = useState("loss");

  useEffect(() => {
    Promise.all(ids.map(async (id) => ({ job: await api.get<Job>(`/api/jobs/${id}`), events: await api.get<any[]>(`/api/jobs/${id}/events`) })))
      .then(setRuns);
  }, [params.get("ids")]); // eslint-disable-line

  const label = (r: Run) => `#${r.job.id} ${r.job.name}`;
  const metrics = useMemo(() => {
    const keys = new Set<string>();
    runs.forEach((r) => r.events.filter((e) => e.type === "log").forEach((e) => Object.keys(e).forEach((k) => typeof e[k] === "number" && !["t", "step", "epoch"].includes(k) && keys.add(k))));
    return ["loss", "eval_loss", "learning_rate", "grad_norm", "mean_token_accuracy", ...[...keys].sort()].filter((k, i, a) => keys.has(k) && a.indexOf(k) === i);
  }, [runs]);
  const data = useMemo(() => {
    const m = new Map<number, any>();
    runs.forEach((r) => r.events.filter((e) => e.type === "log" && e[metric] !== undefined).forEach((e) => {
      m.set(e.step, { ...m.get(e.step), step: e.step, [label(r)]: e[metric] });
    }));
    return [...m.values()].sort((a, b) => a.step - b.step);
  }, [runs, metric]);

  const final = (r: Run) => {
    const res = r.events.find((e) => e.type === "train_result") ?? {};
    const logs = r.events.filter((e) => e.type === "log");
    const lastEval = [...logs].reverse().find((e) => e.eval_loss !== undefined);
    const gpu = [...r.events].reverse().find((e) => e.type === "gpu");
    return { train_loss: res.train_loss, last_loss: [...logs].reverse().find((e) => e.loss !== undefined)?.loss,
             eval_loss: lastEval?.eval_loss, runtime: res.train_runtime, vram: gpu?.mem_gb };
  };
  const keys = [...new Set(runs.flatMap((r) => Object.keys(r.job.config)))].filter((k) => !HIDDEN.has(k)).sort();
  const differs = (k: string) => new Set(runs.map((r) => JSON.stringify(r.job.config[k]))).size > 1;
  const [onlyDiff, setOnlyDiff] = useState(true);
  const fmt = (v: any) => (v == null ? "—" : typeof v === "number" ? (Math.abs(v) < 0.01 && v !== 0 ? v.toExponential(2) : +v.toFixed(4)) : String(v));

  // Interpreting the numbers: who wins, what invalidates the comparison and what to look at next
  const verdict = useMemo(() => verdictForTraining(
    runs.filter((r) => r.job.type === "train").map((r) => {
      const f = final(r);
      const logs = r.events.filter((e) => e.type === "log");
      return {
        id: r.job.id, label: label(r), status: r.job.status,
        eval_loss: f.eval_loss, last_loss: f.last_loss, train_loss: f.train_loss,
        runtime: f.runtime, vram: f.vram,
        dataset: r.job.config.dataset_name, dataset_rows: r.job.config.dataset_rows,
        steps: [...logs].reverse().find((e) => e.step !== undefined)?.step,
        evalCurve: logs.filter((e) => e.eval_loss !== undefined).map((e) => ({ step: e.step, value: e.eval_loss })),
        diffs: Object.fromEntries(keys.filter(differs).map((k) => [k, r.job.config[k]])),
      };
    })), [runs, keys]);

  return (
    <>
      <div className="page-head">
        <div>
          <Link to="/training" className="sub row" style={{ gap: 4 }}><ArrowLeft size={14} /> Training</Link>
          <h1>Compare training runs</h1>
          <div className="sub">{runs.length} jobs</div>
        </div>
      </div>
      {runs.length < 2 && <div className="card empty">Select at least two training runs in the Training list.</div>}
      {runs.length >= 2 && runs.some((r) => r.job.type === "bench") && (() => {
        const benchRuns = runs.filter((r) => r.job.type === "bench");
        const tasks = [...new Set(benchRuns.flatMap((r) => r.events.filter((e) => e.type === "bench_task").map((e) => e.task)))];
        const val = (r: Run, t: string) => r.events.find((e) => e.type === "bench_task" && e.task === t)?.value;
        return (
          <div className="card">
            <h3>Benchmarks</h3>
            <table>
              <thead><tr><th>Task</th>{benchRuns.map((r, i) => <th key={r.job.id} style={{ color: COLORS[i % COLORS.length] }}>{r.job.config.model_name}</th>)}{benchRuns.length === 2 && <th>Difference</th>}</tr></thead>
              <tbody>{tasks.map((t) => {
                const vs = benchRuns.map((r) => val(r, t));
                const d = vs.length === 2 && vs[0] != null && vs[1] != null ? vs[1] - vs[0] : null;
                return (<tr key={t}><td><b>{t}</b></td>{vs.map((v, i) => <td key={i}>{v != null ? `${(v * 100).toFixed(1)}%` : "—"}</td>)}
                  {benchRuns.length === 2 && <td style={{ color: d == null ? undefined : d >= -0.01 ? "var(--ok)" : "var(--err)" }}>{d == null ? "—" : `${d >= 0 ? "+" : ""}${(d * 100).toFixed(1)} pts`}</td>}</tr>);
              })}</tbody>
            </table>
            {benchRuns.length === 2 && <div className="sub" style={{ marginTop: 6, fontSize: 12 }}>Drops of more than 2–3 points after fine-tuning usually mean catastrophic forgetting: use fewer epochs or a lower learning rate, or mix in general data.</div>}
          </div>
        );
      })()}
      {runs.length >= 2 && runs.some((r) => r.job.type === "train") && (
        <>
          <VerdictPanel findings={verdict} />
          <div className="card">
            <div className="row between"><h3 style={{ margin: 0 }}>Curves</h3>
              <select value={metric} onChange={(e) => setMetric(e.target.value)} style={{ width: "auto" }}>{metrics.map((m) => <option key={m}>{m}</option>)}</select></div>
            <ResponsiveContainer width="100%" height={320}>
              <LineChart data={data}>
                <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                <XAxis dataKey="step" stroke="var(--muted)" fontSize={12} />
                <YAxis stroke="var(--muted)" fontSize={12} domain={["auto", "auto"]} tickFormatter={(v) => (Math.abs(v) < 0.01 && v !== 0 ? v.toExponential(0) : v)} />
                <Tooltip contentStyle={{ background: "var(--panel)", border: "1px solid var(--border)" }} />
                <Legend />
                {runs.map((r, i) => <Line key={r.job.id} type="monotone" dataKey={label(r)} stroke={COLORS[i % COLORS.length]} dot={false} strokeWidth={2} connectNulls isAnimationActive={false} />)}
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="card">
            <h3>Results</h3>
            <table>
              <thead><tr><th>Job</th><th>Status</th><th>Mean loss</th><th>Final loss</th><th>Eval loss</th><th>Duration</th><th>Peak VRAM</th></tr></thead>
              <tbody>{runs.map((r, i) => { const f = final(r); return (
                <tr key={r.job.id}><td><span style={{ color: COLORS[i % COLORS.length] }}>●</span> <Link to={`/training/${r.job.id}`}>{label(r)}</Link></td>
                  <td><StatusBadge status={r.job.status} /></td><td>{fmt(f.train_loss)}</td><td>{fmt(f.last_loss)}</td><td>{fmt(f.eval_loss)}</td>
                  <td>{f.runtime ? `${f.runtime.toFixed(0)} s` : duration(r.job.started_at, r.job.finished_at)}</td><td>{f.vram ? `${f.vram} GB` : "—"}</td></tr>); })}</tbody>
            </table>
          </div>
          <div className="card">
            <div className="row between"><h3 style={{ margin: 0 }}>Configuration</h3>
              <label className="check"><input type="checkbox" checked={onlyDiff} onChange={(e) => setOnlyDiff(e.target.checked)} /> Differences only</label></div>
            <div style={{ overflowX: "auto" }}>
              <table style={{ marginTop: 8 }}>
                <thead><tr><th>Parameter</th>{runs.map((r, i) => <th key={r.job.id} style={{ color: COLORS[i % COLORS.length] }}>#{r.job.id}</th>)}</tr></thead>
                <tbody>{keys.filter((k) => !onlyDiff || differs(k)).map((k) => (
                  <tr key={k} style={{ background: differs(k) ? "color-mix(in srgb, var(--warn) 7%, transparent)" : undefined }}>
                    <td className="mono">{k}</td>{runs.map((r) => <td key={r.job.id} className="mono truncate" style={{ maxWidth: 220 }} title={String(r.job.config[k])}>{fmt(r.job.config[k])}</td>)}</tr>
                ))}</tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </>
  );
}

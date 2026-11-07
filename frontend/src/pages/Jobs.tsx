import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { BarChart3, Boxes, Eraser, GitCompare, GraduationCap, Lock, Ruler, Trash2 } from "lucide-react";
import { api, duration, type Job } from "../api";
import { Check, Field, Help, Modal, Progress, Spinner, StatusBadge, useData, useDialog, useToast } from "../components/ui";

/** What each kind of job is, in plain words. */
const KINDS: Record<string, { label: string; icon: any; desc: string }> = {
  train: { label: "Training", icon: GraduationCap, desc: "Fine-tuning a language model" },
  embed: { label: "Embeddings", icon: Ruler, desc: "Fine-tuning the RAG search model" },
  bench: { label: "Benchmark", icon: BarChart3, desc: "Standard test to measure general skills" },
  export: { label: "Export", icon: Boxes, desc: "LoRA merge or GGUF conversion" },
};

/** One-line summary of what defines that job. */
function detail(j: Job): string {
  if (j.type === "train") {
    const m = (j.config.method ?? "").toUpperCase();
    return [m, j.config.backend === "unsloth" ? "Unsloth" : null, j.config.base_model?.split("/").pop()].filter(Boolean).join(" · ");
  }
  if (j.type === "export") return j.config.gguf ? `GGUF ${j.config.gguf_quant}` : "LoRA merge";
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

  const remove = async (j: Job) => {
    const info = await api.get<any>(`/api/jobs/${j.id}/output`);
    const isProtected = info.models?.length;
    const msg = !info.exists ? "This job left no files: only the history row is deleted."
      : isProtected ? `It takes ${info.size_mb} MB, but those files belong to ${info.models.join(", ")}, which is registered in Models. The row is deleted and the files are kept.`
      : `The row and its ${info.size_mb} MB of files will be deleted; no registered model uses them.`;
    if (!(await dialog.confirm(`Delete “${j.name}”`, msg))) return;
    const r = await api.del<any>(`/api/jobs/${j.id}?delete_output=${!isProtected}`);
    notify(r.deleted_output ? `Deleted · ${r.freed_mb || "<0.1"} MB freed` : "Deleted from the history");
    reload();
  };

  const shown = useMemo(() => (jobs ?? []).filter((j) => (!kind || j.type === kind) && (!status || j.status === status)), [jobs, kind, status]);
  const running = (jobs ?? []).filter((j) => j.status === "running" || j.status === "queued");
  // Only training curves and benchmark results make sense side by side
  const comparable = (j: Job) => j.type === "train" || j.type === "bench";

  return (
    <>
      <div className="page-head">
        <div><h1>Jobs</h1><div className="sub">Everything you've launched: training runs, embedding fine-tunes, benchmarks and exports</div></div>
        <div className="row" style={{ gap: 8 }}>
          <button onClick={() => setClean(true)}><Eraser size={14} /> Clean up history</button>
          <button className="primary" disabled={cmp.length < 2} onClick={() => nav(`/training/compare?ids=${cmp.join(",")}`)}>
            <GitCompare size={14} /> Compare ({cmp.length})
          </button>
        </div>
      </div>

      <Help title="What is this screen, and what is “Compare” for?">
        <p>It's the <b>history</b>: every training run, benchmark, embedding fine-tune or export you launch leaves a row here. Nothing is configured on this screen; it's for looking things up. Click a name to open its details, with its curves, its exact configuration and its full log.</p>
        <p style={{ marginTop: 8 }}><b>Compare</b>: tick <b>two or more</b> training runs to put them side by side: overlaid loss curves, the hyperparameters that differ and the final metrics. It's how you answer “did raising the learning rate help?” or “did rank 32 beat rank 16?” without opening tabs.</p>
        <ul>
          <li>You can only compare <b>training runs with training runs and benchmarks with benchmarks</b>; exports have no metrics to compare, so they have no checkbox.</li>
          <li>For the comparison to mean anything, the runs must have used <b>the same dataset</b>; otherwise you're comparing data, not configurations.</li>
          <li>If you want to try many configurations at once, use <b>Training → Sweep</b>: it launches one per combination and ranks them for you.</li>
        </ul>
      </Help>

      {!!running.length && (
        <div className="card" style={{ borderColor: "var(--accent)" }}>
          <b>Running right now</b>
          {running.map((j) => (
            <div key={j.id} style={{ marginTop: 6 }}>
              <Link to={`/training/${j.id}`}>{j.name}</Link> <span className="sub">· {detail(j)}</span>
              {j.progress?.max_steps ? (
                <><Progress value={j.progress.step / j.progress.max_steps} />
                  <div className="sub" style={{ fontSize: 11 }}>{j.progress.step}/{j.progress.max_steps} steps</div></>
              ) : <div className="sub" style={{ fontSize: 12 }}>en cola</div>}
            </div>
          ))}
        </div>
      )}

      <div className="card">
        <div className="row" style={{ gap: 10, alignItems: "flex-end" }}>
          <Field label="Type">
            <select value={kind} onChange={(e) => setKind(e.target.value)}>
              <option value="">All</option>
              {Object.entries(KINDS).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}
            </select>
          </Field>
          <Field label="Status">
            <select value={status} onChange={(e) => setStatus(e.target.value)}>
              <option value="">All</option>
              <option value="completed">Completed</option>
              <option value="failed">Failed</option>
              <option value="running">Running</option>
              <option value="cancelled">Cancelled</option>
            </select>
          </Field>
          <div className="spacer" />
          <span className="sub">{shown.length} de {jobs?.length ?? 0}</span>
        </div>

        {!shown.length ? <div className="empty">{jobs?.length ? "No jobs match that filter" : "You haven't launched anything yet"}</div> : (
          <table style={{ marginTop: 10 }}>
            <thead><tr><th style={{ width: 28 }} title="Tick two or more to compare them" /><th>Name</th><th>Type</th><th>Duration</th><th>Status</th><th /></tr></thead>
            <tbody>
              {shown.map((j) => {
                const K = KINDS[j.type];
                const Icon = K?.icon ?? GraduationCap;
                return (
                  <tr key={j.id}>
                    <td>{comparable(j) && (
                      <input type="checkbox" checked={cmp.includes(j.id)} title="Compare with another"
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
                    <td><button className="icon" title="Delete this job" onClick={() => remove(j)}><Trash2 size={13} /></button></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
        {cmp.length === 1 && <div className="sub" style={{ marginTop: 8 }}>Tick at least one more to compare them.</div>}
      </div>

      {clean && <Cleanup onClose={() => { setClean(false); reload(); }} />}
    </>
  );
}

/** Bulk cleanup. It always shows first what will be deleted and how much disk it would really free. */
function Cleanup({ onClose }: { onClose: () => void }) {
  const [statuses, setStatuses] = useState<string[]>(["failed", "cancelled"]);
  const [days, setDays] = useState(0);
  const [outputs, setOutputs] = useState(false);
  const [preview, setPreview] = useState<any>(null);
  const [purge, setPurge] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const { notify } = useToast();

  const body = () => ({ statuses: statuses.length ? statuses : null, older_than_days: days || null,
                        delete_outputs: outputs, dry_run: true });

  const preview_ = async () => {
    setBusy(true);
    try { setPreview(await api.post<any>("/api/jobs/cleanup", body())); }
    catch (e: any) { notify(e.message, "err"); } finally { setBusy(false); }
  };

  const removeSelected = async () => {
    setBusy(true);
    try {
      const r = await api.post<any>("/api/jobs/cleanup", { ...body(), dry_run: false });
      notify(`${r.count} job(s) deleted${r.deleted_outputs ? ` · ${r.deleted_outputs} folder(s), ${r.freed_mb} MB freed` : ""}`);
      onClose();
    } catch (e: any) { notify(e.message, "err"); } finally { setBusy(false); }
  };

  const previewPurge = async () => {
    setBusy(true);
    try { setPurge(await api.post<any>("/api/jobs/purge-outputs", { dry_run: true })); }
    catch (e: any) { notify(e.message, "err"); } finally { setBusy(false); }
  };

  const doPurge = async () => {
    setBusy(true);
    try {
      const r = await api.post<any>("/api/jobs/purge-outputs", { dry_run: false });
      notify(`${(r.freed_mb / 1000).toFixed(2)} GB freed · the history is kept`);
      setPurge(null);
      onClose();
    } catch (e: any) { notify(e.message, "err"); } finally { setBusy(false); }
  };

  const toggle = (s: string) => setStatuses((p) => (p.includes(s) ? p.filter((x) => x !== s) : [...p, s]));

  return (
    <Modal title="Clean up history" onClose={onClose} wide>
      <p className="sub">
        Choose which jobs to remove from the history. Jobs that are <b>running or queued</b> are never touched, and
        the files of a job whose model is registered in <b>Models</b> are always kept,
        so that model isn't left pointing at nothing.
      </p>
      <Field label="Statuses to delete">
        <div className="row" style={{ gap: 14, flexWrap: "wrap" }}>
          {[["failed", "Failed"], ["cancelled", "Cancelled"], ["completed", "Completed"]].map(([k, l]) => (
            <Check key={k} label={l} checked={statuses.includes(k)} onChange={() => toggle(k)} />
          ))}
        </div>
      </Field>
      <Field label="Only those older than…" hint="0 = no age limit">
        <input type="number" min={0} value={days} onChange={(e) => { setDays(+e.target.value); setPreview(null); }} />
      </Field>
      <Check label="Also delete their files from disk (adapters, checkpoints)" checked={outputs}
        onChange={(v) => { setOutputs(v); setPreview(null); }} />

      <div className="row" style={{ marginTop: 10 }}>
        <button onClick={preview_} disabled={busy || !statuses.length}>{busy ? <Spinner /> : <Eraser size={13} />} See what would be deleted</button>
      </div>

      <div className="card" style={{ marginTop: 12 }}>
        <b>Only free disk space</b>
        <div className="sub" style={{ marginTop: 4 }}>
          Deletes the weights and checkpoints of every job but <b>keeps their rows</b>, with their curves and
          metrics. It's almost always what you want: weights take gigabytes and are rarely run again,
          while what each run measured weighs nothing and is the reason you ran it.
        </div>
        <div className="row" style={{ marginTop: 8, gap: 8 }}>
          <button onClick={previewPurge} disabled={busy}>{busy ? <Spinner /> : null} See how much would be freed</button>
          {purge != null && purge.count > 0 && (
            <button className="danger" onClick={doPurge}>
              <Trash2 size={13} /> Free {(purge.freed_mb / 1000).toFixed(2)} GB from {purge.count} job(s)
            </button>
          )}
        </div>
        {purge != null && <div className="sub" style={{ marginTop: 6 }}>
          {purge.count ? `${purge.count} job(s) with weights on disk · ${(purge.freed_mb / 1000).toFixed(2)} GB` : "There are no weights left to delete."}
        </div>}
      </div>

      {preview && (
        <div className="card" style={{ marginTop: 10 }}>
          {!preview.count ? <div className="empty">No job matches those criteria</div> : (<>
            <b>{preview.count} job(s)</b>
            <div className="sub">
              {outputs
                ? preview.freeable_mb ? `${preview.freeable_mb} MB of disk would be freed.` : "No disk would be freed: all their files are protected."
                : "They're only removed from the history; the files stay on disk."}
            </div>
            {!!preview.protected?.length && (
              <div className="sub" style={{ marginTop: 6 }}>
                <Lock size={12} /> Files kept because a model is registered: {preview.protected.join(", ")}
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
            {preview.jobs.length > 40 && <div className="sub">…and {preview.jobs.length - 40} more</div>}
          </>)}
        </div>
      )}

      <div className="row end" style={{ marginTop: 10 }}>
        <button onClick={onClose}>Cancel</button>
        <button className="danger" disabled={busy || !preview?.count} onClick={removeSelected}>
          <Trash2 size={13} /> Delete {preview?.count ?? 0} job(s)
        </button>
      </div>
    </Modal>
  );
}

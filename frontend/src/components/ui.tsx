import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { Eraser, HelpCircle, ListChecks, Loader2, Sliders, X } from "lucide-react";
import { api, type Task } from "../api";

export function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="field">
      {label}
      {children}
      {hint && <span className="hint">{hint}</span>}
    </label>
  );
}

export function Check({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="check">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} /> {label}
    </label>
  );
}

const STATUS: Record<string, string> = { completed: "ok", running: "accent", queued: "warn", failed: "err", cancelled: "" };
const STATUS_LABEL: Record<string, string> = { completed: "completed", running: "running", queued: "queued", failed: "failed", cancelled: "cancelled" };
export const StatusBadge = ({ status }: { status: string }) => <span className={`badge ${STATUS[status] ?? ""}`}>{STATUS_LABEL[status] ?? status}</span>;

export function Modal({ title, onClose, children, wide }: { title: string; onClose: () => void; children: ReactNode; wide?: boolean }) {
  useEffect(() => {
    const h = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [onClose]);
  return (
    <div className="modal-bg" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className={`modal ${wide ? "wide" : ""}`}>
        <div className="row between" style={{ marginBottom: 14 }}>
          <h2 style={{ margin: 0 }}>{title}</h2>
          <button className="ghost" onClick={onClose} aria-label="Close"><X size={16} /></button>
        </div>
        {children}
      </div>
    </div>
  );
}

export const Spinner = ({ size = 14 }: { size?: number }) => <Loader2 size={size} className="spin" />;

export function Progress({ value }: { value: number }) {
  return <div className="bar"><div style={{ width: `${Math.round(value * 100)}%` }} /></div>;
}

/** Data hook with reload and optional polling. */
export function useData<T>(url: string | null, poll?: number): [T | undefined, () => void, string | null] {
  const [data, setData] = useState<T>();
  const [err, setErr] = useState<string | null>(null);
  const urlRef = useRef(url);
  urlRef.current = url;
  const load = useCallback(() => {
    if (!url) return;
    api.get<T>(url).then((d) => { if (urlRef.current === url) { setData(d); setErr(null); } }).catch((e) => setErr(e.message));
  }, [url]);
  useEffect(() => {
    setData(undefined);
    load();
    if (!poll) return;
    const t = setInterval(load, poll);
    return () => clearInterval(t);
  }, [load, poll]);
  return [data, load, err];
}

// ---------- Notifications and background tasks ----------
type Toast = { id: number; text: string; kind: "info" | "err"; task?: Task };
const ToastCtx = createContext<{
  notify: (text: string, kind?: "info" | "err") => void;
  track: (t: Task, onDone?: (t: Task) => void) => void;
}>({ notify: () => {}, track: () => {} });

export const useToast = () => useContext(ToastCtx);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const nextId = useRef(1);
  const remove = (id: number) => setToasts((ts) => ts.filter((t) => t.id !== id));

  const notify = useCallback((text: string, kind: "info" | "err" = "info") => {
    const id = nextId.current++;
    setToasts((ts) => [...ts, { id, text, kind }]);
    setTimeout(() => remove(id), kind === "err" ? 9000 : 4000);
  }, []);

  const track = useCallback((task: Task, onDone?: (t: Task) => void) => {
    const id = nextId.current++;
    setToasts((ts) => [...ts, { id, text: task.title, kind: "info", task }]);
    const tick = async () => {
      try {
        const t = await api.get<Task>(`/api/tasks/${task.id}`);
        setToasts((ts) => ts.map((x) => (x.id === id ? { ...x, task: t } : x)));
        if (t.status === "running") return void setTimeout(tick, 1000);
        remove(id);
        if (t.status === "completed") { notify(`✓ ${t.title}`); onDone?.(t); }
        else notify(`${t.title}: ${t.error?.split("\n")[0]}`, "err");
      } catch (e: any) { remove(id); notify(e.message, "err"); }
    };
    setTimeout(tick, 600);
  }, [notify]);

  return (
    <ToastCtx.Provider value={{ notify, track }}>
      {children}
      <div className="toasts">
        {toasts.map((t) => (
          <div key={t.id} className={`toast ${t.kind === "err" ? "err" : ""}`}>
            {t.task ? (
              <div className="stack" style={{ gap: 6 }}>
                <div className="row"><Spinner /> <b className="truncate">{t.task.title}</b></div>
                {t.task.message && <div className="sub truncate">{t.task.message}</div>}
                {t.task.progress > 0 && <Progress value={t.task.progress} />}
              </div>
            ) : t.text}
          </div>
        ))}
      </div>
    </ToastCtx.Provider>
  );
}

// ---------- Dialogs (replace the native confirm/prompt) ----------
type Choice = { key: string; label: string; kind?: "primary" | "danger" };
type DialogReq =
  | { type: "choose"; title: string; message?: ReactNode; choices: Choice[]; resolve: (k: string | null) => void }
  | { type: "prompt"; title: string; message?: ReactNode; value: string; placeholder?: string; resolve: (v: string | null) => void };

const DialogCtx = createContext<{
  confirm: (title: string, message?: ReactNode, opts?: { label?: string; danger?: boolean }) => Promise<boolean>;
  choose: (title: string, message: ReactNode, choices: Choice[]) => Promise<string | null>;
  prompt: (title: string, opts?: { message?: ReactNode; value?: string; placeholder?: string }) => Promise<string | null>;
}>({ confirm: async () => false, choose: async () => null, prompt: async () => null });

export const useDialog = () => useContext(DialogCtx);

export function DialogProvider({ children }: { children: ReactNode }) {
  const [req, setReq] = useState<DialogReq | null>(null);
  const [value, setValue] = useState("");
  const choose = useCallback((title: string, message: ReactNode, choices: Choice[]) =>
    new Promise<string | null>((resolve) => setReq({ type: "choose", title, message, choices, resolve })), []);
  const confirm = useCallback(async (title: string, message?: ReactNode, opts?: { label?: string; danger?: boolean }) =>
    (await choose(title, message, [{ key: "ok", label: opts?.label ?? "OK", kind: opts?.danger === false ? "primary" : "danger" }])) === "ok", [choose]);
  const prompt = useCallback((title: string, opts?: { message?: ReactNode; value?: string; placeholder?: string }) =>
    new Promise<string | null>((resolve) => { setValue(opts?.value ?? ""); setReq({ type: "prompt", title, message: opts?.message, value: opts?.value ?? "", placeholder: opts?.placeholder, resolve }); }), []);
  const close = (result: string | null) => { req?.resolve(result as never); setReq(null); };

  return (
    <DialogCtx.Provider value={{ confirm, choose, prompt }}>
      {children}
      {req && (
        <Modal title={req.title} onClose={() => close(null)}>
          <div className="stack">
            {req.message && <div className="sub" style={{ whiteSpace: "pre-wrap" }}>{req.message}</div>}
            {req.type === "prompt" && (
              <input autoFocus value={value} placeholder={req.placeholder} onChange={(e) => setValue(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && value.trim() && close(value.trim())} />
            )}
            <div className="row" style={{ justifyContent: "flex-end" }}>
              <button onClick={() => close(null)}>Cancel</button>
              {req.type === "choose"
                ? req.choices.map((c, i) => (
                  <button key={c.key} autoFocus={i === req.choices.length - 1} className={c.kind ?? ""} onClick={() => close(c.key)}>{c.label}</button>))
                : <button className="primary" disabled={!value.trim()} onClick={() => close(value.trim())}>OK</button>}
            </div>
          </div>
        </Modal>
      )}
    </DialogCtx.Provider>
  );
}

// ---------- Short per-screen help ----------
export function Help({ title, children, defaultOpen = false }: { title: string; children: ReactNode; defaultOpen?: boolean }) {
  const key = `llmt.help.${title}`;
  const [open, setOpen] = useState(() => {
    try { const v = localStorage.getItem(key); return v === null ? defaultOpen : v === "1"; } catch { return defaultOpen; }
  });
  const toggle = () => { setOpen(!open); try { localStorage.setItem(key, open ? "0" : "1"); } catch { /* no storage */ } };
  return (
    <div className="help">
      <button className="ghost sm" onClick={toggle}><HelpCircle size={14} /> {title}</button>
      {open && <div className="help-body">{children}</div>}
    </div>
  );
}

// ---------- Background task center ----------
const KIND_LABEL: Record<string, string> = {
  "dataset-import": "Import dataset", synth: "Synthetic dataset", download: "Download", load: "Load model",
  "rag-ingest": "Index documents", "rag-web": "Download pages", "rag-enrich": "Questions per chunk",
  "rag-summaries": "Summaries", "rag-graph": "Knowledge graph", "rag-images": "Describe images",
  "rag-reindex": "Re-index", "rag-bench": "Retrieval benchmark", eval: "Evaluation",
  "eval-qgen": "Generate questions", setup: "Install",
};

export function TaskCenter() {
  const [tasks, reload] = useData<Task[]>("/api/tasks", 2000);
  const [open, setOpen] = useState(false);
  const running = (tasks ?? []).filter((t) => t.status === "running");
  const recent = (tasks ?? []).slice(0, 8);
  if (!tasks?.length) return null;
  const cancel = async (id: string) => { await api.post(`/api/tasks/${id}/cancel`); reload(); };
  return (
    <div className="taskcenter">
      <button className="ghost sm" style={{ width: "100%", justifyContent: "flex-start" }} onClick={() => setOpen(!open)}>
        {running.length ? <Spinner /> : <ListChecks size={14} />}
        <span className="nav-label">{running.length ? `${running.length} task(s)` : "Tasks"}</span>
      </button>
      {open && (
        <div className="tasklist">
          {recent.map((t) => (
            <div key={t.id} className="task">
              <div className="row between" style={{ gap: 4 }}>
                <span className="truncate" title={t.title}>{KIND_LABEL[t.kind] ?? t.kind}</span>
                {t.status === "running"
                  ? <button className="ghost sm" title="Cancel" onClick={() => cancel(t.id)}><X size={11} /></button>
                  : <span className={`badge ${t.status === "completed" ? "ok" : "err"}`}>{t.status === "completed" ? "ok" : "error"}</span>}
              </div>
              {t.status === "running" && <><Progress value={t.progress} /><div className="sub truncate" style={{ fontSize: 11 }}>{t.message}</div></>}
              {t.status === "failed" && <div className="sub truncate" style={{ fontSize: 11 }} title={t.error ?? ""}>{t.error?.split("\n")[0]}</div>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------- Simple / full mode ----------
// Simple mode hides the settings you almost never need to touch and leaves them at their defaults,
// which are the same values full mode would use if you didn't change them. It doesn't change what is sent,
// only what you see. It's remembered per browser.
const ModeCtx = createContext<{ simple: boolean; setSimple: (v: boolean) => void }>({ simple: true, setSimple: () => {} });
export const useMode = () => useContext(ModeCtx);

export function ModeProvider({ children }: { children: ReactNode }) {
  const [simple, setSimpleState] = useState(() => {
    try { return localStorage.getItem("llmt-mode") !== "full"; } catch { return true; }
  });
  const setSimple = useCallback((v: boolean) => {
    setSimpleState(v);
    try { localStorage.setItem("llmt-mode", v ? "simple" : "full"); } catch { /* private mode */ }
  }, []);
  return <ModeCtx.Provider value={{ simple, setSimple }}>{children}</ModeCtx.Provider>;
}

/** Wraps the settings that only show in full mode. */
export function Adv({ children }: { children: ReactNode }) {
  const { simple } = useMode();
  return simple ? null : <>{children}</>;
}

/** Sidebar switch. */
export function ModeToggle() {
  const { simple, setSimple } = useMode();
  // Two buttons instead of one toggle: a bare “Simple mode” button doesn't say whether that's the current
  // state or what clicking will do, and that's exactly the doubt a switch shouldn't leave.
  return (
    <div className="modes" title="How many settings the forms show">
      <Sliders size={13} />
      <button className={simple ? "active" : ""} onClick={() => setSimple(true)}
        title="Only the settings that change the result; everything else uses sensible defaults">
        <span className="nav-label">Simple</span>
      </button>
      <button className={simple ? "" : "active"} onClick={() => setSimple(false)}
        title="Every parameter in view">
        <span className="nav-label">Full</span>
      </button>
    </div>
  );
}

/** A note shown only in simple mode: says which values sit behind the hidden settings,
  *  so simple mode isn't a black box. */
export function SimpleNote({ children }: { children: ReactNode }) {
  const { simple, setSimple } = useMode();
  if (!simple) return null;
  return (
    <div className="simple-note">
      <Sliders size={13} />
      <span>{children} <button className="linky" onClick={() => setSimple(false)}>Show all parameters</button></span>
    </div>
  );
}

/** VRAM status and a button to free it, in the sidebar.
  *  With 24 GB and 17 GB models, knowing what takes what and being able to release it is part of daily work. */
export function VramPanel({ gpu }: { gpu: { mem_used_gb: number; mem_total_gb: number } | undefined }) {
  const [v, reload] = useData<{
    libre_gb: number | null; total_gb: number | null; nuestro_gb: number;
    modelo: string | null; rag_modelos: string[];
  }>("/api/models/inference/vram", 5000);
  const [busy, setBusy] = useState(false);
  const { notify } = useToast();
  if (!gpu) return null;

  const oursLoaded = (v?.nuestro_gb ?? 0) >= 0.3;
  const freeVram = async () => {
    setBusy(true);
    try {
      const r = await api.post<any>("/api/models/inference/unload?everything=true");
      notify(r.liberado_gb > 0 ? `${r.liberado_gb} GB freed · ${r.libre_gb} GB free now`
                               : "Nothing of ours was in VRAM");
      reload();
    } catch (e: any) { notify(e.message, "err"); } finally { setBusy(false); }
  };

  const detail = [
    v?.modelo ? `model: ${v.modelo}` : null,
    v?.rag_modelos?.length ? `RAG: ${v.rag_modelos.join(", ")}` : null,
    `this app uses ${v?.nuestro_gb ?? 0} GB of the total`,
  ].filter(Boolean).join("\n");

  return (
    <>
      <div className="row between">
        <span title={detail}>VRAM</span>
        <span>{gpu.mem_used_gb.toFixed(1)} / {gpu.mem_total_gb.toFixed(0)} GB</span>
      </div>
      <div className="bar" title={detail}>
        <div style={{ width: `${(gpu.mem_used_gb / gpu.mem_total_gb) * 100}%` }} />
        {/* The part we use, to tell it apart from what other programs use */}
        {!!v?.total_gb && oursLoaded && (
          <div className="bar-ours" style={{ width: `${(v.nuestro_gb / v.total_gb) * 100}%` }} />
        )}
      </div>
      {oursLoaded && (
        <button className="mode-toggle" style={{ marginTop: 6 }} disabled={busy} onClick={freeVram}
          title={`Unloads the model and the RAG search models from VRAM.\n${detail}`}>
          <Eraser size={13} />
          <span className="nav-label">{busy ? "Freeing…" : `Free VRAM (${v?.nuestro_gb} GB)`}</span>
        </button>
      )}
    </>
  );
}

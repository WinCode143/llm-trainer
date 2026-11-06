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
const STATUS_ES: Record<string, string> = { completed: "completado", running: "en curso", queued: "en cola", failed: "fallido", cancelled: "cancelado" };
export const StatusBadge = ({ status }: { status: string }) => <span className={`badge ${STATUS[status] ?? ""}`}>{STATUS_ES[status] ?? status}</span>;

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
          <button className="ghost" onClick={onClose} aria-label="Cerrar"><X size={16} /></button>
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

/** Hook de datos con recarga y polling opcional. */
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

// ---------- Notificaciones y tareas en segundo plano ----------
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

// ---------- Diálogos (reemplazan confirm/prompt nativos) ----------
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
    (await choose(title, message, [{ key: "ok", label: opts?.label ?? "Aceptar", kind: opts?.danger === false ? "primary" : "danger" }])) === "ok", [choose]);
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
              <button onClick={() => close(null)}>Cancelar</button>
              {req.type === "choose"
                ? req.choices.map((c, i) => (
                  <button key={c.key} autoFocus={i === req.choices.length - 1} className={c.kind ?? ""} onClick={() => close(c.key)}>{c.label}</button>))
                : <button className="primary" disabled={!value.trim()} onClick={() => close(value.trim())}>Aceptar</button>}
            </div>
          </div>
        </Modal>
      )}
    </DialogCtx.Provider>
  );
}

// ---------- Ayuda breve por pantalla ----------
export function Help({ title, children, defaultOpen = false }: { title: string; children: ReactNode; defaultOpen?: boolean }) {
  const key = `llmt.help.${title}`;
  const [open, setOpen] = useState(() => {
    try { const v = localStorage.getItem(key); return v === null ? defaultOpen : v === "1"; } catch { return defaultOpen; }
  });
  const toggle = () => { setOpen(!open); try { localStorage.setItem(key, open ? "0" : "1"); } catch { /* sin almacenamiento */ } };
  return (
    <div className="help">
      <button className="ghost sm" onClick={toggle}><HelpCircle size={14} /> {title}</button>
      {open && <div className="help-body">{children}</div>}
    </div>
  );
}

// ---------- Centro de tareas en segundo plano ----------
const KIND_LABEL: Record<string, string> = {
  "dataset-import": "Importar dataset", synth: "Dataset sintético", download: "Descarga", load: "Cargar modelo",
  "rag-ingest": "Indexar documentos", "rag-web": "Descargar páginas", "rag-enrich": "Preguntas por fragmento",
  "rag-summaries": "Resúmenes", "rag-graph": "Grafo de conocimiento", "rag-images": "Describir imágenes",
  "rag-reindex": "Re-indexar", "rag-bench": "Benchmark de recuperación", eval: "Evaluación",
  "eval-qgen": "Generar preguntas", setup: "Instalación",
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
        <span className="nav-label">{running.length ? `${running.length} tarea(s)` : "Tareas"}</span>
      </button>
      {open && (
        <div className="tasklist">
          {recent.map((t) => (
            <div key={t.id} className="task">
              <div className="row between" style={{ gap: 4 }}>
                <span className="truncate" title={t.title}>{KIND_LABEL[t.kind] ?? t.kind}</span>
                {t.status === "running"
                  ? <button className="ghost sm" title="Cancelar" onClick={() => cancel(t.id)}><X size={11} /></button>
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

// ---------- Modo simple / completo ----------
// El modo simple oculta los parámetros que casi nunca hay que tocar y deja sus valores por defecto,
// que son los mismos que usaría el modo completo si no los cambias. No cambia lo que se envía: sólo
// lo que se ve. Se guarda por navegador.
const ModeCtx = createContext<{ simple: boolean; setSimple: (v: boolean) => void }>({ simple: true, setSimple: () => {} });
export const useMode = () => useContext(ModeCtx);

export function ModeProvider({ children }: { children: ReactNode }) {
  const [simple, setSimpleState] = useState(() => {
    try { return localStorage.getItem("llmt-mode") !== "full"; } catch { return true; }
  });
  const setSimple = useCallback((v: boolean) => {
    setSimpleState(v);
    try { localStorage.setItem("llmt-mode", v ? "simple" : "full"); } catch { /* modo privado */ }
  }, []);
  return <ModeCtx.Provider value={{ simple, setSimple }}>{children}</ModeCtx.Provider>;
}

/** Envuelve los ajustes que sólo se ven en modo completo. */
export function Adv({ children }: { children: ReactNode }) {
  const { simple } = useMode();
  return simple ? null : <>{children}</>;
}

/** Interruptor de la barra lateral. */
export function ModeToggle() {
  const { simple, setSimple } = useMode();
  // Dos botones en vez de uno que alterna: «Modo simple» a secas no dice si es el estado actual
  // o lo que pasará al pulsarlo, y es justo la duda que no debe tener un interruptor.
  return (
    <div className="modos" title="Cuántos ajustes se ven en los formularios">
      <Sliders size={13} />
      <button className={simple ? "activo" : ""} onClick={() => setSimple(true)}
        title="Sólo los ajustes que cambian el resultado; el resto usa valores por defecto sensatos">
        <span className="nav-label">Simple</span>
      </button>
      <button className={simple ? "" : "activo"} onClick={() => setSimple(false)}
        title="Todos los parámetros a la vista">
        <span className="nav-label">Completo</span>
      </button>
    </div>
  );
}

/** Nota visible sólo en modo simple: dice qué valores se están usando tras los ajustes ocultos,
 *  para que el modo simple no sea una caja negra. */
export function SimpleNote({ children }: { children: ReactNode }) {
  const { simple, setSimple } = useMode();
  if (!simple) return null;
  return (
    <div className="simple-note">
      <Sliders size={13} />
      <span>{children} <button className="linky" onClick={() => setSimple(false)}>Ver todos los parámetros</button></span>
    </div>
  );
}

/** Estado de la VRAM y botón para vaciarla, en la barra lateral.
 *  Con 24 GB y modelos de 17, saber quién ocupa qué y poder soltarlo es parte del trabajo diario. */
export function VramPanel({ gpu }: { gpu: { mem_used_gb: number; mem_total_gb: number } | undefined }) {
  const [v, reload] = useData<{
    libre_gb: number | null; total_gb: number | null; nuestro_gb: number;
    modelo: string | null; rag_modelos: string[];
  }>("/api/models/inference/vram", 5000);
  const [busy, setBusy] = useState(false);
  const { notify } = useToast();
  if (!gpu) return null;

  const ocupamos = (v?.nuestro_gb ?? 0) >= 0.3;
  const vaciar = async () => {
    setBusy(true);
    try {
      const r = await api.post<any>("/api/models/inference/unload?everything=true");
      notify(r.liberado_gb > 0 ? `${r.liberado_gb} GB liberados · quedan ${r.libre_gb} GB libres`
                               : "No había nada nuestro en la VRAM");
      reload();
    } catch (e: any) { notify(e.message, "err"); } finally { setBusy(false); }
  };

  const detalle = [
    v?.modelo ? `modelo: ${v.modelo}` : null,
    v?.rag_modelos?.length ? `RAG: ${v.rag_modelos.join(", ")}` : null,
    `esta app usa ${v?.nuestro_gb ?? 0} GB del total`,
  ].filter(Boolean).join("\n");

  return (
    <>
      <div className="row between">
        <span title={detalle}>VRAM</span>
        <span>{gpu.mem_used_gb.toFixed(1)} / {gpu.mem_total_gb.toFixed(0)} GB</span>
      </div>
      <div className="bar" title={detalle}>
        <div style={{ width: `${(gpu.mem_used_gb / gpu.mem_total_gb) * 100}%` }} />
        {/* La parte que ocupamos nosotros, para distinguirla de lo que usan otros programas */}
        {!!v?.total_gb && ocupamos && (
          <div className="bar-nuestro" style={{ width: `${(v.nuestro_gb / v.total_gb) * 100}%` }} />
        )}
      </div>
      {ocupamos && (
        <button className="mode-toggle" style={{ marginTop: 6 }} disabled={busy} onClick={vaciar}
          title={`Saca de la VRAM el modelo y los modelos de búsqueda del RAG.\n${detalle}`}>
          <Eraser size={13} />
          <span className="nav-label">{busy ? "Vaciando…" : `Vaciar VRAM (${v?.nuestro_gb} GB)`}</span>
        </button>
      )}
    </>
  );
}

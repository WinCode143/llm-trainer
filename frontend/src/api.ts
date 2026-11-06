export type Dataset = { id: number; name: string; format: "chat" | "text" | "preference"; path: string; num_rows: number; source: string; created_at: string };
export type Model = { id: number; name: string; kind: "hf" | "lora" | "merged" | "gguf" | "embedding"; path: string; base_model: string | null; job_id: number | null; created_at: string };
export type Job = {
  id: number; name: string; type: "train" | "export" | "embed" | "bench"; status: "queued" | "running" | "completed" | "failed" | "cancelled";
  config: Record<string, any>; result: Record<string, any>; error: string | null; created_at: string; started_at: string | null; finished_at: string | null;
  progress?: { step: number; max_steps: number; epoch?: number } | null;
};
export type Task = { id: string; kind: string; title: string; status: "running" | "completed" | "failed"; progress: number; message: string; result: any; error: string | null };
export type Collection = {
  id: number; name: string; embed_model: string; chunk_size: number; chunk_overlap: number; num_chunks: number; documents: string[]; created_at: string;
  search_mode: "hybrid" | "vector" | "bm25"; reranker: string | null; top_k: number; min_score: number; fallback: "no_context" | "refuse";
  chunk_template: string; query_mode: string; parser?: string;
  features?: { questions: number; summaries: number; images: number; chunks: number; graph: boolean };
  doc_meta: Record<string, { tags?: string[]; template?: string }> | null;
  originals?: Record<string, boolean>;
};
export type EvalRun = { id: number; name: string; status: string; config: any; results: any[]; summary: Record<string, any>; error: string | null; created_at: string };
export type Msg = { role: "system" | "user" | "assistant"; content: string };

async function handle<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try {
      const j = await r.json();
      msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j);
    } catch { /* cuerpo no JSON */ }
    throw new Error(msg);
  }
  return r.json();
}

export const api = {
  get: <T,>(url: string) => fetch(url).then((r) => handle<T>(r)),
  post: <T,>(url: string, body?: unknown) =>
    fetch(url, { method: "POST", headers: { "content-type": "application/json" }, body: body === undefined ? undefined : JSON.stringify(body) }).then((r) => handle<T>(r)),
  put: <T,>(url: string, body: unknown) =>
    fetch(url, { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }).then((r) => handle<T>(r)),
  patch: <T,>(url: string, body: unknown) =>
    fetch(url, { method: "PATCH", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }).then((r) => handle<T>(r)),
  form: <T,>(url: string, fd: FormData) => fetch(url, { method: "POST", body: fd }).then((r) => handle<T>(r)),
  del: <T,>(url: string) => fetch(url, { method: "DELETE" }).then((r) => handle<T>(r)),
};

/** POST que devuelve NDJSON; llama onEvent por cada línea. */
export async function streamNDJSON(url: string, body: unknown, onEvent: (e: any) => void, signal?: AbortSignal) {
  const r = await fetch(url, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body), signal });
  if (!r.ok || !r.body) await handle(r);
  const reader = r.body!.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, i).trim();
      buf = buf.slice(i + 1);
      if (line) onEvent(JSON.parse(line));
    }
  }
}

/** Espera a que una tarea en segundo plano termine. */
export async function waitTask(id: string, onProgress?: (t: Task) => void): Promise<Task> {
  for (;;) {
    const t = await api.get<Task>(`/api/tasks/${id}`);
    onProgress?.(t);
    if (t.status !== "running") {
      if (t.status === "failed") throw new Error(t.error?.split("\n")[0] ?? "La tarea falló");
      return t;
    }
    await new Promise((r) => setTimeout(r, 1000));
  }
}

export const fmtDate = (s: string | null) => (s ? new Date(s.endsWith("Z") ? s : s + "Z").toLocaleString() : "—");
export const duration = (a: string | null, b: string | null) => {
  if (!a) return "—";
  const ms = (b ? new Date(b + (b.endsWith("Z") ? "" : "Z")) : new Date()).getTime() - new Date(a + (a.endsWith("Z") ? "" : "Z")).getTime();
  const s = Math.max(0, Math.round(ms / 1000));
  return s < 60 ? `${s}s` : s < 3600 ? `${Math.floor(s / 60)}m ${s % 60}s` : `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
};

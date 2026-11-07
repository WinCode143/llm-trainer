import { useEffect, useRef, useState } from "react";
import { Copy, ImagePlus, MessageSquarePlus, RotateCcw, Send, Square, Trash2, X } from "lucide-react";
import { api, type Collection, type Model, type Msg, streamNDJSON } from "../api";
import { AssistantText } from "../components/Markdownish";
import { Adv, Field, Help, Spinner, useData, useToast } from "../components/ui";

type Part = { type: "text"; text: string } | { type: "image"; url: string };
type ChatMsg = Omit<Msg, "content"> & { content: string | Part[]; sources?: any[]; ragMode?: string; stats?: any; error?: string };
const msgText = (c: string | Part[]) => (typeof c === "string" ? c : c.filter((p) => p.type === "text").map((p: any) => p.text).join(" "));
const msgImages = (c: string | Part[]) => (typeof c === "string" ? [] : c.filter((p) => p.type === "image").map((p: any) => p.url));
export type GenParams = { max_new_tokens: number; temperature: number; top_p: number; top_k: number; repetition_penalty: number; enable_thinking: boolean | null };
export const DEFAULT_PARAMS: GenParams = { max_new_tokens: 512, temperature: 0.7, top_p: 0.9, top_k: 50, repetition_penalty: 1.05, enable_thinking: null };

export function ParamsEditor({ p, set }: { p: GenParams; set: (p: GenParams) => void }) {
  const row = (k: Exclude<keyof GenParams, "enable_thinking">, label: string, min: number, max: number, step: number) => (
    <Field label={`${label}: ${p[k]}`}>
      <input type="range" min={min} max={max} step={step} value={p[k]} onChange={(e) => set({ ...p, [k]: +e.target.value })} />
    </Field>
  );
  return (
    <div className="stack">
      <Field label="Reasoning (<think>)" hint="For reasoning models such as Qwen3; others ignore it">
        <select value={String(p.enable_thinking)} onChange={(e) => set({ ...p, enable_thinking: e.target.value === "null" ? null : e.target.value === "true" })}>
          <option value="null">Model default</option><option value="true">On</option><option value="false">Off</option>
        </select>
      </Field>
      {row("temperature", "Temperature", 0, 2, 0.05)}
      <Adv>
        {row("top_p", "Top-p", 0, 1, 0.01)}
        {row("top_k", "Top-k", 0, 200, 1)}
        {row("repetition_penalty", "Repetition penalty", 1, 2, 0.01)}
      </Adv>
      {row("max_new_tokens", "Max new tokens", 16, 4096, 16)}
    </div>
  );
}

/** Per-browser persistent state (the playground's conversation and settings). */
function usePersisted<T>(key: string, initial: T): [T, (v: T | ((p: T) => T)) => void] {
  const [v, setV] = useState<T>(() => {
    try { const raw = localStorage.getItem(key); return raw ? { ...initial as object, ...JSON.parse(raw) } as T : initial; } catch { return initial; }
  });
  useEffect(() => { try { localStorage.setItem(key, JSON.stringify(v)); } catch { /* no storage */ } }, [key, v]);
  return [v, setV];
}

export default function Playground() {
  const [tab, setTab] = useState<"chat" | "compare">("chat");
  return (
    <>
      <div className="page-head">
        <div><h1>Playground</h1><div className="sub">Chat with your models, with or without RAG, and compare them</div></div>
        <div className="tabs" style={{ margin: 0, border: 0 }}>
          <button className={`tab ${tab === "chat" ? "active" : ""}`} onClick={() => setTab("chat")}>Chat</button>
          <button className={`tab ${tab === "compare" ? "active" : ""}`} onClick={() => setTab("compare")}>A/B compare</button>
        </div>
      </div>
      <Help title="What is the Playground for?">
        <p>It's where you <b>try out</b> what you train. Load a model (a base model or your adapter) and talk to it.</p>
        <ul>
          <li><b>RAG</b>: pick a collection and the model answers from your documents, citing the sources. If nothing is relevant, it tells you and answers without context.</li>
          <li><b>Images</b>: if you load a vision model, you can attach images.</li>
          <li><b>A/B compare</b>: the same prompt on several models, one after another (only one fits in VRAM), to see whether fine-tuning helped.</li>
          <li>Conversations are saved automatically, and you can copy or regenerate the last reply.</li>
        </ul>
      </Help>
      {tab === "chat" ? <Chat /> : <Compare />}
    </>
  );
}

function Chat() {
  const [models] = useData<Model[]>("/api/models");
  const [cols] = useData<Collection[]>("/api/rag/collections");
  const [status, reloadStatus] = useData<any>("/api/models/inference/status", 3000);
  const [saved, setSaved] = usePersisted("llmt.playground", { convId: null as number | null, system: "", params: DEFAULT_PARAMS, rag: "" as number | "", ragK: 0, ragTags: [] as string[] });
  const [msgs, setMsgs] = useState<ChatMsg[]>([]);
  const [convs, reloadConvs] = useData<{ id: number; title: string; n: number; updated_at: string }[]>("/api/conversations");
  const [convQ, setConvQ] = useState("");
  const convId = saved.convId;
  const setConvId = (id: number | null) => setSaved((p) => ({ ...p, convId: id }));
  const [input, setInput] = useState("");
  const [images, setImages] = useState<string[]>([]);
  const { system, params, rag, ragK } = saved; // ragK 0 = use the collection's top_k
  const setSystem = (v: string) => setSaved((p) => ({ ...p, system: v }));
  const setParams = (v: GenParams) => setSaved((p) => ({ ...p, params: v }));
  const setRag = (v: number | "") => setSaved((p) => ({ ...p, rag: v }));
  const setRagK = (v: number) => setSaved((p) => ({ ...p, ragK: v }));
  const ragTags = saved.ragTags ?? [];
  const toggleTag = (t: string) => setSaved((p) => ({ ...p, ragTags: ragTags.includes(t) ? ragTags.filter((x) => x !== t) : [...ragTags, t] }));
  const collTags = [...new Set(Object.values(cols?.find((c) => c.id === rag)?.doc_meta ?? {}).flatMap((m) => m?.tags ?? []))].sort();
  const [busy, setBusy] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const { notify, track } = useToast();
  const loaded = status?.loaded;

  useEffect(() => { logRef.current?.scrollTo({ top: logRef.current.scrollHeight }); }, [msgs]);
  // Load the selected conversation
  useEffect(() => {
    if (!convId) { setMsgs([]); return; }
    api.get<any>(`/api/conversations/${convId}`).then((c) => setMsgs(c.messages)).catch(() => setConvId(null));
  }, [convId]); // eslint-disable-line
  // Save to the server when each reply finishes (not on every token)
  const persist = async (all: ChatMsg[]) => {
    const settings = { model: status?.loaded?.name, rag, system, params };
    if (convId) await api.put(`/api/conversations/${convId}`, { messages: all, settings });
    else setConvId((await api.post<any>("/api/conversations", { messages: all, settings })).id);
    reloadConvs();
  };
  const newConv = () => { setConvId(null); setMsgs([]); };
  const delConv = async (id: number) => { await api.del(`/api/conversations/${id}`); if (id === convId) newConv(); reloadConvs(); };

  const loadModel = async (id: number) => {
    try { track(await api.post("/api/models/inference/load", { model_id: id }), reloadStatus); } catch (e: any) { notify(e.message, "err"); }
  };

  const send = async (override?: ChatMsg[]) => {
    if ((!override && !input.trim()) || busy) return;
    const content: string | Part[] = images.length
      ? [...images.map((url) => ({ type: "image" as const, url })), { type: "text" as const, text: input }]
      : input;
    const history: ChatMsg[] = override ?? [...msgs, { role: "user", content }];
    setMsgs([...history, { role: "assistant", content: "" }]);
    if (!override) { setInput(""); setImages([]); }
    setBusy(true);
    abort.current = new AbortController();
    // The reply accumulates locally: that way exactly what was shown gets saved
    let answer: ChatMsg = { role: "assistant", content: "" };
    const upd = (f: (m: ChatMsg) => ChatMsg) => { answer = f(answer); setMsgs([...history, answer]); };
    try {
      await streamNDJSON("/api/chat/stream", {
        messages: history.map(({ role, content }) => ({ role, content })), params,
        system_prompt: system || null, rag_collection_id: rag || null, rag_k: ragK || null, rag_tags: rag && ragTags.length ? ragTags.filter((t) => collTags.includes(t)) : null,
      }, (e) => {
        if (e.type === "token") upd((m) => ({ ...m, content: m.content + e.text }));
        else if (e.type === "sources") upd((m) => ({ ...m, sources: e.sources, ragMode: e.rag_mode }));
        else if (e.type === "done") upd((m) => ({ ...m, stats: e }));
        else if (e.type === "error") upd((m) => ({ ...m, error: e.message }));
      }, abort.current.signal);
    } catch (e: any) {
      if (e.name !== "AbortError") upd((m) => ({ ...m, error: e.message }));
    } finally {
      setBusy(false);
      persist([...history, answer]).catch(() => {});
    }
  };

  return (
    <div className="split" style={{ gridTemplateColumns: "300px minmax(0,1fr)" }}>
      <div className="card stack">
        <div className="row between">
          <b>Conversations</b>
          <button className="sm" onClick={newConv}><MessageSquarePlus size={13} /> New</button>
        </div>
        {(convs?.length ?? 0) > 5 && <input value={convQ} onChange={(e) => setConvQ(e.target.value)} placeholder="Search…" />}
        <div className="conv-list">
          {!convs?.length && <div className="sub" style={{ fontSize: 12 }}>They're saved automatically</div>}
          {convs?.filter((c) => !convQ || c.title.toLowerCase().includes(convQ.toLowerCase())).slice(0, 30).map((c) => (
            <div key={c.id} className={`conv ${c.id === convId ? "active" : ""}`} onClick={() => setConvId(c.id)}>
              <span className="truncate">{c.title}</span>
              <button className="ghost sm" onClick={(e) => { e.stopPropagation(); delConv(c.id); }} title="Delete"><Trash2 size={12} /></button>
            </div>
          ))}
        </div>
        <Field label="Model">
          <select value={loaded?.model_id ?? ""} onChange={(e) => e.target.value && loadModel(+e.target.value)} disabled={!!status?.loading}>
            <option value="">{status?.loading ? `Loading ${status.loading}…` : "— load a model —"}</option>
            {models?.filter((m) => m.kind !== "embedding").map((m) => <option key={m.id} value={m.id}>{m.name} [{m.kind}]</option>)}
          </select>
        </Field>
        {status?.loading && <div className="row sub"><Spinner /> Loading into VRAM…</div>}
        <Field label="System prompt"><textarea rows={3} value={system} onChange={(e) => setSystem(e.target.value)} placeholder="You are a helpful assistant…" /></Field>
        <Field label="RAG" hint="Retrieves context from a collection before answering">
          <select value={rag} onChange={(e) => setRag(e.target.value ? +e.target.value : "")}>
            <option value="">Sin RAG</option>
            {cols?.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
          </select>
        </Field>
        {rag && collTags.length > 0 && (
          <Field label="Only search documents tagged">
            <div className="row" style={{ gap: 4 }}>{collTags.map((t) => <button key={t} className={`sm ${ragTags.includes(t) ? "primary" : ""}`} onClick={() => toggleTag(t)}>{t}</button>)}</div>
          </Field>
        )}
        {rag && <Field label={`Chunks to retrieve: ${ragK || "the collection's default"}`}><input type="range" min={0} max={12} value={ragK} onChange={(e) => setRagK(+e.target.value)} /></Field>}
        <ParamsEditor p={params} set={setParams} />

      </div>

      <div className="card chat">
        <div className="chat-log" ref={logRef}>
          {!msgs.length && <div className="empty">{loaded ? `Send a message to talk to ${loaded.name}` : "Load a model to start"}</div>}
          {msgs.map((m, i) => (
            <div key={i} className={`msg ${m.role}`}>
              {msgImages(m.content).length > 0 && (
                <div className="row" style={{ gap: 6, marginBottom: 6 }}>
                  {msgImages(m.content).map((u, j) => <img key={j} src={u} alt="" style={{ maxHeight: 130, borderRadius: 8 }} />)}
                </div>
              )}
              {m.role === "assistant" ? <AssistantText text={msgText(m.content)} streaming={busy && i === msgs.length - 1} /> : msgText(m.content)}
              {m.role === "assistant" && busy && i === msgs.length - 1 && !m.content && <Spinner />}
              {m.error && <div className="error-box" style={{ marginTop: 6 }}>{m.error}</div>}
              {m.ragMode && m.ragMode !== "rag" && (
                <div className="meta" title="No chunk passed the collection's relevance threshold">
                  {m.ragMode === "no_context" ? "⚠ No relevant context: answered without RAG" : "⚠ No relevant context: it said the answer isn't in the documents"}
                </div>
              )}
              {m.sources && m.sources.length > 0 && (
                <div className="sources">
                  {m.sources.map((s, j) => (
                    <details key={j} className="source">
                      <summary>{s.image ? "🖼 " : ""}[{j + 1}] {s.source}{s.page ? ` · p. ${s.page}` : ""}{s.section ? ` · ${s.section}` : ""} · score {s.score?.toFixed?.(3) ?? s.score}</summary>
                      <div style={{ whiteSpace: "pre-wrap", marginTop: 6 }}>{s.text}</div>
                      {s.image && rag && <img src={`/api/rag/collections/${rag}/images/${s.image}`} alt="" style={{ maxHeight: 220, marginTop: 6, borderRadius: 8 }} />}
                    </details>
                  ))}
                </div>
              )}
              {m.role === "assistant" && !(busy && i === msgs.length - 1) && m.content && (
                <div className="meta row" style={{ gap: 2 }}>
                  {m.stats && <span>{m.stats.elapsed_s}s · 1.er token {m.stats.ttft_s}s · {m.stats.chunks_per_s} frag./s</span>}
                  <span className="spacer" />
                  <button className="ghost sm" title="Copy" onClick={() => navigator.clipboard?.writeText(msgText(m.content).replace(/^\s*<think>[\s\S]*?<\/think>\s*/, "")).then(() => notify("Copied"))}><Copy size={12} /></button>
                  {i === msgs.length - 1 && <button className="ghost sm" title="Regenerate" onClick={() => send(msgs.slice(0, -1))}><RotateCcw size={12} /></button>}
                </div>
              )}
            </div>
          ))}
        </div>
        {images.length > 0 && (
          <div className="row" style={{ gap: 6, padding: "6px 0" }}>
            {images.map((u, i) => (
              <div key={i} style={{ position: "relative" }}>
                <img src={u} alt="" style={{ height: 56, borderRadius: 6 }} />
                <button className="ghost sm" style={{ position: "absolute", top: -6, right: -6 }} onClick={() => setImages(images.filter((_, j) => j !== i))}><X size={11} /></button>
              </div>
            ))}
          </div>
        )}
        <div className="chat-input">
          <label className="btn" title={loaded?.vision ? "Attach an image" : "The loaded model doesn't accept images"} style={{ opacity: loaded?.vision ? 1 : 0.5, alignSelf: "flex-end", cursor: loaded?.vision ? "pointer" : "not-allowed" }}>
            <ImagePlus size={15} />
            <input type="file" accept="image/*" multiple hidden disabled={!loaded?.vision} onChange={(e) => {
              for (const f of Array.from(e.target.files ?? [])) {
                const r = new FileReader();
                r.onload = () => setImages((p) => [...p, String(r.result)]);
                r.readAsDataURL(f);
              }
              e.target.value = "";
            }} />
          </label>
          <textarea value={input} onChange={(e) => setInput(e.target.value)} placeholder="Message… (Enter sends, Shift+Enter for a new line)"
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }} />
          {busy ? <button onClick={() => abort.current?.abort()}><Square size={14} /></button>
            : <button className="primary" onClick={() => send()} disabled={!loaded || !input.trim()}><Send size={14} /></button>}
        </div>
      </div>
    </div>
  );
}

function Compare() {
  const [models] = useData<Model[]>("/api/models");
  const [sel, setSel] = useState<number[]>([]);
  const [prompt, setPrompt] = useState("");
  const [system, setSystem] = useState("");
  const [params, setParams] = useState<GenParams>({ ...DEFAULT_PARAMS, temperature: 0.2 });
  const [results, setResults] = useState<Record<number, { name: string; text?: string; latency_s?: number; loading?: boolean }>>({});
  const [busy, setBusy] = useState(false);
  const { notify } = useToast();

  const run = async () => {
    setBusy(true);
    setResults(Object.fromEntries(sel.map((id) => [id, { name: models!.find((m) => m.id === id)!.name }])));
    try {
      await streamNDJSON("/api/chat/compare", { model_ids: sel, prompt, system_prompt: system || null, params }, (e) => {
        if (e.type === "loading") setResults((r) => ({ ...r, [e.model_id]: { ...r[e.model_id], loading: true } }));
        else if (e.type === "result") setResults((r) => ({ ...r, [e.model_id]: { name: e.name, text: e.text, latency_s: e.latency_s } }));
        else if (e.type === "error") notify(e.message, "err");
      });
    } catch (e: any) { notify(e.message, "err"); } finally { setBusy(false); }
  };

  return (
    <div className="split" style={{ gridTemplateColumns: "300px minmax(0,1fr)" }}>
      <div className="card stack">
        <div className="sub">Models load one after another (only one fits in VRAM), so each comparison takes as long as the loads do.</div>
        <Field label="Models to compare">
          <div className="stack" style={{ gap: 4 }}>
            {models?.filter((m) => m.kind !== "embedding").map((m) => (
              <label key={m.id} className="check">
                <input type="checkbox" checked={sel.includes(m.id)} onChange={(e) => setSel(e.target.checked ? [...sel, m.id] : sel.filter((x) => x !== m.id))} />
                {m.name} <span className="sub">[{m.kind}]</span>
              </label>
            ))}
          </div>
        </Field>
        <Field label="System prompt"><textarea rows={2} value={system} onChange={(e) => setSystem(e.target.value)} /></Field>
        <ParamsEditor p={params} set={setParams} />
      </div>
      <div className="stack" style={{ gap: 16 }}>
        <div className="card stack">
          <Field label="Prompt"><textarea rows={4} value={prompt} onChange={(e) => setPrompt(e.target.value)} /></Field>
          <div className="row"><div className="spacer" /><button className="primary" disabled={busy || sel.length < 1 || !prompt} onClick={run}>{busy ? <Spinner /> : <Send size={14} />} Compare</button></div>
        </div>
        <div className="grid g2">
          {Object.entries(results).map(([id, r]) => (
            <div key={id} className="card">
              <div className="row between"><h3>{r.name}</h3>{r.latency_s !== undefined && <span className="sub">{r.latency_s}s</span>}</div>
              <div style={{ whiteSpace: "pre-wrap" }}>{r.text !== undefined ? <AssistantText text={r.text} /> : r.loading ? <span className="row sub"><Spinner /> generating…</span> : <span className="sub">waiting</span>}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

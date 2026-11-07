import { useState } from "react";
import { Download, FolderInput, PackageOpen, Play, Power, Trash2 } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import { api, fmtDate, type Model } from "../api";
import { Adv, Field, Help, Modal, SimpleNote, useData, useDialog, useToast } from "../components/ui";

const KIND: Record<string, [string, string]> = { hf: ["HF", "accent"], lora: ["LoRA", "ok"], merged: ["Merged", "warn"], gguf: ["GGUF", ""], embedding: ["Embeddings", "err"] };

/** The screen's three families. They're separate because they're used differently: you chat with or train
  *  a base model, a trained one is a result of yours, and an embedding model doesn't generate text. */
const FAMILIES = {
  base: { label: "Base",
    desc: "Models you downloaded and start from: chat with them or use them as the starting point for training. Hugging Face repos keep their weights in the shared cache, not in the app's folder." },
  trained: { label: "Trained",
    desc: "What came out of your training runs and exports. A LoRA is small but needs its base model; a merged model and an exported GGUF work on their own and are as big as the base." },
  embeddings: { label: "Embeddings",
    desc: "RAG search models. They don't generate text: they turn text into vectors. They're applied by re-indexing a collection." },
} as const;
type Family = keyof typeof FAMILIES;

/** Which tab each model goes to.
  *  It looks at the ORIGIN, not the format: a GGUF can be something you exported (it has a job_id)
  *  or a model you downloaded already quantized, and classifying by format put the downloaded ones
  *  under “Trained”, where they don't belong. */
function familyOf(m: Model): Family {
  if (m.kind === "embedding") return "embeddings";
  if (m.job_id || m.kind === "lora") return "trained";
  return "base";
}


export default function Models() {
  const [models, reload] = useData<Model[]>("/api/models");
  const [sys] = useData<any>("/api/system", 2000);
  const [suggested] = useData<any[]>("/api/models/suggested");
  const [sizes] = useData<Record<string, { mb: number | null; cached: boolean; used_by: string[] }>>("/api/models/sizes");
  const [modal, setModal] = useState<null | "download" | "register" | { export: Model }>(null);
  const [quant, setQuant] = useState("none");
  const [fam, setFam] = useState<Family>("base");
  const [engine, setEngine] = useState("auto");
  const { notify, track } = useToast();
  const { choose } = useDialog();
  const nav = useNavigate();
  const loaded = sys?.inference?.loaded;
  const shown = (models ?? []).filter((m) => familyOf(m) === fam);

  const load = async (m: Model) => {
    try { track(await api.post("/api/models/inference/load", { model_id: m.id, quantization: quant, engine: m.kind === "gguf" ? "auto" : engine })); } catch (e: any) { notify(e.message, "err"); }
  };
  const del = async (m: Model) => {
    const inData = m.path.includes("/data/");
    const r = await choose(`Remove “${m.name}”`, inData ? "You can remove it from the registry only, or delete its files from disk too." : "It will be removed from the registry; its files (outside data/) aren't touched.",
      [{ key: "reg", label: "Remove from the registry" }, ...(inData ? [{ key: "files", label: "Delete the files too", kind: "danger" as const }] : [])]);
    if (!r) return;
    await api.del(`/api/models/${m.id}?delete_files=${r === "files"}`);
    reload();
  };

  return (
    <>
      <div className="page-head">
        <div><h1>Models</h1><div className="sub">The base models you start from, what you trained, and the RAG search models</div></div>
        <div className="row">
          <button onClick={() => setModal("register")}><FolderInput size={14} /> Registrar</button>
          <button className="primary" onClick={() => setModal("download")}><Download size={14} /> Download from HF</button>
        </div>
      </div>

      <Help title="What's on this screen?">
        <p>Three families that are used differently, each in its own tab:</p>
        <ul>
          <li><b>Base</b>: full models from Hugging Face. You train from them and can chat with them directly. They take no space here: the weights live in the Hugging Face cache and are downloaded the first time.</li>
          <li><b>Trained</b>: what your training runs produce. A <b>LoRA</b> is just the adapter (a few MB) and needs its base model to work; a <b>merged</b> model already contains the base and stands alone (that's why it's as big as the base); a <b>GGUF</b> is the compressed version for llama.cpp, Ollama or LM Studio.</li>
          <li><b>Embeddings</b>: they don't generate text. They turn text into vectors so RAG can search, and they're applied by re-indexing a collection.</li>
        </ul>
        <p style={{ marginTop: 8 }}>None of these models is needed for the app or the automated tests to work: they're the results of what you've trained. You can delete the ones you won't run again; each run's metrics and curves stay in <b>Jobs</b> even if you delete its weights.</p>
        <p style={{ marginTop: 8 }}><b>Only one model fits in VRAM.</b> Engines: <b>transformers</b> (supports 4/8-bit and vision), <b>llama.cpp</b> (for GGUF) and <b>vLLM</b> (16-bit, much faster and answers in batches: ideal for evaluations; takes ~1 min to start).</p>
      </Help>
      <div className="card row between">
        <div>
          <b>Model in memory:</b> {sys?.inference?.loading ? `loading ${sys.inference.loading}…` : loaded ? <>{loaded.name} <span className="sub">({loaded.backend}, {loaded.quantization})</span></> : <span className="sub">none</span>}
          {sys?.inference?.loading && engine === "vllm" && <div className="sub" style={{ fontSize: 12 }}>vLLM takes ~1 min to start the first time</div>}
        </div>
        <div className="row">
          <Adv>
            <Field label=""><select value={engine} onChange={(e) => setEngine(e.target.value)} title="Inference engine">
              <option value="auto">Motor: transformers</option>
              <option value="vllm" disabled={!sys?.vllm}>Engine: vLLM (fast, batched){sys?.vllm ? "" : " — not installed"}</option>
            </select></Field>
            <Field label=""><select value={quant} onChange={(e) => setQuant(e.target.value)} disabled={engine === "vllm"} title={engine === "vllm" ? "vLLM doesn't support bitsandbytes: loads in 16-bit" : "Quantization when loading (HF models)"}>
              <option value="none">Load in 16-bit</option><option value="8bit">Load in 8-bit</option><option value="4bit">Load in 4-bit</option>
            </select></Field>
          </Adv>
          {loaded && <button title="Also unloads the RAG search models, which stay cached"
            onClick={() => api.post("/api/models/inference/unload?everything=true")
              .then((r: any) => notify(r.liberado_gb > 0 ? `${r.liberado_gb} GB freed` : "VRAM freed"))}>
            <Power size={14} /> Free VRAM</button>}
        </div>
      </div>

      <SimpleNote>Models load with transformers in 16-bit (GGUFs always with llama.cpp). Pick the engine and quantization in full mode.</SimpleNote>

      <div className="tabs">
        {(Object.keys(FAMILIES) as Family[]).map((k) => {
          const f = FAMILIES[k];
          const n = (models ?? []).filter((m) => familyOf(m) === k).length;
          return (
            <button key={k} className={`tab ${fam === k ? "active" : ""}`} onClick={() => setFam(k)} title={f.desc}>
              {f.label} <span className="sub">({n})</span>
            </button>
          );
        })}
      </div>

      <div className="card">
        <div className="row between" style={{ marginBottom: 8, alignItems: "flex-start" }}>
          <div className="sub" style={{ maxWidth: "70ch" }}>{FAMILIES[fam].desc}</div>
          {(() => {
            const t = shown.reduce((a, m) => a + (sizes?.[m.id]?.mb ?? 0), 0);
            return t > 0 ? <span className="sub" style={{ whiteSpace: "nowrap" }}>
              {t >= 1000 ? `${(t / 1000).toFixed(1)} GB` : `${t.toFixed(0)} MB`} en disco</span> : null;
          })()}
        </div>
        {models?.length && !shown.length ? (
          <div className="empty" style={{ fontSize: 13 }}>Nothing in this tab yet.</div>
        ) : !models?.length ? (
          <div className="empty">
            No models registered. Download one from Hugging Face or register a repo to use as a base.
            <div className="row" style={{ justifyContent: "center", marginTop: 12 }}>
              {suggested?.slice(0, 3).map((s) => <button key={s.repo} className="sm" onClick={() => api.post("/api/models", { name: s.repo.split("/")[1], path: s.repo }).then(reload)}>+ {s.repo}</button>)}
            </div>
          </div>
        ) : (
          <table>
            <thead><tr><th>Name</th><th>Type</th><th>Base</th><th>Size</th><th>Created</th><th></th></tr></thead>
            <tbody>
              {shown.map((m) => (
                <tr key={m.id} className={loaded?.model_id === m.id ? "selected" : ""}>
                  <td><b>{m.name}</b></td>
                  <td><span className={`badge ${KIND[m.kind][1]}`}>{KIND[m.kind][0]}</span></td>
                  <td className="sub truncate" style={{ maxWidth: 200 }}>{m.base_model ?? "—"}</td>
                  <td className="sub" title={m.path}>
                    {(() => {
                      const z = sizes?.[m.id];
                      if (!z) return "…";
                      if (z.cached) return <span title="The weights are in the Hugging Face cache">in HF cache</span>;
                      return <>{z.mb! >= 1000 ? `${(z.mb! / 1000).toFixed(1)} GB` : `${z.mb!.toFixed(0)} MB`}</>;
                    })()}
                    {!!sizes?.[m.id]?.used_by?.length && (
                      <div style={{ fontSize: 11 }}>used by: {sizes[m.id].used_by.join(", ")}</div>
                    )}
                  </td>
                  <td className="sub">{fmtDate(m.created_at)}</td>
                  <td>
                    <div className="row" style={{ justifyContent: "flex-end", flexWrap: "nowrap" }}>
                      {m.kind === "embedding" ? <button className="sm" onClick={() => nav("/rag")} title="Use it by re-indexing a collection">Use in RAG</button> : <>
                      <button className="sm" disabled={sys?.gpu_job_running} onClick={() => load(m)} title="Load into VRAM"><Play size={13} /> Load</button>
                      <button className="sm" onClick={() => { load(m); nav("/playground"); }} disabled={sys?.gpu_job_running}>Chat</button>
                      {m.kind !== "gguf" && <button className="sm" onClick={() => setModal({ export: m })} title="Merge / GGUF"><PackageOpen size={13} /></button>}
                      {m.kind === "gguf" && <button className="sm" title="Register it in Ollama with one click"
                        onClick={() => api.post<any>(`/api/models/${m.id}/ollama`)
                          .then((r) => notify(`Done · ${r.comando}`))
                          .catch((e) => notify(e.message, "err"))}>Ollama</button>}</>}
                      <button className="sm danger" onClick={() => del(m)}><Trash2 size={13} /></button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {modal === "download" && <DownloadModal suggested={suggested ?? []} onClose={() => setModal(null)} onDone={reload} />}
      {modal === "register" && <RegisterModal onClose={() => setModal(null)} onDone={() => { setModal(null); reload(); }} />}
      {modal && typeof modal === "object" && <ExportModal model={modal.export} ggufAvailable={sys?.gguf_export} onClose={() => setModal(null)} />}
    </>
  );
}

function DownloadModal({ suggested, onClose, onDone }: { suggested: any[]; onClose: () => void; onDone: () => void }) {
  const [repo, setRepo] = useState("");
  const [pattern, setPattern] = useState("");
  const { track, notify } = useToast();
  const go = async (r = repo) => {
    try {
      track(await api.post("/api/models/download", { repo: r, allow_patterns: pattern ? pattern.split(",").map((s) => s.trim()) : null }), onDone);
      onClose();
    } catch (e: any) { notify(e.message, "err"); }
  };
  return (
    <Modal title="Download from Hugging Face" onClose={onClose}>
      <div className="stack">
        <Field label="Repository"><input value={repo} onChange={(e) => setRepo(e.target.value)} placeholder="Qwen/Qwen3-8B" /></Field>
        <Field label="File filter (optional)" hint="For GGUF repos, e.g. *Q4_K_M.gguf"><input value={pattern} onChange={(e) => setPattern(e.target.value)} /></Field>
        <div className="row"><div className="spacer" /><button className="primary" disabled={!repo} onClick={() => go()}>Download</button></div>
        <h3 style={{ marginTop: 8 }}>Suggested for a 24 GB GPU</h3>
        <table><tbody>
          {suggested.map((s) => (
            <tr key={s.repo}><td><b>{s.repo}</b><div className="sub" style={{ fontSize: 12 }}>{s.note}</div></td><td>{s.size}</td>
              <td><button className="sm" onClick={() => setRepo(s.repo)}>Choose</button></td></tr>
          ))}
        </tbody></table>
        <div className="sub">Tip: you don't need to download before training. If you set a repo as the base model, it's downloaded to the HF cache automatically.</div>
      </div>
    </Modal>
  );
}

function RegisterModal({ onClose, onDone }: { onClose: () => void; onDone: () => void }) {
  const [name, setName] = useState("");
  const [path, setPath] = useState("");
  const [base, setBase] = useState("");
  const { notify } = useToast();
  const go = async () => {
    try { await api.post("/api/models", { name: name || path.split("/").filter(Boolean).pop(), path, base_model: base || null }); onDone(); } catch (e: any) { notify(e.message, "err"); }
  };
  return (
    <Modal title="Register a model" onClose={onClose}>
      <div className="stack">
        <Field label="HF repo or local path" hint="An HF folder, a folder with adapter_config.json (LoRA) or a .gguf file. The type is detected automatically."><input value={path} onChange={(e) => setPath(e.target.value)} placeholder="meta-llama/Llama-3.1-8B-Instruct  or  /path/to/model.gguf" /></Field>
        <Field label="Name"><input value={name} onChange={(e) => setName(e.target.value)} placeholder="auto" /></Field>
        <Field label="Base model (LoRA only, optional)"><input value={base} onChange={(e) => setBase(e.target.value)} /></Field>
        <div className="row"><div className="spacer" /><button className="primary" disabled={!path} onClick={go}>Registrar</button></div>
      </div>
    </Modal>
  );
}

function ExportModal({ model, ggufAvailable, onClose }: { model: Model; ggufAvailable: boolean; onClose: () => void }) {
  const [gguf, setGguf] = useState(model.kind !== "lora");
  const [q, setQ] = useState("q4_k_m");
  const { notify } = useToast();
  const nav = useNavigate();
  const go = async () => {
    try { const j = await api.post<any>("/api/jobs/export", { model_id: model.id, gguf, gguf_quant: q }); nav(`/training/${j.id}`); } catch (e: any) { notify(e.message, "err"); }
  };
  return (
    <Modal title={`Export ${model.name}`} onClose={onClose}>
      <div className="stack">
        {model.kind === "lora" && <div>The LoRA adapter will be merged with <code>{model.base_model}</code> in 16-bit, creating a standalone model.</div>}
        <label className="check"><input type="checkbox" checked={gguf} onChange={(e) => setGguf(e.target.checked)} disabled={model.kind !== "lora"} /> Convert to GGUF (llama.cpp / Ollama / LM Studio)</label>
        {gguf && (
          <>
            <Field label="GGUF quantization">
              <select value={q} onChange={(e) => setQ(e.target.value)}>
                {["q4_k_m", "q5_k_m", "q6_k", "q8_0", "f16", "bf16"].map((x) => <option key={x}>{x}</option>)}
              </select>
            </Field>
            {!ggufAvailable && <div className="error-box">llama.cpp is needed to export to GGUF. <Link to="/system">Install it from System</Link> with one click (~6 min).</div>}
            <div className="sub" style={{ fontSize: 12 }}>q4_k_m: the best size/quality balance · q8_0: nearly lossless · f16: unquantized. A <code>Modelfile</code> for Ollama is generated too.</div>
          </>
        )}
        <div className="row"><div className="spacer" /><button className="primary" onClick={go} disabled={gguf && !ggufAvailable}>Export</button></div>
      </div>
    </Modal>
  );
}

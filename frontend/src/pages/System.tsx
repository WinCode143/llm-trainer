import { useState } from "react";
import { Archive, CheckCircle2, Download, Trash2, XCircle } from "lucide-react";
import { api, type Collection, fmtDate, type Model } from "../api";
import { Markdown } from "../components/Markdownish";
import { Help, useData, useDialog, useToast } from "../components/ui";

type Tool = { key: string; label: string; desc: string; setup?: string; manual?: string };
const TOOLS: Tool[] = [
  { key: "unsloth", label: "Unsloth", desc: "Training engine ~2× faster with ~50% less VRAM (separate environment).", setup: "unsloth" },
  { key: "vllm", label: "vLLM (fast GRPO)", desc: "Generation engine for GRPO: produces the completions much faster than transformers (separate environment, ~10 GB).", setup: "vllm" },
  { key: "llama_cpp", label: "llama.cpp (GGUF export)", desc: "Converts and quantizes models to GGUF for Ollama, LM Studio or llama.cpp.", setup: "llamacpp" },
  { key: "llama_cpp_python", label: "GGUF inference on the GPU", desc: "Loads .gguf models in the Playground and the API (Vulkan). Installed together with llama.cpp.", setup: "llamacpp" },
  { key: "sandbox", label: "Code sandbox (bubblewrap)", desc: "Runs agent code without access to your files or the network, with memory and time limits.", manual: "sudo apt install bubblewrap  (Arch: sudo pacman -S bubblewrap)" },
  { key: "tesseract", label: "Tesseract OCR", desc: "Extracts text from scanned PDFs when indexing them for RAG.", manual: "sudo apt install tesseract-ocr  (Arch: sudo pacman -S tesseract tesseract-data-eng)" },
  { key: "hf_token", label: "Hugging Face token", desc: "Needed for gated models (Llama, Gemma) and faster downloads.", manual: "Add LLMT_HF_TOKEN=hf_… to the .env file and restart" },
];

export default function System() {
  const [sys, reload] = useData<any>("/api/system", 5000);
  const [models] = useData<Model[]>("/api/models");
  const [cols] = useData<Collection[]>("/api/rag/collections");
  const { track, notify } = useToast();
  const [installing, setInstalling] = useState<string | null>(null);

  const install = async (setup: string) => {
    setInstalling(setup);
    try {
      track(await api.post(`/api/system/setup/${setup}`), () => { setInstalling(null); reload(); });
    } catch (e: any) { notify(e.message, "err"); setInstalling(null); }
  };

  const base = `${location.protocol}//${location.hostname}:${location.port || (location.protocol === "https:" ? 443 : 80)}/v1`;
  const model = sys?.inference?.loaded?.name ?? models?.[0]?.name ?? "my-model";
  const coll = cols?.[0]?.name ?? "my-collection";
  const key = sys?.api_key_required ? "YOUR_API_KEY" : "not-needed";

  return (
    <>
      <div className="page-head"><div><h1>System</h1><div className="sub">Optional tools, OpenAI-compatible API and configuration</div></div></div>

      <Help title="What gets installed here, and why?">
        <ul>
          <li><b>Unsloth</b>: trains ~2× faster with half the VRAM.</li>
          <li><b>vLLM</b>: generates much faster (GRPO and batch evaluations).</li>
          <li><b>llama.cpp</b>: export to GGUF and use those models in the app.</li>
        </ul>
        <p style={{ marginTop: 8 }}>Each one installs into its own environment so their versions don't clash. Below you'll find how to connect your models to other apps (OpenAI-compatible API) and to external agents (MCP).</p>
      </Help>
      <div className="card">
        <h2>Tools</h2>
        <table><tbody>
          {TOOLS.map((t) => {
            const ok = !!sys?.tools?.[t.key];
            return (
              <tr key={t.key}>
                <td style={{ width: 28 }}>{ok ? <CheckCircle2 size={18} color="var(--ok)" /> : <XCircle size={18} color="var(--muted)" />}</td>
                <td><b>{t.label}</b><div className="sub" style={{ fontSize: 12.5 }}>{t.desc}</div></td>
                <td style={{ textAlign: "right", verticalAlign: "middle" }}>
                  {ok ? <span className="badge ok">installed</span>
                    : t.setup ? <button className="sm primary" disabled={!!installing} onClick={() => install(t.setup!)}><Download size={13} /> {installing === t.setup ? "Installing…" : "Install"}</button>
                    : <code style={{ fontSize: 12 }}>{t.manual}</code>}
                </td>
              </tr>
            );
          })}
        </tbody></table>
        <div className="sub" style={{ fontSize: 12, marginTop: 8 }}>Installing compiles on your machine (llama.cpp takes ~6 min). Everything stays in <code>tools/</code> and <code>backend/</code>; no sudo needed.</div>
      </div>

      <div className="card">
        <div className="row between"><h2 style={{ margin: 0 }}>OpenAI-compatible API</h2>
          <span className="badge accent">{sys?.api_key_required ? "API key required" : "no authentication (localhost only)"}</span></div>
        <p className="sub" style={{ marginTop: 8 }}>
          Use your models from Open WebUI, LibreChat, Continue, LangChain or the OpenAI SDK. The <code>model</code> field accepts the name of any registered model
          (it's loaded automatically); any other name uses the loaded model. Extension: <code>rag_collection</code> applies a collection's RAG.
        </p>
        <div className="grid g2" style={{ marginTop: 10 }}>
          <Snippet title="Python (openai SDK)" code={`from openai import OpenAI\nclient = OpenAI(base_url="${base}", api_key="${key}")\n\nr = client.chat.completions.create(\n    model="${model}",\n    messages=[{"role": "user", "content": "Hello"}],\n    extra_body={"rag_collection": "${coll}"},  # optional\n)\nprint(r.choices[0].message.content)`} />
          <Snippet title="curl" code={`curl ${base}/chat/completions \\\n  -H "Content-Type: application/json" \\\n  -H "Authorization: Bearer ${key}" \\\n  -d '{"model": "${model}", "stream": true,\n       "messages": [{"role": "user", "content": "Hello"}]}'`} />
        </div>
        <div className="kv" style={{ marginTop: 12 }}>
          <span>Base URL</span><code>{base}</code>
          <span>Endpoints</span><span className="mono">GET /models · POST /chat/completions (stream) · POST /embeddings</span>
          <span>Open WebUI</span><span>Settings → Connections → OpenAI API → URL <code>{base}</code></span>
          <span>API key</span><span>{sys?.api_key_required ? "Set in .env (LLMT_API_KEY)" : <>Optional: set <code>LLMT_API_KEY</code> in <code>.env</code> if you expose the app on your network (<code>LLMT_HOST=0.0.0.0</code>)</>}</span>
        </div>
      </div>

      <div className="card">
        <div className="sub">The MCP server, external MCP servers and helper models now live under <b>Connections</b>.</div>
      </div>

      <Backups />

      <div className="card">
        <h2>Configuration</h2>
        <div className="kv">
          <span>Data folder</span><code>{sys?.data_dir}</code>
          <span>Free disk</span><span>{sys?.disk_free_gb} GB</span>
          <span>GPU</span><span>{sys?.gpus?.[0] ? `${sys.gpus[0].name} · ${sys.gpus[0].mem_total_gb.toFixed(0)} GB` : "not detected"}</span>
          <span>.env file</span><span className="sub">Copy <code>.env.example</code> to <code>.env</code> for the HF token, API key, host and paths. Needs a restart.</span>
        </div>
      </div>
    </>
  );
}

function Snippet({ title, code }: { title: string; code: string }) {
  return (
    <div style={{ minWidth: 0 }}>
      <h3 style={{ margin: 0 }}>{title}</h3>
      <Markdown text={"```\n" + code + "\n```"} />
    </div>
  );
}

/** Backups of what can't be replaced (database, datasets and RAG). */
function Backups() {
  const [data, reload] = useData<any>("/api/system/backups");
  const [busy, setBusy] = useState(false);
  const [checking, setChecking] = useState<string | null>(null);
  const [checks, setChecks] = useState<Record<string, any>>({});
  const { track, notify } = useToast();
  const { confirm } = useDialog();
  const run = async () => {
    setBusy(true);
    try { track(await api.post("/api/system/backups", { note: "manual backup" }), () => { setBusy(false); reload(); }); }
    catch (e: any) { notify(e.message, "err"); setBusy(false); }
  };
  const del = async (name: string) => {
    if (!(await confirm(`Delete backup ${name}?`, undefined, { label: "Delete" }))) return;
    await api.del(`/api/system/backups/${name}`); reload();
  };
  const help = async (name: string) => {
    const r = await api.get<any>(`/api/system/backups/${name}/restore-help`);
    await confirm("How to restore this backup", r.instructions, { label: "Got it", danger: false });
  };
  const verify = async (name: string) => {
    setChecking(name);
    try {
      const t = await api.post<any>(`/api/system/backups/${name}/verify`);
      track(t, (done) => {
        setChecks((c) => ({ ...c, [name]: done.result }));
        notify(done.result?.ok ? `${name}: the backup opens correctly` : `${name}: ${done.result?.error}`,
               done.result?.ok ? "info" : "err");
      });
    } catch (e: any) { notify(e.message, "err"); } finally { setChecking(null); }
  };
  return (
    <div className="card">
      <div className="row between">
        <h2 style={{ margin: 0 }}><Archive size={16} /> Backups</h2>
        <button className="primary" disabled={busy} onClick={run}>{busy ? "Creating…" : "Back up now"}</button>
      </div>
      <p className="sub" style={{ marginTop: 8 }}>
        Includes the <b>database</b> (datasets, models, jobs, collections, agents, conversations and evaluations),
        the <b>dataset files</b> and the <b>whole RAG</b> (indexes, graphs and original documents).
        It doesn't include model weights or checkpoints: they're large and can be downloaded or trained again.
        {data?.daily ? " A backup is created automatically every 24 h" : " Automatic backups are off"} and the last {data?.keep ?? 7} are kept in <code>{data?.dir}</code>.
      </p>
      {!data?.backups?.length ? <div className="empty">No backups yet.</div> : (
        <table>
          <thead><tr><th>File</th><th>Size</th><th>Date</th><th>Note</th><th></th></tr></thead>
          <tbody>
            {data.backups.map((b: any) => (
              <tr key={b.file}>
                <td className="mono">{b.file}</td><td>{b.size_mb} MB</td><td className="sub">{fmtDate(b.created_at)}</td>
                <td className="sub">{b.note}</td>
                <td style={{ textAlign: "right", whiteSpace: "nowrap" }}>
                  <button className="sm ghost" disabled={checking === b.file} onClick={() => verify(b.file)}>
                    {checking === b.file ? "Verifying…" : "Verify"}
                  </button>
                  <button className="sm ghost" onClick={() => help(b.file)}>Restore…</button>
                  <button className="sm ghost danger" onClick={() => del(b.file)}><Trash2 size={13} /></button>
                </td>
              </tr>
            )).flatMap((row: any, i: number) => {
              const b = data.backups[i];
              const c = checks[b.file];
              return c ? [row, (
                <tr key={`${b.file}-check`}>
                  <td colSpan={5} className="sub" style={{ paddingTop: 0 }}>
                    {c.ok ? (
                      <>✓ Opens correctly · {Object.entries(c.rows).map(([t, n]: any) => `${t}: ${n}`).join(" · ")}
                        {" · "}{c.dataset_files} dataset file(s)
                        {c.missing_tables?.length ? ` · ⚠ predates: ${c.missing_tables.join(", ")}` : ""}</>
                    ) : <span style={{ color: "var(--err)" }}>✗ {c.error}</span>}
                  </td>
                </tr>
              )] : [row];
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}

import { useState } from "react";
import { Archive, CheckCircle2, Download, Trash2, XCircle } from "lucide-react";
import { api, type Collection, fmtDate, type Model } from "../api";
import { Markdown } from "../components/Markdownish";
import { Help, useData, useDialog, useToast } from "../components/ui";

type Tool = { key: string; label: string; desc: string; setup?: string; manual?: string };
const TOOLS: Tool[] = [
  { key: "unsloth", label: "Unsloth", desc: "Motor de entrenamiento ~2× más rápido y con ~50 % menos VRAM (entorno aislado).", setup: "unsloth" },
  { key: "vllm", label: "vLLM (GRPO rápido)", desc: "Motor de generación para GRPO: genera las respuestas mucho más rápido que transformers (entorno aislado, ~10 GB).", setup: "vllm" },
  { key: "llama_cpp", label: "llama.cpp (exportación GGUF)", desc: "Convierte y cuantiza modelos a GGUF para Ollama, LM Studio o llama.cpp.", setup: "llamacpp" },
  { key: "llama_cpp_python", label: "Inferencia GGUF en GPU", desc: "Carga modelos .gguf en el Playground y la API (Vulkan). Se instala junto con llama.cpp.", setup: "llamacpp" },
  { key: "sandbox", label: "Aislamiento de código (bubblewrap)", desc: "Ejecuta el código de los agentes sin acceso a tus archivos ni a la red, con límites de memoria y tiempo.", manual: "sudo pacman -S bubblewrap" },
  { key: "tesseract", label: "Tesseract OCR", desc: "Extrae texto de PDFs escaneados al indexarlos en RAG.", manual: "sudo pacman -S tesseract tesseract-data-spa" },
  { key: "hf_token", label: "Token de Hugging Face", desc: "Necesario para modelos con licencia (Llama, Gemma) y descargas más rápidas.", manual: "Añade LLMT_HF_TOKEN=hf_… al archivo .env y reinicia" },
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
  const model = sys?.inference?.loaded?.name ?? models?.[0]?.name ?? "mi-modelo";
  const coll = cols?.[0]?.name ?? "mi-coleccion";
  const key = sys?.api_key_required ? "TU_API_KEY" : "no-necesaria";

  return (
    <>
      <div className="page-head"><div><h1>Sistema</h1><div className="sub">Herramientas opcionales, API compatible con OpenAI y configuración</div></div></div>

      <Help title="¿Qué se instala aquí y para qué?">
        <ul>
          <li><b>Unsloth</b>: entrena ~2× más rápido y con la mitad de VRAM.</li>
          <li><b>vLLM</b>: genera mucho más rápido (GRPO y evaluaciones por lotes).</li>
          <li><b>llama.cpp</b>: exportar a GGUF y usar esos modelos en la app.</li>
        </ul>
        <p style={{ marginTop: 8 }}>Cada uno se instala en su propio entorno para que no choquen sus versiones. Debajo tienes cómo conectar tus modelos a otras apps (API compatible con OpenAI) y a agentes externos (MCP).</p>
      </Help>
      <div className="card">
        <h2>Herramientas</h2>
        <table><tbody>
          {TOOLS.map((t) => {
            const ok = !!sys?.tools?.[t.key];
            return (
              <tr key={t.key}>
                <td style={{ width: 28 }}>{ok ? <CheckCircle2 size={18} color="var(--ok)" /> : <XCircle size={18} color="var(--muted)" />}</td>
                <td><b>{t.label}</b><div className="sub" style={{ fontSize: 12.5 }}>{t.desc}</div></td>
                <td style={{ textAlign: "right", verticalAlign: "middle" }}>
                  {ok ? <span className="badge ok">instalado</span>
                    : t.setup ? <button className="sm primary" disabled={!!installing} onClick={() => install(t.setup!)}><Download size={13} /> {installing === t.setup ? "Instalando…" : "Instalar"}</button>
                    : <code style={{ fontSize: 12 }}>{t.manual}</code>}
                </td>
              </tr>
            );
          })}
        </tbody></table>
        <div className="sub" style={{ fontSize: 12, marginTop: 8 }}>La instalación compila en tu máquina (llama.cpp tarda ~6 min). Todo queda en <code>tools/</code> y <code>backend/</code>; no necesita sudo.</div>
      </div>

      <div className="card">
        <div className="row between"><h2 style={{ margin: 0 }}>API compatible con OpenAI</h2>
          <span className="badge accent">{sys?.api_key_required ? "requiere API key" : "sin autenticación (sólo localhost)"}</span></div>
        <p className="sub" style={{ marginTop: 8 }}>
          Usa tus modelos desde Open WebUI, LibreChat, Continue, LangChain o el SDK de OpenAI. El campo <code>model</code> acepta el nombre de cualquier modelo registrado
          (se carga automáticamente); cualquier otro nombre usa el modelo cargado. Extensión propia: <code>rag_collection</code> aplica el RAG de una colección.
        </p>
        <div className="grid g2" style={{ marginTop: 10 }}>
          <Snippet title="Python (SDK openai)" code={`from openai import OpenAI\nclient = OpenAI(base_url="${base}", api_key="${key}")\n\nr = client.chat.completions.create(\n    model="${model}",\n    messages=[{"role": "user", "content": "Hola"}],\n    extra_body={"rag_collection": "${coll}"},  # opcional\n)\nprint(r.choices[0].message.content)`} />
          <Snippet title="curl" code={`curl ${base}/chat/completions \\\n  -H "Content-Type: application/json" \\\n  -H "Authorization: Bearer ${key}" \\\n  -d '{"model": "${model}", "stream": true,\n       "messages": [{"role": "user", "content": "Hola"}]}'`} />
        </div>
        <div className="kv" style={{ marginTop: 12 }}>
          <span>Base URL</span><code>{base}</code>
          <span>Endpoints</span><span className="mono">GET /models · POST /chat/completions (stream) · POST /embeddings</span>
          <span>Open WebUI</span><span>Ajustes → Conexiones → OpenAI API → URL <code>{base}</code></span>
          <span>API key</span><span>{sys?.api_key_required ? "Definida en .env (LLMT_API_KEY)" : <>Opcional: define <code>LLMT_API_KEY</code> en <code>.env</code> si expones la app en tu red (<code>LLMT_HOST=0.0.0.0</code>)</>}</span>
        </div>
      </div>

      <div className="card">
        <div className="sub">El servidor MCP, los servidores MCP externos y los modelos ayudantes están ahora en <b>Conexiones</b>.</div>
      </div>

      <Backups />

      <div className="card">
        <h2>Configuración</h2>
        <div className="kv">
          <span>Carpeta de datos</span><code>{sys?.data_dir}</code>
          <span>Disco libre</span><span>{sys?.disk_free_gb} GB</span>
          <span>GPU</span><span>{sys?.gpus?.[0] ? `${sys.gpus[0].name} · ${sys.gpus[0].mem_total_gb.toFixed(0)} GB` : "no detectada"}</span>
          <span>Archivo .env</span><span className="sub">Copia <code>.env.example</code> a <code>.env</code> para token de HF, API key, host y rutas. Requiere reiniciar.</span>
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

/** Copias de seguridad de lo irreemplazable (base de datos, datasets y RAG). */
function Backups() {
  const [data, reload] = useData<any>("/api/system/backups");
  const [busy, setBusy] = useState(false);
  const [checking, setChecking] = useState<string | null>(null);
  const [checks, setChecks] = useState<Record<string, any>>({});
  const { track, notify } = useToast();
  const { confirm } = useDialog();
  const run = async () => {
    setBusy(true);
    try { track(await api.post("/api/system/backups", { note: "copia manual" }), () => { setBusy(false); reload(); }); }
    catch (e: any) { notify(e.message, "err"); setBusy(false); }
  };
  const del = async (name: string) => {
    if (!(await confirm(`¿Borrar la copia ${name}?`, undefined, { label: "Borrar" }))) return;
    await api.del(`/api/system/backups/${name}`); reload();
  };
  const help = async (name: string) => {
    const r = await api.get<any>(`/api/system/backups/${name}/restore-help`);
    await confirm("Cómo restaurar esta copia", r.instructions, { label: "Entendido", danger: false });
  };
  const verify = async (name: string) => {
    setChecking(name);
    try {
      const t = await api.post<any>(`/api/system/backups/${name}/verify`);
      track(t, (done) => {
        setChecks((c) => ({ ...c, [name]: done.result }));
        notify(done.result?.ok ? `${name}: la copia abre correctamente` : `${name}: ${done.result?.error}`,
               done.result?.ok ? "info" : "err");
      });
    } catch (e: any) { notify(e.message, "err"); } finally { setChecking(null); }
  };
  return (
    <div className="card">
      <div className="row between">
        <h2 style={{ margin: 0 }}><Archive size={16} /> Copias de seguridad</h2>
        <button className="primary" disabled={busy} onClick={run}>{busy ? "Creando…" : "Crear copia ahora"}</button>
      </div>
      <p className="sub" style={{ marginTop: 8 }}>
        Incluye <b>base de datos</b> (datasets, modelos, trabajos, colecciones, agentes, conversaciones y evaluaciones),
        los <b>archivos de dataset</b> y el <b>RAG completo</b> (índices, grafos y documentos originales).
        No incluye los pesos de los modelos ni los checkpoints: ocupan mucho y se pueden volver a descargar o entrenar.
        {data?.daily ? " Se crea una copia automática cada 24 h" : " Las copias automáticas están desactivadas"} y se conservan las {data?.keep ?? 7} últimas en <code>{data?.dir}</code>.
      </p>
      {!data?.backups?.length ? <div className="empty">Todavía no hay copias.</div> : (
        <table>
          <thead><tr><th>Archivo</th><th>Tamaño</th><th>Fecha</th><th>Nota</th><th></th></tr></thead>
          <tbody>
            {data.backups.map((b: any) => (
              <tr key={b.file}>
                <td className="mono">{b.file}</td><td>{b.size_mb} MB</td><td className="sub">{fmtDate(b.created_at)}</td>
                <td className="sub">{b.note}</td>
                <td style={{ textAlign: "right", whiteSpace: "nowrap" }}>
                  <button className="sm ghost" disabled={checking === b.file} onClick={() => verify(b.file)}>
                    {checking === b.file ? "Verificando…" : "Verificar"}
                  </button>
                  <button className="sm ghost" onClick={() => help(b.file)}>Restaurar…</button>
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
                      <>✓ Abre correctamente · {Object.entries(c.rows).map(([t, n]: any) => `${t}: ${n}`).join(" · ")}
                        {" · "}{c.dataset_files} archivo(s) de dataset
                        {c.missing_tables?.length ? ` · ⚠ es anterior a: ${c.missing_tables.join(", ")}` : ""}</>
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

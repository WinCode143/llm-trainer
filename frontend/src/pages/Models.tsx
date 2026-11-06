import { useState } from "react";
import { Download, FolderInput, PackageOpen, Play, Power, Trash2 } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import { api, fmtDate, type Model } from "../api";
import { Adv, Field, Help, Modal, SimpleNote, useData, useDialog, useToast } from "../components/ui";

const KIND: Record<string, [string, string]> = { hf: ["HF", "accent"], lora: ["LoRA", "ok"], merged: ["Merged", "warn"], gguf: ["GGUF", ""], embedding: ["Embeddings", "err"] };

/** Las tres familias de la pantalla. Se separan porque se usan de forma distinta: con un base se
 *  chatea o se entrena, un entrenado es un resultado tuyo y un embedding no genera texto. */
const FAMILIAS = {
  base: { label: "Base",
    desc: "Modelos que has descargado y de los que partes: para chatear con ellos o usarlos como punto de partida de un entrenamiento. Los repos de Hugging Face guardan sus pesos en la caché compartida, no en la carpeta de la app." },
  entrenados: { label: "Entrenados",
    desc: "Lo que ha salido de tus entrenamientos y exportaciones. Un LoRA pesa poco pero necesita su modelo base; un fusionado y un GGUF exportado funcionan solos y ocupan como el base." },
  embeddings: { label: "Embeddings",
    desc: "Modelos de búsqueda del RAG. No generan texto: convierten texto en vectores. Se aplican re-indexando una colección." },
} as const;
type Familia = keyof typeof FAMILIAS;

/** A qué pestaña va cada modelo.
 *  Se mira el ORIGEN, no el formato: un GGUF puede ser tanto algo que exportaste tú (lleva job_id)
 *  como un modelo que te bajaste ya cuantizado, y clasificar por formato metía los descargados
 *  en «Entrenados», donde no pintan nada. */
function familiaDe(m: Model): Familia {
  if (m.kind === "embedding") return "embeddings";
  if (m.job_id || m.kind === "lora") return "entrenados";
  return "base";
}


export default function Models() {
  const [models, reload] = useData<Model[]>("/api/models");
  const [sys] = useData<any>("/api/system", 2000);
  const [suggested] = useData<any[]>("/api/models/suggested");
  const [sizes] = useData<Record<string, { mb: number | null; cached: boolean; used_by: string[] }>>("/api/models/sizes");
  const [modal, setModal] = useState<null | "download" | "register" | { export: Model }>(null);
  const [quant, setQuant] = useState("none");
  const [fam, setFam] = useState<Familia>("base");
  const [engine, setEngine] = useState("auto");
  const { notify, track } = useToast();
  const { choose } = useDialog();
  const nav = useNavigate();
  const loaded = sys?.inference?.loaded;
  const visibles = (models ?? []).filter((m) => familiaDe(m) === fam);

  const load = async (m: Model) => {
    try { track(await api.post("/api/models/inference/load", { model_id: m.id, quantization: quant, engine: m.kind === "gguf" ? "auto" : engine })); } catch (e: any) { notify(e.message, "err"); }
  };
  const del = async (m: Model) => {
    const inData = m.path.includes("/data/");
    const r = await choose(`Quitar «${m.name}»`, inData ? "Puedes quitarlo sólo del registro o borrar también sus archivos del disco." : "Se quitará del registro; los archivos (externos a data/) no se tocan.",
      [{ key: "reg", label: "Quitar del registro" }, ...(inData ? [{ key: "files", label: "Borrar archivos también", kind: "danger" as const }] : [])]);
    if (!r) return;
    await api.del(`/api/models/${m.id}?delete_files=${r === "files"}`);
    reload();
  };

  return (
    <>
      <div className="page-head">
        <div><h1>Modelos</h1><div className="sub">Los modelos base de los que partes, lo que has entrenado y los modelos de búsqueda del RAG</div></div>
        <div className="row">
          <button onClick={() => setModal("register")}><FolderInput size={14} /> Registrar</button>
          <button className="primary" onClick={() => setModal("download")}><Download size={14} /> Descargar de HF</button>
        </div>
      </div>

      <Help title="¿Qué hay en esta pantalla?">
        <p>Tres familias que se usan de forma distinta, cada una en su pestaña:</p>
        <ul>
          <li><b>Base</b>: modelos completos de Hugging Face. Son de los que partes para entrenar y con los que puedes chatear directamente. No ocupan sitio aquí: los pesos viven en la caché de Hugging Face y se descargan la primera vez.</li>
          <li><b>Entrenados</b>: lo que sale de tus entrenamientos. Un <b>LoRA</b> es sólo el adaptador —pocos MB— y necesita su modelo base para funcionar; un <b>fusionado</b> ya lleva el base dentro y es independiente (por eso ocupa igual que el base); un <b>GGUF</b> es la versión comprimida para llama.cpp, Ollama o LM Studio.</li>
          <li><b>Embeddings</b>: no generan texto. Convierten texto en vectores para que el RAG pueda buscar, y se aplican re-indexando una colección.</li>
        </ul>
        <p style={{ marginTop: 8 }}>Ninguno de estos modelos hace falta para que la aplicación funcione ni para las pruebas automáticas: son los resultados de lo que has ido entrenando. Puedes borrar los que no vayas a volver a ejecutar; las métricas y curvas de cada entrenamiento se conservan en <b>Trabajos</b> aunque borres sus pesos.</p>
        <p style={{ marginTop: 8 }}><b>Sólo cabe un modelo en VRAM.</b> Motores: <b>transformers</b> (admite 4/8 bits y visión), <b>llama.cpp</b> (para GGUF) y <b>vLLM</b> (16 bits, mucho más rápido y responde por lotes: ideal para evaluaciones; tarda ~1 min en arrancar).</p>
      </Help>
      <div className="card row between">
        <div>
          <b>Modelo en memoria:</b> {sys?.inference?.loading ? `cargando ${sys.inference.loading}…` : loaded ? <>{loaded.name} <span className="sub">({loaded.backend}, {loaded.quantization})</span></> : <span className="sub">ninguno</span>}
          {sys?.inference?.loading && engine === "vllm" && <div className="sub" style={{ fontSize: 12 }}>vLLM tarda ~1 min en arrancar la primera vez</div>}
        </div>
        <div className="row">
          <Adv>
            <Field label=""><select value={engine} onChange={(e) => setEngine(e.target.value)} title="Motor de inferencia">
              <option value="auto">Motor: transformers</option>
              <option value="vllm" disabled={!sys?.vllm}>Motor: vLLM (rápido, por lotes){sys?.vllm ? "" : " — no instalado"}</option>
            </select></Field>
            <Field label=""><select value={quant} onChange={(e) => setQuant(e.target.value)} disabled={engine === "vllm"} title={engine === "vllm" ? "vLLM no admite bitsandbytes: carga en 16-bit" : "Cuantización al cargar (modelos HF)"}>
              <option value="none">Carga en 16-bit</option><option value="8bit">Carga en 8-bit</option><option value="4bit">Carga en 4-bit</option>
            </select></Field>
          </Adv>
          {loaded && <button title="Saca también los modelos de búsqueda del RAG, que se quedan cacheados"
            onClick={() => api.post("/api/models/inference/unload?everything=true")
              .then((r: any) => notify(r.liberado_gb > 0 ? `${r.liberado_gb} GB liberados` : "VRAM liberada"))}>
            <Power size={14} /> Vaciar VRAM</button>}
        </div>
      </div>

      <SimpleNote>Los modelos se cargan con transformers en 16 bits (los GGUF siempre con llama.cpp). El motor y la cuantización se eligen en modo completo.</SimpleNote>

      <div className="tabs">
        {(Object.keys(FAMILIAS) as Familia[]).map((k) => {
          const f = FAMILIAS[k];
          const n = (models ?? []).filter((m) => familiaDe(m) === k).length;
          return (
            <button key={k} className={`tab ${fam === k ? "active" : ""}`} onClick={() => setFam(k)} title={f.desc}>
              {f.label} <span className="sub">({n})</span>
            </button>
          );
        })}
      </div>

      <div className="card">
        <div className="row between" style={{ marginBottom: 8, alignItems: "flex-start" }}>
          <div className="sub" style={{ maxWidth: "70ch" }}>{FAMILIAS[fam].desc}</div>
          {(() => {
            const t = visibles.reduce((a, m) => a + (sizes?.[m.id]?.mb ?? 0), 0);
            return t > 0 ? <span className="sub" style={{ whiteSpace: "nowrap" }}>
              {t >= 1000 ? `${(t / 1000).toFixed(1)} GB` : `${t.toFixed(0)} MB`} en disco</span> : null;
          })()}
        </div>
        {models?.length && !visibles.length ? (
          <div className="empty" style={{ fontSize: 13 }}>Nada en esta pestaña todavía.</div>
        ) : !models?.length ? (
          <div className="empty">
            No hay modelos registrados. Descarga uno de Hugging Face o registra un repo para usarlo como base.
            <div className="row" style={{ justifyContent: "center", marginTop: 12 }}>
              {suggested?.slice(0, 3).map((s) => <button key={s.repo} className="sm" onClick={() => api.post("/api/models", { name: s.repo.split("/")[1], path: s.repo }).then(reload)}>+ {s.repo}</button>)}
            </div>
          </div>
        ) : (
          <table>
            <thead><tr><th>Nombre</th><th>Tipo</th><th>Base</th><th>Tamaño</th><th>Creado</th><th></th></tr></thead>
            <tbody>
              {visibles.map((m) => (
                <tr key={m.id} className={loaded?.model_id === m.id ? "selected" : ""}>
                  <td><b>{m.name}</b></td>
                  <td><span className={`badge ${KIND[m.kind][1]}`}>{KIND[m.kind][0]}</span></td>
                  <td className="sub truncate" style={{ maxWidth: 200 }}>{m.base_model ?? "—"}</td>
                  <td className="sub" title={m.path}>
                    {(() => {
                      const z = sizes?.[m.id];
                      if (!z) return "…";
                      if (z.cached) return <span title="Los pesos están en la caché de Hugging Face">en caché HF</span>;
                      return <>{z.mb! >= 1000 ? `${(z.mb! / 1000).toFixed(1)} GB` : `${z.mb!.toFixed(0)} MB`}</>;
                    })()}
                    {!!sizes?.[m.id]?.used_by?.length && (
                      <div style={{ fontSize: 11 }}>en uso: {sizes[m.id].used_by.join(", ")}</div>
                    )}
                  </td>
                  <td className="sub">{fmtDate(m.created_at)}</td>
                  <td>
                    <div className="row" style={{ justifyContent: "flex-end", flexWrap: "nowrap" }}>
                      {m.kind === "embedding" ? <button className="sm" onClick={() => nav("/rag")} title="Úsalo re-indexando una colección">Usar en RAG</button> : <>
                      <button className="sm" disabled={sys?.gpu_job_running} onClick={() => load(m)} title="Cargar en VRAM"><Play size={13} /> Cargar</button>
                      <button className="sm" onClick={() => { load(m); nav("/playground"); }} disabled={sys?.gpu_job_running}>Chat</button>
                      {m.kind !== "gguf" && <button className="sm" onClick={() => setModal({ export: m })} title="Fusionar / GGUF"><PackageOpen size={13} /></button>}
                      {m.kind === "gguf" && <button className="sm" title="Registrarlo en Ollama con un clic"
                        onClick={() => api.post<any>(`/api/models/${m.id}/ollama`)
                          .then((r) => notify(`Listo · ${r.comando}`))
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
    <Modal title="Descargar de Hugging Face" onClose={onClose}>
      <div className="stack">
        <Field label="Repositorio"><input value={repo} onChange={(e) => setRepo(e.target.value)} placeholder="Qwen/Qwen3-8B" /></Field>
        <Field label="Filtro de archivos (opcional)" hint="Para repos GGUF, p. ej. *Q4_K_M.gguf"><input value={pattern} onChange={(e) => setPattern(e.target.value)} /></Field>
        <div className="row"><div className="spacer" /><button className="primary" disabled={!repo} onClick={() => go()}>Descargar</button></div>
        <h3 style={{ marginTop: 8 }}>Sugeridos para una GPU de 24 GB</h3>
        <table><tbody>
          {suggested.map((s) => (
            <tr key={s.repo}><td><b>{s.repo}</b><div className="sub" style={{ fontSize: 12 }}>{s.note}</div></td><td>{s.size}</td>
              <td><button className="sm" onClick={() => setRepo(s.repo)}>Elegir</button></td></tr>
          ))}
        </tbody></table>
        <div className="sub">Consejo: no hace falta descargar antes de entrenar. Si pones un repo como modelo base, se descarga automáticamente a la caché de HF.</div>
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
    <Modal title="Registrar modelo" onClose={onClose}>
      <div className="stack">
        <Field label="Repo de HF o ruta local" hint="Carpeta HF, carpeta con adapter_config.json (LoRA) o archivo .gguf. El tipo se detecta solo."><input value={path} onChange={(e) => setPath(e.target.value)} placeholder="meta-llama/Llama-3.1-8B-Instruct  ó  /ruta/a/modelo.gguf" /></Field>
        <Field label="Nombre"><input value={name} onChange={(e) => setName(e.target.value)} placeholder="auto" /></Field>
        <Field label="Modelo base (sólo LoRA, opcional)"><input value={base} onChange={(e) => setBase(e.target.value)} /></Field>
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
    <Modal title={`Exportar ${model.name}`} onClose={onClose}>
      <div className="stack">
        {model.kind === "lora" && <div>Se fusionará el adaptador LoRA con <code>{model.base_model}</code> en 16-bit, creando un modelo independiente.</div>}
        <label className="check"><input type="checkbox" checked={gguf} onChange={(e) => setGguf(e.target.checked)} disabled={model.kind !== "lora"} /> Convertir a GGUF (llama.cpp / Ollama / LM Studio)</label>
        {gguf && (
          <>
            <Field label="Cuantización GGUF">
              <select value={q} onChange={(e) => setQ(e.target.value)}>
                {["q4_k_m", "q5_k_m", "q6_k", "q8_0", "f16", "bf16"].map((x) => <option key={x}>{x}</option>)}
              </select>
            </Field>
            {!ggufAvailable && <div className="error-box">Falta llama.cpp para exportar a GGUF. <Link to="/system">Instálalo desde Sistema</Link> con un clic (~6 min).</div>}
            <div className="sub" style={{ fontSize: 12 }}>q4_k_m: el mejor equilibrio tamaño/calidad · q8_0: casi sin pérdida · f16: sin cuantizar. Se genera también un <code>Modelfile</code> para Ollama.</div>
          </>
        )}
        <div className="row"><div className="spacer" /><button className="primary" onClick={go} disabled={gguf && !ggufAvailable}>Exportar</button></div>
      </div>
    </Modal>
  );
}

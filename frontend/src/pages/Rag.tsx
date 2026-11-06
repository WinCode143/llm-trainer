import { useState } from "react";
import { Download, Eye, FileText, Globe, Image as ImageIcon, Layers, Pencil, Plus, Save, Search, Sparkles, Tag, Trash2, Upload, X } from "lucide-react";
import { api, type Collection, fmtDate } from "../api";
import { GraphView } from "../components/GraphView";
import { Adv, Check, Field, Help, Modal, SimpleNote, Spinner, useData, useDialog, useToast } from "../components/ui";
import { HelperPicker } from "../components/Helper";

const MODES: Record<string, string> = { hybrid: "Híbrido (BM25 + vectores)", vector: "Sólo vectores", bm25: "Sólo BM25 (palabras)" };
const DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3";
type Options = { templates: Record<string, string>; query_modes: Record<string, string> };
/** Nombre legible de un documento (las URLs se muestran decodificadas). */
const prettyDoc = (d: string) => { try { return d.startsWith("http") ? decodeURIComponent(d) : d; } catch { return d; } };
const allTags = (c: Collection) => [...new Set(Object.values(c.doc_meta ?? {}).flatMap((m) => m?.tags ?? []))].sort();

export default function Rag() {
  const [cols, reload] = useData<Collection[]>("/api/rag/collections");
  const [sel, setSel] = useState<number | null>(null);
  const [creating, setCreating] = useState(false);
  const cur = cols?.find((c) => c.id === sel) ?? cols?.[0];

  return (
    <>
      <div className="page-head">
        <div><h1>RAG</h1><div className="sub">Colecciones de documentos con búsqueda híbrida, reranking y umbral de relevancia</div></div>
        <button className="primary" onClick={() => setCreating(true)}><Plus size={14} /> Nueva colección</button>
      </div>
      <Help title="¿Qué es RAG y qué hace cada pestaña?">
        <p>RAG = el modelo <b>busca en tus documentos antes de responder</b>. Así contesta con información tuya y cita de dónde la sacó, sin necesidad de entrenarlo.</p>
        <p style={{ marginTop: 6 }}>El proceso: los documentos se trocean en fragmentos → cada fragmento se convierte en un vector (embedding) → al preguntar se buscan los más parecidos (por significado y por palabras) → un <b>reranker</b> los reordena → los mejores se le pasan al modelo como contexto.</p>
        <ul>
          <li><b>Documentos</b>: subir archivos o páginas web, etiquetarlos y lanzar los extras (preguntas por fragmento, resúmenes, descripción de imágenes).</li>
          <li><b>Probar búsqueda</b>: ver qué se recupera y con qué puntuación, antes de preguntar al modelo.</li>
          <li><b>Grafo</b>: entidades y relaciones extraídas de los documentos (GraphRAG), para preguntas que cruzan varios fragmentos.</li>
          <li><b>Benchmark</b>: compara modos de búsqueda con tus propias preguntas y te dice cuál acierta más.</li>
          <li><b>Ajustes</b>: modo de búsqueda, reranker, umbral, plantilla de troceo y modelo de embeddings.</li>
        </ul>
      </Help>
      {(cols?.length ?? 0) >= 2 && <CompararColecciones cols={cols ?? []} />}

      <div className="split">
        <div className="card" style={{ padding: 8 }}>
          {!cols?.length && <div className="empty">Sin colecciones</div>}
          <table><tbody>
            {cols?.map((c) => (
              <tr key={c.id} className={`clickable ${cur?.id === c.id ? "selected" : ""}`} onClick={() => setSel(c.id)}>
                <td><b>{c.name}</b><div className="sub" style={{ fontSize: 12 }}>{c.documents.length} docs · {c.num_chunks} fragmentos{c.reranker ? " · rerank" : ""}</div>
                  <div className="row" style={{ gap: 3, marginTop: 3 }}>
                    {c.features?.questions ? <span className="badge ok" title={`${c.features.questions} fragmentos con preguntas generadas`}>preguntas</span> : null}
                    {c.features?.summaries ? <span className="badge ok" title={`${c.features.summaries} resúmenes`}>resúmenes</span> : null}
                    {c.features?.images ? <span className="badge ok" title={`${c.features.images} imágenes descritas`}>imágenes</span> : null}
                    {c.features?.graph ? <span className="badge ok" title="Grafo de conocimiento construido">grafo</span> : null}
                  </div></td>
              </tr>
            ))}
          </tbody></table>
        </div>
        {cur ? <CollectionDetail key={cur.id} c={cur} onChange={reload} /> : <div className="card empty">Crea una colección y sube PDF, DOCX, Markdown, TXT o HTML.</div>}
      </div>
      {creating && <CreateModal onClose={() => setCreating(false)} onDone={(c) => { setCreating(false); reload(); setSel(c.id); }} />}
    </>
  );
}

function CollectionDetail({ c, onChange }: { c: Collection; onChange: () => void }) {
  const [tab, setTab] = useState<"docs" | "search" | "graph" | "bench" | "settings">("docs");
  const { confirm } = useDialog();
  return (
    <div className="stack" style={{ gap: 16 }}>
      <div className="card">
        <div className="row between" style={{ flexWrap: "nowrap", alignItems: "flex-start" }}>
          <div style={{ minWidth: 0 }}>
            <h2 style={{ marginBottom: 2 }}>{c.name}</h2>
            <div className="sub">
              {MODES[c.search_mode]} · {c.reranker ? <>rerank <code>{c.reranker.split("/").pop()}</code></> : "sin rerank"} · top-{c.top_k}
              {c.query_mode && c.query_mode !== "simple" && <> · consulta <b>{c.query_mode}</b></>}
              {c.min_score > 0 && <> · umbral {c.min_score}</>} · embeddings <code>{c.embed_model.split("/").pop()}</code> · {fmtDate(c.created_at)}
            </div>
          </div>
          <button className="danger" onClick={async () => (await confirm(`¿Borrar la colección «${c.name}»?`, `${c.documents.length} documentos y ${c.num_chunks} fragmentos indexados.`, { label: "Borrar" })) && api.del(`/api/rag/collections/${c.id}`).then(onChange)}><Trash2 size={14} /></button>
        </div>
        <div className="tabs" style={{ marginTop: 12, marginBottom: 0 }}>
          {([["docs", "Documentos"], ["search", "Probar búsqueda"], ["graph", "Grafo"], ["bench", "Benchmark"], ["settings", "Ajustes"]] as const).map(([k, l]) => (
            <button key={k} className={`tab ${tab === k ? "active" : ""}`} onClick={() => setTab(k)}>{l}</button>
          ))}
        </div>
      </div>
      {tab === "docs" && <Documents c={c} onChange={onChange} />}
      {tab === "search" && <SearchTester c={c} />}
      {tab === "graph" && <Graph c={c} />}
      {tab === "bench" && <Benchmark c={c} />}
      {tab === "settings" && <Settings c={c} onChange={onChange} />}
    </div>
  );
}

// ---------------- Documentos + visor de chunks ----------------
function Documents({ c, onChange }: { c: Collection; onChange: () => void }) {
  const [drag, setDrag] = useState(false);
  const [viewing, setViewing] = useState<string | null>(null);
  const [opts] = useData<Options>("/api/rag/options");
  const [template, setTemplate] = useState(c.chunk_template || "general");
  const [tags, setTags] = useState("");
  const [urls, setUrls] = useState("");
  const [sitemap, setSitemap] = useState("");
  const [status] = useData<any>("/api/models/inference/status", 5000);
  const { track, notify } = useToast();
  const { confirm, prompt } = useDialog();
  const upload = async (files: FileList | File[]) => {
    const fd = new FormData();
    for (const f of Array.from(files)) fd.append("files", f);
    fd.append("template", template);
    if (tags.trim()) fd.append("tags", tags);
    try { track(await api.form(`/api/rag/collections/${c.id}/documents`, fd), onChange); } catch (e: any) { notify(e.message, "err"); }
  };
  const addUrls = async () => {
    try {
      const t = await api.post<any>(`/api/rag/collections/${c.id}/urls`, {
        urls: urls.split(/\s+/).filter(Boolean), sitemap: sitemap || null, template, tags: tags.split(",").map((x) => x.trim()).filter(Boolean),
      });
      track(t, (done) => done.result?.ingest_task && track({ ...t, id: done.result.ingest_task, title: `Indexar páginas en ${c.name}` }, onChange));
      setUrls(""); setSitemap("");
    } catch (e: any) { notify(e.message, "err"); }
  };
  const run = async (what: "enrich" | "summaries" | "images") => {
    if (!status?.loaded) return notify("Carga primero un modelo (Playground o Modelos): se usa para generar el texto", "err");
    const ok = await confirm(
      what === "enrich" ? "Generar preguntas por fragmento" : what === "images" ? "Describir imágenes de los PDF" : "Crear resúmenes jerárquicos",
      what === "images"
        ? `Se extraen las imágenes de los PDF indexados y el modelo de visión cargado (${status.loaded.name}) las describe. Cada descripción se indexa como fragmento buscable y la imagen se muestra como fuente. Requiere haber subido el PDF con esta versión (se guarda el original).`
        : what === "enrich"
        ? `El modelo cargado (${status.loaded.name}) escribirá 3 preguntas y palabras clave por fragmento sin enriquecer. Se indexan como vectores propios y en BM25, y sirven como pares para ajustar embeddings. Tarda ~2–6 s por fragmento (${c.num_chunks} en total). Mide el efecto con Benchmark: con modelos pequeños (<7B) puede no mejorar la búsqueda.`
        : `El modelo cargado (${status.loaded.name}) resumirá grupos de fragmentos y cada documento completo. Permite responder preguntas globales («¿de qué trata…?», «resume…»).`,
      { label: "Empezar", danger: false });
    if (!ok) return;
    try { track(await api.post(`/api/rag/collections/${c.id}/${what}`, {}), onChange); } catch (e: any) { notify(e.message, "err"); }
  };
  const editTags = async (doc: string) => {
    const cur = c.doc_meta?.[doc]?.tags ?? [];
    const v = await prompt(`Etiquetas de «${doc}»`, { message: "Separadas por comas. Sirven para filtrar la búsqueda (Playground, API, MCP).", value: cur.join(", ") });
    if (v === null) return;
    await api.patch(`/api/rag/collections/${c.id}/documents`, { filename: doc, tags: v.split(",").map((x) => x.trim()).filter(Boolean) });
    onChange();
  };
  return (
    <div className="card">
      <label
        className="card empty" style={{ borderStyle: "dashed", borderColor: drag ? "var(--accent)" : undefined, cursor: "pointer", display: "block" }}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); upload(e.dataTransfer.files); }}>
        <Upload size={18} /><div>Arrastra documentos o haz clic — PDF (con OCR si está escaneado), DOCX, PPTX, XLSX, MD, TXT, HTML, CSV</div>
        <div className="sub" style={{ fontSize: 12 }}>Se trocean por secciones (títulos) y cada fragmento se indexa con su ruta «documento › sección»</div>
        <input type="file" multiple hidden accept=".pdf,.docx,.pptx,.xlsx,.md,.txt,.html,.htm,.csv,.json" onChange={(e) => e.target.files && upload(e.target.files)} />
      </label>
      <div className="grid g2" style={{ marginTop: 10 }}>
        <Field label="Plantilla de troceo" hint={opts?.templates[template]}>
          <select value={template} onChange={(e) => setTemplate(e.target.value)}>
            {Object.entries(opts?.templates ?? { general: "" }).map(([k, d]) => <option key={k} value={k}>{k} — {d}</option>)}
          </select>
        </Field>
        <Field label="Etiquetas para lo que subas" hint="Separadas por comas (p. ej. «legal, 2025»)"><input value={tags} onChange={(e) => setTags(e.target.value)} /></Field>
      </div>
      <details style={{ marginTop: 10 }}>
        <summary style={{ cursor: "pointer" }}><Globe size={13} /> <b>Añadir páginas web</b> <span className="sub">— URLs sueltas o un sitemap completo</span></summary>
        <div className="stack" style={{ marginTop: 8 }}>
          <textarea rows={3} value={urls} onChange={(e) => setUrls(e.target.value)} placeholder={"https://ejemplo.com/pagina-1\nhttps://ejemplo.com/manual.pdf"} />
          <div className="row" style={{ flexWrap: "nowrap" }}>
            <input value={sitemap} onChange={(e) => setSitemap(e.target.value)} placeholder="https://ejemplo.com/sitemap.xml (opcional, hasta 30 páginas)" />
            <button className="primary" disabled={!urls.trim() && !sitemap.trim()} onClick={addUrls}>Descargar e indexar</button>
          </div>
        </div>
      </details>
      <div className="row" style={{ marginTop: 12 }}>
        <button onClick={() => run("enrich")} title="El LLM genera preguntas por fragmento"><Sparkles size={14} /> Generar preguntas por fragmento</button>
        <button onClick={() => run("summaries")} title="Resúmenes por grupos y por documento (tipo RAPTOR)"><Layers size={14} /> Crear resúmenes jerárquicos</button>
        <button onClick={() => run("images")} title="Requiere un modelo de visión cargado (Qwen-VL…)" disabled={!status?.loaded?.vision}><ImageIcon size={14} /> Describir imágenes de PDF</button>
        <span className="sub" style={{ fontSize: 12 }}>{status?.loaded ? `usará ${status.loaded.name}${status.loaded.vision ? " (con visión)" : ""}` : "requieren un modelo cargado"}</span>
      </div>
      {c.documents.length > 0 && (
        <table style={{ marginTop: 10 }}><tbody>
          {c.documents.map((d) => (
            <tr key={d}><td><FileText size={13} /> <span title={d}>{prettyDoc(d)}</span>
                <div className="row" style={{ gap: 4, marginTop: 2 }}>
                  {c.doc_meta?.[d]?.template && c.doc_meta[d].template !== "general" && <span className="badge">{c.doc_meta[d].template}</span>}
                  {(c.doc_meta?.[d]?.tags ?? []).map((t) => <span key={t} className="badge accent">{t}</span>)}
                  {c.originals && c.originals[d] === false && (
                    <span className="badge" title="No se conserva el archivo original: se puede buscar y responder con normalidad, pero no volver a trocearlo con otra plantilla, cambiar de parser ni extraer sus imágenes. Vuelve a subirlo si lo necesitas.">sin original</span>
                  )}
                </div></td>
              <td style={{ textAlign: "right", whiteSpace: "nowrap" }}>
                <button className="sm ghost" onClick={() => editTags(d)} title="Etiquetas"><Tag size={13} /></button>
                <button className="sm ghost" onClick={() => setViewing(d)}><Eye size={13} /> Fragmentos</button>
                <button className="sm ghost danger" onClick={async () => (await confirm(`¿Quitar «${d}» de la colección?`, undefined, { label: "Quitar" })) && api.del(`/api/rag/collections/${c.id}/documents?filename=${encodeURIComponent(d)}`).then(onChange)}><Trash2 size={13} /></button>
              </td></tr>
          ))}
        </tbody></table>
      )}
      {viewing && <ChunkViewer c={c} source={viewing} onClose={() => { setViewing(null); onChange(); }} />}
    </div>
  );
}

function ChunkViewer({ c, source, onClose }: { c: Collection; source: string; onClose: () => void }) {
  const [page, setPage] = useState(0);
  const [data, reload] = useData<{ total: number; chunks: any[] }>(`/api/rag/collections/${c.id}/chunks?source=${encodeURIComponent(source)}&offset=${page * 20}&limit=20`);
  const [editing, setEditing] = useState<{ id: string; text: string } | null>(null);
  const { notify } = useToast();
  const save = async () => {
    try { await api.put(`/api/rag/collections/${c.id}/chunks/${editing!.id}`, { text: editing!.text }); setEditing(null); reload(); notify("Fragmento actualizado y re-indexado"); } catch (e: any) { notify(e.message, "err"); }
  };
  return (
    <Modal title={`${prettyDoc(source)} — ${data?.total ?? "…"} fragmentos`} onClose={onClose} wide>
      <div className="stack">
        {data?.chunks.map((ch) => (
          <div key={ch.id} className="source">
            <div className="row between">
              <span className="sub" style={{ fontSize: 12 }}>{ch.level ? <span className="badge warn">resumen N{ch.level}</span> : `#${ch.chunk}`}{ch.page ? ` · pág. ${ch.page}` : ""}{ch.section ? ` · ${ch.section}` : ""} · {ch.text.length} car.{ch.edited ? " · editado" : ""}</span>
              <div className="row" style={{ gap: 2 }}>
                {editing?.id === ch.id
                  ? <><button className="sm primary" onClick={save}><Save size={12} /></button><button className="sm ghost" onClick={() => setEditing(null)}><X size={12} /></button></>
                  : <button className="sm ghost" onClick={() => setEditing({ id: ch.id, text: ch.text })}><Pencil size={12} /></button>}
                <button className="sm ghost danger" onClick={() => api.del(`/api/rag/collections/${c.id}/chunks/${ch.id}`).then(reload)}><Trash2 size={12} /></button>
              </div>
            </div>
            {editing?.id === ch.id
              ? <textarea rows={8} value={editing?.text ?? ""} onChange={(e) => setEditing({ id: ch.id, text: e.target.value })} style={{ marginTop: 6 }} />
              : <div style={{ whiteSpace: "pre-wrap", fontSize: 13, marginTop: 4 }}>{ch.text}</div>}
            {ch.questions && <div className="sub" style={{ fontSize: 12, marginTop: 6, borderTop: "1px dashed var(--border)", paddingTop: 4 }}>
              <Sparkles size={11} /> {ch.questions.split("\n").join(" · ")}{ch.keywords ? <> · <i>{ch.keywords}</i></> : null}</div>}
          </div>
        ))}
        <div className="row" style={{ justifyContent: "center" }}>
          <button className="sm" disabled={page === 0} onClick={() => setPage(page - 1)}>Anterior</button>
          <span className="sub">{page * 20 + 1}–{Math.min((page + 1) * 20, data?.total ?? 0)}</span>
          <button className="sm" disabled={(page + 1) * 20 >= (data?.total ?? 0)} onClick={() => setPage(page + 1)}>Siguiente</button>
        </div>
      </div>
    </Modal>
  );
}

// ---------------- Probador de búsqueda ----------------
function SearchTester({ c }: { c: Collection }) {
  const [q, setQ] = useState("");
  const [k, setK] = useState(c.top_k);
  const [mode, setMode] = useState<string>(c.search_mode);
  const [rerank, setRerank] = useState(!!c.reranker);
  const [qmode, setQmode] = useState(c.query_mode || "simple");
  const [ftags, setFtags] = useState<string[]>([]);
  const [opts] = useData<Options>("/api/rag/options");
  const [res, setRes] = useState<{ hits: any[]; ms: number } | null>(null);
  const [busy, setBusy] = useState(false);
  const { notify } = useToast();
  const search = async () => {
    setBusy(true);
    try {
      setRes(await api.post(`/api/rag/collections/${c.id}/search`, { query: q, k, mode, reranker: rerank ? c.reranker || DEFAULT_RERANKER : "", min_score: 0, query_mode: qmode, tags: ftags.length ? ftags : null }));
    } catch (e: any) { notify(e.message, "err"); } finally { setBusy(false); }
  };
  return (
    <div className="card stack">
      <div className="row" style={{ flexWrap: "nowrap" }}>
        <input value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && search()} placeholder="Escribe una pregunta…" />
        <button className="primary" onClick={search} disabled={!q || busy}>{busy ? <Spinner /> : <Search size={14} />}</button>
      </div>
      <div className="row" style={{ gap: 14 }}>
        <select value={mode} onChange={(e) => setMode(e.target.value)} style={{ width: "auto" }}>
          {Object.entries(MODES).map(([k, l]) => <option key={k} value={k}>{l}</option>)}
        </select>
        <label className="check"><input type="checkbox" checked={rerank} onChange={(e) => setRerank(e.target.checked)} /> Reranking</label>
        <select value={qmode} onChange={(e) => setQmode(e.target.value)} style={{ width: "auto" }} title={opts?.query_modes[qmode]}>
          {Object.keys(opts?.query_modes ?? { simple: "" }).map((m) => <option key={m} value={m}>consulta: {m}</option>)}
        </select>
        <label className="check">top-k <input type="number" value={k} onChange={(e) => setK(+e.target.value)} style={{ width: 60 }} /></label>
        {res && <span className="sub">{res.hits.length} resultados · {res.ms} ms</span>}
      </div>
      {allTags(c).length > 0 && (
        <div className="row" style={{ gap: 6 }}><span className="sub">Filtrar:</span>
          {allTags(c).map((t) => <button key={t} className={`sm ${ftags.includes(t) ? "primary" : ""}`} onClick={() => setFtags(ftags.includes(t) ? ftags.filter((x) => x !== t) : [...ftags, t])}><Tag size={11} /> {t}</button>)}
        </div>
      )}
      {qmode !== "simple" && <div className="sub" style={{ fontSize: 12 }}>{opts?.query_modes[qmode]} — necesita un modelo cargado.</div>}
      {(res?.hits[0]?.rewritten_queries || res?.hits[0]?.agent_queries) && (
        <div className="source sub" style={{ fontSize: 12 }}><b>{res!.hits[0].agent_queries ? "Búsquedas del agente" : "Consultas generadas"}:</b>
          <ol style={{ margin: "4px 0 0", paddingLeft: 18 }}>{(res!.hits[0].agent_queries ?? res!.hits[0].rewritten_queries).map((x: string, i: number) => <li key={i}>{x}</li>)}</ol></div>
      )}
      {res && !res.hits.length && <div className="empty">Sin resultados</div>}
      {res?.hits.map((h, i) => {
        const pass = h.score >= (c.min_score || 0);
        return (
          <div key={h.id} className="source" style={{ opacity: pass ? 1 : 0.55 }}>
            <div className="row between">
              <b>[{i + 1}] {prettyDoc(h.source)}{h.page ? ` · pág. ${h.page}` : ""}</b>
              <div className="row" style={{ gap: 4 }}>
                {h.rerank_score != null && <span className="badge ok" title="Score del reranker (0–1): lo que usa el umbral">rerank {h.rerank_score.toFixed(3)}</span>}
                <span className="badge accent" title="Similitud coseno con la pregunta">cos {h.vector_score?.toFixed(3)}</span>
                {h.bm25_rank && <span className="badge" title="Posición en BM25 (palabras)">bm25 #{h.bm25_rank}</span>}
                {h.vector_rank && <span className="badge" title="Posición por vectores">vec #{h.vector_rank}</span>}
                {!pass && <span className="badge err" title={`Por debajo del umbral ${c.min_score}`}>bajo umbral</span>}
              </div>
            </div>
            {h.section && <div className="sub" style={{ fontSize: 12 }}>{h.section}</div>}
            {h.image && <img src={`/api/rag/collections/${c.id}/images/${h.image}`} alt="" style={{ maxHeight: 180, borderRadius: 8, marginTop: 6 }} />}
            <div style={{ whiteSpace: "pre-wrap", fontSize: 13, marginTop: 4 }}>{h.text}</div>
          </div>
        );
      })}
    </div>
  );
}

// ---------------- Benchmark de recuperación ----------------
function parseQs(text: string) {
  return text.split("\n").filter((l) => l.trim()).map((l) => {
    const [question, kw] = l.split("|").map((s) => s.trim());
    return { question, keywords: kw ? kw.split(",").map((s) => s.trim()).filter(Boolean) : [] };
  });
}

function Benchmark({ c }: { c: Collection }) {
  const [text, setText] = useState("");
  const [k, setK] = useState(c.top_k);
  const [res, setRes] = useState<any[] | null>(null);
  const [busy, setBusy] = useState(false);
  const { notify } = useToast();
  const qs = parseQs(text);
  const run = async () => {
    setBusy(true);
    try {
      const t = await api.post<any>(`/api/rag/collections/${c.id}/benchmark`, { questions: qs, k });
      for (;;) {
        const s = await api.get<any>(`/api/tasks/${t.id}`);
        if (s.status === "completed") { setRes(s.result); break; }
        if (s.status === "failed") throw new Error(s.error?.split("\n")[0]);
        await new Promise((r) => setTimeout(r, 1000));
      }
    } catch (e: any) { notify(e.message, "err"); } finally { setBusy(false); }
  };
  const best = res ? Math.max(...res.map((r) => r.mrr)) : 0;
  return (
    <div className="card stack">
      <div className="sub">Compara vectores, BM25, híbrido e híbrido + reranking sobre tus propias preguntas. Un acierto = algún fragmento del top-k contiene todas las palabras clave.</div>
      <Field label={`Preguntas (${qs.length})`} hint="Una por línea: pregunta | palabra clave 1, palabra clave 2">
        <textarea rows={7} className="mono" value={text} onChange={(e) => setText(e.target.value)}
          placeholder={"¿Qué volcán destruyó Pompeya? | Vesubio\n¿Cuántos días de vacaciones hay? | 23"} />
      </Field>
      <div className="row">
        <label className="check">top-k <input type="number" value={k} onChange={(e) => setK(+e.target.value)} style={{ width: 60 }} /></label>
        <div className="spacer" />
        <button className="primary" disabled={busy || !qs.length || qs.some((q) => !q.keywords.length)} onClick={run}>{busy ? <Spinner /> : null} Ejecutar benchmark</button>
      </div>
      {qs.some((q) => !q.keywords.length) && text && <div className="sub">Cada línea necesita al menos una palabra clave tras «|».</div>}
      {res && (
        <>
          <table>
            <thead><tr><th>Configuración</th><th title="Fracción de preguntas con la respuesta en el top-k">Acierto@{k}</th><th title="Mean Reciprocal Rank: 1 = la respuesta siempre en 1.ª posición">MRR</th><th>Latencia media</th></tr></thead>
            <tbody>{res.map((r) => (
              <tr key={r.label} className={r.mrr === best ? "selected" : ""}><td><b>{r.label}</b>{r.mrr === best && " ★"}</td>
                <td>{Math.round(r.hit_rate * 100)}%</td><td>{r.mrr.toFixed(3)}</td><td>{r.latency_ms} ms</td></tr>
            ))}</tbody>
          </table>
          <h3 style={{ marginTop: 8 }}>Posición de la respuesta por pregunta</h3>
          <table>
            <thead><tr><th>Pregunta</th>{res.map((r) => <th key={r.label} style={{ textAlign: "center" }}>{r.label}</th>)}</tr></thead>
            <tbody>{res[0].per_question.map((pq: any, i: number) => (
              <tr key={i}><td>{pq.question}</td>
                {res.map((r) => { const rank = r.per_question[i].rank; return (
                  <td key={r.label} style={{ textAlign: "center", color: rank === 1 ? "var(--ok)" : rank ? "var(--warn)" : "var(--err)" }}>{rank ? `#${rank}` : "✗"}</td>); })}
              </tr>
            ))}</tbody>
          </table>
        </>
      )}
    </div>
  );
}

// ---------------- Ajustes ----------------
function Settings({ c, onChange }: { c: Collection; onChange: () => void }) {
  const [rerankers] = useData<any[]>("/api/rag/rerankers");
  const [opts] = useData<Options>("/api/rag/options");
  const [s, setS] = useState<{ search_mode: string; reranker: string; top_k: number; min_score: number; fallback: string; chunk_template: string; query_mode: string; parser: string }>(
    { search_mode: c.search_mode, reranker: c.reranker ?? "", top_k: c.top_k, min_score: c.min_score, fallback: c.fallback, chunk_template: c.chunk_template || "general", query_mode: c.query_mode || "simple", parser: (c as any).parser || "auto" });
  const { notify } = useToast();
  const save = async () => {
    try { await api.patch(`/api/rag/collections/${c.id}`, s); notify("Ajustes guardados"); onChange(); } catch (e: any) { notify(e.message, "err"); }
  };
  return (
    <div className="card stack">
      <div className="grid g2">
        <Adv><Field label="Modo de búsqueda" hint="Híbrido combina significado (vectores) y palabras exactas (BM25): mejor con nombres propios, códigos y cifras">
          <select value={s.search_mode} onChange={(e) => setS({ ...s, search_mode: e.target.value })}>
            {Object.entries(MODES).map(([k, l]) => <option key={k} value={k}>{l}</option>)}
          </select>
        </Field>
        <Field label="Reranker" hint="Reordena ~20 candidatos con un cross-encoder. Más preciso (+~300 ms) y da scores fiables para el umbral">
          <select value={s.reranker} onChange={(e) => setS({ ...s, reranker: e.target.value, min_score: e.target.value && !s.min_score ? 0.02 : e.target.value ? s.min_score : 0 })}>
            <option value="">Sin reranking</option>
            {rerankers?.map((r) => <option key={r.name} value={r.name}>{r.name} — {r.note}</option>)}
          </select>
        </Field>
        </Adv>
        <Field label="Fragmentos por respuesta (top-k)"><input type="number" min={1} max={20} value={s.top_k} onChange={(e) => setS({ ...s, top_k: +e.target.value })} /></Field>
        <Adv><Field label={`Umbral de relevancia: ${s.min_score}`} hint={s.reranker ? "Sobre el score del reranker. 0.02 descarta lo irrelevante sin perder paráfrasis" : "⚠ Sin reranker el umbral se aplica al coseno, que apenas discrimina (casi todo da 0.7–0.9)"}>
          <input type="range" min={0} max={s.reranker ? 0.5 : 1} step={0.01} value={s.min_score} onChange={(e) => setS({ ...s, min_score: +e.target.value })} />
        </Field>
        <Field label="Modo de consulta" hint={opts?.query_modes[s.query_mode]}>
          <select value={s.query_mode} onChange={(e) => setS({ ...s, query_mode: e.target.value })}>
            {Object.keys(opts?.query_modes ?? { simple: "" }).map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        </Field>
        <Field label="Plantilla de troceo por defecto" hint="Para documentos nuevos; los ya indexados no cambian">
          <select value={s.chunk_template} onChange={(e) => setS({ ...s, chunk_template: e.target.value })}>
            {Object.entries(opts?.templates ?? { general: "" }).map(([k, d]) => <option key={k} value={k}>{k} — {d}</option>)}
          </select>
        </Field>
        <Field label="Parser de documentos" hint="Docling analiza el layout y la estructura de tablas con modelos de visión (más lento, mejor con PDFs complejos). PPTX/XLSX siempre usan Docling.">
          <select value={s.parser} onChange={(e) => setS({ ...s, parser: e.target.value })}>
            <option value="auto">Rápido (pymupdf4llm + OCR Tesseract)</option>
            <option value="docling">Docling (layout y tablas complejas)</option>
          </select>
        </Field>
        </Adv>
        <Field label="Si ningún fragmento supera el umbral" hint="Evita que el modelo se bloquee (o invente) con contexto irrelevante">
          <select value={s.fallback} onChange={(e) => setS({ ...s, fallback: e.target.value })}>
            <option value="no_context">Responder sin RAG (con su propio conocimiento)</option>
            <option value="refuse">Decir que no está en los documentos</option>
          </select>
        </Field>
      </div>
      <SimpleNote>Búsqueda {s.search_mode}{s.reranker ? " con reranker" : ""}, modo de consulta «{s.query_mode}» y umbral {s.min_score}: los valores que mejor salieron en nuestro benchmark (1,00 de acierto, 0,95 de MRR).</SimpleNote>
      <div className="sub" style={{ fontSize: 12 }}>El tamaño de chunk ({c.chunk_size}/{c.chunk_overlap}) no se puede cambiar sin volver a subir los documentos.</div>
      <Reindex c={c} onChange={onChange} />
      <div className="row"><div className="spacer" /><button className="primary" onClick={save}><Save size={14} /> Guardar</button></div>
      <ChunkPreview size={c.chunk_size} overlap={c.chunk_overlap} />
    </div>
  );
}

function ChunkPreview({ size: s0, overlap: o0 }: { size: number; overlap: number }) {
  const [text, setText] = useState("");
  const [size, setSize] = useState(s0);
  const [overlap, setOverlap] = useState(o0);
  const [chunks, setChunks] = useState<string[] | null>(null);
  return (
    <details>
      <summary style={{ cursor: "pointer" }}><b>Laboratorio de chunking</b> <span className="sub">— prueba cómo se trocea un texto</span></summary>
      <div className="stack" style={{ marginTop: 12 }}>
        <textarea rows={5} value={text} onChange={(e) => setText(e.target.value)} placeholder="Pega texto aquí…" />
        <div className="row">
          <Field label="Tamaño"><input type="number" value={size} onChange={(e) => setSize(+e.target.value)} /></Field>
          <Field label="Solape"><input type="number" value={overlap} onChange={(e) => setOverlap(+e.target.value)} /></Field>
          <button style={{ alignSelf: "flex-end" }} onClick={async () => setChunks(await api.post("/api/rag/chunk-preview", { text, chunk_size: size, chunk_overlap: overlap }))}>Trocear</button>
        </div>
        {chunks?.map((c, i) => <div key={i} className="source"><span className="badge">{i + 1} · {c.length} car.</span><div style={{ whiteSpace: "pre-wrap", marginTop: 4 }}>{c}</div></div>)}
      </div>
    </details>
  );
}

function CreateModal({ onClose, onDone }: { onClose: () => void; onDone: (c: Collection) => void }) {
  const [rerankers] = useData<any[]>("/api/rag/rerankers");
  const [name, setName] = useState("");
  const [model, setModel] = useState("");
  const [size, setSize] = useState(800);
  const [overlap, setOverlap] = useState(120);
  const [reranker, setReranker] = useState(DEFAULT_RERANKER);
  const { notify } = useToast();
  const go = async () => {
    try {
      onDone(await api.post("/api/rag/collections", {
        name, embed_model: model || null, chunk_size: size, chunk_overlap: overlap, search_mode: "hybrid",
        reranker: reranker || null, min_score: reranker ? 0.02 : 0,
      }));
    } catch (e: any) { notify(e.message, "err"); }
  };
  return (
    <Modal title="Nueva colección" onClose={onClose}>
      <div className="stack">
        <Field label="Nombre"><input value={name} onChange={(e) => setName(e.target.value)} /></Field>
        <Field label="Modelo de embeddings" hint="Se puede cambiar luego, pero obliga a re-indexar toda la colección">
          <EmbedPicker value={model} onChange={setModel} />
        </Field>
        <Adv>
          <Field label="Reranker" hint="Recomendado: mejora mucho la precisión y permite filtrar contexto irrelevante">
            <select value={reranker} onChange={(e) => setReranker(e.target.value)}>
              <option value="">Sin reranking</option>
              {rerankers?.map((r) => <option key={r.name} value={r.name}>{r.name} — {r.note}</option>)}
            </select>
          </Field>
          <div className="grid g2">
            <Field label="Tamaño de chunk (caracteres)"><input type="number" value={size} onChange={(e) => setSize(+e.target.value)} /></Field>
            <Field label="Solape (caracteres)"><input type="number" value={overlap} onChange={(e) => setOverlap(+e.target.value)} /></Field>
          </div>
        </Adv>
        <SimpleNote>Se crea con búsqueda híbrida, reranker y troceo de 800 caracteres, que es lo que mejor nos ha funcionado. Todo se puede cambiar luego en Ajustes.</SimpleNote>
        <div className="row"><div className="spacer" /><button className="primary" disabled={!name} onClick={go}>Crear</button></div>
      </div>
    </Modal>
  );
}

function Reindex({ c, onChange }: { c: Collection; onChange: () => void }) {
  const [m, setM] = useState(c.embed_model);
  const { track, notify } = useToast();
  const go = async () => {
    try { track(await api.post(`/api/rag/collections/${c.id}/reindex`, { embed_model: m }), onChange); } catch (e: any) { notify(e.message, "err"); }
  };
  return (
    <div className="row source" style={{ padding: "8px 10px", flexWrap: "nowrap" }}>
      <span className="sub" style={{ whiteSpace: "nowrap" }}>Modelo de embeddings:</span>
      <EmbedPicker value={m} onChange={setM} current={c.embed_model} />
      <button className="sm" disabled={m === c.embed_model} onClick={go}>Re-indexar</button>
    </div>
  );
}


/** Selector de modelo de embeddings con descarga desde el catálogo.
 *  Cambiar de modelo obliga a re-indexar: los vectores viejos no son comparables con los nuevos. */
function EmbedPicker({ value, onChange, current }: { value: string; onChange: (v: string) => void; current?: string }) {
  const [models, reload] = useData<any[]>("/api/rag/embed-models");
  const { track, notify } = useToast();
  const sel = models?.find((m) => m.name === value);
  const pending = sel && !sel.downloaded;
  const download = async () => {
    try { track(await api.post(`/api/rag/embed-models/download`, { name: value }), () => reload()); }
    catch (e: any) { notify(e.message, "err"); }
  };
  return (
    <div className="stack" style={{ gap: 4 }}>
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {current && !models?.some((x) => x.name === current) && <option value={current}>{current}</option>}
        {!current && <option value="">Por defecto (multilingual-e5-small)</option>}
        {models?.map((x) => (
          <option key={x.name} value={x.name}>
            {x.tuned ? "★ " : x.downloaded ? "" : "⬇ "}{x.name.split("/").slice(-1)[0]}
            {x.dim ? ` · ${x.dim}d` : ""}{x.params ? ` · ${x.params}` : ""}{x.note ? ` — ${x.note}` : ""}
          </option>
        ))}
      </select>
      {pending && (
        <div className="row" style={{ gap: 6, alignItems: "center" }}>
          <span className="sub">Aún no está descargado.</span>
          <button className="sm" onClick={download}><Download size={12} /> Descargar ahora</button>
        </div>
      )}
      {sel?.dim && <div className="sub">Vector de {sel.dim} dimensiones. Los marcados con ⬇ se descargan solos la primera vez que se usan.</div>}
    </div>
  );
}

// ---------------- Grafo de conocimiento (GraphRAG) ----------------
function Graph({ c }: { c: Collection }) {
  const [g, reload] = useData<any>(`/api/rag/collections/${c.id}/graph?limit=250`);
  const [status] = useData<any>("/api/models/inference/status", 5000);
  const [q, setQ] = useState("");
  const [res, setRes] = useState<any>(null);
  const [maxChunks, setMaxChunks] = useState(60);
  const [src, setSrc] = useState("");
  const { track, notify } = useToast();
  const { confirm } = useDialog();
  const [helper, setHelper] = useState<number | null>(null);

  const build = async () => {
    if (!status?.loaded && !helper) return notify("Carga un modelo o elige un ayudante: el grafo lo extrae un LLM", "err");
    const ok = await confirm("Construir el grafo de conocimiento",
      `El modelo (${helper ? "ayudante externo" : status.loaded.name}) extraerá entidades y relaciones de ${maxChunks || "todos los"} fragmentos${src ? ` de ${src}` : ""}, y resumirá las comunidades detectadas. Tarda ~5-10 s por fragmento. Vuelve a construirlo para rehacerlo.`,
      { label: "Construir", danger: false });
    if (!ok) return;
    try { track(await api.post(`/api/rag/collections/${c.id}/graph`, { max_chunks: maxChunks || null, sources: src ? [src] : null, helper }), reload); } catch (e: any) { notify(e.message, "err"); }
  };
  const search = async () => {
    try { setRes(await api.get<any>(`/api/rag/collections/${c.id}/graph/query?q=${encodeURIComponent(q)}`)); } catch (e: any) { notify(e.message, "err"); }
  };

  return (
    <div className="card stack">
      <div className="sub">
        El grafo es <b>sólo de «{c.name}»</b>: cada colección tiene el suyo y no se mezclan.
        {g?.stats && <> Está construido a partir de <b>{g.stats.chunks ?? "?"} de los {c.num_chunks} fragmentos</b> de
        esta colección{(g.stats.chunks ?? 0) < c.num_chunks ? " — el resto no está en el grafo, sube «Fragmentos máx.» y reconstruye si quieres cubrirlos" : ""}.</>}
      </div>
      <div className="sub">El LLM extrae <b>entidades y relaciones</b> de cada fragmento y las une en un grafo. Responde mejor preguntas relacionales o de varios saltos («¿qué relación hay entre X e Y?», «¿quién participó en…?»), donde la información está repartida entre fragmentos. Para usarlo al responder, pon el modo de consulta en <b>graph</b> (pestaña Ajustes).</div>
      <div className="row">
        <Field label="Fragmentos máx." hint="0 = todos"><input type="number" value={maxChunks} onChange={(e) => setMaxChunks(+e.target.value)} style={{ width: 90 }} /></Field>
        <Field label="Sólo un documento (opcional)">
          <select value={src} onChange={(e) => setSrc(e.target.value)}><option value="">todos</option>{c.documents.map((d) => <option key={d} value={d}>{prettyDoc(d)}</option>)}</select>
        </Field>
        <HelperPicker value={helper} onChange={setHelper}
          hint="Extraer entidades es donde más se equivoca un modelo pequeño" />
        <button className="primary" style={{ alignSelf: "flex-end" }} onClick={build}>{g?.stats ? "Reconstruir" : "Construir"} grafo</button>
      </div>
      {!g?.stats ? <div className="empty">Aún no hay grafo para esta colección.</div> : (
        <>
          <div className="row" style={{ gap: 16 }}>
            <span className="sub">{g.stats.entities} entidades · {g.stats.edges} relaciones · {g.stats.communities} temas · {g.stats.chunks} fragmentos analizados{g.stats.failed ? ` · ${g.stats.failed} fallos de extracción` : ""}</span>
          </div>
          <div className="row" style={{ flexWrap: "nowrap" }}>
            <input value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && search()} placeholder="Probar una pregunta sobre el grafo…" />
            <button className="primary" onClick={search} disabled={!q}><Search size={14} /></button>
          </div>
          {res?.context && (
            <div className="source" style={{ fontSize: 13 }}>
              <div className="sub" style={{ fontSize: 12 }}>Entidades semilla: <b>{res.seeds.join(", ")}</b> · vecinos: {res.neighbors.slice(0, 6).join(", ")} · {res.relations} relaciones{res.communities?.length ? ` · temas: ${res.communities.join(", ")}` : ""}</div>
              <div style={{ whiteSpace: "pre-wrap", marginTop: 6 }}>{res.context}</div>
            </div>
          )}
          {res && !res.context && <div className="sub">Ninguna entidad del grafo coincide con esa pregunta.</div>}
          <div style={{ border: "1px solid var(--border)", borderRadius: 8, overflow: "hidden" }}>
            <GraphView nodes={g.nodes} links={g.links} highlight={res?.seeds} />
          </div>
          <div className="sub" style={{ fontSize: 12 }}>Tamaño = nº de conexiones · color = tema (comunidad) · pasa el ratón para ver descripción y vecinos · rojo = entidades de tu consulta</div>
          {g.communities?.length > 0 && (
            <details>
              <summary style={{ cursor: "pointer" }}><b>Temas detectados ({g.communities.length})</b></summary>
              <div className="stack" style={{ marginTop: 8 }}>
                {g.communities.map((cm: any) => (
                  <div key={cm.id} className="source"><b>{cm.title}</b> <span className="badge">{cm.size} entidades</span>
                    {cm.summary && <div className="sub" style={{ marginTop: 4 }}>{cm.summary}</div>}</div>
                ))}
              </div>
            </details>
          )}
        </>
      )}
    </div>
  );
}

/** Las mismas preguntas contra varias colecciones.
 *  Compara sólo la RECUPERACIÓN: el modelo no interviene, así que mide de verdad si cambiar el
 *  troceo, el reranker o los embeddings mejoró algo, sin que la respuesta lo enturbie. */
function CompararColecciones({ cols }: { cols: Collection[] }) {
  const [abierto, setAbierto] = useState(false);
  const [sel, setSel] = useState<number[]>([]);
  const [txt, setTxt] = useState("");
  const [res, setRes] = useState<any>(null);
  const { track, notify } = useToast();
  const preguntas = txt.split("\n").map((l) => l.trim()).filter(Boolean);

  const go = async () => {
    try {
      const t = await api.post<any>("/api/rag/compare", { collection_ids: sel, questions: preguntas });
      track(t, (done) => setRes(done.result));
    } catch (e: any) { notify(e.message, "err"); }
  };

  if (!abierto) {
    return (
      <div className="card" style={{ padding: "8px 12px" }}>
        <button className="ghost sm" onClick={() => setAbierto(true)}>
          <Search size={13} /> Comparar colecciones con las mismas preguntas
        </button>
      </div>
    );
  }
  return (
    <div className="card stack">
      <div className="row between">
        <b><Search size={14} /> Comparar recuperación</b>
        <button className="sm ghost" onClick={() => setAbierto(false)}>Cerrar</button>
      </div>
      <div className="sub">
        Lanza las mismas preguntas contra varias colecciones y enseña qué recupera cada una. No
        interviene el modelo: mide la búsqueda, que es lo que cambias cuando tocas el troceo, el
        reranker o los embeddings.
      </div>
      <Field label="Colecciones">
        <div className="row" style={{ gap: 14, flexWrap: "wrap" }}>
          {cols.map((c) => (
            <Check key={c.id} label={`${c.name} (${c.num_chunks})`} checked={sel.includes(c.id)}
              onChange={(v) => setSel(v ? [...sel, c.id] : sel.filter((x) => x !== c.id))} />
          ))}
        </div>
      </Field>
      <Field label={`Preguntas (${preguntas.length})`} hint="Una por línea">
        <textarea rows={4} value={txt} onChange={(e) => setTxt(e.target.value)}
          placeholder={"¿Cuál es el plazo de devolución?\n¿Qué garantía tienen los productos?"} />
      </Field>
      <div className="row end">
        <button className="primary" disabled={sel.length < 2 || !preguntas.length} onClick={go}>
          Comparar {sel.length} colecciones
        </button>
      </div>
      {res && (
        <table style={{ marginTop: 8 }}>
          <thead><tr><th>Colección</th><th>Ajustes</th><th>Fragmentos</th><th>Sin resultados</th><th>Tiempo</th></tr></thead>
          <tbody>
            {res.colecciones.map((c: any) => (
              <tr key={c.id}>
                <td><b>{c.coleccion}</b></td>
                <td className="sub">{c.ajustes.busqueda}{c.ajustes.reranker ? " + rerank" : ""} · k={c.ajustes.top_k} · troceo {c.ajustes.troceo}</td>
                <td>{c.fragmentos_media}</td>
                <td style={{ color: c.sin_resultados ? "var(--err)" : undefined }}>{c.sin_resultados}</td>
                <td>{c.segundos_media}s</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

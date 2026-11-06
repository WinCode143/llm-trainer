import { useState } from "react";
import { ChevronLeft, ChevronRight, Plus, Scissors, Trash2, Upload } from "lucide-react";
import { api, type Collection, type Dataset, fmtDate } from "../api";
import { Adv, Field, Help, Modal, SimpleNote, useData, useDialog, useToast } from "../components/ui";
import { HelperPicker } from "../components/Helper";

const FORMAT_LABEL = { chat: "Chat / instrucciones", text: "Texto plano", preference: "Preferencias (DPO)" };

export default function Datasets() {
  const [list, reload] = useData<Dataset[]>("/api/datasets");
  const [sel, setSel] = useState<number | null>(null);
  const [showImport, setShowImport] = useState(false);
  const current = list?.find((d) => d.id === sel) ?? list?.[0];

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Datasets</h1>
          <div className="sub">Importa, inspecciona y prepara datos de entrenamiento</div>
        </div>
        <button className="primary" onClick={() => setShowImport(true)}><Plus size={15} /> Nuevo dataset</button>
      </div>
      <Help title="¿Qué es un dataset y cómo se prepara?">
        <p>Un dataset son los <b>ejemplos de los que aprende el modelo</b>: pares de pregunta y respuesta con el estilo y el conocimiento que quieres que imite.</p>
        <ul>
          <li><b>Chat / instrucciones</b>: lo normal para enseñar a responder. Formatos aceptados: Alpaca (<code>instruction</code>/<code>output</code>), ShareGPT, OpenAI (<code>messages</code>).</li>
          <li><b>Preferencias</b> (<code>prompt</code>/<code>chosen</code>/<code>rejected</code>): para DPO, ORPO y KTO, que enseñan <i>qué respuesta es mejor</i>.</li>
          <li><b>Texto plano</b>: para preentrenamiento continuo sobre un dominio.</li>
        </ul>
        <p style={{ marginTop: 8 }}>De dónde sacarlo: subir un archivo, importar de Hugging Face, escribirlo a mano, o <b>generarlo</b> desde tus documentos RAG o desde un dominio (botón «Nuevo dataset» → Sintético). Con 50–100 ejemplos buenos ya se nota; divide en train/eval para poder medir.</p>
      </Help>
      <div className="split">
        <div className="card" style={{ padding: 8 }}>
          {!list?.length && <div className="empty">Sin datasets todavía</div>}
          <table>
            <tbody>
              {list?.map((d) => (
                <tr key={d.id} className={`clickable ${current?.id === d.id ? "selected" : ""}`} onClick={() => setSel(d.id)}>
                  <td>
                    <div><b>{d.name}</b></div>
                    <div className="sub" style={{ fontSize: 12 }}>{d.num_rows.toLocaleString()} filas · {FORMAT_LABEL[d.format]}</div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {current ? <DatasetDetail key={current.id} ds={current} onChange={reload} /> : <div className="card empty">Crea o importa un dataset para empezar.</div>}
      </div>
      {showImport && <ImportModal onClose={() => setShowImport(false)} onDone={(d) => { reload(); setSel(d?.id ?? null); setShowImport(false); }} />}
    </>
  );
}

function DatasetDetail({ ds, onChange }: { ds: Dataset; onChange: () => void }) {
  const [page, setPage] = useState(0);
  const PAGE = 10;
  const [rows] = useData<{ total: number; rows: any[] }>(`/api/datasets/${ds.id}/rows?offset=${page * PAGE}&limit=${PAGE}&v=${ds.num_rows}`);
  const [stats] = useData<any>(`/api/datasets/${ds.id}/stats?v=${ds.num_rows}`);
  const { notify } = useToast();
  const { confirm, prompt } = useDialog();
  const [adding, setAdding] = useState(false);

  const remove = async () => {
    if (!(await confirm(`¿Borrar el dataset «${ds.name}»?`, `${ds.num_rows} filas. No se puede deshacer.`, { label: "Borrar" }))) return;
    await api.del(`/api/datasets/${ds.id}`);
    onChange();
  };
  const split = async () => {
    const r = await prompt("Dividir en train / eval", { message: "Proporción de filas que irán al conjunto de evaluación (0–1). Se crearán dos datasets nuevos: -train y -eval.", value: "0.1" });
    if (!r) return;
    try {
      await api.post(`/api/datasets/${ds.id}/split`, { eval_ratio: parseFloat(r) });
      notify("Creados los datasets -train y -eval");
      onChange();
    } catch (e: any) { notify(e.message, "err"); }
  };

  return (
    <div className="stack" style={{ gap: 16 }}>
      <div className="card">
        <div className="row between">
          <div>
            <h2 style={{ marginBottom: 2 }}>{ds.name}</h2>
            <div className="sub">{FORMAT_LABEL[ds.format]} · {ds.source} · {fmtDate(ds.created_at)}</div>
          </div>
          <div className="row">
            <button onClick={() => setAdding(true)}><Plus size={14} /> Añadir ejemplo</button>
            <button onClick={split}><Scissors size={14} /> Dividir train/eval</button>
            <button className="danger" onClick={remove}><Trash2 size={14} /></button>
          </div>
        </div>
        {stats && (
          <div className="grid g4" style={{ marginTop: 14 }}>
            <div className="stat"><div className="label">Filas</div><div className="value">{stats.rows.toLocaleString()}</div></div>
            <div className="stat"><div className="label">≈ Tokens totales</div><div className="value">{stats.approx_tokens_total.toLocaleString()}</div></div>
            <div className="stat"><div className="label">Caracteres (media / p95)</div><div className="value">{stats.chars_avg} / {stats.chars_p95}</div></div>
            <div className="stat"><div className="label">{stats.turns_avg ? "Turnos por ejemplo" : "Máximo"}</div><div className="value">{stats.turns_avg ?? stats.chars_max}</div></div>
          </div>
        )}
      </div>

      <div className="card">
        <div className="row between" style={{ marginBottom: 10 }}>
          <h3 style={{ margin: 0 }}>Vista previa</h3>
          <div className="row">
            <button className="sm" disabled={page === 0} onClick={() => setPage(page - 1)}><ChevronLeft size={14} /></button>
            <span className="sub">{page * PAGE + 1}–{Math.min((page + 1) * PAGE, rows?.total ?? 0)} de {rows?.total ?? 0}</span>
            <button className="sm" disabled={(page + 1) * PAGE >= (rows?.total ?? 0)} onClick={() => setPage(page + 1)}><ChevronRight size={14} /></button>
          </div>
        </div>
        <div className="stack">
          {rows?.rows.map((r, i) => <RowView key={i} row={r} format={ds.format} />)}
        </div>
      </div>
      {adding && <AddRowModal ds={ds} onClose={() => setAdding(false)} onDone={() => { setAdding(false); onChange(); }} />}
    </div>
  );
}

function RowView({ row, format }: { row: any; format: Dataset["format"] }) {
  if (format === "chat")
    return (
      <div className="stack" style={{ gap: 6, borderBottom: "1px solid var(--border)", paddingBottom: 10 }}>
        {row.messages.map((m: any, i: number) => (
          <div key={i} className="row" style={{ alignItems: "flex-start", flexWrap: "nowrap" }}>
            <span className={`badge ${m.role === "assistant" ? "ok" : m.role === "system" ? "warn" : "accent"}`} style={{ minWidth: 72, textAlign: "center" }}>{m.role}</span>
            <div style={{ whiteSpace: "pre-wrap", flex: 1 }}>{m.content}</div>
          </div>
        ))}
      </div>
    );
  if (format === "preference")
    return (
      <div className="stack" style={{ gap: 6, borderBottom: "1px solid var(--border)", paddingBottom: 10 }}>
        <div><span className="badge accent">prompt</span> {row.prompt}</div>
        <div><span className="badge ok">chosen</span> {row.chosen}</div>
        <div><span className="badge err">rejected</span> {row.rejected}</div>
      </div>
    );
  return <div style={{ whiteSpace: "pre-wrap", borderBottom: "1px solid var(--border)", paddingBottom: 10 }}>{row.text}</div>;
}

function AddRowModal({ ds, onClose, onDone }: { ds: Dataset; onClose: () => void; onDone: () => void }) {
  const [a, setA] = useState("");
  const [b, setB] = useState("");
  const [c, setC] = useState("");
  const { notify } = useToast();
  const save = async () => {
    const row = ds.format === "chat" ? { instruction: a, output: b } : ds.format === "text" ? { text: a } : { prompt: a, chosen: b, rejected: c };
    try { await api.post(`/api/datasets/${ds.id}/append`, { rows: [row] }); onDone(); } catch (e: any) { notify(e.message, "err"); }
  };
  return (
    <Modal title="Añadir ejemplo" onClose={onClose}>
      <div className="stack">
        <Field label={ds.format === "chat" ? "Usuario" : ds.format === "text" ? "Texto" : "Prompt"}><textarea value={a} onChange={(e) => setA(e.target.value)} /></Field>
        {ds.format !== "text" && <Field label={ds.format === "chat" ? "Asistente" : "Respuesta preferida (chosen)"}><textarea value={b} onChange={(e) => setB(e.target.value)} /></Field>}
        {ds.format === "preference" && <Field label="Respuesta rechazada (rejected)"><textarea value={c} onChange={(e) => setC(e.target.value)} /></Field>}
        <div className="row"><div className="spacer" /><button className="primary" onClick={save} disabled={!a}>Guardar</button></div>
      </div>
    </Modal>
  );
}

function ImportModal({ onClose, onDone }: { onClose: () => void; onDone: (d?: Dataset) => void }) {
  const [tab, setTab] = useState<"file" | "hf" | "manual" | "synth" | "plantilla">("file");
  const [manualRows, setManualRows] = useState([{ a: "", b: "", c: "" }]);
  const [name, setName] = useState("");
  const [format, setFormat] = useState("");
  const [system, setSystem] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [repo, setRepo] = useState("");
  const [split, setSplit] = useState("train");
  const [subset, setSubset] = useState("");
  const [limit, setLimit] = useState("");
  const [busy, setBusy] = useState(false);
  const [cols] = useData<Collection[]>("/api/rag/collections");
  const [collId, setCollId] = useState<number | "">("");
  const [nChunks, setNChunks] = useState(20);
  const [pairs, setPairs] = useState(3);
  const [synthSrc, setSynthSrc] = useState<"docs" | "topics">("docs");
  const [domain, setDomain] = useState("");
  const [styles, setStyles] = useState<string[]>(["hecho", "explicacion"]);
  const [judgeMin, setJudgeMin] = useState(0);
  const [helper, setHelper] = useState<number | null>(null);
  const [plantilla, setPlantilla] = useState<string | null>(null);
  const [plantillas] = useData<any[]>("/api/datasets/plantillas");
  const { notify, track } = useToast();

  const submit = async () => {
    setBusy(true);
    try {
      if (tab === "file" && file) {
        const fd = new FormData();
        fd.append("file", file);
        fd.append("name", name || file.name.replace(/\.[^.]+$/, ""));
        if (format) fd.append("format", format);
        if (system) fd.append("system_prompt", system);
        onDone(await api.form<Dataset>("/api/datasets/upload", fd));
      } else if (tab === "hf") {
        const t = await api.post<any>("/api/datasets/import-hf", {
          repo, name: name || repo.split("/").pop(), split, subset: subset || null, limit: limit ? +limit : null,
          format: format || null, system_prompt: system || null,
        });
        track(t, (done) => onDone(done.result));
        onClose();
      } else if (tab === "manual") {
        const rows = manualRows.filter((r) => r.a.trim() && (format === "text" || r.b.trim())).map((r) =>
          format === "preference" ? { prompt: r.a, chosen: r.b, rejected: r.c }
            : format === "text" ? { text: r.a } : { instruction: r.a, output: r.b, ...(system ? { system } : {}) });
        if (!rows.length) throw new Error("Escribe al menos un ejemplo completo");
        onDone(await api.post<Dataset>("/api/datasets/manual", { name, format: format || "chat", rows }));
      } else if (tab === "plantilla") {
        if (!plantilla) throw new Error("Elige una plantilla");
        onDone(await api.post<Dataset>(`/api/datasets/plantillas/${plantilla}?name=${encodeURIComponent(name)}`));
      } else if (tab === "synth") {
        const t = await api.post<any>("/api/datasets/synthesize", {
          source: synthSrc, collection_id: collId || null, domain: domain || null, name, num_chunks: nChunks, num_topics: nChunks,
          pairs_per_chunk: pairs, styles, judge_min_score: judgeMin, system_prompt: system || null, helper,
        });
        track(t, () => onDone());
        onClose();
      }
    } catch (e: any) { notify(e.message, "err"); } finally { setBusy(false); }
  };

  return (
    <Modal title="Nuevo dataset" onClose={onClose}>
      <div className="tabs">
        <button className={`tab ${tab === "file" ? "active" : ""}`} onClick={() => setTab("file")}>Archivo</button>
        <button className={`tab ${tab === "hf" ? "active" : ""}`} onClick={() => setTab("hf")}>Hugging Face</button>
        <button className={`tab ${tab === "manual" ? "active" : ""}`} onClick={() => setTab("manual")}>A mano</button>
        <button className={`tab ${tab === "synth" ? "active" : ""}`} onClick={() => setTab("synth")}>Sintético</button>
        <button className={`tab ${tab === "plantilla" ? "active" : ""}`} onClick={() => setTab("plantilla")}>Plantilla</button>
      </div>
      <div className="stack">
        <Field label="Nombre"><input value={name} onChange={(e) => setName(e.target.value)} placeholder="mi-dataset" /></Field>
        {tab === "plantilla" && (
          <div className="stack">
            <div className="sub">
              Empieza desde un caso de uso. Cada plantilla crea un dataset con ejemplos reales ya
              escritos para que veas la forma exacta que necesitan tus datos: luego los editas,
              añades los tuyos y entrenas.
            </div>
            {!plantillas?.length && <div className="empty">Cargando…</div>}
            {plantillas?.map((t) => (
              <label key={t.key} className={`option ${plantilla === t.key ? "selected" : ""}`} onClick={() => { setPlantilla(t.key); setName(t.key); }}>
                <input type="radio" checked={plantilla === t.key} readOnly />
                <div>
                  <b>{t.nombre}</b> <span className="badge">{t.metodo.toUpperCase()}</span>
                  <div className="sub">{t.descripcion}</div>
                  <div className="sub" style={{ marginTop: 4 }}><b>Cuántos:</b> {t.cuantos}</div>
                  <div className="sub"><b>Consejo:</b> {t.consejo}</div>
                </div>
              </label>
            ))}
            <Field label="Nombre del dataset"><input value={name} onChange={(e) => setName(e.target.value)} /></Field>
          </div>
        )}
        {tab === "manual" && (
          <div className="stack">
            <div className="sub">Escribe los ejemplos uno a uno. Puedes añadir más después desde la ficha del dataset («Añadir ejemplo»).</div>
            {manualRows.map((r, i) => (
              <div key={i} className="source stack" style={{ gap: 6 }}>
                <div className="row between"><b style={{ fontSize: 12 }}>Ejemplo {i + 1}</b>
                  {manualRows.length > 1 && <button className="ghost sm danger" onClick={() => setManualRows(manualRows.filter((_, j) => j !== i))}>quitar</button>}</div>
                <Field label={format === "preference" ? "Prompt" : format === "text" ? "Texto" : "Usuario pregunta"}>
                  <textarea rows={2} value={r.a} onChange={(e) => setManualRows(manualRows.map((x, j) => (j === i ? { ...x, a: e.target.value } : x)))} /></Field>
                {format !== "text" && (
                  <Field label={format === "preference" ? "Respuesta preferida" : "Respuesta que debe aprender"}>
                    <textarea rows={2} value={r.b} onChange={(e) => setManualRows(manualRows.map((x, j) => (j === i ? { ...x, b: e.target.value } : x)))} /></Field>
                )}
                {format === "preference" && (
                  <Field label="Respuesta rechazada">
                    <textarea rows={2} value={r.c} onChange={(e) => setManualRows(manualRows.map((x, j) => (j === i ? { ...x, c: e.target.value } : x)))} /></Field>
                )}
              </div>
            ))}
            <button onClick={() => setManualRows([...manualRows, { a: "", b: "", c: "" }])}><Plus size={13} /> Añadir otro ejemplo</button>
          </div>
        )}
        {tab === "file" && (
          <Field label="Archivo" hint="JSONL, JSON, CSV, Parquet o TXT. Formatos: Alpaca (instruction/output), ShareGPT (conversations), OpenAI (messages), DPO (prompt/chosen/rejected), texto (text).">
            <input type="file" accept=".jsonl,.json,.csv,.parquet,.txt,.md" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          </Field>
        )}
        {tab === "hf" && (
          <>
            <Field label="Repositorio" hint="p. ej. mlabonne/FineTome-100k, HuggingFaceH4/ultrafeedback_binarized"><input value={repo} onChange={(e) => setRepo(e.target.value)} placeholder="usuario/dataset" /></Field>
            <div className="grid g3">
              <Field label="Límite de filas"><input type="number" value={limit} onChange={(e) => setLimit(e.target.value)} placeholder="todas" /></Field>
              <Adv>
                <Field label="Split"><input value={split} onChange={(e) => setSplit(e.target.value)} /></Field>
                <Field label="Subset (opcional)"><input value={subset} onChange={(e) => setSubset(e.target.value)} /></Field>
              </Adv>
            </div>
          </>
        )}
        {tab === "synth" && (
          <>
            <div className="sub">Usa el modelo cargado para generar pares pregunta/respuesta. Se eliminan duplicados semánticos y, opcionalmente, un juez descarta los pares de baja calidad.</div>
            <div className="row" style={{ gap: 16 }}>
              <label className="check"><input type="radio" checked={synthSrc === "docs"} onChange={() => setSynthSrc("docs")} /> Desde documentos (colección RAG)</label>
              <label className="check"><input type="radio" checked={synthSrc === "topics"} onChange={() => setSynthSrc("topics")} /> Desde un dominio (árbol de temas)</label>
            </div>
            {synthSrc === "docs" ? (
              <Field label="Colección RAG">
                <select value={collId} onChange={(e) => setCollId(e.target.value ? +e.target.value : "")}>
                  <option value="">— elige —</option>
                  {cols?.map((c) => <option key={c.id} value={c.id}>{c.name} ({c.num_chunks} chunks)</option>)}
                </select>
              </Field>
            ) : (
              <Field label="Dominio" hint="El modelo propone subtemas y genera pares para cada uno"><input value={domain} onChange={(e) => setDomain(e.target.value)} placeholder="Atención al cliente de una tienda online de electrónica" /></Field>
            )}
            <div className="grid g3">
              <Field label={synthSrc === "docs" ? "Fragmentos a muestrear" : "Subtemas"}><input type="number" value={nChunks} onChange={(e) => setNChunks(+e.target.value)} /></Field>
              <Field label={synthSrc === "docs" ? "Pares por fragmento" : "Pares por subtema"}><input type="number" value={pairs} onChange={(e) => setPairs(+e.target.value)} /></Field>
              <Field label={`Filtro del juez: ${judgeMin ? `≥ ${judgeMin}/5` : "desactivado"}`}><input type="range" min={0} max={5} step={1} value={judgeMin} onChange={(e) => setJudgeMin(+e.target.value)} /></Field>
              <HelperPicker value={helper} onChange={setHelper}
                hint="Genera con un modelo más grande que el que vas a entrenar: es como se hacen los datos sintéticos buenos" />
            </div>
            <Adv>
              <Field label="Tipos de pregunta (se alternan para dar variedad)">
                <div className="row" style={{ gap: 12 }}>
                  {[["hecho", "Datos concretos"], ["explicacion", "Explicaciones"], ["comparacion", "Comparaciones"], ["procedimiento", "Procedimientos"], ["aplicacion", "Casos prácticos"]].map(([k, l]) => (
                    <label key={k} className="check"><input type="checkbox" checked={styles.includes(k)} onChange={(e) => setStyles(e.target.checked ? [...styles, k] : styles.filter((x) => x !== k))} />{l}</label>
                  ))}
                </div>
              </Field>
            </Adv>
            <SimpleNote>Se generan preguntas de datos y de explicación, se quitan las casi repetidas y se descartan las que el juez puntúe bajo.</SimpleNote>
          </>
        )}
        {tab !== "synth" && (
          <Field label="Formato" hint={tab === "manual" ? "Define qué campos se piden arriba" : "Autodetectado si se deja vacío"}>
            <select value={format} onChange={(e) => setFormat(e.target.value)}>
              <option value="">{tab === "manual" ? "Chat / instrucciones" : "Autodetectar"}</option>
              <option value="chat">Chat / instrucciones</option>
              <option value="text">Texto plano (preentrenamiento continuo)</option>
              <option value="preference">Preferencias (DPO)</option>
            </select>
          </Field>
        )}
        <Field label="System prompt (opcional)" hint="Se añade a cada conversación que no tenga uno">
          <textarea value={system} onChange={(e) => setSystem(e.target.value)} rows={2} />
        </Field>
        <div className="row">
          <div className="spacer" />
          <button className="primary" disabled={busy || (tab === "file" ? !file : tab === "hf" ? !repo : tab === "manual" ? !name || !manualRows.some((r) => r.a.trim()) : !name || (synthSrc === "docs" ? !collId : !domain))} onClick={submit}>
            <Upload size={14} /> {tab === "synth" ? "Generar" : tab === "manual" ? "Crear" : "Importar"}
          </button>
        </div>
      </div>
    </Modal>
  );
}

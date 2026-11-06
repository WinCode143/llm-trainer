import { useEffect, useState } from "react";
import { Bot, ChevronRight, Plug, Plus, Save, Send, Square, Trash2, Wrench } from "lucide-react";
import { api, type Collection, type Model, streamNDJSON } from "../api";
import { HelperPicker } from "../components/Helper";
import { AssistantText } from "../components/Markdownish";
import { Adv, Field, Help, Modal, Spinner, useData, useDialog, useToast } from "../components/ui";

type ToolSpec = { id: string; label: string; desc: string; icon: string; function: string | null; config: any[] };
type Agent = { id: number; name: string; description: string; model_id: number | null; helper?: number | null; system_prompt: string;
  tools: { id: string; config: Record<string, any> }[]; max_steps: number; max_tokens: number; temperature: number };

const EMPTY: Omit<Agent, "id"> = { name: "", description: "", model_id: null, helper: null, system_prompt: "Eres un asistente útil. Usa las herramientas disponibles cuando aporten información que no tienes.", tools: [], max_steps: 5, max_tokens: 700, temperature: 0.3 };

export default function Agents() {
  const [list, reload] = useData<Agent[]>("/api/agents");
  const [sel, setSel] = useState<number | "new">("new");
  const current = sel === "new" ? null : list?.find((a) => a.id === sel);
  const [importar, setImportar] = useState(false);
  return (
    <>
      <div className="page-head">
        <div><h1>Agentes</h1><div className="sub">Un modelo + las herramientas que elijas: buscar en tus documentos, en la web, ejecutar código, leer archivos o cualquier servidor MCP</div></div>
        <div className="row" style={{ gap: 8 }}>
          <button onClick={() => setImportar(true)}>Importar de CrewAI</button>
          <button className="primary" onClick={() => setSel("new")}><Plus size={14} /> Nuevo agente</button>
        </div>
      </div>
      <Help title="¿Qué es un agente?">
        <p>Un agente es <b>un modelo que puede usar herramientas</b>. En vez de responder de memoria, decide llamar a una herramienta (buscar en tus documentos, buscar en la web, calcular…), lee el resultado y sigue hasta darte la respuesta.</p>
        <ul>
          <li>Creas un perfil: <b>modelo + instrucciones + herramientas marcadas</b>. Cada herramienta tiene sus propios ajustes.</li>
          <li><b>Pasos máximos</b> limita cuántas veces puede llamar a herramientas antes de responder.</li>
          <li>Con <b>Servidor MCP externo</b> añades de golpe las herramientas de cualquier servidor compatible.</li>
          <li>Abajo tienes un panel para probarlo: verás cada llamada, sus argumentos y el resultado.</li>
        </ul>
        <p style={{ marginTop: 8 }}>La herramienta <b>Ejecutar Python</b> corre aislada con bubblewrap: el código no ve tus archivos ni el proyecto, no tiene red salvo que la actives, y se le aplican límites de memoria y tiempo. Puedes darle una carpeta de datos en sólo lectura (aparece dentro como <code>/datos</code>).</p>
        <p style={{ marginTop: 8 }}>Con modelos pequeños (menos de 7B) los agentes se equivocan a menudo al elegir herramienta o al leer el resultado.</p>
      </Help>
      <div className="split" style={{ gridTemplateColumns: "230px minmax(0,1fr)" }}>
        <div className="card" style={{ padding: 8 }}>
          {!list?.length && <div className="empty" style={{ fontSize: 13 }}>Sin agentes</div>}
          <table><tbody>
            {list?.map((a) => (
              <tr key={a.id} className={`clickable ${sel === a.id ? "selected" : ""}`} onClick={() => setSel(a.id)}>
                <td><b>{a.name}</b><div className="sub" style={{ fontSize: 12 }}>{a.tools.length} herramienta(s)</div></td>
              </tr>
            ))}
          </tbody></table>
        </div>
        <Editor key={sel} agent={current} onSaved={(a) => { reload(); setSel(a.id); }} onDeleted={() => { reload(); setSel("new"); }} />
      </div>
      {importar && <ImportarCrew onClose={() => setImportar(false)} onDone={() => { setImportar(false); reload(); }} />}
    </>
  );
}

function Editor({ agent, onSaved, onDeleted }: { agent: Agent | null | undefined; onSaved: (a: Agent) => void; onDeleted: () => void }) {
  const [catalog] = useData<ToolSpec[]>("/api/agents/tools");
  const [models] = useData<Model[]>("/api/models");
  const [cols] = useData<Collection[]>("/api/rag/collections");
  const [f, setF] = useState<Omit<Agent, "id">>(agent ?? EMPTY);
  const [dirty, setDirty] = useState(false);
  const { notify } = useToast();
  const { confirm } = useDialog();
  useEffect(() => { setF(agent ?? EMPTY); setDirty(false); }, [agent]);
  const set = (patch: Partial<Agent>) => { setF((p) => ({ ...p, ...patch })); setDirty(true); };

  const selected = (id: string) => f.tools.find((t) => t.id === id);
  const toggle = (spec: ToolSpec) => {
    if (selected(spec.id)) set({ tools: f.tools.filter((t) => t.id !== spec.id) });
    else set({ tools: [...f.tools, { id: spec.id, config: Object.fromEntries(spec.config.filter((c) => c.default !== undefined).map((c) => [c.key, c.default])) }] });
  };
  const setCfg = (id: string, key: string, v: any) => set({ tools: f.tools.map((t) => (t.id === id ? { ...t, config: { ...t.config, [key]: v } } : t)) });

  const save = async () => {
    try {
      const saved = agent ? await api.put<Agent>(`/api/agents/${agent.id}`, f) : await api.post<Agent>("/api/agents", f);
      setDirty(false); notify("Agente guardado"); onSaved(saved);
    } catch (e: any) { notify(e.message, "err"); }
  };
  const del = async () => {
    if (!agent || !(await confirm(`¿Borrar el agente «${agent.name}»?`, undefined, { label: "Borrar" }))) return;
    await api.del(`/api/agents/${agent.id}`); onDeleted();
  };

  return (
    <div className="stack" style={{ gap: 16 }}>
      <div className="card stack">
        <div className="row between">
          <h2 style={{ margin: 0 }}>{agent ? agent.name : "Nuevo agente"}</h2>
          <div className="row">
            {agent && <button className="danger" onClick={del}><Trash2 size={14} /></button>}
            <button className="primary" disabled={!f.name || !dirty} onClick={save}><Save size={14} /> Guardar</button>
          </div>
        </div>
        <div className="grid g2">
          <Field label="Nombre"><input value={f.name} onChange={(e) => set({ name: e.target.value })} placeholder="investigador" /></Field>
          <Field label="Modelo" hint="Se carga automáticamente al ejecutar el agente">
            <select value={f.model_id ?? ""} onChange={(e) => set({ model_id: e.target.value ? +e.target.value : null })}>
              <option value="">El que esté cargado</option>
              {models?.filter((m) => m.kind !== "embedding").map((m) => <option key={m.id} value={m.id}>{m.name} [{m.kind}]</option>)}
            </select>
          </Field>
        </div>
        <Field label="Instrucciones (system prompt)"><textarea rows={3} value={f.system_prompt} onChange={(e) => set({ system_prompt: e.target.value })} /></Field>
        <div className="grid g3">
          <Field label={`Pasos máximos: ${f.max_steps}`} hint="Cada paso = una respuesta del modelo (puede llamar a herramientas)"><input type="range" min={1} max={12} value={f.max_steps} onChange={(e) => set({ max_steps: +e.target.value })} /></Field>
          <HelperPicker value={f.helper ?? null} onChange={(helper) => set({ helper })}
            hint="Con un ayudante externo el agente no usa la GPU. A cambio pierde el tool-calling nativo: las herramientas se describen en el prompt." />
          <Adv>
            <Field label={`Temperatura: ${f.temperature}`}><input type="range" min={0} max={1.5} step={0.05} value={f.temperature} onChange={(e) => set({ temperature: +e.target.value })} /></Field>
            <Field label={`Máx. tokens por paso: ${f.max_tokens}`}><input type="range" min={128} max={2048} step={64} value={f.max_tokens} onChange={(e) => set({ max_tokens: +e.target.value })} /></Field>
          </Adv>
        </div>
      </div>

      <div className="card">
        <div className="row between"><h3 style={{ margin: 0 }}><Wrench size={15} /> Herramientas</h3><span className="sub">{f.tools.length} activas</span></div>
        <div className="grid g2" style={{ marginTop: 10 }}>
          {catalog?.map((spec) => {
            const on = selected(spec.id);
            return (
              <div key={spec.id} className="source" style={{ borderColor: on ? "var(--accent)" : undefined }}>
                <label className="check" style={{ alignItems: "flex-start" }}>
                  <input type="checkbox" checked={!!on} onChange={() => toggle(spec)} />
                  <span><b>{spec.label}</b>{spec.function && <code style={{ marginLeft: 6, fontSize: 11 }}>{spec.function}</code>}
                    <div className="sub" style={{ fontSize: 12 }}>{spec.desc}</div></span>
                </label>
                {on && spec.config.length > 0 && (
                  <div className="stack" style={{ marginTop: 8, gap: 6 }}>
                    {spec.config.map((c: any) => (
                      <ConfigField key={c.key} c={c} value={on.config[c.key]} cols={cols ?? []} onChange={(v) => setCfg(spec.id, c.key, v)} />
                    ))}
                    {spec.id === "mcp" && <McpProbe url={on.config.url} />}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {agent && <TestPanel agent={agent} dirty={dirty} />}
      {agent && <EvalPanel agent={agent} dirty={dirty} />}
    </div>
  );
}

function ConfigField({ c, value, onChange, cols }: { c: any; value: any; onChange: (v: any) => void; cols: Collection[] }) {
  const common = { value: value ?? c.default ?? "", onChange: (e: any) => onChange(c.type === "number" ? +e.target.value : e.target.value) };
  if (c.type === "collection")
    return <Field label={c.label}><select value={value ?? ""} onChange={(e) => onChange(e.target.value ? +e.target.value : null)}>
      <option value="">— elige —</option>{cols.map((x) => <option key={x.id} value={x.id}>{x.name}</option>)}</select></Field>;
  if (c.type === "mcp_server")
    return <McpServerSelect label={c.label} value={value} onChange={onChange} />;
  if (c.type === "select")
    return <Field label={c.label}><select {...common}>{c.options.map((o: string) => <option key={o}>{o}</option>)}</select></Field>;
  if (c.type === "bool")
    return <label className="check"><input type="checkbox" checked={!!value} onChange={(e) => onChange(e.target.checked)} /> {c.label}</label>;
  if (c.type === "tags")
    return <Field label={c.label} hint="separadas por comas"><input value={(value ?? []).join(", ")} onChange={(e) => onChange(e.target.value.split(",").map((x: string) => x.trim()).filter(Boolean))} /></Field>;
  return <Field label={c.label} hint={c.type === "path" ? "El agente no puede salir de esta carpeta" : undefined}>
    <input type={c.type === "number" ? "number" : "text"} placeholder={c.placeholder} {...common} /></Field>;
}

function McpProbe({ url }: { url: string }) {
  const [res, setRes] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const probe = async () => {
    setBusy(true);
    try {
      const r = await api.post<any>("/api/agents/tools/mcp/probe", { url });
      setRes(`${r.tools.length} herramientas: ${r.tools.map((t: any) => t.name).join(", ")}`);
    } catch (e: any) { setRes(e.message); } finally { setBusy(false); }
  };
  return (
    <div className="row">
      <button className="sm" disabled={!url || busy} onClick={probe}>{busy ? <Spinner /> : <Plug size={12} />} Probar conexión</button>
      {res && <span className="sub" style={{ fontSize: 12 }}>{res}</span>}
    </div>
  );
}

type Ev = { type: string; [k: string]: any };

function TestPanel({ agent, dirty }: { agent: Agent; dirty: boolean }) {
  const [input, setInput] = useState("");
  const [events, setEvents] = useState<Ev[]>([]);
  const [answer, setAnswer] = useState("");
  const [busy, setBusy] = useState(false);
  const [history, setHistory] = useState<{ role: string; content: string }[]>([]);
  const abort = useState(() => ({ ctrl: null as AbortController | null }))[0];
  const { notify } = useToast();

  const send = async () => {
    if (!input.trim()) return;
    const msgs = [...history, { role: "user", content: input }];
    setHistory(msgs); setInput(""); setEvents([]); setAnswer(""); setBusy(true);
    abort.ctrl = new AbortController();
    let acc = "";
    try {
      await streamNDJSON(`/api/agents/${agent.id}/run`, { messages: msgs }, (e: Ev) => {
        if (e.type === "token") { acc += e.text; setAnswer(acc.replace(/<tool_call>[\s\S]*?(<\/tool_call>|$)/g, "")); }
        else { setEvents((p) => [...p, e]); if (e.type === "tool_call") acc = ""; }
        if (e.type === "done") { setAnswer(e.final); setHistory((h) => [...h, { role: "assistant", content: e.final }]); }
        if (e.type === "error") notify(e.message, "err");
      }, abort.ctrl.signal);
    } catch (e: any) { if (e.name !== "AbortError") notify(e.message, "err"); } finally { setBusy(false); }
  };

  const calls = events.filter((e) => e.type === "tool_call" || e.type === "tool_result");
  const toolsEv = events.find((e) => e.type === "tools");
  const done = events.find((e) => e.type === "done");

  return (
    <div className="card stack">
      <div className="row between">
        <h3 style={{ margin: 0 }}><Bot size={15} /> Probar agente</h3>
        {toolsEv && <span className="sub" style={{ fontSize: 12 }}>{toolsEv.tools.length} herramientas cargadas · {toolsEv.native ? "tool-calling nativo" : "instrucciones en el prompt"}</span>}
      </div>
      {dirty && <div className="sub">Guarda los cambios para probarlos.</div>}
      <div className="row" style={{ flexWrap: "nowrap" }}>
        <input value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && send()}
          placeholder="Escribe una petición para el agente…" />
        {busy ? <button onClick={() => abort.ctrl?.abort()}><Square size={14} /></button>
          : <button className="primary" onClick={send} disabled={!input.trim()}><Send size={14} /></button>}
      </div>
      {history.length > 0 && <div className="sub" style={{ fontSize: 12 }}>Turnos en el contexto: {history.length} · <button className="ghost sm" onClick={() => { setHistory([]); setEvents([]); setAnswer(""); }}>reiniciar</button></div>}
      {calls.length > 0 && (
        <div className="stack" style={{ gap: 4 }}>
          {calls.map((e, i) => e.type === "tool_call" ? (
            <div key={i} className="row" style={{ gap: 6 }}>
              <ChevronRight size={13} /><b>{e.name}</b>
              <code style={{ fontSize: 12 }}>{JSON.stringify(e.arguments).slice(0, 120)}</code>
              {!e.known && <span className="badge err">no existe</span>}
            </div>
          ) : (
            <details key={i} className="source" style={{ fontSize: 12.5 }}>
              <summary className="sub">resultado de {e.name} · {e.seconds}s · {e.result.length} caracteres</summary>
              <div style={{ whiteSpace: "pre-wrap", marginTop: 6 }}>{e.result.slice(0, 2000)}</div>
            </details>
          ))}
        </div>
      )}
      {(answer || busy) && (
        <div className="msg assistant" style={{ maxWidth: "100%" }}>
          {answer ? <AssistantText text={answer} /> : <Spinner />}
          {done && <div className="meta">{done.steps} paso(s){done.limit_reached ? " · límite alcanzado" : ""}</div>}
        </div>
      )}
      {!events.length && !busy && <div className="sub" style={{ fontSize: 12 }}>Ejemplos: «busca en mis documentos cuánto dura la garantía», «busca en la web el precio actual del litio y calcula el coste de 300 kg», «lista los archivos y resume el más reciente».</div>}
    </div>
  );
}

/** Desplegable con los servidores MCP registrados en Conexiones. */
function McpServerSelect({ label, value, onChange }: { label: string; value: any; onChange: (v: any) => void }) {
  const [lista] = useData<{ id: number; name: string; transport: string; enabled: boolean }[]>("/api/connections/mcp-servers");
  return (
    <Field label={label} hint={lista?.length ? "Se registran en Conexiones" : "Aún no hay ninguno: añádelos en Conexiones"}>
      <select value={value ?? ""} onChange={(e) => onChange(e.target.value ? +e.target.value : null)}>
        <option value="">— elige —</option>
        {lista?.filter((m) => m.enabled).map((m) => <option key={m.id} value={m.id}>{m.name} ({m.transport})</option>)}
      </select>
    </Field>
  );
}

const EJEMPLO_CASOS = `¿Cuál es el plazo de devolución? | rag_search | 30
¿Cuánto es 1234 por 56? | calculator | 69104
Hola, ¿qué tal? | ninguna`;

/** ¿Elige la herramienta correcta? Casos de prueba con la herramienta esperada y palabras clave. */
function EvalPanel({ agent, dirty }: { agent: Agent; dirty: boolean }) {
  const [txt, setTxt] = useState(EJEMPLO_CASOS);
  const [runs, reload] = useData<any[]>(`/api/agents/${agent.id}/evals`);
  const [abierto, setAbierto] = useState(false);
  const { track, notify } = useToast();

  const casos = txt.split("\n").map((l) => l.trim()).filter(Boolean).map((l) => {
    const [question, tool = "", claves = ""] = l.split("|").map((x) => x.trim());
    return { question, tool, keywords: claves.split(",").map((x) => x.trim()).filter(Boolean) };
  });
  const ultimo = runs?.[0];

  const lanzar = async () => {
    if (dirty) return notify("Guarda el agente antes de evaluarlo", "err");
    try {
      const r = await api.post<any>(`/api/agents/${agent.id}/eval`, { name: "", cases: casos });
      track(r.task, () => reload());
    } catch (e: any) { notify(e.message, "err"); }
  };

  if (!abierto) {
    return (
      <div className="card" style={{ padding: "8px 12px" }}>
        <button className="ghost sm" onClick={() => setAbierto(true)}>
          Evaluar el agente · ¿elige la herramienta correcta?
          {ultimo?.summary?.casos ? ` — última: ${Math.round(ultimo.summary.herramienta_correcta * 100)}%` : ""}
        </button>
      </div>
    );
  }
  return (
    <div className="card stack">
      <div className="row between">
        <b>Evaluar el agente</b>
        <button className="sm ghost" onClick={() => setAbierto(false)}>Cerrar</button>
      </div>
      <div className="sub">
        Con modelos pequeños lo que falla no suele ser el texto, sino la <b>decisión de qué herramienta usar</b>.
        Escribe preguntas con la herramienta que debería elegir y mide cuántas acierta. Cambiar el modelo, las
        instrucciones o quitar herramientas y volver a medir es la forma de mejorar un agente sin ir a ciegas.
      </div>
      <Field label={`Casos (${casos.length})`}
        hint={`Uno por línea: pregunta | herramienta esperada | palabras clave. Herramientas de este agente: ${agent.tools.map((t) => t.id).join(", ") || "ninguna"}, o «ninguna».`}>
        <textarea rows={6} className="mono" value={txt} onChange={(e) => setTxt(e.target.value)} />
      </Field>
      <div className="row end">
        <button className="primary" disabled={!casos.length} onClick={lanzar}>Evaluar {casos.length} casos</button>
      </div>

      {ultimo?.results?.length > 0 && (<>
        <div className="row" style={{ gap: 18, flexWrap: "wrap" }}>
          <Metrica titulo="Herramienta correcta" valor={`${Math.round(ultimo.summary.herramienta_correcta * 100)}%`} />
          <Metrica titulo="A la primera" valor={`${Math.round(ultimo.summary.a_la_primera * 100)}%`} />
          {ultimo.summary.palabras_clave != null && <Metrica titulo="Palabras clave" valor={`${Math.round(ultimo.summary.palabras_clave * 100)}%`} />}
          <Metrica titulo="Pasos" valor={ultimo.summary.pasos_media} />
          <Metrica titulo="Por caso" valor={`${ultimo.summary.segundos_media}s`} />
        </div>
        {!!ultimo.summary.fallos_frecuentes?.length && (
          <div className="warn">
            <b>Fallos que se repiten:</b>
            {ultimo.summary.fallos_frecuentes.map(([k, n]: [string, number]) => <div key={k}>{k} ×{n}</div>)}
          </div>
        )}
        <table>
          <thead><tr><th /><th>Pregunta</th><th>Esperaba</th><th>Usó</th><th>Pasos</th></tr></thead>
          <tbody>
            {ultimo.results.map((f: any, i: number) => (
              <tr key={i} title={f.respuesta}>
                <td>{f.acierto ? "✓" : "✗"}</td>
                <td>{f.question}</td>
                <td className="mono sub">{f.esperada}</td>
                <td className="mono sub" style={{ color: f.acierto ? undefined : "var(--err)" }}>{f.llamadas.join(", ") || "ninguna"}</td>
                <td className="sub">{f.pasos}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {runs && runs.length > 1 && (
          <div className="sub">Historial: {runs.slice(0, 6).map((r) => `${Math.round((r.summary?.herramienta_correcta ?? 0) * 100)}%`).join(" · ")}</div>
        )}
      </>)}
    </div>
  );
}

const Metrica = ({ titulo, valor }: { titulo: string; valor: any }) => (
  <div><div className="sub" style={{ fontSize: 11 }}>{titulo}</div><div style={{ fontSize: 20, fontWeight: 600 }}>{valor}</div></div>
);

/** Importa agentes de un agents.yaml de CrewAI. Los roles pasan tal cual; las herramientas, sólo si hay equivalente. */
function ImportarCrew({ onClose, onDone }: { onClose: () => void; onDone: () => void }) {
  const [txt, setTxt] = useState("");
  const [res, setRes] = useState<any>(null);
  const [models] = useData<Model[]>("/api/models");
  const [modelo, setModelo] = useState<number | "">("");
  const { notify } = useToast();
  const go = async () => {
    try { setRes(await api.post("/api/agents/import/crewai", { yaml: txt, model_id: modelo || null })); }
    catch (e: any) { notify(e.message, "err"); }
  };
  return (
    <Modal title="Importar agentes de CrewAI" onClose={res ? onDone : onClose} wide>
      {!res ? (<>
        <p className="sub">
          Pega el contenido de un <code>agents.yaml</code> de CrewAI. El rol, el objetivo y el trasfondo se convierten en
          las instrucciones del agente. Sus herramientas son código de CrewAI y no se pueden traer: se traducen las que
          tienen equivalente aquí (búsqueda web, leer y escribir archivos, Python, SQL, documentos) y del resto se avisa.
        </p>
        <Field label="agents.yaml"><textarea rows={12} className="mono" value={txt} onChange={(e) => setTxt(e.target.value)}
          placeholder={"researcher:\n  role: Investigador senior\n  goal: Encontrar lo último sobre {topic}\n  backstory: Llevas años…\n  tools:\n    - SerperDevTool"} /></Field>
        <Field label="Modelo para los agentes importados">
          <select value={modelo} onChange={(e) => setModelo(e.target.value ? +e.target.value : "")}>
            <option value="">el que esté cargado</option>
            {models?.filter((m) => m.kind !== "embedding").map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
          </select>
        </Field>
        <div className="row end"><button onClick={onClose}>Cancelar</button><button className="primary" disabled={!txt.trim()} onClick={go}>Importar</button></div>
      </>) : (<>
        <p><b>{res.creados.length} agente(s) creados:</b></p>
        {res.creados.map((a: any) => <div key={a.id}>· {a.name} <span className="sub">{a.tools.join(", ") || "sin herramientas"}</span></div>)}
        {!!res.avisos.length && <div className="warn" style={{ marginTop: 10 }}>{res.avisos.map((w: string, i: number) => <div key={i}>⚠ {w}</div>)}</div>}
        <div className="row end" style={{ marginTop: 10 }}><button className="primary" onClick={onDone}>Hecho</button></div>
      </>)}
    </Modal>
  );
}

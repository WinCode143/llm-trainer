import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Check as CheckIcon, Code2, LayoutGrid, Play, Plus, Save, Square, Trash2, Workflow, X } from "lucide-react";
import { api, type Collection, streamNDJSON } from "../api";
import { autoLayout, FlowCanvas, type FlowNode, type Spec } from "../components/FlowCanvas";
import { HelperPicker } from "../components/Helper";
import { TimeTravel, VarField } from "../components/FlowDebug";
import { AssistantText } from "../components/Markdownish";
import { Adv, Check, Field, Help, Modal, SimpleNote, Spinner, useData, useDialog, useToast } from "../components/ui";

type NodeType = { label: string; desc: string; icon: string };
type Meta = { types: Record<string, NodeType>; templates: Record<string, { name: string; description: string }> };
type FlowRow = { id: number; name: string; description: string; nodes: number; updated_at: string };
type Flow = { id: number; name: string; description: string; spec: Spec; warnings?: string[] };
type ToolSpec = { id: string; label: string; desc: string; config: any[] };
type AgentRow = { id: number; name: string };
type Trace = { node: string; node_type: string; label: string; output: string; info: any };
type Interrupt = { node: string; label: string; message: string; preview: string; editable: boolean };

const EMPTY_SPEC: Spec = {
  nodes: [
    { id: "in", type: "input", label: "Entrada", x: 60, y: 160, config: { fields: [{ key: "entrada", label: "Pregunta" }] } },
    { id: "fin", type: "output", label: "Salida", x: 460, y: 160, config: { template: "{entrada}" } },
  ],
  edges: [{ from: "in", to: "fin", when: null }],
};

/** Identificador corto y único para un nodo nuevo del tipo dado. */
function newId(spec: Spec, type: string): string {
  let i = 1;
  while (spec.nodes.some((n) => n.id === `${type}${i}`)) i++;
  return `${type}${i}`;
}

const DEFAULT_CONFIG: Record<string, any> = {
  input: { fields: [{ key: "entrada", label: "Pregunta" }] },
  llm: { prompt: "{entrada}", system: "", max_tokens: 600, temperature: 0.3, json: false },
  tool: { tool_id: "", args: {}, config: {} },
  rag: { collection_id: null, query: "{entrada}", k: 4 },
  agent: { agent_id: null, prompt: "{entrada}" },
  router: { mode: "expr", branches: [{ label: "sí", when: "True" }, { label: "no", when: "True" }] },
  approval: { message: "¿Apruebas este resultado?", preview: "{entrada}", editable: true },
  output: { template: "{entrada}" },
  foreach: { items_key: "", mode: "llm", prompt: "{item}", parallel: 3, on_error: "null", max_tokens: 400, max_items: 0 },
  subflow: { flow_id: null, input: "{entrada}" },
  code: { code: "# Las variables del flujo están en `vars`.\n# Devuelve un dict en `resultado` y cada clave pasa a ser una variable.\nresultado = {\"total\": len(vars.get(\"entrada\", \"\"))}", timeout_s: 20, memory_mb: 1024, allow_network: false },
};

export default function Flows() {
  const [meta] = useData<Meta>("/api/flows/node-types");
  const [list, reload] = useData<FlowRow[]>("/api/flows");
  const [sel, setSel] = useState<number | null>(null);
  const [showNew, setShowNew] = useState(false);
  const { notify } = useToast();
  const dialog = useDialog();

  useEffect(() => { if (sel == null && list?.length) setSel(list[0].id); }, [list, sel]);

  const create = async (key: string | null, name: string) => {
    const tpl = key ? await api.get<any>(`/api/flows/templates/${key}`) : null;
    const f = await api.post<Flow>("/api/flows", {
      name, description: tpl?.description ?? "", spec: tpl?.spec ?? EMPTY_SPEC,
    });
    setShowNew(false);
    reload();
    setSel(f.id);
  };

  const remove = async (id: number, name: string) => {
    if (!(await dialog.confirm("Borrar flujo", `¿Borrar «${name}»? También se pierden sus ejecuciones.`))) return;
    await api.del(`/api/flows/${id}`);
    setSel(null);
    reload();
    notify("Flujo borrado");
  };

  return (
    <>
      <div className="page-head">
        <div><h1>Flujos</h1><div className="sub">Encadena modelo, documentos, herramientas y agentes en un diagrama, con bifurcaciones y aprobaciones tuyas</div></div>
        <button className="primary" onClick={() => setShowNew(true)}><Plus size={14} /> Nuevo flujo</button>
      </div>

      <Help title="¿Qué es un flujo y cuándo conviene sobre un agente?">
        <p>Un <b>agente</b> decide solo qué hacer en cada paso: es flexible pero impredecible. Un <b>flujo</b> fija el camino de antemano: siempre pasa por los mismos nodos, en el mismo orden, y tú decides dónde puede ramificarse. Sirve para procesos que deben salir igual cada vez (responder con citas verificadas, clasificar y derivar, redactar y aprobar antes de guardar).</p>
        <ul>
          <li><b>Entrada</b> define los campos que rellenas al lanzarlo. Cada nodo guarda su resultado en una variable.</li>
          <li>En cualquier texto puedes escribir <code>{"{variable}"}</code> y se sustituye por el valor: <code>{"{entrada}"}</code>, <code>{"{buscar_salida}"}</code>…</li>
          <li><b>Modelo</b> con «devolver JSON» activado vuelca cada clave del JSON como variable propia, lista para usar en condiciones.</li>
          <li><b>Bifurcación</b> elige rama por condición (<code>apoyada == True</code>) o dejando que el modelo escoja una etiqueta.</li>
          <li><b>Aprobación</b> detiene el flujo y espera tu visto bueno; puedes editar el texto antes de continuar.</li>
        </ul>
        <p style={{ marginTop: 8 }}>La generación usa el modelo que tengas cargado en <b>Playground</b> o <b>Modelos</b>, salvo que elijas un <b>modelo ayudante</b> por nodo. Por debajo el flujo se compila a un grafo de <b>LangGraph</b> con punto de guardado.</p>
        <ul>
          <li><b>Para cada</b> repite algo por cada elemento de una lista, de uno en uno o varios a la vez.</li>
          <li><b>Subflujo</b> mete un flujo entero dentro de otro, para reutilizarlo sin copiar nodos.</li>
          <li><b>Código</b> ejecuta Python aislado para transformar variables; si devuelves un diccionario, cada clave pasa a ser una variable.</li>
          <li>Tras ejecutar aparece <b>Rebobinar</b>: vuelves a cualquier paso, cambias lo que quieras y sigues desde ahí sin repetir lo anterior.</li>
          <li><b>Exportar</b> descarga el flujo como código LangGraph, por si algún día quieres salir de aquí.</li>
        </ul>
      </Help>

      <div className="split" style={{ gridTemplateColumns: "230px minmax(0,1fr)" }}>
        <div className="card" style={{ padding: 8 }}>
          {!list?.length && <div className="empty" style={{ fontSize: 13 }}>Sin flujos todavía</div>}
          <table><tbody>
            {list?.map((f) => (
              <tr key={f.id} className={`clickable ${sel === f.id ? "selected" : ""}`} onClick={() => setSel(f.id)}>
                <td><b>{f.name}</b><div className="sub" style={{ fontSize: 12 }}>{f.nodes} nodo(s)</div></td>
                <td style={{ width: 28 }}>
                  <button className="icon" title="Borrar" onClick={(e) => { e.stopPropagation(); remove(f.id, f.name); }}><Trash2 size={13} /></button>
                </td>
              </tr>
            ))}
          </tbody></table>
        </div>
        {sel != null && meta ? <Editor key={sel} id={sel} meta={meta} onSaved={reload} /> : <div className="card empty">Elige un flujo o crea uno nuevo</div>}
      </div>

      {showNew && meta && <NewFlow meta={meta} onCreate={create} onClose={() => setShowNew(false)} />}
    </>
  );
}

function NewFlow({ meta, onCreate, onClose }: { meta: Meta; onCreate: (k: string | null, n: string) => Promise<void>; onClose: () => void }) {
  const [key, setKey] = useState<string | null>(Object.keys(meta.templates)[0] ?? null);
  const [name, setName] = useState(key ? meta.templates[key].name : "Mi flujo");
  const [busy, setBusy] = useState(false);
  const pick = (k: string | null) => { setKey(k); setName(k ? meta.templates[k].name : "Mi flujo"); };
  return (
    <Modal title="Nuevo flujo" onClose={onClose}>
      <div className="sub" style={{ marginBottom: 8 }}>Empieza desde una plantilla ya montada o desde un flujo vacío.</div>
      <div className="stack">
        {Object.entries(meta.templates).map(([k, t]) => (
          <label key={k} className={`option ${key === k ? "selected" : ""}`} onClick={() => pick(k)}>
            <input type="radio" checked={key === k} readOnly />
            <div><b>{t.name}</b><div className="sub">{t.description}</div></div>
          </label>
        ))}
        <label className={`option ${key === null ? "selected" : ""}`} onClick={() => pick(null)}>
          <input type="radio" checked={key === null} readOnly />
          <div><b>En blanco</b><div className="sub">Sólo entrada y salida; añade los nodos que quieras</div></div>
        </label>
      </div>
      <Field label="Nombre"><input value={name} onChange={(e) => setName(e.target.value)} /></Field>
      <div className="row end" style={{ marginTop: 10 }}>
        <button onClick={onClose}>Cancelar</button>
        <button className="primary" disabled={!name.trim() || busy}
          onClick={async () => { setBusy(true); try { await onCreate(key, name.trim()); } finally { setBusy(false); } }}>
          {busy ? <Spinner /> : <Plus size={14} />} Crear
        </button>
      </div>
    </Modal>
  );
}

function Editor({ id, meta, onSaved }: { id: number; meta: Meta; onSaved: () => void }) {
  const [flow, reloadFlow] = useData<Flow>(`/api/flows/${id}`);
  const [spec, setSpec] = useState<Spec | null>(null);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [dirty, setDirty] = useState(false);
  const [tab, setTab] = useState<"editor" | "runs">("editor");
  const { notify } = useToast();

  useEffect(() => {
    if (!flow) return;
    setSpec(flow.spec?.nodes ? flow.spec : EMPTY_SPEC);
    setName(flow.name);
    setDescription(flow.description);
    setDirty(false);
  }, [flow]);

  const update = useCallback((s: Spec) => { setSpec(s); setDirty(true); }, []);

  const save = async () => {
    if (!spec) return;
    const saved = await api.put<Flow>(`/api/flows/${id}`, { name, description, spec });
    setDirty(false);
    onSaved();
    notify(saved.warnings?.length ? `Guardado · ${saved.warnings.length} aviso(s)` : "Flujo guardado");
    reloadFlow();
  };

  const exportar = async () => {
    const r = await api.get<{ filename: string; code: string }>(`/api/flows/${id}/export`);
    const url = URL.createObjectURL(new Blob([r.code], { type: "text/x-python" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = r.filename;
    a.click();
    URL.revokeObjectURL(url);
    notify(`${r.filename} descargado`);
  };

  const addNode = (type: string) => {
    if (!spec) return;
    const nid = newId(spec, type);
    const maxX = Math.max(0, ...spec.nodes.map((n) => n.x ?? 0));
    update({ ...spec, nodes: [...spec.nodes, {
      id: nid, type, label: meta.types[type].label, x: maxX + 220, y: 160,
      config: JSON.parse(JSON.stringify(DEFAULT_CONFIG[type] ?? {})),
    }] });
    setSelected(nid);
  };

  const patchNode = (nid: string, patch: Partial<FlowNode>) => {
    if (!spec) return;
    update({ ...spec, nodes: spec.nodes.map((n) => (n.id === nid ? { ...n, ...patch } : n)) });
  };

  const [run, setRun] = useState<RunState | null>(null);
  if (!spec) return <div className="card empty"><Spinner /> Cargando…</div>;
  const node = spec.nodes.find((n) => n.id === selected) ?? null;

  return (
    <div className="stack">
      <div className="card">
        <div className="row" style={{ gap: 8, alignItems: "flex-end" }}>
          <Field label="Nombre"><input value={name} onChange={(e) => { setName(e.target.value); setDirty(true); }} /></Field>
          <Field label="Descripción"><input value={description} onChange={(e) => { setDescription(e.target.value); setDirty(true); }} /></Field>
          <button title="Reordenar los nodos por capas" onClick={() => update(autoLayout(spec))}>
            <LayoutGrid size={14} /> Ordenar
          </button>
          <button title="Descargar el flujo como código LangGraph" onClick={exportar}><Code2 size={14} /> Exportar</button>
          <button className="primary" onClick={save} disabled={!dirty}><Save size={14} /> Guardar</button>
        </div>
        {!!flow?.warnings?.length && (
          <div className="warn" style={{ marginTop: 8 }}>
            <b>Hay que revisar {flow.warnings.length} cosa(s) antes de que funcione:</b>
            {flow.warnings.map((w, i) => <div key={i}>⚠ {w}</div>)}
          </div>
        )}
        <div className="tabs" style={{ marginTop: 10 }}>
          <button className={tab === "editor" ? "active" : ""} onClick={() => setTab("editor")}>Diagrama</button>
          <button className={tab === "runs" ? "active" : ""} onClick={() => setTab("runs")}>Ejecuciones</button>
        </div>
      </div>

      {tab === "runs" ? <Runs flowId={id} /> : (
        <>
          <div className="card">
            <div className="sub" style={{ marginBottom: 6 }}>Añade nodos y arrastra de un punto a otro para conectarlos. Selecciona una conexión o un nodo y pulsa Supr para borrarlo.</div>
            <div className="palette">
              {Object.entries(meta.types).map(([t, d]) => (
                <button key={t} title={d.desc} onClick={() => addNode(t)}><Plus size={12} /> {d.label}</button>
              ))}
            </div>
          </div>

          <div className="split" style={{ gridTemplateColumns: "minmax(0,1fr) 340px" }}>
            <FlowCanvas spec={spec} onChange={update} selected={selected} onSelect={setSelected}
              types={meta.types} running={run?.current ?? null} done={run?.done ?? []} tokens={run?.tokens} />
            <div className="card" style={{ maxHeight: 560, overflow: "auto" }}>
              <SimpleNote>Se ocultan los ajustes finos del nodo (etiqueta, nombre de la variable de salida,
                instrucciones del sistema, límites y reintentos). Los valores por defecto funcionan.</SimpleNote>
              {node ? <NodeProps node={node} spec={spec} meta={meta} onChange={(p) => patchNode(node.id, p)} onSpec={update} flowId={id} />
                : <div className="empty" style={{ fontSize: 13 }}>Selecciona un nodo para configurarlo</div>}
            </div>
          </div>

          <RunPanel flowId={id} spec={spec} dirty={dirty} onState={setRun} />
        </>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ propiedades por tipo de nodo */

function NodeProps({ node, spec, meta, onChange, onSpec, flowId }: {
  node: FlowNode; spec: Spec; meta: Meta; onChange: (p: Partial<FlowNode>) => void; onSpec: (s: Spec) => void;
  flowId: number;   // para no ofrecerse a sí mismo como subflujo
}) {
  const cfg = node.config || {};
  const set = (patch: any) => onChange({ config: { ...cfg, ...patch } });
  const [tools] = useData<ToolSpec[]>(node.type === "tool" || node.type === "foreach" ? "/api/agents/tools" : null);
  const [colls] = useData<Collection[]>(node.type === "rag" ? "/api/rag/collections" : null);
  const [agents] = useData<AgentRow[]>(node.type === "agent" ? "/api/agents" : null);
  const [flows] = useData<FlowRow[]>(node.type === "subflow" || node.type === "foreach" ? "/api/flows" : null);
  const outKey = cfg.output_key || `${node.id}_salida`;
  // Variables que el nodo puede usar: las de los nodos anteriores más las del JSON que declaren
  const vars = useMemo(() => {
    const out = new Set<string>(["entrada"]);
    for (const n of spec.nodes) {
      if (n.id === node.id) continue;
      out.add((n.config?.output_key as string) || `${n.id}_salida`);
      if (n.type === "rag") out.add(`${(n.config?.output_key as string) || `${n.id}_salida`}_fuentes`);
      for (const f of (n.config?.fields ?? []) as any[]) if (f.key) out.add(f.key);
    }
    return [...out];
  }, [spec.nodes, node.id]);

  return (
    <div className="stack">
      <div className="row between">
        <b>{meta.types[node.type]?.label ?? node.type}</b>
        <code className="sub">{node.id}</code>
      </div>
      <div className="sub">{meta.types[node.type]?.desc}</div>
      <Adv><Field label="Etiqueta"><input value={node.label ?? ""} onChange={(e) => onChange({ label: e.target.value })} /></Field></Adv>

      {node.type === "input" && (
        <Field label="Campos del formulario" hint="Cada campo se convierte en una variable con ese nombre">
          <KeyList items={cfg.fields ?? []} onChange={(fields) => set({ fields })}
            cols={[["key", "variable"], ["label", "etiqueta"]]} addLabel="Añadir campo" />
        </Field>
      )}

      {node.type === "llm" && (<>
        <Adv>
          <Field label="Instrucciones del sistema" hint="Opcional; define el papel del modelo">
            <VarField rows={2} value={cfg.system ?? ""} onChange={(system) => set({ system })} vars={vars} />
          </Field>
        </Adv>
        <Field label="Prompt" hint="Escribe { para insertar una variable">
          <VarField rows={6} value={cfg.prompt ?? ""} onChange={(prompt) => set({ prompt })} vars={vars} />
        </Field>
        <Check label="Devolver JSON (cada clave pasa a ser una variable)" checked={!!cfg.json} onChange={(json) => set({ json })} />
        <Adv>
          <div className="row" style={{ gap: 8 }}>
            <Field label="Tokens máx."><input type="number" value={cfg.max_tokens ?? 600} onChange={(e) => set({ max_tokens: +e.target.value })} /></Field>
            <Field label="Temperatura"><input type="number" step="0.1" value={cfg.temperature ?? 0.3} onChange={(e) => set({ temperature: +e.target.value })} /></Field>
          </div>
        </Adv>
        <HelperPicker value={cfg.helper ?? null} onChange={(helper) => set({ helper })}
          hint="Por nodo: uno barato para clasificar, uno potente para redactar" />
      </>)}

      {node.type === "tool" && (<>
        <Field label="Herramienta">
          <select value={cfg.tool_id ?? ""} onChange={(e) => set({ tool_id: e.target.value, args: {}, config: {} })}>
            <option value="">— elige —</option>
            {tools?.map((t) => <option key={t.id} value={t.id}>{t.label}</option>)}
          </select>
        </Field>
        {(() => {
          const t = tools?.find((x) => x.id === cfg.tool_id);
          if (!t) return null;
          if (t.id === "mcp") return <McpToolPicker cfg={cfg} set={set} />;
          return (<>
            <div className="sub">{t.desc}</div>
            <Field label="Argumentos" hint="Puedes usar {variables} en los valores">
              <KeyList items={Object.entries(cfg.args ?? {}).map(([k, v]) => ({ key: k, value: String(v) }))}
                onChange={(rows) => set({ args: Object.fromEntries(rows.filter((r: any) => r.key).map((r: any) => [r.key, r.value])) })}
                cols={[["key", "argumento"], ["value", "valor"]]} addLabel="Añadir argumento" />
            </Field>
            {!!t.config?.length && (
              <Field label="Ajustes de la herramienta">
                <div className="stack">
                  {t.config.map((c: any) => (
                    <ConfigField key={c.key} spec={c} value={(cfg.config ?? {})[c.key]}
                      onChange={(v) => set({ config: { ...(cfg.config ?? {}), [c.key]: v } })} />
                  ))}
                </div>
              </Field>
            )}
          </>);
        })()}
      </>)}

      {node.type === "rag" && (<>
        <Field label="Colección">
          <select value={cfg.collection_id ?? ""} onChange={(e) => set({ collection_id: e.target.value ? +e.target.value : null })}>
            <option value="">— elige —</option>
            {colls?.map((c) => <option key={c.id} value={c.id}>{c.name} ({c.num_chunks} fragmentos)</option>)}
          </select>
        </Field>
        <Field label="Consulta"><VarField rows={2} value={cfg.query ?? "{entrada}"} onChange={(query) => set({ query })} vars={vars} /></Field>
        <Field label="Fragmentos a recuperar"><input type="number" value={cfg.k ?? 4} onChange={(e) => set({ k: +e.target.value })} /></Field>
      </>)}

      {node.type === "agent" && (<>
        <Field label="Agente">
          <select value={cfg.agent_id ?? ""} onChange={(e) => set({ agent_id: e.target.value ? +e.target.value : null })}>
            <option value="">— elige —</option>
            {agents?.map((a) => <option key={a.id} value={a.id}>{a.name}</option>)}
          </select>
        </Field>
        <Field label="Encargo"><VarField rows={4} value={cfg.prompt ?? "{entrada}"} onChange={(prompt) => set({ prompt })} vars={vars} /></Field>
      </>)}

      {node.type === "router" && (<>
        <Field label="Cómo decide">
          <select value={cfg.mode ?? "expr"} onChange={(e) => set({ mode: e.target.value })}>
            <option value="expr">Por condición</option>
            <option value="llm">Que elija el modelo</option>
          </select>
        </Field>
        {cfg.mode === "llm" && <Field label="Pregunta al modelo"><VarField rows={3} value={cfg.prompt ?? "{entrada}"} onChange={(prompt) => set({ prompt })} vars={vars} /></Field>}
        {cfg.mode === "llm" && <HelperPicker value={cfg.helper ?? null} onChange={(helper) => set({ helper })} />}
        <Field label="Ramas" hint={cfg.mode === "llm" ? "El modelo elige una de estas etiquetas" : "Se toma la primera cuya condición se cumpla, p. ej. apoyada == True"}>
          <KeyList items={cfg.branches ?? []} onChange={(branches) => set({ branches })}
            cols={cfg.mode === "llm" ? [["label", "etiqueta"]] : [["label", "etiqueta"], ["when", "condición"]]}
            addLabel="Añadir rama" />
        </Field>
        <div className="sub">Conecta este nodo a los siguientes y pon en cada conexión la etiqueta de su rama (campo «Cuando» abajo).</div>
        <EdgeLabels spec={spec} nodeId={node.id} onSpec={onSpec} />
      </>)}

      {node.type === "approval" && (<>
        <Field label="Mensaje"><VarField rows={2} value={cfg.message ?? ""} onChange={(message) => set({ message })} vars={vars} /></Field>
        <Field label="Qué mostrar para revisar"><VarField rows={4} value={cfg.preview ?? ""} onChange={(preview) => set({ preview })} vars={vars} /></Field>
        <Check label="Permitir editar el texto antes de continuar" checked={!!cfg.editable} onChange={(editable) => set({ editable })} />
        {cfg.editable && <Adv><Field label="Variable que se reemplaza al editar" hint="Por defecto, la salida de este nodo">
          <input value={cfg.edit_key ?? ""} placeholder={outKey} onChange={(e) => set({ edit_key: e.target.value })} /></Field></Adv>}
      </>)}

      {node.type === "foreach" && (<>
        <Field label="Lista sobre la que repetir" hint="Variable con una lista; si es texto, se usa cada línea">
          <select value={cfg.items_key ?? ""} onChange={(e) => set({ items_key: e.target.value })}>
            <option value="">— elige —</option>
            {vars.map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </Field>
        <Field label="Qué hacer con cada elemento">
          <select value={cfg.mode ?? "llm"} onChange={(e) => set({ mode: e.target.value })}>
            <option value="llm">Pedírselo al modelo</option>
            <option value="tool">Ejecutar una herramienta</option>
            <option value="subflow">Ejecutar un flujo entero</option>
          </select>
        </Field>
        {(cfg.mode ?? "llm") === "llm" && (<>
          <Field label="Prompt" hint="{item} es el elemento y {indice} su posición">
            <VarField rows={4} value={cfg.prompt ?? "{item}"} onChange={(prompt) => set({ prompt })} vars={["item", "indice", ...vars]} />
          </Field>
          <HelperPicker value={cfg.helper ?? null} onChange={(helper) => set({ helper })}
            hint="Con un ayudante externo el paralelismo es real; el modelo local atiende de uno en uno" />
        </>)}
        {cfg.mode === "tool" && (<>
          <Field label="Herramienta">
            <select value={cfg.tool_id ?? ""} onChange={(e) => set({ tool_id: e.target.value, args: {} })}>
              <option value="">— elige —</option>
              {tools?.map((t) => <option key={t.id} value={t.id}>{t.label}</option>)}
            </select>
          </Field>
          <Field label="Argumentos" hint="Usa {item} para el elemento actual">
            <KeyList items={Object.entries(cfg.args ?? {}).map(([k, v]) => ({ key: k, value: String(v) }))}
              onChange={(rows) => set({ args: Object.fromEntries(rows.filter((r: any) => r.key).map((r: any) => [r.key, r.value])) })}
              cols={[["key", "argumento"], ["value", "valor"]]} addLabel="Añadir argumento" />
          </Field>
        </>)}
        {cfg.mode === "subflow" && (<>
          <Field label="Flujo que se ejecuta por elemento">
            <select value={cfg.flow_id ?? ""} onChange={(e) => set({ flow_id: e.target.value ? +e.target.value : null })}>
              <option value="">— elige —</option>
              {flows?.filter((f) => f.id !== flowId).map((f) => <option key={f.id} value={f.id}>{f.name}</option>)}
            </select>
          </Field>
          <Field label="Qué se le pasa"><VarField rows={2} value={cfg.input ?? "{item}"} onChange={(input) => set({ input })} vars={["item", "indice", ...vars]} /></Field>
        </>)}
        <Adv>
          <div className="row" style={{ gap: 8 }}>
            <Field label="A la vez" hint="1 = uno detrás de otro"><input type="number" min={1} max={10} value={cfg.parallel ?? 1} onChange={(e) => set({ parallel: +e.target.value })} /></Field>
            <Field label="Máx. elementos" hint="0 = todos"><input type="number" min={0} value={cfg.max_items ?? 0} onChange={(e) => set({ max_items: +e.target.value })} /></Field>
          </div>
        </Adv>
        <Field label="Si uno falla">
          <select value={cfg.on_error ?? "stop"} onChange={(e) => set({ on_error: e.target.value })}>
            <option value="stop">Detener todo</option>
            <option value="null">Continuar y dejarlo vacío</option>
            <option value="skip">Continuar y descartarlo</option>
          </select>
        </Field>
        <div className="sub">Además de la lista, tendrás <code>{`{${outKey}_texto}`}</code> con todo junto y <code>{`{${outKey}_n}`}</code> con cuántos salieron.</div>
      </>)}

      {node.type === "subflow" && (<>
        <Field label="Flujo a ejecutar">
          <select value={cfg.flow_id ?? ""} onChange={(e) => set({ flow_id: e.target.value ? +e.target.value : null })}>
            <option value="">— elige —</option>
            {flows?.filter((f) => f.id !== flowId).map((f) => <option key={f.id} value={f.id}>{f.name}</option>)}
          </select>
        </Field>
        <Field label="Qué se le pasa como entrada"><VarField rows={2} value={cfg.input ?? "{entrada}"} onChange={(input) => set({ input })} vars={vars} /></Field>
        <div className="sub">El subflujo se ejecuta entero y devuelve su salida. Un flujo no puede llamarse a sí mismo.</div>
      </>)}

      {node.type === "code" && (<>
        <Field label="Código Python" hint="Las variables están en `vars`; pon el resultado en `resultado`">
          <textarea rows={12} className="mono" value={cfg.code ?? ""} onChange={(e) => set({ code: e.target.value })} />
        </Field>
        <Adv>
          <div className="row" style={{ gap: 8 }}>
            <Field label="Segundos máx."><input type="number" value={cfg.timeout_s ?? 20} onChange={(e) => set({ timeout_s: +e.target.value })} /></Field>
            <Field label="Memoria (MB)"><input type="number" value={cfg.memory_mb ?? 1024} onChange={(e) => set({ memory_mb: +e.target.value })} /></Field>
          </div>
        </Adv>
        <Check label="Permitir acceso a la red" checked={!!cfg.allow_network} onChange={(allow_network) => set({ allow_network })} />
        <div className="sub">Corre aislado con bubblewrap: no ve tus archivos ni el proyecto. Si <code>resultado</code> es un diccionario, cada clave pasa a ser una variable del flujo.</div>
      </>)}

      {node.type === "output" && (
        <Field label="Texto final"><VarField rows={5} value={cfg.template ?? "{entrada}"} onChange={(template) => set({ template })} vars={vars} /></Field>
      )}

      <Adv>
        <Field label="Reintentos si falla" hint="Útil en nodos que salen a internet; espera cada vez un poco más">
          <input type="number" min={0} max={5} value={cfg.retries ?? 0} onChange={(e) => set({ retries: +e.target.value })} />
        </Field>
      </Adv>
      <Adv>
        <Field label="Guardar resultado en" hint="Nombre de la variable que otros nodos pueden usar">
          <input value={cfg.output_key ?? ""} placeholder={`${node.id}_salida`} onChange={(e) => set({ output_key: e.target.value })} />
        </Field>
      </Adv>

    </div>
  );
}

/** Etiqueta de cada conexión que sale de un nodo de bifurcación. */
function EdgeLabels({ spec, nodeId, onSpec }: { spec: Spec; nodeId: string; onSpec: (s: Spec) => void }) {
  const outs = spec.edges.map((e, i) => ({ e, i })).filter(({ e }) => e.from === nodeId);
  if (!outs.length) return <div className="sub">Este nodo aún no tiene conexiones de salida.</div>;
  const setWhen = (idx: number, when: string) =>
    onSpec({ ...spec, edges: spec.edges.map((e, j) => (j === idx ? { ...e, when } : e)) });
  return (
    <div className="stack">
      {outs.map(({ e, i }) => (
        <div key={i} className="row" style={{ gap: 6, alignItems: "center" }}>
          <span className="sub" style={{ minWidth: 80 }}>→ {spec.nodes.find((n) => n.id === e.to)?.label ?? e.to}</span>
          <input value={e.when ?? ""} placeholder="etiqueta de la rama" onChange={(ev) => setWhen(i, ev.target.value)} />
        </div>
      ))}
    </div>
  );
}

/** Nodo «Herramienta» con un servidor MCP: se elige el servidor, luego UNA de sus herramientas,
 *  y sus argumentos salen del esquema que publica el servidor. */
function McpToolPicker({ cfg, set }: { cfg: any; set: (p: any) => void }) {
  const [servers] = useData<{ id: number; name: string; transport: string; enabled: boolean }[]>("/api/connections/mcp-servers");
  const [lista, setLista] = useState<any[] | null>(null);
  const [error, setError] = useState("");
  const [cargando, setCargando] = useState(false);
  const conf = cfg.config ?? {};

  const cargar = async (id: number) => {
    setCargando(true); setError("");
    try {
      const r = await api.post<any>(`/api/connections/mcp-servers/${id}/test`);
      if (r.ok) setLista(r.tools); else { setLista(null); setError(r.error); }
    } finally { setCargando(false); }
  };
  useEffect(() => { if (conf.server_id) cargar(conf.server_id); }, [conf.server_id]);

  const elegida = lista?.find((t) => t.name === conf.tool);
  const props: Record<string, any> = elegida?.parameters?.properties ?? {};
  const requeridos: string[] = elegida?.parameters?.required ?? [];

  return (<>
    <Field label="Servidor MCP" hint={servers?.length ? "Se registran en Conexiones" : "Aún no hay ninguno: regístralos en Conexiones"}>
      <select value={conf.server_id ?? ""} onChange={(e) => { setLista(null); set({ config: { server_id: e.target.value ? +e.target.value : null, tool: "" }, args: {} }); }}>
        <option value="">— elige —</option>
        {servers?.filter((m) => m.enabled).map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
      </select>
    </Field>
    {cargando && <div className="sub"><Spinner /> Preguntando al servidor qué herramientas tiene…</div>}
    {error && <div className="warn">{error}</div>}
    {lista && (
      <Field label="Herramienta">
        <select value={conf.tool ?? ""} onChange={(e) => {
          const t = lista.find((x) => x.name === e.target.value);
          // Los argumentos obligatorios aparecen ya como filas, vacíos, para no tener que adivinar sus nombres
          const req: string[] = t?.parameters?.required ?? [];
          set({ config: { ...conf, tool: e.target.value }, args: Object.fromEntries(req.map((k) => [k, cfg.args?.[k] ?? ""])) });
        }}>
          <option value="">— elige —</option>
          {lista.map((t) => <option key={t.name} value={t.name}>{t.name}</option>)}
        </select>
      </Field>
    )}
    {elegida && (<>
      <div className="sub">{elegida.description}</div>
      <Field label="Argumentos" hint="Puedes usar {variables}. Números, true/false y JSON ([…], {…}) se convierten solos">
        <KeyList items={Object.entries(cfg.args ?? {}).map(([k, v]) => ({ key: k, value: typeof v === "string" ? v : JSON.stringify(v) }))}
          onChange={(rows) => set({ args: Object.fromEntries(rows.filter((r: any) => r.key).map((r: any) => [r.key, r.value])) })}
          cols={[["key", "argumento"], ["value", "valor"]]} addLabel="Añadir argumento" />
      </Field>
      {!!Object.keys(props).length && (
        <div className="sub" style={{ fontSize: 11 }}>
          {Object.entries(props).map(([k, p]: any) => (
            <div key={k}><code>{k}</code>{requeridos.includes(k) ? " *" : ""} {p.type ? `(${p.type})` : ""} {String(p.description ?? "").slice(0, 120)}</div>
          ))}
        </div>
      )}
    </>)}
  </>);
}

/** Campo generado a partir del catálogo de herramientas (mismo formato que en Agentes). */
function ConfigField({ spec, value, onChange }: { spec: any; value: any; onChange: (v: any) => void }) {
  const v = value ?? spec.default ?? "";
  if (spec.type === "bool") return <Check label={spec.label} checked={!!v} onChange={onChange} />;
  if (spec.type === "select") return (
    <Field label={spec.label} hint={spec.hint}>
      <select value={v} onChange={(e) => onChange(e.target.value)}>
        {(spec.options ?? []).map((o: any) => <option key={o.value ?? o} value={o.value ?? o}>{o.label ?? o}</option>)}
      </select>
    </Field>
  );
  return (
    <Field label={spec.label} hint={spec.hint}>
      <input type={spec.type === "int" || spec.type === "number" ? "number" : "text"} value={v}
        onChange={(e) => onChange(spec.type === "int" || spec.type === "number" ? +e.target.value : e.target.value)} />
    </Field>
  );
}

/** Lista editable de filas con columnas fijas (campos de entrada, ramas, argumentos…). */
function KeyList({ items, onChange, cols, addLabel }: {
  items: any[]; onChange: (v: any[]) => void; cols: [string, string][]; addLabel: string;
}) {
  const rows = items ?? [];
  const patch = (i: number, k: string, v: string) => onChange(rows.map((r, j) => (j === i ? { ...r, [k]: v } : r)));
  return (
    <div className="stack">
      {rows.map((r, i) => (
        <div key={i} className="row" style={{ gap: 6 }}>
          {cols.map(([k, ph]) => (
            <input key={k} value={r[k] ?? ""} placeholder={ph} onChange={(e) => patch(i, k, e.target.value)} />
          ))}
          <button className="icon" title="Quitar" onClick={() => onChange(rows.filter((_, j) => j !== i))}><X size={13} /></button>
        </div>
      ))}
      <button onClick={() => onChange([...rows, Object.fromEntries(cols.map(([k]) => [k, ""]))])}><Plus size={12} /> {addLabel}</button>
    </div>
  );
}

/* ------------------------------------------------------------------ ejecución */

type RunState = { current: string | null; done: string[]; tokens: Record<string, string> };

function RunPanel({ flowId, spec, dirty, onState }: {
  flowId: number; spec: Spec; dirty: boolean; onState: (s: RunState | null) => void;
}) {
  const fields = useMemo(() => spec.nodes.filter((n) => n.type === "input").flatMap((n) => (n.config?.fields ?? []) as any[]), [spec]);
  const [values, setValues] = useState<Record<string, string>>({});
  const [trace, setTrace] = useState<Trace[]>([]);
  const [output, setOutput] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [interrupt, setInterrupt] = useState<Interrupt | null>(null);
  const [runId, setRunId] = useState<number | null>(null);
  const [, setTokens] = useState<Record<string, string>>({});
  const abort = useRef<AbortController | null>(null);
  const { notify } = useToast();

  const consume = async (url: string, body: unknown) => {
    setBusy(true);
    setError("");
    setInterrupt(null);
    abort.current = new AbortController();
    const done: string[] = [];
    const vivo: Record<string, string> = {};
    try {
      await streamNDJSON(url, body, (ev) => {
        if (ev.run_id) setRunId(ev.run_id);
        if (ev.type === "token") {
          // El nodo va escribiendo: se ve en el lienzo mientras genera
          vivo[ev.node] = (vivo[ev.node] ?? "") + ev.text;
          setTokens({ ...vivo });
          onState({ current: ev.node, done: [...done], tokens: { ...vivo } });
        } else if (ev.type === "node") {
          done.push(ev.node);
          delete vivo[ev.node];          // ya terminó: deja de mostrarse el texto parcial
          setTokens({ ...vivo });
          setTrace((t) => [...t, ev as Trace]);
          onState({ current: ev.node, done: [...done], tokens: { ...vivo } });
        } else if (ev.type === "interrupt") {
          setInterrupt(ev.data);
          onState({ current: ev.data.node, done: [...done], tokens: {} });
        } else if (ev.type === "done") {
          setOutput(ev.vars?._salida ?? "");
          onState({ current: null, done: [...done], tokens: {} });
        } else if (ev.type === "error") {
          setError(ev.message);
          onState({ current: null, done: [...done], tokens: {} });
        }
      }, abort.current.signal);
    } catch (e: any) {
      if (e.name !== "AbortError") setError(String(e.message ?? e));
    } finally {
      setBusy(false);
    }
  };

  const start = async () => {
    if (dirty) { notify("Guarda el flujo antes de ejecutarlo", "err"); return; }
    setTrace([]);
    setOutput("");
    const inputs = { ...values, _entrada: values[fields[0]?.key] ?? "" };
    await consume(`/api/flows/${flowId}/run`, { inputs });
  };

  const resume = async (approved: boolean, comment: string, edited: string | null) => {
    if (runId == null) return;
    await consume(`/api/flows/runs/${runId}/resume`, { approved, comment, edited });
  };

  const replay = async (checkpoint_id: string, edits: Record<string, string>) => {
    if (runId == null) return;
    setTrace([]);
    setOutput("");
    await consume(`/api/flows/runs/${runId}/replay`, { checkpoint_id, edits });
  };

  return (
    <div className="card">
      <div className="row between" style={{ marginBottom: 8 }}>
        <b><Workflow size={14} /> Ejecutar</b>
        {busy
          ? <button onClick={() => abort.current?.abort()}><Square size={13} /> Detener</button>
          : <button className="primary" onClick={start}><Play size={13} /> Ejecutar flujo</button>}
      </div>
      {!fields.length && <div className="sub">El nodo de entrada no tiene campos definidos.</div>}
      <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
        {fields.map((f) => (
          <Field key={f.key} label={f.label || f.key}>
            <input value={values[f.key] ?? ""} onChange={(e) => setValues({ ...values, [f.key]: e.target.value })}
              onKeyDown={(e) => { if (e.key === "Enter" && !busy) start(); }} />
          </Field>
        ))}
      </div>

      {!!trace.length && (
        <div className="stack" style={{ marginTop: 10 }}>
          {trace.map((t, i) => (
            <details key={i} className="trace">
              <summary><b>{t.label}</b> <span className="sub">{t.node_type}</span></summary>
              <AssistantText text={String(t.output ?? "").slice(0, 4000)} />
              {!!t.info && Object.keys(t.info).length > 0 && (
                <div className="sub" style={{ marginTop: 4 }}>
                  {Object.entries(t.info).map(([k, v]) => <div key={k}>{k}: {typeof v === "string" ? v : JSON.stringify(v)}</div>)}
                </div>
              )}
            </details>
          ))}
        </div>
      )}
      {busy && !interrupt && <div className="sub" style={{ marginTop: 8 }}><Spinner /> Ejecutando…</div>}
      {!!error && <div className="error" style={{ marginTop: 8 }}>{error}</div>}
      {!!output && (
        <div style={{ marginTop: 10 }}>
          <div className="sub">Resultado</div>
          <AssistantText text={output} />
        </div>
      )}

      {runId != null && !busy && !interrupt && <div style={{ marginTop: 12 }}><TimeTravel runId={runId} onReplay={replay} /></div>}

      {interrupt && <ApprovalModal data={interrupt} onClose={() => setInterrupt(null)} onResume={resume} />}
    </div>
  );
}

function ApprovalModal({ data, onClose, onResume }: {
  data: Interrupt; onClose: () => void; onResume: (ok: boolean, comment: string, edited: string | null) => Promise<void>;
}) {
  const [text, setText] = useState(data.preview ?? "");
  const [comment, setComment] = useState("");
  const [busy, setBusy] = useState(false);
  const act = async (ok: boolean) => {
    setBusy(true);
    onClose();
    await onResume(ok, comment, data.editable && text !== data.preview ? text : null);
    setBusy(false);
  };
  return (
    <Modal title={`Aprobación · ${data.label}`} onClose={onClose} wide>
      <p>{data.message}</p>
      {data.editable
        ? <Field label="Puedes corregirlo antes de continuar"><textarea rows={14} value={text} onChange={(e) => setText(e.target.value)} /></Field>
        : <div className="preview"><AssistantText text={data.preview} /></div>}
      <Field label="Comentario (opcional)"><input value={comment} onChange={(e) => setComment(e.target.value)} /></Field>
      <div className="row end" style={{ marginTop: 10 }}>
        <button disabled={busy} onClick={() => act(false)}><X size={14} /> Rechazar</button>
        <button className="primary" disabled={busy} onClick={() => act(true)}><CheckIcon size={14} /> Aprobar y continuar</button>
      </div>
    </Modal>
  );
}

function Runs({ flowId }: { flowId: number }) {
  const [runs] = useData<any[]>(`/api/flows/${flowId}/runs`, 5000);
  const [open, setOpen] = useState<number | null>(null);
  if (!runs?.length) return <div className="card empty">Todavía no has ejecutado este flujo</div>;
  const cur = runs.find((r) => r.id === open);
  return (
    <div className="card">
      <table>
        <thead><tr><th>#</th><th>Estado</th><th>Nodos</th><th>Entrada</th><th>Cuándo</th></tr></thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.id} className="clickable" onClick={() => setOpen(r.id === open ? null : r.id)}>
              <td>{r.id}</td>
              <td><span className={`badge ${r.status === "completed" ? "ok" : r.status === "failed" ? "err" : ""}`}>
                {{ completed: "completado", failed: "fallido", waiting: "esperando aprobación", running: "en curso" }[r.status as string] ?? r.status}</span></td>
              <td>{(r.trace ?? []).length}</td>
              <td className="ellipsis" style={{ maxWidth: 260 }}>{r.inputs?._entrada ?? ""}</td>
              <td className="sub">{new Date(r.created_at).toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {cur && (
        <div className="stack" style={{ marginTop: 10 }}>
          {!!cur.error && <div className="error">{cur.error}</div>}
          {(cur.trace ?? []).map((t: Trace, i: number) => (
            <details key={i} className="trace"><summary><b>{t.label}</b> <span className="sub">{t.node_type}</span></summary>
              <AssistantText text={String(t.output ?? "").slice(0, 4000)} /></details>
          ))}
          {!!cur.state?._salida && (<><div className="sub">Resultado</div><AssistantText text={cur.state._salida} /></>)}
        </div>
      )}
    </div>
  );
}

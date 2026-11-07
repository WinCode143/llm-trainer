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
    { id: "in", type: "input", label: "Input", x: 60, y: 160, config: { fields: [{ key: "input", label: "Question" }] } },
    { id: "fin", type: "output", label: "Output", x: 460, y: 160, config: { template: "{input}" } },
  ],
  edges: [{ from: "in", to: "fin", when: null }],
};

/** Short, unique id for a new node of the given type. */
function newId(spec: Spec, type: string): string {
  let i = 1;
  while (spec.nodes.some((n) => n.id === `${type}${i}`)) i++;
  return `${type}${i}`;
}

const DEFAULT_CONFIG: Record<string, any> = {
  input: { fields: [{ key: "input", label: "Question" }] },
  llm: { prompt: "{input}", system: "", max_tokens: 600, temperature: 0.3, json: false },
  tool: { tool_id: "", args: {}, config: {} },
  rag: { collection_id: null, query: "{input}", k: 4 },
  agent: { agent_id: null, prompt: "{input}" },
  router: { mode: "expr", branches: [{ label: "yes", when: "True" }, { label: "no", when: "True" }] },
  approval: { message: "Do you approve this result?", preview: "{input}", editable: true },
  output: { template: "{input}" },
  foreach: { items_key: "", mode: "llm", prompt: "{item}", parallel: 3, on_error: "null", max_tokens: 400, max_items: 0 },
  subflow: { flow_id: null, input: "{input}" },
  code: { code: "# The flow's variables are in `vars`.\n# Return a dict in `result` and each key becomes a variable.\nresult = {\"total\": len(vars.get(\"input\", \"\"))}", timeout_s: 20, memory_mb: 1024, allow_network: false },
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
    if (!(await dialog.confirm("Delete flow", `Delete “${name}”? Its runs are lost too.`))) return;
    await api.del(`/api/flows/${id}`);
    setSel(null);
    reload();
    notify("Flow deleted");
  };

  return (
    <>
      <div className="page-head">
        <div><h1>Flows</h1><div className="sub">Chain models, documents, tools and agents in a diagram, with branches and your own approvals</div></div>
        <button className="primary" onClick={() => setShowNew(true)}><Plus size={14} /> New flow</button>
      </div>

      <Help title="What is a flow, and when is it better than an agent?">
        <p>An <b>agent</b> decides on its own what to do at each step: flexible but unpredictable. A <b>flow</b> fixes the path in advance: it always goes through the same nodes, in the same order, and you decide where it can branch. It's for processes that must come out the same every time (answering with verified citations, classifying and routing, drafting and approving before saving).</p>
        <ul>
          <li><b>Input</b> defines the fields you fill in when you run it. Each node stores its result in a variable.</li>
          <li>In any text you can write <code>{"{variable}"}</code> and it's replaced by its value: <code>{"{input}"}</code>, <code>{"{search_output}"}</code>…</li>
          <li><b>Model</b> with “return JSON” on turns each JSON key into its own variable, ready to use in conditions.</li>
          <li><b>Branch</b> picks a path by condition (<code>supported == True</code>) or by letting the model choose a label.</li>
          <li><b>Approval</b> pauses the flow and waits for your go-ahead; you can edit the text before continuing.</li>
        </ul>
        <p style={{ marginTop: 8 }}>Generation uses the model you have loaded in the <b>Playground</b> or <b>Models</b>, unless you pick a <b>helper model</b> per node. Under the hood the flow compiles to a <b>LangGraph</b> graph with checkpointing.</p>
        <ul>
          <li><b>For each</b> repeats something for every item in a list, one at a time or several at once.</li>
          <li><b>Sub-flow</b> puts a whole flow inside another, to reuse it without copying nodes.</li>
          <li><b>Code</b> runs sandboxed Python to transform variables; if you return a dict, each key becomes a variable.</li>
          <li>After a run, <b>Rewind</b> appears: go back to any step, change whatever you want and continue from there without repeating what came before.</li>
          <li><b>Export</b> downloads the flow as LangGraph code, in case you ever want to move off this app.</li>
        </ul>
      </Help>

      <div className="split" style={{ gridTemplateColumns: "230px minmax(0,1fr)" }}>
        <div className="card" style={{ padding: 8 }}>
          {!list?.length && <div className="empty" style={{ fontSize: 13 }}>No flows yet</div>}
          <table><tbody>
            {list?.map((f) => (
              <tr key={f.id} className={`clickable ${sel === f.id ? "selected" : ""}`} onClick={() => setSel(f.id)}>
                <td><b>{f.name}</b><div className="sub" style={{ fontSize: 12 }}>{f.nodes} node(s)</div></td>
                <td style={{ width: 28 }}>
                  <button className="icon" title="Delete" onClick={(e) => { e.stopPropagation(); remove(f.id, f.name); }}><Trash2 size={13} /></button>
                </td>
              </tr>
            ))}
          </tbody></table>
        </div>
        {sel != null && meta ? <Editor key={sel} id={sel} meta={meta} onSaved={reload} /> : <div className="card empty">Choose a flow or create a new one</div>}
      </div>

      {showNew && meta && <NewFlow meta={meta} onCreate={create} onClose={() => setShowNew(false)} />}
    </>
  );
}

function NewFlow({ meta, onCreate, onClose }: { meta: Meta; onCreate: (k: string | null, n: string) => Promise<void>; onClose: () => void }) {
  const [key, setKey] = useState<string | null>(Object.keys(meta.templates)[0] ?? null);
  const [name, setName] = useState(key ? meta.templates[key].name : "My flow");
  const [busy, setBusy] = useState(false);
  const pick = (k: string | null) => { setKey(k); setName(k ? meta.templates[k].name : "My flow"); };
  return (
    <Modal title="New flow" onClose={onClose}>
      <div className="sub" style={{ marginBottom: 8 }}>Start from a ready-made template or from an empty flow.</div>
      <div className="stack">
        {Object.entries(meta.templates).map(([k, t]) => (
          <label key={k} className={`option ${key === k ? "selected" : ""}`} onClick={() => pick(k)}>
            <input type="radio" checked={key === k} readOnly />
            <div><b>{t.name}</b><div className="sub">{t.description}</div></div>
          </label>
        ))}
        <label className={`option ${key === null ? "selected" : ""}`} onClick={() => pick(null)}>
          <input type="radio" checked={key === null} readOnly />
          <div><b>Blank</b><div className="sub">Just input and output; add the nodes you want</div></div>
        </label>
      </div>
      <Field label="Name"><input value={name} onChange={(e) => setName(e.target.value)} /></Field>
      <div className="row end" style={{ marginTop: 10 }}>
        <button onClick={onClose}>Cancel</button>
        <button className="primary" disabled={!name.trim() || busy}
          onClick={async () => { setBusy(true); try { await onCreate(key, name.trim()); } finally { setBusy(false); } }}>
          {busy ? <Spinner /> : <Plus size={14} />} Create
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
    notify(saved.warnings?.length ? `Saved · ${saved.warnings.length} warning(s)` : "Flow saved");
    reloadFlow();
  };

  const exportFlow = async () => {
    const r = await api.get<{ filename: string; code: string }>(`/api/flows/${id}/export`);
    const url = URL.createObjectURL(new Blob([r.code], { type: "text/x-python" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = r.filename;
    a.click();
    URL.revokeObjectURL(url);
    notify(`${r.filename} downloaded`);
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
  if (!spec) return <div className="card empty"><Spinner /> Loading…</div>;
  const node = spec.nodes.find((n) => n.id === selected) ?? null;

  return (
    <div className="stack">
      <div className="card">
        <div className="row" style={{ gap: 8, alignItems: "flex-end" }}>
          <Field label="Name"><input value={name} onChange={(e) => { setName(e.target.value); setDirty(true); }} /></Field>
          <Field label="Description"><input value={description} onChange={(e) => { setDescription(e.target.value); setDirty(true); }} /></Field>
          <button title="Arrange the nodes in layers" onClick={() => update(autoLayout(spec))}>
            <LayoutGrid size={14} /> Arrange
          </button>
          <button title="Download the flow as LangGraph code" onClick={exportFlow}><Code2 size={14} /> Export</button>
          <button className="primary" onClick={save} disabled={!dirty}><Save size={14} /> Save</button>
        </div>
        {!!flow?.warnings?.length && (
          <div className="warn" style={{ marginTop: 8 }}>
            <b>{flow.warnings.length} thing(s) need fixing before it works:</b>
            {flow.warnings.map((w, i) => <div key={i}>⚠ {w}</div>)}
          </div>
        )}
        <div className="tabs" style={{ marginTop: 10 }}>
          <button className={tab === "editor" ? "active" : ""} onClick={() => setTab("editor")}>Diagram</button>
          <button className={tab === "runs" ? "active" : ""} onClick={() => setTab("runs")}>Runs</button>
        </div>
      </div>

      {tab === "runs" ? <Runs flowId={id} /> : (
        <>
          <div className="card">
            <div className="sub" style={{ marginBottom: 6 }}>Add nodes and drag from one handle to another to connect them. Select a connection or a node and press Delete to remove it.</div>
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
              <SimpleNote>The node's fine-grained settings are hidden (label, output variable name,
                system instructions, limits and retries). The defaults work.</SimpleNote>
              {node ? <NodeProps node={node} spec={spec} meta={meta} onChange={(p) => patchNode(node.id, p)} onSpec={update} flowId={id} />
                : <div className="empty" style={{ fontSize: 13 }}>Select a node to configure it</div>}
            </div>
          </div>

          <RunPanel flowId={id} spec={spec} dirty={dirty} onState={setRun} />
        </>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ properties per node type */

function NodeProps({ node, spec, meta, onChange, onSpec, flowId }: {
  node: FlowNode; spec: Spec; meta: Meta; onChange: (p: Partial<FlowNode>) => void; onSpec: (s: Spec) => void;
  flowId: number;   // so it doesn't offer itself as a sub-flow
}) {
  const cfg = node.config || {};
  const set = (patch: any) => onChange({ config: { ...cfg, ...patch } });
  const [tools] = useData<ToolSpec[]>(node.type === "tool" || node.type === "foreach" ? "/api/agents/tools" : null);
  const [colls] = useData<Collection[]>(node.type === "rag" ? "/api/rag/collections" : null);
  const [agents] = useData<AgentRow[]>(node.type === "agent" ? "/api/agents" : null);
  const [flows] = useData<FlowRow[]>(node.type === "subflow" || node.type === "foreach" ? "/api/flows" : null);
  const outKey = cfg.output_key || `${node.id}_output`;
  // Variables the node can use: those of the earlier nodes plus the JSON keys they declare
  const vars = useMemo(() => {
    const out = new Set<string>(["input"]);
    for (const n of spec.nodes) {
      if (n.id === node.id) continue;
      out.add((n.config?.output_key as string) || `${n.id}_output`);
      if (n.type === "rag") out.add(`${(n.config?.output_key as string) || `${n.id}_output`}_sources`);
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
      <Adv><Field label="Label"><input value={node.label ?? ""} onChange={(e) => onChange({ label: e.target.value })} /></Field></Adv>

      {node.type === "input" && (
        <Field label="Form fields" hint="Each field becomes a variable with that name">
          <KeyList items={cfg.fields ?? []} onChange={(fields) => set({ fields })}
            cols={[["key", "variable"], ["label", "label"]]} addLabel="Add field" />
        </Field>
      )}

      {node.type === "llm" && (<>
        <Adv>
          <Field label="System instructions" hint="Optional; sets the model's role">
            <VarField rows={2} value={cfg.system ?? ""} onChange={(system) => set({ system })} vars={vars} />
          </Field>
        </Adv>
        <Field label="Prompt" hint="Type { to insert a variable">
          <VarField rows={6} value={cfg.prompt ?? ""} onChange={(prompt) => set({ prompt })} vars={vars} />
        </Field>
        <Check label="Return JSON (each key becomes a variable)" checked={!!cfg.json} onChange={(json) => set({ json })} />
        <Adv>
          <div className="row" style={{ gap: 8 }}>
            <Field label="Max tokens"><input type="number" value={cfg.max_tokens ?? 600} onChange={(e) => set({ max_tokens: +e.target.value })} /></Field>
            <Field label="Temperature"><input type="number" step="0.1" value={cfg.temperature ?? 0.3} onChange={(e) => set({ temperature: +e.target.value })} /></Field>
          </div>
        </Adv>
        <HelperPicker value={cfg.helper ?? null} onChange={(helper) => set({ helper })}
          hint="Per node: a cheap one to classify, a strong one to write" />
      </>)}

      {node.type === "tool" && (<>
        <Field label="Tool">
          <select value={cfg.tool_id ?? ""} onChange={(e) => set({ tool_id: e.target.value, args: {}, config: {} })}>
            <option value="">— choose —</option>
            {tools?.map((t) => <option key={t.id} value={t.id}>{t.label}</option>)}
          </select>
        </Field>
        {(() => {
          const t = tools?.find((x) => x.id === cfg.tool_id);
          if (!t) return null;
          if (t.id === "mcp") return <McpToolPicker cfg={cfg} set={set} />;
          return (<>
            <div className="sub">{t.desc}</div>
            <Field label="Arguments" hint="You can use {variables} in the values">
              <KeyList items={Object.entries(cfg.args ?? {}).map(([k, v]) => ({ key: k, value: String(v) }))}
                onChange={(rows) => set({ args: Object.fromEntries(rows.filter((r: any) => r.key).map((r: any) => [r.key, r.value])) })}
                cols={[["key", "argument"], ["value", "value"]]} addLabel="Add argument" />
            </Field>
            {!!t.config?.length && (
              <Field label="Tool settings">
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
        <Field label="Collection">
          <select value={cfg.collection_id ?? ""} onChange={(e) => set({ collection_id: e.target.value ? +e.target.value : null })}>
            <option value="">— choose —</option>
            {colls?.map((c) => <option key={c.id} value={c.id}>{c.name} ({c.num_chunks} chunks)</option>)}
          </select>
        </Field>
        <Field label="Query"><VarField rows={2} value={cfg.query ?? "{input}"} onChange={(query) => set({ query })} vars={vars} /></Field>
        <Field label="Chunks to retrieve"><input type="number" value={cfg.k ?? 4} onChange={(e) => set({ k: +e.target.value })} /></Field>
      </>)}

      {node.type === "agent" && (<>
        <Field label="Agent">
          <select value={cfg.agent_id ?? ""} onChange={(e) => set({ agent_id: e.target.value ? +e.target.value : null })}>
            <option value="">— choose —</option>
            {agents?.map((a) => <option key={a.id} value={a.id}>{a.name}</option>)}
          </select>
        </Field>
        <Field label="Task"><VarField rows={4} value={cfg.prompt ?? "{input}"} onChange={(prompt) => set({ prompt })} vars={vars} /></Field>
      </>)}

      {node.type === "router" && (<>
        <Field label="How it decides">
          <select value={cfg.mode ?? "expr"} onChange={(e) => set({ mode: e.target.value })}>
            <option value="expr">By condition</option>
            <option value="llm">Let the model choose</option>
          </select>
        </Field>
        {cfg.mode === "llm" && <Field label="Question for the model"><VarField rows={3} value={cfg.prompt ?? "{input}"} onChange={(prompt) => set({ prompt })} vars={vars} /></Field>}
        {cfg.mode === "llm" && <HelperPicker value={cfg.helper ?? null} onChange={(helper) => set({ helper })} />}
        <Field label="Branches" hint={cfg.mode === "llm" ? "The model picks one of these labels" : "The first one whose condition holds is taken, e.g. supported == True"}>
          <KeyList items={cfg.branches ?? []} onChange={(branches) => set({ branches })}
            cols={cfg.mode === "llm" ? [["label", "label"]] : [["label", "label"], ["when", "condition"]]}
            addLabel="Add branch" />
        </Field>
        <div className="sub">Connect this node to the next ones and set each connection to its branch's label (the “When” field below).</div>
        <EdgeLabels spec={spec} nodeId={node.id} onSpec={onSpec} />
      </>)}

      {node.type === "approval" && (<>
        <Field label="Message"><VarField rows={2} value={cfg.message ?? ""} onChange={(message) => set({ message })} vars={vars} /></Field>
        <Field label="What to show for review"><VarField rows={4} value={cfg.preview ?? ""} onChange={(preview) => set({ preview })} vars={vars} /></Field>
        <Check label="Allow editing the text before continuing" checked={!!cfg.editable} onChange={(editable) => set({ editable })} />
        {cfg.editable && <Adv><Field label="Variable replaced when editing" hint="By default, this node's output">
          <input value={cfg.edit_key ?? ""} placeholder={outKey} onChange={(e) => set({ edit_key: e.target.value })} /></Field></Adv>}
      </>)}

      {node.type === "foreach" && (<>
        <Field label="List to repeat over" hint="A variable holding a list; if it's text, each line is used">
          <select value={cfg.items_key ?? ""} onChange={(e) => set({ items_key: e.target.value })}>
            <option value="">— choose —</option>
            {vars.map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </Field>
        <Field label="What to do with each item">
          <select value={cfg.mode ?? "llm"} onChange={(e) => set({ mode: e.target.value })}>
            <option value="llm">Ask the model</option>
            <option value="tool">Run a tool</option>
            <option value="subflow">Run a whole flow</option>
          </select>
        </Field>
        {(cfg.mode ?? "llm") === "llm" && (<>
          <Field label="Prompt" hint="{item} is the item and {index} its position">
            <VarField rows={4} value={cfg.prompt ?? "{item}"} onChange={(prompt) => set({ prompt })} vars={["item", "index", ...vars]} />
          </Field>
          <HelperPicker value={cfg.helper ?? null} onChange={(helper) => set({ helper })}
            hint="With an external helper the parallelism is real; the local model serves one at a time" />
        </>)}
        {cfg.mode === "tool" && (<>
          <Field label="Tool">
            <select value={cfg.tool_id ?? ""} onChange={(e) => set({ tool_id: e.target.value, args: {} })}>
              <option value="">— choose —</option>
              {tools?.map((t) => <option key={t.id} value={t.id}>{t.label}</option>)}
            </select>
          </Field>
          <Field label="Arguments" hint="Use {item} for the current item">
            <KeyList items={Object.entries(cfg.args ?? {}).map(([k, v]) => ({ key: k, value: String(v) }))}
              onChange={(rows) => set({ args: Object.fromEntries(rows.filter((r: any) => r.key).map((r: any) => [r.key, r.value])) })}
              cols={[["key", "argument"], ["value", "value"]]} addLabel="Add argument" />
          </Field>
        </>)}
        {cfg.mode === "subflow" && (<>
          <Field label="Flow run for each item">
            <select value={cfg.flow_id ?? ""} onChange={(e) => set({ flow_id: e.target.value ? +e.target.value : null })}>
              <option value="">— choose —</option>
              {flows?.filter((f) => f.id !== flowId).map((f) => <option key={f.id} value={f.id}>{f.name}</option>)}
            </select>
          </Field>
          <Field label="What it receives"><VarField rows={2} value={cfg.input ?? "{item}"} onChange={(input) => set({ input })} vars={["item", "index", ...vars]} /></Field>
        </>)}
        <Adv>
          <div className="row" style={{ gap: 8 }}>
            <Field label="At once" hint="1 = one after another"><input type="number" min={1} max={10} value={cfg.parallel ?? 1} onChange={(e) => set({ parallel: +e.target.value })} /></Field>
            <Field label="Max items" hint="0 = all"><input type="number" min={0} value={cfg.max_items ?? 0} onChange={(e) => set({ max_items: +e.target.value })} /></Field>
          </div>
        </Adv>
        <Field label="If one fails">
          <select value={cfg.on_error ?? "stop"} onChange={(e) => set({ on_error: e.target.value })}>
            <option value="stop">Stop everything</option>
            <option value="null">Continue and leave it empty</option>
            <option value="skip">Continue and drop it</option>
          </select>
        </Field>
        <div className="sub">Besides the list, you get <code>{`{${outKey}_text}`}</code> with everything joined and <code>{`{${outKey}_n}`}</code> with how many came out.</div>
      </>)}

      {node.type === "subflow" && (<>
        <Field label="Flow to run">
          <select value={cfg.flow_id ?? ""} onChange={(e) => set({ flow_id: e.target.value ? +e.target.value : null })}>
            <option value="">— choose —</option>
            {flows?.filter((f) => f.id !== flowId).map((f) => <option key={f.id} value={f.id}>{f.name}</option>)}
          </select>
        </Field>
        <Field label="What it receives as input"><VarField rows={2} value={cfg.input ?? "{input}"} onChange={(input) => set({ input })} vars={vars} /></Field>
        <div className="sub">The sub-flow runs in full and returns its output. A flow can't call itself.</div>
      </>)}

      {node.type === "code" && (<>
        <Field label="Python code" hint="The variables are in `vars`; put the result in `result`">
          <textarea rows={12} className="mono" value={cfg.code ?? ""} onChange={(e) => set({ code: e.target.value })} />
        </Field>
        <Adv>
          <div className="row" style={{ gap: 8 }}>
            <Field label="Max seconds"><input type="number" value={cfg.timeout_s ?? 20} onChange={(e) => set({ timeout_s: +e.target.value })} /></Field>
            <Field label="Memory (MB)"><input type="number" value={cfg.memory_mb ?? 1024} onChange={(e) => set({ memory_mb: +e.target.value })} /></Field>
          </div>
        </Adv>
        <Check label="Allow network access" checked={!!cfg.allow_network} onChange={(allow_network) => set({ allow_network })} />
        <div className="sub">Runs sandboxed with bubblewrap: it can't see your files or the project. If <code>result</code> is a dict, each key becomes a flow variable.</div>
      </>)}

      {node.type === "output" && (
        <Field label="Final text"><VarField rows={5} value={cfg.template ?? "{input}"} onChange={(template) => set({ template })} vars={vars} /></Field>
      )}

      <Adv>
        <Field label="Retries on failure" hint="Useful for nodes that go out to the internet; it waits a little longer each time">
          <input type="number" min={0} max={5} value={cfg.retries ?? 0} onChange={(e) => set({ retries: +e.target.value })} />
        </Field>
      </Adv>
      <Adv>
        <Field label="Save the result as" hint="Name of the variable other nodes can use">
          <input value={cfg.output_key ?? ""} placeholder={`${node.id}_output`} onChange={(e) => set({ output_key: e.target.value })} />
        </Field>
      </Adv>

    </div>
  );
}

/** The label of each connection leaving a branch node. */
function EdgeLabels({ spec, nodeId, onSpec }: { spec: Spec; nodeId: string; onSpec: (s: Spec) => void }) {
  const outs = spec.edges.map((e, i) => ({ e, i })).filter(({ e }) => e.from === nodeId);
  if (!outs.length) return <div className="sub">This node has no outgoing connections yet.</div>;
  const setWhen = (idx: number, when: string) =>
    onSpec({ ...spec, edges: spec.edges.map((e, j) => (j === idx ? { ...e, when } : e)) });
  return (
    <div className="stack">
      {outs.map(({ e, i }) => (
        <div key={i} className="row" style={{ gap: 6, alignItems: "center" }}>
          <span className="sub" style={{ minWidth: 80 }}>→ {spec.nodes.find((n) => n.id === e.to)?.label ?? e.to}</span>
          <input value={e.when ?? ""} placeholder="branch label" onChange={(ev) => setWhen(i, ev.target.value)} />
        </div>
      ))}
    </div>
  );
}

/** “Tool” node with an MCP server: you pick the server, then ONE of its tools,
  *  and its arguments come from the schema the server publishes. */
function McpToolPicker({ cfg, set }: { cfg: any; set: (p: any) => void }) {
  const [servers] = useData<{ id: number; name: string; transport: string; enabled: boolean }[]>("/api/connections/mcp-servers");
  const [tools, setTools] = useState<any[] | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const conf = cfg.config ?? {};

  const loadTools = async (id: number) => {
    setLoading(true); setError("");
    try {
      const r = await api.post<any>(`/api/connections/mcp-servers/${id}/test`);
      if (r.ok) setTools(r.tools); else { setTools(null); setError(r.error); }
    } finally { setLoading(false); }
  };
  useEffect(() => { if (conf.server_id) loadTools(conf.server_id); }, [conf.server_id]);

  const chosen = tools?.find((t) => t.name === conf.tool);
  const props: Record<string, any> = chosen?.parameters?.properties ?? {};
  const required: string[] = chosen?.parameters?.required ?? [];

  return (<>
    <Field label="MCP server" hint={servers?.length ? "They're registered under Connections" : "None yet: register them under Connections"}>
      <select value={conf.server_id ?? ""} onChange={(e) => { setTools(null); set({ config: { server_id: e.target.value ? +e.target.value : null, tool: "" }, args: {} }); }}>
        <option value="">— choose —</option>
        {servers?.filter((m) => m.enabled).map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
      </select>
    </Field>
    {loading && <div className="sub"><Spinner /> Asking the server which tools it has…</div>}
    {error && <div className="warn">{error}</div>}
    {tools && (
      <Field label="Tool">
        <select value={conf.tool ?? ""} onChange={(e) => {
          const t = tools.find((x) => x.name === e.target.value);
          // Required arguments show up as empty rows, so you don't have to guess their names
          const req: string[] = t?.parameters?.required ?? [];
          set({ config: { ...conf, tool: e.target.value }, args: Object.fromEntries(req.map((k) => [k, cfg.args?.[k] ?? ""])) });
        }}>
          <option value="">— choose —</option>
          {tools.map((t) => <option key={t.name} value={t.name}>{t.name}</option>)}
        </select>
      </Field>
    )}
    {chosen && (<>
      <div className="sub">{chosen.description}</div>
      <Field label="Arguments" hint="You can use {variables}. Numbers, true/false and JSON ([…], {…}) are converted automatically">
        <KeyList items={Object.entries(cfg.args ?? {}).map(([k, v]) => ({ key: k, value: typeof v === "string" ? v : JSON.stringify(v) }))}
          onChange={(rows) => set({ args: Object.fromEntries(rows.filter((r: any) => r.key).map((r: any) => [r.key, r.value])) })}
          cols={[["key", "argument"], ["value", "value"]]} addLabel="Add argument" />
      </Field>
      {!!Object.keys(props).length && (
        <div className="sub" style={{ fontSize: 11 }}>
          {Object.entries(props).map(([k, p]: any) => (
            <div key={k}><code>{k}</code>{required.includes(k) ? " *" : ""} {p.type ? `(${p.type})` : ""} {String(p.description ?? "").slice(0, 120)}</div>
          ))}
        </div>
      )}
    </>)}
  </>);
}

/** A field generated from the tool catalog (same format as in Agents). */
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

/** Editable list of rows with fixed columns (input fields, branches, arguments…). */
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
          <button className="icon" title="Remove" onClick={() => onChange(rows.filter((_, j) => j !== i))}><X size={13} /></button>
        </div>
      ))}
      <button onClick={() => onChange([...rows, Object.fromEntries(cols.map(([k]) => [k, ""]))])}><Plus size={12} /> {addLabel}</button>
    </div>
  );
}

/* ------------------------------------------------------------------ running */

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
    const live: Record<string, string> = {};
    try {
      await streamNDJSON(url, body, (ev) => {
        if (ev.run_id) setRunId(ev.run_id);
        if (ev.type === "token") {
          // The node writes as it goes: you see it on the canvas while it generates
          live[ev.node] = (live[ev.node] ?? "") + ev.text;
          setTokens({ ...live });
          onState({ current: ev.node, done: [...done], tokens: { ...live } });
        } else if (ev.type === "node") {
          done.push(ev.node);
          delete live[ev.node];          // it's done: stop showing the partial text
          setTokens({ ...live });
          setTrace((t) => [...t, ev as Trace]);
          onState({ current: ev.node, done: [...done], tokens: { ...live } });
        } else if (ev.type === "interrupt") {
          setInterrupt(ev.data);
          onState({ current: ev.data.node, done: [...done], tokens: {} });
        } else if (ev.type === "done") {
          setOutput(ev.vars?._output ?? "");
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
    if (dirty) { notify("Save the flow before running it", "err"); return; }
    setTrace([]);
    setOutput("");
    const inputs = { ...values, _input: values[fields[0]?.key] ?? "" };
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
        <b><Workflow size={14} /> Run</b>
        {busy
          ? <button onClick={() => abort.current?.abort()}><Square size={13} /> Stop</button>
          : <button className="primary" onClick={start}><Play size={13} /> Run flow</button>}
      </div>
      {!fields.length && <div className="sub">The input node has no fields defined.</div>}
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
      {busy && !interrupt && <div className="sub" style={{ marginTop: 8 }}><Spinner /> Running…</div>}
      {!!error && <div className="error" style={{ marginTop: 8 }}>{error}</div>}
      {!!output && (
        <div style={{ marginTop: 10 }}>
          <div className="sub">Result</div>
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
    <Modal title={`Approval · ${data.label}`} onClose={onClose} wide>
      <p>{data.message}</p>
      {data.editable
        ? <Field label="You can correct it before continuing"><textarea rows={14} value={text} onChange={(e) => setText(e.target.value)} /></Field>
        : <div className="preview"><AssistantText text={data.preview} /></div>}
      <Field label="Comment (optional)"><input value={comment} onChange={(e) => setComment(e.target.value)} /></Field>
      <div className="row end" style={{ marginTop: 10 }}>
        <button disabled={busy} onClick={() => act(false)}><X size={14} /> Reject</button>
        <button className="primary" disabled={busy} onClick={() => act(true)}><CheckIcon size={14} /> Approve and continue</button>
      </div>
    </Modal>
  );
}

function Runs({ flowId }: { flowId: number }) {
  const [runs] = useData<any[]>(`/api/flows/${flowId}/runs`, 5000);
  const [open, setOpen] = useState<number | null>(null);
  if (!runs?.length) return <div className="card empty">You haven't run this flow yet</div>;
  const cur = runs.find((r) => r.id === open);
  return (
    <div className="card">
      <table>
        <thead><tr><th>#</th><th>Status</th><th>Nodes</th><th>Input</th><th>When</th></tr></thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.id} className="clickable" onClick={() => setOpen(r.id === open ? null : r.id)}>
              <td>{r.id}</td>
              <td><span className={`badge ${r.status === "completed" ? "ok" : r.status === "failed" ? "err" : ""}`}>
                {{ completed: "completed", failed: "failed", waiting: "waiting for approval", running: "running" }[r.status as string] ?? r.status}</span></td>
              <td>{(r.trace ?? []).length}</td>
              <td className="ellipsis" style={{ maxWidth: 260 }}>{r.inputs?._input ?? ""}</td>
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
          {!!cur.state?._output && (<><div className="sub">Result</div><AssistantText text={cur.state._output} /></>)}
        </div>
      )}
    </div>
  );
}

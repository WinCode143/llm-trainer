import { useEffect, useState } from "react";
import { Bot, ChevronRight, Plug, Plus, Save, Send, Square, Trash2, Wrench } from "lucide-react";
import { api, type Collection, type Model, streamNDJSON } from "../api";
import { HelperPicker } from "../components/Helper";
import { AssistantText } from "../components/Markdownish";
import { Adv, Field, Help, Modal, Spinner, useData, useDialog, useToast } from "../components/ui";

type ToolSpec = { id: string; label: string; desc: string; icon: string; function: string | null; config: any[] };
type Agent = { id: number; name: string; description: string; model_id: number | null; helper?: number | null; system_prompt: string;
  tools: { id: string; config: Record<string, any> }[]; max_steps: number; max_tokens: number; temperature: number };

const EMPTY: Omit<Agent, "id"> = { name: "", description: "", model_id: null, helper: null, system_prompt: "You are a helpful assistant. Use the available tools when they provide information you don't have.", tools: [], max_steps: 5, max_tokens: 700, temperature: 0.3 };

export default function Agents() {
  const [list, reload] = useData<Agent[]>("/api/agents");
  const [sel, setSel] = useState<number | "new">("new");
  const current = sel === "new" ? null : list?.find((a) => a.id === sel);
  const [importing, setImporting] = useState(false);
  return (
    <>
      <div className="page-head">
        <div><h1>Agents</h1><div className="sub">A model + the tools you choose: search your documents or the web, run code, read files or use any MCP server</div></div>
        <div className="row" style={{ gap: 8 }}>
          <button onClick={() => setImporting(true)}>Import from CrewAI</button>
          <button className="primary" onClick={() => setSel("new")}><Plus size={14} /> New agent</button>
        </div>
      </div>
      <Help title="What is an agent?">
        <p>An agent is <b>a model that can use tools</b>. Instead of answering from memory, it decides to call a tool (search your documents, search the web, calculate…), reads the result and keeps going until it has your answer.</p>
        <ul>
          <li>You create a profile: <b>model + instructions + ticked tools</b>. Each tool has its own settings.</li>
          <li><b>Max steps</b> limits how many times it can call tools before answering.</li>
          <li>With <b>External MCP server</b> you add all the tools of any compatible server at once.</li>
          <li>Below there's a panel to try it: you'll see every call, its arguments and the result.</li>
        </ul>
        <p style={{ marginTop: 8 }}>The <b>Run Python</b> tool runs sandboxed with bubblewrap: the code can't see your files or the project, has no network unless you enable it, and gets memory and time limits. You can give it a read-only data folder (it shows up inside as <code>/data</code>).</p>
        <p style={{ marginTop: 8 }}>With small models (under 7B) agents often get it wrong when choosing a tool or reading the result.</p>
      </Help>
      <div className="split" style={{ gridTemplateColumns: "230px minmax(0,1fr)" }}>
        <div className="card" style={{ padding: 8 }}>
          {!list?.length && <div className="empty" style={{ fontSize: 13 }}>No agents</div>}
          <table><tbody>
            {list?.map((a) => (
              <tr key={a.id} className={`clickable ${sel === a.id ? "selected" : ""}`} onClick={() => setSel(a.id)}>
                <td><b>{a.name}</b><div className="sub" style={{ fontSize: 12 }}>{a.tools.length} tool(s)</div></td>
              </tr>
            ))}
          </tbody></table>
        </div>
        <Editor key={sel} agent={current} onSaved={(a) => { reload(); setSel(a.id); }} onDeleted={() => { reload(); setSel("new"); }} />
      </div>
      {importing && <ImportCrew onClose={() => setImporting(false)} onDone={() => { setImporting(false); reload(); }} />}
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
      setDirty(false); notify("Agent saved"); onSaved(saved);
    } catch (e: any) { notify(e.message, "err"); }
  };
  const del = async () => {
    if (!agent || !(await confirm(`Delete the agent “${agent.name}”?`, undefined, { label: "Delete" }))) return;
    await api.del(`/api/agents/${agent.id}`); onDeleted();
  };

  return (
    <div className="stack" style={{ gap: 16 }}>
      <div className="card stack">
        <div className="row between">
          <h2 style={{ margin: 0 }}>{agent ? agent.name : "New agent"}</h2>
          <div className="row">
            {agent && <button className="danger" onClick={del}><Trash2 size={14} /></button>}
            <button className="primary" disabled={!f.name || !dirty} onClick={save}><Save size={14} /> Save</button>
          </div>
        </div>
        <div className="grid g2">
          <Field label="Name"><input value={f.name} onChange={(e) => set({ name: e.target.value })} placeholder="researcher" /></Field>
          <Field label="Model" hint="Loaded automatically when the agent runs">
            <select value={f.model_id ?? ""} onChange={(e) => set({ model_id: e.target.value ? +e.target.value : null })}>
              <option value="">Whichever is loaded</option>
              {models?.filter((m) => m.kind !== "embedding").map((m) => <option key={m.id} value={m.id}>{m.name} [{m.kind}]</option>)}
            </select>
          </Field>
        </div>
        <Field label="Instructions (system prompt)"><textarea rows={3} value={f.system_prompt} onChange={(e) => set({ system_prompt: e.target.value })} /></Field>
        <div className="grid g3">
          <Field label={`Max steps: ${f.max_steps}`} hint="Each step = one model reply (it may call tools)"><input type="range" min={1} max={12} value={f.max_steps} onChange={(e) => set({ max_steps: +e.target.value })} /></Field>
          <HelperPicker value={f.helper ?? null} onChange={(helper) => set({ helper })}
            hint="With an external helper the agent doesn't use the GPU. In exchange it loses native tool calling: the tools are described in the prompt." />
          <Adv>
            <Field label={`Temperature: ${f.temperature}`}><input type="range" min={0} max={1.5} step={0.05} value={f.temperature} onChange={(e) => set({ temperature: +e.target.value })} /></Field>
            <Field label={`Max tokens per step: ${f.max_tokens}`}><input type="range" min={128} max={2048} step={64} value={f.max_tokens} onChange={(e) => set({ max_tokens: +e.target.value })} /></Field>
          </Adv>
        </div>
      </div>

      <div className="card">
        <div className="row between"><h3 style={{ margin: 0 }}><Wrench size={15} /> Tools</h3><span className="sub">{f.tools.length} enabled</span></div>
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
      <option value="">— choose —</option>{cols.map((x) => <option key={x.id} value={x.id}>{x.name}</option>)}</select></Field>;
  if (c.type === "mcp_server")
    return <McpServerSelect label={c.label} value={value} onChange={onChange} />;
  if (c.type === "select")
    return <Field label={c.label}><select {...common}>{c.options.map((o: string) => <option key={o}>{o}</option>)}</select></Field>;
  if (c.type === "bool")
    return <label className="check"><input type="checkbox" checked={!!value} onChange={(e) => onChange(e.target.checked)} /> {c.label}</label>;
  if (c.type === "tags")
    return <Field label={c.label} hint="comma-separated"><input value={(value ?? []).join(", ")} onChange={(e) => onChange(e.target.value.split(",").map((x: string) => x.trim()).filter(Boolean))} /></Field>;
  return <Field label={c.label} hint={c.type === "path" ? "The agent can't leave this folder" : undefined}>
    <input type={c.type === "number" ? "number" : "text"} placeholder={c.placeholder} {...common} /></Field>;
}

function McpProbe({ url }: { url: string }) {
  const [res, setRes] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const probe = async () => {
    setBusy(true);
    try {
      const r = await api.post<any>("/api/agents/tools/mcp/probe", { url });
      setRes(`${r.tools.length} tools: ${r.tools.map((t: any) => t.name).join(", ")}`);
    } catch (e: any) { setRes(e.message); } finally { setBusy(false); }
  };
  return (
    <div className="row">
      <button className="sm" disabled={!url || busy} onClick={probe}>{busy ? <Spinner /> : <Plug size={12} />} Test connection</button>
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
        <h3 style={{ margin: 0 }}><Bot size={15} /> Try the agent</h3>
        {toolsEv && <span className="sub" style={{ fontSize: 12 }}>{toolsEv.tools.length} tools loaded · {toolsEv.native ? "native tool calling" : "instructions in the prompt"}</span>}
      </div>
      {dirty && <div className="sub">Save your changes to try them.</div>}
      <div className="row" style={{ flexWrap: "nowrap" }}>
        <input value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && send()}
          placeholder="Write a request for the agent…" />
        {busy ? <button onClick={() => abort.ctrl?.abort()}><Square size={14} /></button>
          : <button className="primary" onClick={send} disabled={!input.trim()}><Send size={14} /></button>}
      </div>
      {history.length > 0 && <div className="sub" style={{ fontSize: 12 }}>Turns in context: {history.length} · <button className="ghost sm" onClick={() => { setHistory([]); setEvents([]); setAnswer(""); }}>reset</button></div>}
      {calls.length > 0 && (
        <div className="stack" style={{ gap: 4 }}>
          {calls.map((e, i) => e.type === "tool_call" ? (
            <div key={i} className="row" style={{ gap: 6 }}>
              <ChevronRight size={13} /><b>{e.name}</b>
              <code style={{ fontSize: 12 }}>{JSON.stringify(e.arguments).slice(0, 120)}</code>
              {!e.known && <span className="badge err">doesn't exist</span>}
            </div>
          ) : (
            <details key={i} className="source" style={{ fontSize: 12.5 }}>
              <summary className="sub">result of {e.name} · {e.seconds}s · {e.result.length} characters</summary>
              <div style={{ whiteSpace: "pre-wrap", marginTop: 6 }}>{e.result.slice(0, 2000)}</div>
            </details>
          ))}
        </div>
      )}
      {(answer || busy) && (
        <div className="msg assistant" style={{ maxWidth: "100%" }}>
          {answer ? <AssistantText text={answer} /> : <Spinner />}
          {done && <div className="meta">{done.steps} step(s){done.limit_reached ? " · limit reached" : ""}</div>}
        </div>
      )}
      {!events.length && !busy && <div className="sub" style={{ fontSize: 12 }}>Examples: “search my documents for how long the warranty lasts”, “look up the current price of lithium on the web and work out the cost of 300 kg”, “list the files and summarize the most recent one”.</div>}
    </div>
  );
}

/** Dropdown with the MCP servers registered under Connections. */
function McpServerSelect({ label, value, onChange }: { label: string; value: any; onChange: (v: any) => void }) {
  const [list] = useData<{ id: number; name: string; transport: string; enabled: boolean }[]>("/api/connections/mcp-servers");
  return (
    <Field label={label} hint={list?.length ? "They're registered under Connections" : "None yet: add them under Connections"}>
      <select value={value ?? ""} onChange={(e) => onChange(e.target.value ? +e.target.value : null)}>
        <option value="">— choose —</option>
        {list?.filter((m) => m.enabled).map((m) => <option key={m.id} value={m.id}>{m.name} ({m.transport})</option>)}
      </select>
    </Field>
  );
}

const EXAMPLE_CASES = `What is the return period? | rag_search | 30
What is 1234 times 56? | calculator | 69104
Hi, how are you? | none`;

/** Does it pick the right tool? Test cases with the expected tool and keywords. */
function EvalPanel({ agent, dirty }: { agent: Agent; dirty: boolean }) {
  const [txt, setTxt] = useState(EXAMPLE_CASES);
  const [runs, reload] = useData<any[]>(`/api/agents/${agent.id}/evals`);
  const [expanded, setExpanded] = useState(false);
  const { track, notify } = useToast();

  const cases = txt.split("\n").map((l) => l.trim()).filter(Boolean).map((l) => {
    const [question, tool = "", kw = ""] = l.split("|").map((x) => x.trim());
    return { question, tool, keywords: kw.split(",").map((x) => x.trim()).filter(Boolean) };
  });
  const latest = runs?.[0];

  const launch = async () => {
    if (dirty) return notify("Save the agent before evaluating it", "err");
    try {
      const r = await api.post<any>(`/api/agents/${agent.id}/eval`, { name: "", cases: cases });
      track(r.task, () => reload());
    } catch (e: any) { notify(e.message, "err"); }
  };

  if (!expanded) {
    return (
      <div className="card" style={{ padding: "8px 12px" }}>
        <button className="ghost sm" onClick={() => setExpanded(true)}>
          Evaluate the agent · does it pick the right tool?
          {latest?.summary?.casos ? ` — latest: ${Math.round(latest.summary.herramienta_correcta * 100)}%` : ""}
        </button>
      </div>
    );
  }
  return (
    <div className="card stack">
      <div className="row between">
        <b>Evaluate the agent</b>
        <button className="sm ghost" onClick={() => setExpanded(false)}>Close</button>
      </div>
      <div className="sub">
        With small models what fails usually isn't the text but the <b>decision about which tool to use</b>.
        Write questions with the tool it should pick and measure how many it gets right. Changing the model, the
        instructions or removing tools and measuring again is how you improve an agent without flying blind.
      </div>
      <Field label={`Cases (${cases.length})`}
        hint={`One per line: question | expected tool | keywords. This agent's tools: ${agent.tools.map((t) => t.id).join(", ") || "none"}, or “none”.`}>
        <textarea rows={6} className="mono" value={txt} onChange={(e) => setTxt(e.target.value)} />
      </Field>
      <div className="row end">
        <button className="primary" disabled={!cases.length} onClick={launch}>Evaluate {cases.length} cases</button>
      </div>

      {latest?.results?.length > 0 && (<>
        <div className="row" style={{ gap: 18, flexWrap: "wrap" }}>
          <Metric title="Right tool" value={`${Math.round(latest.summary.herramienta_correcta * 100)}%`} />
          <Metric title="First try" value={`${Math.round(latest.summary.a_la_primera * 100)}%`} />
          {latest.summary.palabras_clave != null && <Metric title="Keywords" value={`${Math.round(latest.summary.palabras_clave * 100)}%`} />}
          <Metric title="Steps" value={latest.summary.pasos_media} />
          <Metric title="Per case" value={`${latest.summary.segundos_media}s`} />
        </div>
        {!!latest.summary.fallos_frecuentes?.length && (
          <div className="warn">
            <b>Recurring failures:</b>
            {latest.summary.fallos_frecuentes.map(([k, n]: [string, number]) => <div key={k}>{k} ×{n}</div>)}
          </div>
        )}
        <table>
          <thead><tr><th /><th>Question</th><th>Expected</th><th>Used</th><th>Steps</th></tr></thead>
          <tbody>
            {latest.results.map((f: any, i: number) => (
              <tr key={i} title={f.respuesta}>
                <td>{f.acierto ? "✓" : "✗"}</td>
                <td>{f.question}</td>
                <td className="mono sub">{f.esperada}</td>
                <td className="mono sub" style={{ color: f.acierto ? undefined : "var(--err)" }}>{f.llamadas.join(", ") || "none"}</td>
                <td className="sub">{f.pasos}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {runs && runs.length > 1 && (
          <div className="sub">History: {runs.slice(0, 6).map((r) => `${Math.round((r.summary?.herramienta_correcta ?? 0) * 100)}%`).join(" · ")}</div>
        )}
      </>)}
    </div>
  );
}

const Metric = ({ title, value }: { title: string; value: any }) => (
  <div><div className="sub" style={{ fontSize: 11 }}>{title}</div><div style={{ fontSize: 20, fontWeight: 600 }}>{value}</div></div>
);

/** Imports agents from a CrewAI agents.yaml. Roles carry over as they are; tools only when there's an equivalent. */
function ImportCrew({ onClose, onDone }: { onClose: () => void; onDone: () => void }) {
  const [txt, setTxt] = useState("");
  const [res, setRes] = useState<any>(null);
  const [models] = useData<Model[]>("/api/models");
  const [model, setModel] = useState<number | "">("");
  const { notify } = useToast();
  const go = async () => {
    try { setRes(await api.post("/api/agents/import/crewai", { yaml: txt, model_id: model || null })); }
    catch (e: any) { notify(e.message, "err"); }
  };
  return (
    <Modal title="Import agents from CrewAI" onClose={res ? onDone : onClose} wide>
      {!res ? (<>
        <p className="sub">
          Paste the contents of a CrewAI <code>agents.yaml</code>. The role, goal and backstory become
          the agent's instructions. Its tools are CrewAI code and can't be brought over: the ones with an
          equivalent here are translated (web search, reading and writing files, Python, SQL, documents) and you're warned about the rest.
        </p>
        <Field label="agents.yaml"><textarea rows={12} className="mono" value={txt} onChange={(e) => setTxt(e.target.value)}
          placeholder={"researcher:\n  role: Senior researcher\n  goal: Find the latest on {topic}\n  backstory: You've spent years…\n  tools:\n    - SerperDevTool"} /></Field>
        <Field label="Model for the imported agents">
          <select value={model} onChange={(e) => setModel(e.target.value ? +e.target.value : "")}>
            <option value="">whichever is loaded</option>
            {models?.filter((m) => m.kind !== "embedding").map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
          </select>
        </Field>
        <div className="row end"><button onClick={onClose}>Cancel</button><button className="primary" disabled={!txt.trim()} onClick={go}>Import</button></div>
      </>) : (<>
        <p><b>{res.creados.length} agent(s) created:</b></p>
        {res.creados.map((a: any) => <div key={a.id}>· {a.name} <span className="sub">{a.tools.join(", ") || "no tools"}</span></div>)}
        {!!res.avisos.length && <div className="warn" style={{ marginTop: 10 }}>{res.avisos.map((w: string, i: number) => <div key={i}>⚠ {w}</div>)}</div>}
        <div className="row end" style={{ marginTop: 10 }}><button className="primary" onClick={onDone}>Done</button></div>
      </>)}
    </Modal>
  );
}

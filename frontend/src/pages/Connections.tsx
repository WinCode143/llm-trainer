import { useState } from "react";
import { ArrowDownToLine, ArrowUpFromLine, Plug, Plus, Trash2, Zap } from "lucide-react";
import { api } from "../api";
import { Providers } from "../components/Helper";
import { Triggers } from "../components/Triggers";
import { Check, Field, Help, Spinner, useData, useDialog, useToast } from "../components/ui";

type Mcp = {
  id: number; name: string; transport: "stdio" | "http"; url: string; command: string;
  args: string[]; env: Record<string, string>; enabled: boolean; note: string;
};
type Overview = { mcp_servidor: { ruta: string; herramientas: string[] }; mcp_clientes: number; proveedores: number; api_openai: string };

const EMPTY: { name: string; transport: "stdio" | "http"; url: string; command: string; args: string; env: string; enabled: boolean; note: string } = { name: "", transport: "stdio", url: "", command: "", args: "", env: "", enabled: true, note: "" };

export default function Connections() {
  const [ov] = useData<Overview>("/api/connections/overview");
  const base = `${location.protocol}//${location.host}`;

  return (
    <>
      <div className="page-head">
        <div><h1>Connections</h1><div className="sub">Everything that talks to the outside world: what you lend to other programs and what you use from outside</div></div>
      </div>

      <Help title="What is MCP, and why does it show up in two places?">
        <p><b>MCP</b> (Model Context Protocol) is the standard way for an AI program to use another program's tools. Here it plays <b>two opposite roles</b>:</p>
        <ul>
          <li><b>This app as a server</b> — it lends your collections, datasets and flows to others: Claude Code, Cursor, Claude Desktop. They ask, your app answers.</li>
          <li><b>This app as a client</b> — your agents use tools from external MCP servers: the file system, GitHub, a database, a browser… Register them here once and any agent can use them.</li>
        </ul>
        <p style={{ marginTop: 8 }}>There's no universal format to “import an agent” from another project, because its tools are code tied to each framework. MCP is the closest thing: it's the standard way to share <b>tools</b>, and most well-known projects publish theirs that way.</p>
      </Help>

      <div className="card">
        <div className="row between">
          <h2 style={{ margin: 0 }}><ArrowUpFromLine size={16} /> What you lend · MCP server</h2>
          <span className="badge accent">{ov?.mcp_servidor.herramientas.length ?? "…"} tools</span>
        </div>
        <p className="sub" style={{ marginTop: 6 }}>Any MCP-compatible program can use these tools from your app.</p>
        <div className="chips">{ov?.mcp_servidor.herramientas.map((h) => <code key={h} className="chip">{h}</code>)}</div>
        <div className="grid g2" style={{ marginTop: 10 }}>
          <div><div className="sub">Claude Code</div><pre className="mono snippet">{`claude mcp add --transport http llm-trainer ${base}/mcp/`}</pre></div>
          <div><div className="sub">Any other client (JSON)</div><pre className="mono snippet">{`{"mcpServers": {"llm-trainer": {"type": "http", "url": "${base}/mcp/"}}}`}</pre></div>
        </div>
        <p className="sub" style={{ marginTop: 8 }}>
          There's also an <b>OpenAI-compatible API</b> at <code>{base}/v1</code> for Open WebUI, LibreChat or any
          program that already knows how to talk to OpenAI.
        </p>
      </div>

      <Recipes />

      <Triggers />

      <McpClientes />

      <Providers />
    </>
  );
}

/** Ready-made agents that combine an external MCP server with this app. */
function Recipes() {
  const [list, reload] = useData<any[]>("/api/connections/recipes");
  const [models] = useData<{ id: number; name: string; kind: string }[]>("/api/models");
  const [model, setModel] = useState<number | "">("");
  const [busy, setBusy] = useState<string | null>(null);
  const { notify } = useToast();

  const create = async (r: any) => {
    setBusy(r.key);
    try {
      const a = await api.post<any>(`/api/connections/recipes/${r.key}`, { model_id: model || null });
      notify(`Agent “${a.name}” created. Try it in Agents.`);
      reload();
    } catch (e: any) { notify(e.message, "err"); } finally { setBusy(null); }
  };

  return (
    <div className="card">
      <div className="row between">
        <h2 style={{ margin: 0 }}><Zap size={16} /> Ready-made MCP agents</h2>
        <Field label="Model">
          <select value={model} onChange={(e) => setModel(e.target.value ? +e.target.value : "")}>
            <option value="">whichever is loaded</option>
            {models?.filter((m) => !["lora", "embedding", "reranker"].includes(m.kind))
              .map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
          </select>
        </Field>
      </div>
      <p className="sub" style={{ marginTop: 6 }}>
        A lone MCP server does nothing. Each recipe pairs an external one with this same app (so the agent
        can <b>save</b> what it finds: a dataset, a document) and gives it only the tools it needs,
        which is what improves accuracy the most with local models. It registers any missing servers.
      </p>
      <div className="stack" style={{ marginTop: 8, gap: 6 }}>
        {list?.map((r) => (
          <div key={r.key} className="option" style={{ cursor: "default", alignItems: "center" }}>
            <div style={{ flex: 1 }}>
              <b>{r.nombre}</b> {r.creado && <span className="badge ok">created</span>}
              <div className="sub">{r.da}</div>
              <div className="sub" style={{ fontSize: 11 }}>Try: “{r.ejemplo}” · uses {r.servidores.join(" + ")}</div>
            </div>
            {!r.creado && (
              <button className="sm primary" disabled={busy === r.key} onClick={() => create(r)}>
                {busy === r.key ? <Spinner /> : <Plus size={12} />} Create agent
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

/** External MCP servers, registered once and reusable by every agent. */
function McpClientes() {
  const [list, reload] = useData<Mcp[]>("/api/connections/mcp-servers");
  const [form, setForm] = useState<typeof EMPTY & { id?: number } | null>(null);
  const [probes, setProbes] = useState<Record<number, any>>({});
  const [busy, setBusy] = useState<number | "save" | null>(null);
  const { notify } = useToast();
  const dialog = useDialog();

  const save = async () => {
    if (!form) return;
    setBusy("save");
    const body = {
      ...form,
      // args and env are typed as text so nobody has to edit JSON by hand
      args: form.args.split(/\s+/).filter(Boolean),
      env: Object.fromEntries(form.env.split("\n").map((l) => l.trim()).filter((l) => l.includes("="))
        .map((l) => [l.slice(0, l.indexOf("=")).trim(), l.slice(l.indexOf("=") + 1).trim()])),
    };
    try {
      if (form.id) await api.put(`/api/connections/mcp-servers/${form.id}`, body);
      else await api.post("/api/connections/mcp-servers", body);
      setForm(null);
      reload();
      notify("MCP server saved");
    } catch (e: any) { notify(e.message, "err"); } finally { setBusy(null); }
  };

  const test = async (m: Mcp) => {
    setBusy(m.id);
    try {
      const r = await api.post<any>(`/api/connections/mcp-servers/${m.id}/test`);
      setProbes({ ...probes, [m.id]: r });
      notify(r.ok ? `${m.name}: ${r.tools.length} tools in ${r.seconds}s` : `${m.name}: ${r.error}`, r.ok ? "info" : "err");
    } finally { setBusy(null); }
  };

  const remove = async (m: Mcp) => {
    if (!(await dialog.confirm("Remove server", `Remove “${m.name}”? Agents that use it will lose its tools.`))) return;
    await api.del(`/api/connections/mcp-servers/${m.id}`);
    reload();
  };

  return (
    <div className="card">
      <div className="row between">
        <h2 style={{ margin: 0 }}><ArrowDownToLine size={16} /> What you use · external MCP servers</h2>
        <button className="primary sm" onClick={() => setForm({ ...EMPTY })}><Plus size={13} /> Add</button>
      </div>
      <p className="sub" style={{ marginTop: 6 }}>
        Register them here once and pick them from any agent with the “External MCP server” tool.
        <b>stdio</b> servers are a program the app launches (that's how almost all of them ship: <code>npx …</code>,
        <code> uvx …</code>); <b>HTTP</b> servers are already running at a URL.
      </p>
      <div className="warn" style={{ marginTop: 8 }}>
        A stdio server <b>runs a program on your machine</b> with your permissions. Only register the ones that come
        from a source you trust. Environment variables (tokens) are stored in plain text in the local database.
      </div>

      <Catalog onInstalled={reload} />

      {!list?.length ? <div className="empty" style={{ marginTop: 10 }}>None registered yet</div> : (
        <table style={{ marginTop: 10 }}>
          <thead><tr><th>Name</th><th>Type</th><th>How it's launched</th><th /></tr></thead>
          <tbody>
            {list.map((m) => (
              <tr key={m.id}>
                <td>
                  <b>{m.enabled ? m.name : <s>{m.name}</s>}</b>
                  {probes[m.id] && <div className="sub" style={{ fontSize: 12 }}>
                    {probes[m.id].ok ? `✓ ${probes[m.id].tools.map((t: any) => t.name).join(", ")}` : `✗ ${probes[m.id].error}`}
                  </div>}
                </td>
                <td><span className="badge">{m.transport}</span></td>
                <td className="mono sub ellipsis" style={{ maxWidth: 320 }}>
                  {m.transport === "http" ? m.url : `${m.command} ${m.args.join(" ")}`}
                </td>
                <td style={{ whiteSpace: "nowrap" }}>
                  <button className="sm" disabled={busy === m.id} onClick={() => test(m)}>
                    {busy === m.id ? <Spinner /> : <Zap size={12} />} Test
                  </button>
                  <button className="sm" onClick={() => setForm({ ...m, args: m.args.join(" "),
                    env: Object.entries(m.env).map(([k, v]) => `${k}=${v}`).join("\n") })}>Edit</button>
                  <button className="icon" onClick={() => remove(m)}><Trash2 size={13} /></button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {form && (
        <div className="card" style={{ marginTop: 12 }}>
          <div className="grid g3">
            <Field label="Name"><input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} /></Field>
            <Field label="Type">
              <select value={form.transport} onChange={(e) => setForm({ ...form, transport: e.target.value as any })}>
                <option value="stdio">stdio — a program the app launches</option>
                <option value="http">HTTP — already running at a URL</option>
              </select>
            </Field>
            {form.transport === "http"
              ? <Field label="URL"><input value={form.url} placeholder="http://localhost:3000/mcp/" onChange={(e) => setForm({ ...form, url: e.target.value })} /></Field>
              : <Field label="Command"><input value={form.command} placeholder="npx" onChange={(e) => setForm({ ...form, command: e.target.value })} /></Field>}
          </div>
          {form.transport === "stdio" && (
            <Field label="Arguments" hint="Space-separated, e.g. -y @modelcontextprotocol/server-filesystem /home/me/docs">
              <input className="mono" value={form.args} onChange={(e) => setForm({ ...form, args: e.target.value })} />
            </Field>
          )}
          {form.transport === "stdio"
            ? <Field label="Environment variables" hint="One per line, KEY=value. For tokens: GITHUB_TOKEN=ghp_…">
                <textarea rows={3} className="mono" value={form.env} onChange={(e) => setForm({ ...form, env: e.target.value })} />
              </Field>
            : <Field label="HTTP headers" hint="One per line, Name=value. For a token: Authorization=Bearer hf_…">
                <textarea rows={2} className="mono" value={form.env} onChange={(e) => setForm({ ...form, env: e.target.value })} />
              </Field>}
          <Check label="Enabled" checked={form.enabled} onChange={(enabled) => setForm({ ...form, enabled })} />
          <div className="row end" style={{ marginTop: 8 }}>
            <button onClick={() => setForm(null)}>Cancel</button>
            <button className="primary" disabled={!form.name.trim() || busy === "save"} onClick={save}>
              {busy === "save" ? <Spinner /> : <Plug size={13} />} Save
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

/** Well-known, verified MCP servers you can install with one click. */
function Catalog({ onInstalled }: { onInstalled: () => void }) {
  const [cat, reload] = useData<any[]>("/api/connections/catalog");
  const [expanded, setExpanded] = useState(false);
  const [values, setValues] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState<string | null>(null);
  const { notify } = useToast();

  const install = async (c: any) => {
    setBusy(c.key);
    try {
      const r = await api.post<any>("/api/connections/catalog", { key: c.key, valor: values[c.key] ?? "" });
      const t = await api.post<any>(`/api/connections/mcp-servers/${r.id}/test`);
      notify(t.ok ? `“${r.name}” ready: ${t.tools.length} tools` : `It was registered but doesn't respond: ${t.error}`, t.ok ? "info" : "err");
      reload(); onInstalled();
    } catch (e: any) { notify(e.message, "err"); } finally { setBusy(null); }
  };

  return (
    <div style={{ marginTop: 10 }}>
      <button className="ghost sm" onClick={() => setExpanded(!expanded)}>
        {expanded ? "Hide" : "Show"} the catalog of well-known servers ({cat?.length ?? "…"})
      </button>
      {expanded && (
        <div className="stack" style={{ marginTop: 8, gap: 6 }}>
          <div className="sub">
            Only actively maintained ones: checked on npm and PyPI, excluding those marked as abandoned.
          </div>
          {cat?.map((c) => (
            <div key={c.key} className="option" style={{ cursor: "default", alignItems: "center" }}>
              <div style={{ flex: 1 }}>
                <b>{c.nombre}</b> {c.instalado && <span className="badge ok">installed</span>}
                <div className="sub">{c.da}</div>
                <div className="sub" style={{ fontSize: 11 }}>⚠ {c.riesgo}</div>
                <code className="sub" style={{ fontSize: 11 }}>{c.transport === "http" ? c.url : `${c.command} ${c.args.join(" ")}`}</code>
                {c.param && !c.instalado && (
                  <input style={{ marginTop: 4 }} placeholder={c.param.default ?? c.param.placeholder} value={values[c.key] ?? ""}
                    onChange={(e) => setValues({ ...values, [c.key]: e.target.value })} title={c.param.label} />
                )}
              </div>
              {!c.instalado && (
                <button className="sm primary" disabled={busy === c.key || (c.param && !c.param.default && !values[c.key])} onClick={() => install(c)}>
                  {busy === c.key ? <Spinner /> : <Plus size={12} />} Add
                </button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

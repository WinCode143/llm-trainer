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

const VACIO: { name: string; transport: "stdio" | "http"; url: string; command: string; args: string; env: string; enabled: boolean; note: string } = { name: "", transport: "stdio", url: "", command: "", args: "", env: "", enabled: true, note: "" };

export default function Connections() {
  const [ov] = useData<Overview>("/api/connections/overview");
  const base = `${location.protocol}//${location.host}`;

  return (
    <>
      <div className="page-head">
        <div><h1>Conexiones</h1><div className="sub">Todo lo que habla con el exterior: lo que prestas a otros programas y lo que usas de fuera</div></div>
      </div>

      <Help title="¿Qué es MCP y por qué aparece en dos sitios?">
        <p><b>MCP</b> (Model Context Protocol) es el estándar para que un programa con IA use herramientas de otro. Aquí cumple <b>dos papeles opuestos</b>:</p>
        <ul>
          <li><b>Esta app como servidor</b> — le presta tus colecciones, datasets y flujos a otros: Claude Code, Cursor, Claude Desktop. Ellos preguntan, tu app responde.</li>
          <li><b>Esta app como cliente</b> — tus agentes usan herramientas de servidores MCP externos: el sistema de archivos, GitHub, una base de datos, un navegador… Se registran aquí una vez y cualquier agente los puede usar.</li>
        </ul>
        <p style={{ marginTop: 8 }}>No existe un formato universal para «importar un agente» de otro proyecto, porque sus herramientas son código atado a cada framework. MCP es lo más parecido: es la forma estándar de compartir <b>herramientas</b>, y la mayoría de proyectos conocidos las publican así.</p>
      </Help>

      <div className="card">
        <div className="row between">
          <h2 style={{ margin: 0 }}><ArrowUpFromLine size={16} /> Lo que prestas · servidor MCP</h2>
          <span className="badge accent">{ov?.mcp_servidor.herramientas.length ?? "…"} herramientas</span>
        </div>
        <p className="sub" style={{ marginTop: 6 }}>Cualquier programa compatible con MCP puede usar estas herramientas de tu app.</p>
        <div className="chips">{ov?.mcp_servidor.herramientas.map((h) => <code key={h} className="chip">{h}</code>)}</div>
        <div className="grid g2" style={{ marginTop: 10 }}>
          <div><div className="sub">Claude Code</div><pre className="mono snippet">{`claude mcp add --transport http llm-trainer ${base}/mcp/`}</pre></div>
          <div><div className="sub">Cualquier otro cliente (JSON)</div><pre className="mono snippet">{`{"mcpServers": {"llm-trainer": {"type": "http", "url": "${base}/mcp/"}}}`}</pre></div>
        </div>
        <p className="sub" style={{ marginTop: 8 }}>
          También hay una <b>API compatible con OpenAI</b> en <code>{base}/v1</code> para Open WebUI, LibreChat o cualquier
          programa que ya sepa hablar con OpenAI.
        </p>
      </div>

      <Recetas />

      <Triggers />

      <McpClientes />

      <Providers />
    </>
  );
}

/** Agentes ya armados que combinan un servidor MCP externo con esta app. */
function Recetas() {
  const [lista, reload] = useData<any[]>("/api/connections/recetas");
  const [modelos] = useData<{ id: number; name: string; kind: string }[]>("/api/models");
  const [modelo, setModelo] = useState<number | "">("");
  const [busy, setBusy] = useState<string | null>(null);
  const { notify } = useToast();

  const crear = async (r: any) => {
    setBusy(r.key);
    try {
      const a = await api.post<any>(`/api/connections/recetas/${r.key}`, { model_id: modelo || null });
      notify(`Agente «${a.name}» creado. Pruébalo en Agentes.`);
      reload();
    } catch (e: any) { notify(e.message, "err"); } finally { setBusy(null); }
  };

  return (
    <div className="card">
      <div className="row between">
        <h2 style={{ margin: 0 }}><Zap size={16} /> Agentes listos con MCP</h2>
        <Field label="Modelo">
          <select value={modelo} onChange={(e) => setModelo(e.target.value ? +e.target.value : "")}>
            <option value="">el que esté cargado</option>
            {modelos?.filter((m) => !["lora", "embedding", "reranker"].includes(m.kind))
              .map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
          </select>
        </Field>
      </div>
      <p className="sub" style={{ marginTop: 6 }}>
        Un servidor MCP suelto no hace nada. Cada receta junta uno externo con esta misma app (para que el agente
        pueda <b>guardar</b> lo que encuentra: un dataset, un documento) y le da sólo las herramientas que necesita,
        que con modelos locales es lo que más mejora el acierto. Registra los servidores que falten.
      </p>
      <div className="stack" style={{ marginTop: 8, gap: 6 }}>
        {lista?.map((r) => (
          <div key={r.key} className="option" style={{ cursor: "default", alignItems: "center" }}>
            <div style={{ flex: 1 }}>
              <b>{r.nombre}</b> {r.creado && <span className="badge ok">creado</span>}
              <div className="sub">{r.da}</div>
              <div className="sub" style={{ fontSize: 11 }}>Prueba: «{r.ejemplo}» · usa {r.servidores.join(" + ")}</div>
            </div>
            {!r.creado && (
              <button className="sm primary" disabled={busy === r.key} onClick={() => crear(r)}>
                {busy === r.key ? <Spinner /> : <Plus size={12} />} Crear agente
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

/** Servidores MCP externos, registrados una vez y reutilizables por todos los agentes. */
function McpClientes() {
  const [lista, reload] = useData<Mcp[]>("/api/connections/mcp-servers");
  const [form, setForm] = useState<typeof VACIO & { id?: number } | null>(null);
  const [pruebas, setPruebas] = useState<Record<number, any>>({});
  const [busy, setBusy] = useState<number | "save" | null>(null);
  const { notify } = useToast();
  const dialog = useDialog();

  const guardar = async () => {
    if (!form) return;
    setBusy("save");
    const cuerpo = {
      ...form,
      // args y env se escriben como texto para no obligar a editar JSON a mano
      args: form.args.split(/\s+/).filter(Boolean),
      env: Object.fromEntries(form.env.split("\n").map((l) => l.trim()).filter((l) => l.includes("="))
        .map((l) => [l.slice(0, l.indexOf("=")).trim(), l.slice(l.indexOf("=") + 1).trim()])),
    };
    try {
      if (form.id) await api.put(`/api/connections/mcp-servers/${form.id}`, cuerpo);
      else await api.post("/api/connections/mcp-servers", cuerpo);
      setForm(null);
      reload();
      notify("Servidor MCP guardado");
    } catch (e: any) { notify(e.message, "err"); } finally { setBusy(null); }
  };

  const probar = async (m: Mcp) => {
    setBusy(m.id);
    try {
      const r = await api.post<any>(`/api/connections/mcp-servers/${m.id}/test`);
      setPruebas({ ...pruebas, [m.id]: r });
      notify(r.ok ? `${m.name}: ${r.tools.length} herramientas en ${r.seconds}s` : `${m.name}: ${r.error}`, r.ok ? "info" : "err");
    } finally { setBusy(null); }
  };

  const borrar = async (m: Mcp) => {
    if (!(await dialog.confirm("Quitar servidor", `¿Quitar «${m.name}»? Los agentes que lo usen se quedarán sin sus herramientas.`))) return;
    await api.del(`/api/connections/mcp-servers/${m.id}`);
    reload();
  };

  return (
    <div className="card">
      <div className="row between">
        <h2 style={{ margin: 0 }}><ArrowDownToLine size={16} /> Lo que usas · servidores MCP externos</h2>
        <button className="primary sm" onClick={() => setForm({ ...VACIO })}><Plus size={13} /> Añadir</button>
      </div>
      <p className="sub" style={{ marginTop: 6 }}>
        Regístralos aquí una vez y elígelos desde cualquier agente con la herramienta «Servidor MCP externo».
        Los de tipo <b>stdio</b> son un programa que la app lanza (así se distribuyen casi todos: <code>npx …</code>,
        <code> uvx …</code>); los de tipo <b>HTTP</b> ya están corriendo en una URL.
      </p>
      <div className="warn" style={{ marginTop: 8 }}>
        Un servidor stdio <b>ejecuta un programa en tu máquina</b> con tus permisos. Registra sólo los que vengan
        de una fuente en la que confíes. Las variables de entorno (tokens) se guardan en claro en la base de datos local.
      </div>

      <Catalogo onInstalado={reload} />

      {!lista?.length ? <div className="empty" style={{ marginTop: 10 }}>Ninguno registrado todavía</div> : (
        <table style={{ marginTop: 10 }}>
          <thead><tr><th>Nombre</th><th>Tipo</th><th>Cómo se lanza</th><th /></tr></thead>
          <tbody>
            {lista.map((m) => (
              <tr key={m.id}>
                <td>
                  <b>{m.enabled ? m.name : <s>{m.name}</s>}</b>
                  {pruebas[m.id] && <div className="sub" style={{ fontSize: 12 }}>
                    {pruebas[m.id].ok ? `✓ ${pruebas[m.id].tools.map((t: any) => t.name).join(", ")}` : `✗ ${pruebas[m.id].error}`}
                  </div>}
                </td>
                <td><span className="badge">{m.transport}</span></td>
                <td className="mono sub ellipsis" style={{ maxWidth: 320 }}>
                  {m.transport === "http" ? m.url : `${m.command} ${m.args.join(" ")}`}
                </td>
                <td style={{ whiteSpace: "nowrap" }}>
                  <button className="sm" disabled={busy === m.id} onClick={() => probar(m)}>
                    {busy === m.id ? <Spinner /> : <Zap size={12} />} Probar
                  </button>
                  <button className="sm" onClick={() => setForm({ ...m, args: m.args.join(" "),
                    env: Object.entries(m.env).map(([k, v]) => `${k}=${v}`).join("\n") })}>Editar</button>
                  <button className="icon" onClick={() => borrar(m)}><Trash2 size={13} /></button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {form && (
        <div className="card" style={{ marginTop: 12 }}>
          <div className="grid g3">
            <Field label="Nombre"><input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} /></Field>
            <Field label="Tipo">
              <select value={form.transport} onChange={(e) => setForm({ ...form, transport: e.target.value as any })}>
                <option value="stdio">stdio — un programa que lanza la app</option>
                <option value="http">HTTP — ya corriendo en una URL</option>
              </select>
            </Field>
            {form.transport === "http"
              ? <Field label="URL"><input value={form.url} placeholder="http://localhost:3000/mcp/" onChange={(e) => setForm({ ...form, url: e.target.value })} /></Field>
              : <Field label="Comando"><input value={form.command} placeholder="npx" onChange={(e) => setForm({ ...form, command: e.target.value })} /></Field>}
          </div>
          {form.transport === "stdio" && (
            <Field label="Argumentos" hint="Separados por espacios, p. ej. -y @modelcontextprotocol/server-filesystem /home/yo/docs">
              <input className="mono" value={form.args} onChange={(e) => setForm({ ...form, args: e.target.value })} />
            </Field>
          )}
          {form.transport === "stdio"
            ? <Field label="Variables de entorno" hint="Una por línea, CLAVE=valor. Para tokens: GITHUB_TOKEN=ghp_…">
                <textarea rows={3} className="mono" value={form.env} onChange={(e) => setForm({ ...form, env: e.target.value })} />
              </Field>
            : <Field label="Cabeceras HTTP" hint="Una por línea, Nombre=valor. Para un token: Authorization=Bearer hf_…">
                <textarea rows={2} className="mono" value={form.env} onChange={(e) => setForm({ ...form, env: e.target.value })} />
              </Field>}
          <Check label="Activo" checked={form.enabled} onChange={(enabled) => setForm({ ...form, enabled })} />
          <div className="row end" style={{ marginTop: 8 }}>
            <button onClick={() => setForm(null)}>Cancelar</button>
            <button className="primary" disabled={!form.name.trim() || busy === "save"} onClick={guardar}>
              {busy === "save" ? <Spinner /> : <Plug size={13} />} Guardar
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

/** Servidores MCP conocidos, verificados, instalables con un clic. */
function Catalogo({ onInstalado }: { onInstalado: () => void }) {
  const [cat, reload] = useData<any[]>("/api/connections/catalogo");
  const [abierto, setAbierto] = useState(false);
  const [valores, setValores] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState<string | null>(null);
  const { notify } = useToast();

  const instalar = async (c: any) => {
    setBusy(c.key);
    try {
      const r = await api.post<any>("/api/connections/catalogo", { key: c.key, valor: valores[c.key] ?? "" });
      const t = await api.post<any>(`/api/connections/mcp-servers/${r.id}/test`);
      notify(t.ok ? `«${r.name}» listo: ${t.tools.length} herramientas` : `Se registró pero no responde: ${t.error}`, t.ok ? "info" : "err");
      reload(); onInstalado();
    } catch (e: any) { notify(e.message, "err"); } finally { setBusy(null); }
  };

  return (
    <div style={{ marginTop: 10 }}>
      <button className="ghost sm" onClick={() => setAbierto(!abierto)}>
        {abierto ? "Ocultar" : "Ver"} el catálogo de servidores conocidos ({cat?.length ?? "…"})
      </button>
      {abierto && (
        <div className="stack" style={{ marginTop: 8, gap: 6 }}>
          <div className="sub">
            Sólo los que siguen mantenidos: comprobados en npm y PyPI, sin los que están marcados como abandonados.
          </div>
          {cat?.map((c) => (
            <div key={c.key} className="option" style={{ cursor: "default", alignItems: "center" }}>
              <div style={{ flex: 1 }}>
                <b>{c.nombre}</b> {c.instalado && <span className="badge ok">instalado</span>}
                <div className="sub">{c.da}</div>
                <div className="sub" style={{ fontSize: 11 }}>⚠ {c.riesgo}</div>
                <code className="sub" style={{ fontSize: 11 }}>{c.transport === "http" ? c.url : `${c.command} ${c.args.join(" ")}`}</code>
                {c.param && !c.instalado && (
                  <input style={{ marginTop: 4 }} placeholder={c.param.default ?? c.param.placeholder} value={valores[c.key] ?? ""}
                    onChange={(e) => setValores({ ...valores, [c.key]: e.target.value })} title={c.param.label} />
                )}
              </div>
              {!c.instalado && (
                <button className="sm primary" disabled={busy === c.key || (c.param && !c.param.default && !valores[c.key])} onClick={() => instalar(c)}>
                  {busy === c.key ? <Spinner /> : <Plus size={12} />} Añadir
                </button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

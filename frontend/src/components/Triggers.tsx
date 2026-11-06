import { useState } from "react";
import { Bell, Clock, Link2, Plus, RefreshCw, Send, Trash2, Zap } from "lucide-react";
import { api } from "../api";
import { Check, Field, Spinner, useData, useDialog, useToast } from "./ui";

type Trigger = {
  id: number; name: string; kind: "webhook" | "telegram" | "schedule"; target_kind: "flow" | "agent"; target_id: number;
  config: Record<string, any>; enabled: boolean;
  last_runs: { cuando: string; origen: string; entrada: string; estado: string; salida: string; segundos: number }[];
};

const TIPOS = {
  webhook: { label: "Webhook", icon: Link2, desc: "Una URL secreta: cualquier programa que haga un POST lanza el flujo." },
  telegram: { label: "Telegram", icon: Send, desc: "Un bot: le escribes y responde con el flujo o el agente. No abre puertos en tu PC." },
  schedule: { label: "Programado", icon: Clock, desc: "Se lanza solo, cada día a una hora o cada N minutos." },
} as const;

const VACIO = { name: "", kind: "telegram" as Trigger["kind"], target_kind: "flow" as Trigger["target_kind"], target_id: 0,
  bot_token: "", chats: "", cada_min: 0, diario: "", input: "", avisar_a: 0, enabled: true };

/** Disparadores: lo que conecta un flujo o un agente con el mundo, en vez de correrlo sólo desde la app. */
export function Triggers() {
  const [lista, reload] = useData<Trigger[]>("/api/triggers", 8000);
  const [flows] = useData<{ id: number; name: string }[]>("/api/flows");
  const [agents] = useData<{ id: number; name: string }[]>("/api/agents");
  const [form, setForm] = useState<typeof VACIO & { id?: number } | null>(null);
  const [abierto, setAbierto] = useState<number | null>(null);
  const [busy, setBusy] = useState<number | "save" | null>(null);
  const { notify } = useToast();
  const dialog = useDialog();
  const base = `${location.protocol}//${location.host}`;

  const destino = (t: { target_kind: string; target_id: number }) =>
    (t.target_kind === "agent" ? agents : flows)?.find((x) => x.id === t.target_id)?.name ?? `#${t.target_id}`;
  const bots = (lista ?? []).filter((t) => t.kind === "telegram");

  const guardar = async () => {
    if (!form) return;
    setBusy("save");
    const config: Record<string, any> =
      form.kind === "telegram" ? { bot_token: form.bot_token, allowed_chat_ids: form.chats.split(/[\s,]+/).filter(Boolean).map(Number) }
      : form.kind === "schedule" ? { ...(form.diario ? { diario: form.diario } : { cada_min: form.cada_min }), input: form.input,
                                     ...(form.avisar_a ? { avisar_a: form.avisar_a } : {}) }
      : {};
    try {
      const cuerpo = { name: form.name, kind: form.kind, target_kind: form.target_kind, target_id: form.target_id, config, enabled: form.enabled };
      if (form.id) await api.put(`/api/triggers/${form.id}`, cuerpo);
      else await api.post("/api/triggers", cuerpo);
      setForm(null);
      reload();
      notify("Disparador guardado");
    } catch (e: any) { notify(e.message, "err"); } finally { setBusy(null); }
  };

  const probar = async (t: Trigger) => {
    setBusy(t.id);
    try {
      const r = await api.post<any>(`/api/triggers/${t.id}/probar`);
      if (t.kind === "telegram") notify(r.ok ? `Bot ${r.bot} conectado · ${r.chats_permitidos.length} chat(s) permitido(s)` : r.error, r.ok ? "info" : "err");
      else notify(r.estado === "ok" ? `Respondió en ${r.segundos}s` : `${r.estado}: ${String(r.salida).slice(0, 120)}`, r.estado === "ok" ? "info" : "err");
      reload();
    } finally { setBusy(null); }
  };

  const borrar = async (t: Trigger) => {
    if (!(await dialog.confirm("Borrar disparador", `¿Borrar «${t.name}»?${t.kind === "webhook" ? " Su URL dejará de funcionar." : ""}`))) return;
    await api.del(`/api/triggers/${t.id}`);
    reload();
  };

  const editar = (t: Trigger) => setForm({ ...VACIO, id: t.id, name: t.name, kind: t.kind, target_kind: t.target_kind,
    target_id: t.target_id, enabled: t.enabled, bot_token: t.config.bot_token ?? "",
    chats: (t.config.allowed_chat_ids ?? []).join(", "), cada_min: t.config.cada_min ?? 0, diario: t.config.diario ?? "",
    input: t.config.input ?? "", avisar_a: t.config.avisar_a ?? 0 });

  return (
    <div className="card">
      <div className="row between">
        <h2 style={{ margin: 0 }}><Bell size={16} /> Disparadores</h2>
        <button className="primary sm" onClick={() => setForm({ ...VACIO })}><Plus size={13} /> Añadir</button>
      </div>
      <p className="sub" style={{ marginTop: 6 }}>
        Un flujo que sólo se puede lanzar desde esta pantalla es un prototipo. Un disparador lo conecta con el mundo:
        que responda a un mensaje, a una llamada de otro programa o a una hora.
      </p>
      <div className="warn" style={{ marginTop: 8 }}>
        La app no tiene contraseña: es local por diseño. Cada disparador trae su propio cerrojo — el webhook, un token
        que hace de contraseña; el bot de Telegram, una lista de chats permitidos. <b>Un bot sin lista no ejecuta nada</b>:
        sólo le dice a quien le escribe cuál es su chat_id, para que puedas añadirlo.
      </div>

      {!lista?.length ? <div className="empty" style={{ marginTop: 10 }}>Ninguno todavía</div> : (
        <table style={{ marginTop: 10 }}>
          <thead><tr><th>Nombre</th><th>Tipo</th><th>Lanza</th><th>Última vez</th><th /></tr></thead>
          <tbody>
            {lista.map((t) => {
              const K = TIPOS[t.kind];
              const ult = t.last_runs?.[0];
              return (
                <tr key={t.id}>
                  <td>
                    <button className="param-nombre" onClick={() => setAbierto(abierto === t.id ? null : t.id)}>
                      <b>{t.enabled ? t.name : <s>{t.name}</s>}</b>
                    </button>
                    {abierto === t.id && <Detalle t={t} base={base} onRegenerado={reload} />}
                  </td>
                  <td className="sub"><K.icon size={13} /> {K.label}</td>
                  <td>{t.target_kind === "agent" ? "agente" : "flujo"} <b>{destino(t)}</b></td>
                  <td className="sub" style={{ fontSize: 12 }}>
                    {ult ? <>{new Date(ult.cuando).toLocaleString()} · <span style={{ color: ult.estado === "ok" ? "var(--ok)" : ult.estado === "error" ? "var(--err)" : undefined }}>{ult.estado}</span></> : "nunca"}
                  </td>
                  <td style={{ whiteSpace: "nowrap" }}>
                    <button className="sm" disabled={busy === t.id} onClick={() => probar(t)}>
                      {busy === t.id ? <Spinner /> : <Zap size={12} />} Probar
                    </button>
                    <button className="sm" onClick={() => editar(t)}>Editar</button>
                    <button className="icon" onClick={() => borrar(t)}><Trash2 size={13} /></button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}

      {form && (
        <div className="card" style={{ marginTop: 12 }}>
          {!form.id && (
            <Field label="Tipo">
              <div className="stack" style={{ gap: 4 }}>
                {(Object.keys(TIPOS) as Trigger["kind"][]).map((k) => (
                  <label key={k} className={`option ${form.kind === k ? "selected" : ""}`} onClick={() => setForm({ ...form, kind: k })}>
                    <input type="radio" checked={form.kind === k} readOnly />
                    <div><b>{TIPOS[k].label}</b><div className="sub">{TIPOS[k].desc}</div></div>
                  </label>
                ))}
              </div>
            </Field>
          )}
          <div className="grid g3">
            <Field label="Nombre"><input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} /></Field>
            <Field label="Lanza un…">
              <select value={form.target_kind} onChange={(e) => setForm({ ...form, target_kind: e.target.value as any, target_id: 0 })}>
                <option value="flow">flujo</option><option value="agent">agente</option>
              </select>
            </Field>
            <Field label={form.target_kind === "agent" ? "Agente" : "Flujo"}>
              <select value={form.target_id || ""} onChange={(e) => setForm({ ...form, target_id: +e.target.value })}>
                <option value="">— elige —</option>
                {(form.target_kind === "agent" ? agents : flows)?.map((x) => <option key={x.id} value={x.id}>{x.name}</option>)}
              </select>
            </Field>
          </div>

          {form.kind === "telegram" && (<>
            <div className="param-info" style={{ marginLeft: 0 }}>
              <p><b>Cómo conseguir el token:</b> en Telegram, habla con <code>@BotFather</code>, escribe <code>/newbot</code>,
              elige un nombre y te da un token del estilo <code>123456:ABC-…</code>.</p>
              <p><b>Cómo saber tu chat_id:</b> guarda el bot sin chats permitidos y escríbele cualquier cosa: te contestará con tu número.</p>
            </div>
            <div className="grid g2">
              <Field label="Token del bot"><input type="password" value={form.bot_token} placeholder={form.id ? "vacío = conservar el actual" : ""}
                onChange={(e) => setForm({ ...form, bot_token: e.target.value })} /></Field>
              <Field label="Chats permitidos" hint="chat_id separados por comas. Vacío = el bot no ejecuta nada">
                <input value={form.chats} onChange={(e) => setForm({ ...form, chats: e.target.value })} placeholder="123456789" />
              </Field>
            </div>
          </>)}

          {form.kind === "schedule" && (<>
            <div className="grid g3">
              <Field label="Cada día a las" hint="HH:MM. Tiene preferencia sobre «cada N minutos»">
                <input value={form.diario} placeholder="08:00" onChange={(e) => setForm({ ...form, diario: e.target.value })} />
              </Field>
              <Field label="…o cada N minutos"><input type="number" min={0} value={form.cada_min} onChange={(e) => setForm({ ...form, cada_min: +e.target.value })} /></Field>
              <Field label="Mandar el resultado a" hint="Un resultado que nadie ve no sirve">
                <select value={form.avisar_a || ""} onChange={(e) => setForm({ ...form, avisar_a: +e.target.value })}>
                  <option value="">— nadie (sólo historial) —</option>
                  {bots.map((b) => <option key={b.id} value={b.id}>bot «{b.name}»</option>)}
                </select>
              </Field>
            </div>
            <Field label="Entrada que recibe cada vez"><textarea rows={2} value={form.input} onChange={(e) => setForm({ ...form, input: e.target.value })}
              placeholder="Resume las novedades de hoy" /></Field>
          </>)}

          {form.kind === "webhook" && !form.id && (
            <div className="sub">Al guardar se genera una URL con un token secreto. Trátala como una contraseña.</div>
          )}
          <Check label="Activo" checked={form.enabled} onChange={(enabled) => setForm({ ...form, enabled })} />
          <div className="row end" style={{ marginTop: 8 }}>
            <button onClick={() => setForm(null)}>Cancelar</button>
            <button className="primary" disabled={!form.name.trim() || !form.target_id || busy === "save"} onClick={guardar}>
              {busy === "save" ? <Spinner /> : null} Guardar
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function Detalle({ t, base, onRegenerado }: { t: Trigger; base: string; onRegenerado: () => void }) {
  const { notify } = useToast();
  const url = `${base}/api/hooks/${t.config.token}`;
  return (
    <div className="param-info" style={{ marginLeft: 0, maxWidth: 560 }}>
      {t.kind === "webhook" && (<>
        <p><b>URL</b> (secreta):</p>
        <pre className="mono snippet">{url}</pre>
        <p><b>Ejemplo:</b></p>
        <pre className="mono snippet">{`curl -X POST ${url} \\\n  -H 'content-type: application/json' \\\n  -d '{"input": "tu pregunta"}'`}</pre>
        <p>Sólo es accesible desde este PC: la app escucha en 127.0.0.1. Para usarla desde fuera haría falta un túnel.</p>
        <button className="sm" onClick={async () => {
          await api.post(`/api/triggers/${t.id}/regenerar`);
          notify("Token nuevo: la URL anterior ya no funciona");
          onRegenerado();
        }}><RefreshCw size={12} /> Regenerar token</button>
      </>)}
      {t.kind === "telegram" && (
        <p>Chats permitidos: {(t.config.allowed_chat_ids ?? []).join(", ") || <b style={{ color: "var(--warn)" }}>ninguno — el bot no ejecuta nada</b>}</p>
      )}
      {t.kind === "schedule" && <p>{t.config.diario ? `Cada día a las ${t.config.diario}` : `Cada ${t.config.cada_min} min`} · entrada: «{t.config.input}»</p>}
      <p style={{ marginTop: 8 }}><b>Últimas ejecuciones</b></p>
      {!t.last_runs?.length ? <p>Ninguna todavía.</p> : t.last_runs.slice(0, 8).map((r, i) => (
        <p key={i}>{new Date(r.cuando).toLocaleString()} · {r.origen} · <b>{r.estado}</b> · {r.segundos}s<br />
          <span className="mono">«{r.entrada}» → {r.salida.slice(0, 160)}</span></p>
      ))}
    </div>
  );
}

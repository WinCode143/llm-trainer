import { useState } from "react";
import { Bell, Clock, Link2, Plus, RefreshCw, Send, Trash2, Zap } from "lucide-react";
import { api } from "../api";
import { Check, Field, Spinner, useData, useDialog, useToast } from "./ui";

type Trigger = {
  id: number; name: string; kind: "webhook" | "telegram" | "schedule"; target_kind: "flow" | "agent"; target_id: number;
  config: Record<string, any>; enabled: boolean;
  last_runs: { cuando: string; origen: string; entrada: string; estado: string; salida: string; segundos: number }[];
};

const KINDS = {
  webhook: { label: "Webhook", icon: Link2, desc: "A secret URL: any program that sends a POST runs the flow." },
  telegram: { label: "Telegram", icon: Send, desc: "A bot: you message it and it answers with the flow or agent. Opens no ports on your PC." },
  schedule: { label: "Scheduled", icon: Clock, desc: "Runs by itself, every day at a set time or every N minutes." },
} as const;

const EMPTY = { name: "", kind: "telegram" as Trigger["kind"], target_kind: "flow" as Trigger["target_kind"], target_id: 0,
  bot_token: "", chats: "", cada_min: 0, diario: "", input: "", avisar_a: 0, enabled: true };

/** Triggers: what connects a flow or an agent to the outside world, instead of running it only from the app. */
export function Triggers() {
  const [list, reload] = useData<Trigger[]>("/api/triggers", 8000);
  const [flows] = useData<{ id: number; name: string }[]>("/api/flows");
  const [agents] = useData<{ id: number; name: string }[]>("/api/agents");
  const [form, setForm] = useState<typeof EMPTY & { id?: number } | null>(null);
  const [expanded, setExpanded] = useState<number | null>(null);
  const [busy, setBusy] = useState<number | "save" | null>(null);
  const { notify } = useToast();
  const dialog = useDialog();
  const base = `${location.protocol}//${location.host}`;

  const target = (t: { target_kind: string; target_id: number }) =>
    (t.target_kind === "agent" ? agents : flows)?.find((x) => x.id === t.target_id)?.name ?? `#${t.target_id}`;
  const bots = (list ?? []).filter((t) => t.kind === "telegram");

  const save = async () => {
    if (!form) return;
    setBusy("save");
    const config: Record<string, any> =
      form.kind === "telegram" ? { bot_token: form.bot_token, allowed_chat_ids: form.chats.split(/[\s,]+/).filter(Boolean).map(Number) }
      : form.kind === "schedule" ? { ...(form.diario ? { diario: form.diario } : { cada_min: form.cada_min }), input: form.input,
                                     ...(form.avisar_a ? { avisar_a: form.avisar_a } : {}) }
      : {};
    try {
      const body = { name: form.name, kind: form.kind, target_kind: form.target_kind, target_id: form.target_id, config, enabled: form.enabled };
      if (form.id) await api.put(`/api/triggers/${form.id}`, body);
      else await api.post("/api/triggers", body);
      setForm(null);
      reload();
      notify("Trigger saved");
    } catch (e: any) { notify(e.message, "err"); } finally { setBusy(null); }
  };

  const test = async (t: Trigger) => {
    setBusy(t.id);
    try {
      const r = await api.post<any>(`/api/triggers/${t.id}/test`);
      if (t.kind === "telegram") notify(r.ok ? `Bot ${r.bot} connected · ${r.chats_permitidos.length} allowed chat(s)` : r.error, r.ok ? "info" : "err");
      else notify(r.estado === "ok" ? `Answered in ${r.segundos}s` : `${r.estado}: ${String(r.salida).slice(0, 120)}`, r.estado === "ok" ? "info" : "err");
      reload();
    } finally { setBusy(null); }
  };

  const remove = async (t: Trigger) => {
    if (!(await dialog.confirm("Delete trigger", `Delete “${t.name}”?${t.kind === "webhook" ? " Its URL will stop working." : ""}`))) return;
    await api.del(`/api/triggers/${t.id}`);
    reload();
  };

  const edit = (t: Trigger) => setForm({ ...EMPTY, id: t.id, name: t.name, kind: t.kind, target_kind: t.target_kind,
    target_id: t.target_id, enabled: t.enabled, bot_token: t.config.bot_token ?? "",
    chats: (t.config.allowed_chat_ids ?? []).join(", "), cada_min: t.config.cada_min ?? 0, diario: t.config.diario ?? "",
    input: t.config.input ?? "", avisar_a: t.config.avisar_a ?? 0 });

  return (
    <div className="card">
      <div className="row between">
        <h2 style={{ margin: 0 }}><Bell size={16} /> Triggers</h2>
        <button className="primary sm" onClick={() => setForm({ ...EMPTY })}><Plus size={13} /> Add</button>
      </div>
      <p className="sub" style={{ marginTop: 6 }}>
        A flow you can only run from this screen is a prototype. A trigger connects it to the outside world:
        it can answer a message, a call from another program, or a time of day.
      </p>
      <div className="warn" style={{ marginTop: 8 }}>
        The app has no password: it's local by design. Each trigger brings its own lock — the webhook, a token
        that works as a password; the Telegram bot, a list of allowed chats. <b>A bot without a list runs nothing</b>:
        it only tells whoever messages it what their chat_id is, so you can add it.
      </div>

      {!list?.length ? <div className="empty" style={{ marginTop: 10 }}>None yet</div> : (
        <table style={{ marginTop: 10 }}>
          <thead><tr><th>Name</th><th>Type</th><th>Runs</th><th>Last run</th><th /></tr></thead>
          <tbody>
            {list.map((t) => {
              const K = KINDS[t.kind];
              const last = t.last_runs?.[0];
              return (
                <tr key={t.id}>
                  <td>
                    <button className="param-name" onClick={() => setExpanded(expanded === t.id ? null : t.id)}>
                      <b>{t.enabled ? t.name : <s>{t.name}</s>}</b>
                    </button>
                    {expanded === t.id && <Details t={t} base={base} onRegenerated={reload} />}
                  </td>
                  <td className="sub"><K.icon size={13} /> {K.label}</td>
                  <td>{t.target_kind === "agent" ? "agent" : "flow"} <b>{target(t)}</b></td>
                  <td className="sub" style={{ fontSize: 12 }}>
                    {last ? <>{new Date(last.cuando).toLocaleString()} · <span style={{ color: last.estado === "ok" ? "var(--ok)" : last.estado === "error" ? "var(--err)" : undefined }}>{last.estado}</span></> : "never"}
                  </td>
                  <td style={{ whiteSpace: "nowrap" }}>
                    <button className="sm" disabled={busy === t.id} onClick={() => test(t)}>
                      {busy === t.id ? <Spinner /> : <Zap size={12} />} Test
                    </button>
                    <button className="sm" onClick={() => edit(t)}>Edit</button>
                    <button className="icon" onClick={() => remove(t)}><Trash2 size={13} /></button>
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
            <Field label="Type">
              <div className="stack" style={{ gap: 4 }}>
                {(Object.keys(KINDS) as Trigger["kind"][]).map((k) => (
                  <label key={k} className={`option ${form.kind === k ? "selected" : ""}`} onClick={() => setForm({ ...form, kind: k })}>
                    <input type="radio" checked={form.kind === k} readOnly />
                    <div><b>{KINDS[k].label}</b><div className="sub">{KINDS[k].desc}</div></div>
                  </label>
                ))}
              </div>
            </Field>
          )}
          <div className="grid g3">
            <Field label="Name"><input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} /></Field>
            <Field label="Runs a…">
              <select value={form.target_kind} onChange={(e) => setForm({ ...form, target_kind: e.target.value as any, target_id: 0 })}>
                <option value="flow">flow</option><option value="agent">agent</option>
              </select>
            </Field>
            <Field label={form.target_kind === "agent" ? "Agent" : "Flow"}>
              <select value={form.target_id || ""} onChange={(e) => setForm({ ...form, target_id: +e.target.value })}>
                <option value="">— choose —</option>
                {(form.target_kind === "agent" ? agents : flows)?.map((x) => <option key={x.id} value={x.id}>{x.name}</option>)}
              </select>
            </Field>
          </div>

          {form.kind === "telegram" && (<>
            <div className="param-info" style={{ marginLeft: 0 }}>
              <p><b>How to get the token:</b> in Telegram, talk to <code>@BotFather</code>, send <code>/newbot</code>,
              pick a name and it gives you a token like <code>123456:ABC-…</code>.</p>
              <p><b>How to find your chat_id:</b> save the bot with no allowed chats and send it anything: it will reply with your number.</p>
            </div>
            <div className="grid g2">
              <Field label="Bot token"><input type="password" value={form.bot_token} placeholder={form.id ? "empty = keep the current one" : ""}
                onChange={(e) => setForm({ ...form, bot_token: e.target.value })} /></Field>
              <Field label="Allowed chats" hint="Comma-separated chat_ids. Empty = the bot runs nothing">
                <input value={form.chats} onChange={(e) => setForm({ ...form, chats: e.target.value })} placeholder="123456789" />
              </Field>
            </div>
          </>)}

          {form.kind === "schedule" && (<>
            <div className="grid g3">
              <Field label="Every day at" hint="HH:MM. Takes precedence over “every N minutes”">
                <input value={form.diario} placeholder="08:00" onChange={(e) => setForm({ ...form, diario: e.target.value })} />
              </Field>
              <Field label="…or every N minutes"><input type="number" min={0} value={form.cada_min} onChange={(e) => setForm({ ...form, cada_min: +e.target.value })} /></Field>
              <Field label="Send the result to" hint="A result nobody sees is useless">
                <select value={form.avisar_a || ""} onChange={(e) => setForm({ ...form, avisar_a: +e.target.value })}>
                  <option value="">— nobody (history only) —</option>
                  {bots.map((b) => <option key={b.id} value={b.id}>bot “{b.name}”</option>)}
                </select>
              </Field>
            </div>
            <Field label="Input it receives each time"><textarea rows={2} value={form.input} onChange={(e) => setForm({ ...form, input: e.target.value })}
              placeholder="Summarize today's news" /></Field>
          </>)}

          {form.kind === "webhook" && !form.id && (
            <div className="sub">Saving generates a URL with a secret token. Treat it like a password.</div>
          )}
          <Check label="Enabled" checked={form.enabled} onChange={(enabled) => setForm({ ...form, enabled })} />
          <div className="row end" style={{ marginTop: 8 }}>
            <button onClick={() => setForm(null)}>Cancel</button>
            <button className="primary" disabled={!form.name.trim() || !form.target_id || busy === "save"} onClick={save}>
              {busy === "save" ? <Spinner /> : null} Save
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function Details({ t, base, onRegenerated }: { t: Trigger; base: string; onRegenerated: () => void }) {
  const { notify } = useToast();
  const url = `${base}/api/hooks/${t.config.token}`;
  return (
    <div className="param-info" style={{ marginLeft: 0, maxWidth: 560 }}>
      {t.kind === "webhook" && (<>
        <p><b>URL</b> (secret):</p>
        <pre className="mono snippet">{url}</pre>
        <p><b>Example:</b></p>
        <pre className="mono snippet">{`curl -X POST ${url} \\\n  -H 'content-type: application/json' \\\n  -d '{"input": "your question"}'`}</pre>
        <p>Only reachable from this PC: the app listens on 127.0.0.1. Using it from outside would need a tunnel.</p>
        <button className="sm" onClick={async () => {
          await api.post(`/api/triggers/${t.id}/regenerate-token`);
          notify("New token: the previous URL no longer works");
          onRegenerated();
        }}><RefreshCw size={12} /> Regenerate token</button>
      </>)}
      {t.kind === "telegram" && (
        <p>Allowed chats: {(t.config.allowed_chat_ids ?? []).join(", ") || <b style={{ color: "var(--warn)" }}>none — the bot runs nothing</b>}</p>
      )}
      {t.kind === "schedule" && <p>{t.config.diario ? `Every day at ${t.config.diario}` : `Every ${t.config.cada_min} min`} · input: “{t.config.input}”</p>}
      <p style={{ marginTop: 8 }}><b>Recent runs</b></p>
      {!t.last_runs?.length ? <p>None yet.</p> : t.last_runs.slice(0, 8).map((r, i) => (
        <p key={i}>{new Date(r.cuando).toLocaleString()} · {r.origen} · <b>{r.estado}</b> · {r.segundos}s<br />
          <span className="mono">“{r.entrada}” → {r.salida.slice(0, 160)}</span></p>
      ))}
    </div>
  );
}

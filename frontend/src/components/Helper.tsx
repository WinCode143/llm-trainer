import { useState } from "react";
import { Check as CheckIcon, Plus, Server, Trash2, X, Zap } from "lucide-react";
import { api } from "../api";
import { Check, Field, Spinner, useData, useDialog, useToast } from "./ui";

export type Choice = { id: number | null; label: string; ready: boolean };
export type Provider = {
  id: number; name: string; kind: string; base_url: string; api_key: string; has_key: boolean;
  model: string; enabled: boolean; note: string;
};
type Preset = { key: string; label: string; kind: string; base_url: string; api_key: string; model: string; note: string };

/** “Helper model” picker, used across the screens that generate text.
  *  `null` = the model loaded in VRAM (the default, and free). */
export function HelperPicker({ value, onChange, label = "Helper model", hint }: {
  value: number | null; onChange: (v: number | null) => void; label?: string; hint?: string;
}) {
  const [choices] = useData<Choice[]>("/api/providers/choices", 15000);
  const solo = (choices?.length ?? 0) <= 1;
  return (
    <Field label={label} hint={hint ?? (solo ? "Add external models in System → Helper models" : "Hand this task to a stronger model")}>
      <select value={value ?? ""} onChange={(e) => onChange(e.target.value === "" ? null : +e.target.value)}>
        {choices?.map((c) => (
          <option key={c.id ?? "local"} value={c.id ?? ""} disabled={!c.ready}>
            {c.label}{c.ready ? "" : " — no model loaded"}
          </option>
        ))}
      </select>
    </Field>
  );
}

const EMPTY = { name: "", kind: "openai", base_url: "", api_key: "", model: "", enabled: true, note: "" };

/** Adding and managing external helper models (System screen). */
export function Providers() {
  const [list, reload] = useData<Provider[]>("/api/providers");
  const [meta] = useData<{ presets: Preset[]; kinds: Record<string, any> }>("/api/providers/presets");
  const [form, setForm] = useState<typeof EMPTY & { id?: number } | null>(null);
  const [results, setResults] = useState<Record<number, any>>({});
  const [busy, setBusy] = useState<number | "save" | null>(null);
  const { notify } = useToast();
  const dialog = useDialog();

  const save = async () => {
    if (!form) return;
    setBusy("save");
    try {
      if (form.id) await api.put(`/api/providers/${form.id}`, form);
      else await api.post("/api/providers", form);
      setForm(null);
      reload();
      notify("Helper model saved");
    } catch (e: any) { notify(e.message, "err"); } finally { setBusy(null); }
  };

  const probe = async (p: Provider) => {
    setBusy(p.id);
    try {
      const r = await api.post<any>(`/api/providers/${p.id}/test`);
      setResults({ ...results, [p.id]: r });
      notify(r.ok ? `${p.name} answered in ${r.seconds}s` : `${p.name}: ${r.error}`, r.ok ? "info" : "err");
    } finally { setBusy(null); }
  };

  const remove = async (p: Provider) => {
    if (!(await dialog.confirm("Delete helper", `Delete “${p.name}”? Whatever uses it will go back to the local model.`))) return;
    await api.del(`/api/providers/${p.id}`);
    reload();
  };

  return (
    <div className="card">
      <div className="row between">
        <h2 style={{ margin: 0 }}><Server size={16} /> Helper models</h2>
        <button className="primary sm" onClick={() => setForm({ ...EMPTY })}><Plus size={13} /> Add</button>
      </div>
      <div className="sub" style={{ marginTop: 6 }}>
        External models to hand off the tasks a small model is bad at: generating synthetic
        datasets, extracting the knowledge graph, judging evaluations or running a flow node.
        Any server compatible with the OpenAI API works (Ollama or LM Studio on your PC, vLLM on another machine,
        OpenRouter, Groq, DeepSeek…), and so does the Anthropic API. The model loaded in VRAM stays the default.
      </div>
      <div className="warn" style={{ marginTop: 8 }}>
        Keys are stored <b>in plain text</b> in the local database (<code>data/llmtrainer.db</code>). They never leave your
        machine except towards the provider, but anyone who can read that file can read them. They're included in backups.
      </div>

      {!list?.length && <div className="empty" style={{ fontSize: 13, marginTop: 10 }}>No helpers: everything uses the loaded model</div>}
      {!!list?.length && (
        <table style={{ marginTop: 10 }}>
          <thead><tr><th>Name</th><th>Model</th><th>Endpoint</th><th>Key</th><th /></tr></thead>
          <tbody>
            {list.map((p) => (
              <tr key={p.id}>
                <td>
                  <b>{p.enabled ? p.name : <s>{p.name}</s>}</b>
                  {results[p.id] && <div className="sub" style={{ fontSize: 12 }}>
                    {results[p.id].ok ? `✓ ${results[p.id].seconds}s · “${results[p.id].reply}”` : `✗ ${results[p.id].error}`}
                  </div>}
                </td>
                <td><code>{p.model || "—"}</code></td>
                <td className="sub ellipsis" style={{ maxWidth: 200 }}>{p.base_url}</td>
                <td className="sub">{p.has_key ? p.api_key : "—"}</td>
                <td style={{ whiteSpace: "nowrap" }}>
                  <button className="sm" disabled={busy === p.id} onClick={() => probe(p)}>
                    {busy === p.id ? <Spinner /> : <Zap size={12} />} Test
                  </button>
                  <button className="sm" onClick={() => setForm({ ...p, api_key: "" })}>Edit</button>
                  <button className="icon" onClick={() => remove(p)}><Trash2 size={13} /></button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {form && (
        <div className="card" style={{ marginTop: 12 }}>
          {!form.id && !!meta?.presets?.length && (
            <Field label="Start from a preset" hint="Fills in the URL and a typical model; you can change them afterwards">
              <div className="palette">
                {meta.presets.map((p) => (
                  <button key={p.key} title={p.note} onClick={() => setForm({
                    ...form, name: form.name || p.label, kind: p.kind, base_url: p.base_url,
                    api_key: p.api_key, model: p.model, note: p.note,
                  })}>{p.label}</button>
                ))}
              </div>
            </Field>
          )}
          <div className="grid g3">
            <Field label="Name"><input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} /></Field>
            <Field label="Type">
              <select value={form.kind} onChange={(e) => setForm({ ...form, kind: e.target.value })}>
                {Object.entries(meta?.kinds ?? {}).map(([k, v]: any) => <option key={k} value={k}>{v.label}</option>)}
              </select>
            </Field>
            <Field label="Model" hint="Exact name at that provider">
              <input value={form.model} onChange={(e) => setForm({ ...form, model: e.target.value })} />
            </Field>
          </div>
          <div className="grid g2">
            <Field label="Base URL" hint="Most servers need it to end in /v1">
              <input value={form.base_url} onChange={(e) => setForm({ ...form, base_url: e.target.value })} />
            </Field>
            <Field label="API key" hint={form.id ? "Empty = keep the current one" : "Local servers usually accept anything"}>
              <input type="password" value={form.api_key} onChange={(e) => setForm({ ...form, api_key: e.target.value })} />
            </Field>
          </div>
          <Check label="Enabled" checked={form.enabled} onChange={(enabled) => setForm({ ...form, enabled })} />
          <div className="row end" style={{ marginTop: 8 }}>
            <button onClick={() => setForm(null)}><X size={13} /> Cancel</button>
            <button className="primary" disabled={!form.name.trim() || !form.base_url.trim() || busy === "save"} onClick={save}>
              {busy === "save" ? <Spinner /> : <CheckIcon size={13} />} Save
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

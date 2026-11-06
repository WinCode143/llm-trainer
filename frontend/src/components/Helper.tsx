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

/** Desplegable de «modelo ayudante», repartido por las pantallas que generan texto.
 *  `null` = el modelo cargado en VRAM (opción por defecto y gratuita). */
export function HelperPicker({ value, onChange, label = "Modelo ayudante", hint }: {
  value: number | null; onChange: (v: number | null) => void; label?: string; hint?: string;
}) {
  const [choices] = useData<Choice[]>("/api/providers/choices", 15000);
  const solo = (choices?.length ?? 0) <= 1;
  return (
    <Field label={label} hint={hint ?? (solo ? "Añade modelos externos en Sistema → Modelos ayudantes" : "Delega esta tarea en un modelo más potente")}>
      <select value={value ?? ""} onChange={(e) => onChange(e.target.value === "" ? null : +e.target.value)}>
        {choices?.map((c) => (
          <option key={c.id ?? "local"} value={c.id ?? ""} disabled={!c.ready}>
            {c.label}{c.ready ? "" : " — sin modelo cargado"}
          </option>
        ))}
      </select>
    </Field>
  );
}

const EMPTY = { name: "", kind: "openai", base_url: "", api_key: "", model: "", enabled: true, note: "" };

/** Alta y gestión de modelos ayudantes externos (pantalla Sistema). */
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
      notify("Modelo ayudante guardado");
    } catch (e: any) { notify(e.message, "err"); } finally { setBusy(null); }
  };

  const probe = async (p: Provider) => {
    setBusy(p.id);
    try {
      const r = await api.post<any>(`/api/providers/${p.id}/test`);
      setResults({ ...results, [p.id]: r });
      notify(r.ok ? `${p.name} responde en ${r.seconds}s` : `${p.name}: ${r.error}`, r.ok ? "info" : "err");
    } finally { setBusy(null); }
  };

  const remove = async (p: Provider) => {
    if (!(await dialog.confirm("Borrar ayudante", `¿Borrar «${p.name}»? Los sitios que lo usen volverán al modelo local.`))) return;
    await api.del(`/api/providers/${p.id}`);
    reload();
  };

  return (
    <div className="card">
      <div className="row between">
        <h2 style={{ margin: 0 }}><Server size={16} /> Modelos ayudantes</h2>
        <button className="primary sm" onClick={() => setForm({ ...EMPTY })}><Plus size={13} /> Añadir</button>
      </div>
      <div className="sub" style={{ marginTop: 6 }}>
        Modelos externos a los que delegar las tareas donde un modelo pequeño se queda corto: generar datasets
        sintéticos, extraer el grafo de conocimiento, hacer de juez en las evaluaciones o resolver un nodo de un flujo.
        Vale cualquier servidor compatible con la API de OpenAI (Ollama o LM Studio en tu PC, un vLLM en otra máquina,
        OpenRouter, Groq, DeepSeek…) y la API de Anthropic. El modelo cargado en VRAM sigue siendo el de por defecto.
      </div>
      <div className="warn" style={{ marginTop: 8 }}>
        Las claves se guardan <b>en claro</b> en la base de datos local (<code>data/llmtrainer.db</code>). No salen de tu
        máquina salvo hacia el proveedor, pero quien tenga acceso al archivo puede leerlas. Entra en las copias de seguridad.
      </div>

      {!list?.length && <div className="empty" style={{ fontSize: 13, marginTop: 10 }}>Sin ayudantes: todo usa el modelo cargado</div>}
      {!!list?.length && (
        <table style={{ marginTop: 10 }}>
          <thead><tr><th>Nombre</th><th>Modelo</th><th>Endpoint</th><th>Clave</th><th /></tr></thead>
          <tbody>
            {list.map((p) => (
              <tr key={p.id}>
                <td>
                  <b>{p.enabled ? p.name : <s>{p.name}</s>}</b>
                  {results[p.id] && <div className="sub" style={{ fontSize: 12 }}>
                    {results[p.id].ok ? `✓ ${results[p.id].seconds}s · «${results[p.id].reply}»` : `✗ ${results[p.id].error}`}
                  </div>}
                </td>
                <td><code>{p.model || "—"}</code></td>
                <td className="sub ellipsis" style={{ maxWidth: 200 }}>{p.base_url}</td>
                <td className="sub">{p.has_key ? p.api_key : "—"}</td>
                <td style={{ whiteSpace: "nowrap" }}>
                  <button className="sm" disabled={busy === p.id} onClick={() => probe(p)}>
                    {busy === p.id ? <Spinner /> : <Zap size={12} />} Probar
                  </button>
                  <button className="sm" onClick={() => setForm({ ...p, api_key: "" })}>Editar</button>
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
            <Field label="Empezar desde un atajo" hint="Rellena la URL y el modelo típico; luego lo puedes cambiar">
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
            <Field label="Nombre"><input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} /></Field>
            <Field label="Tipo">
              <select value={form.kind} onChange={(e) => setForm({ ...form, kind: e.target.value })}>
                {Object.entries(meta?.kinds ?? {}).map(([k, v]: any) => <option key={k} value={k}>{v.label}</option>)}
              </select>
            </Field>
            <Field label="Modelo" hint="Nombre exacto en ese proveedor">
              <input value={form.model} onChange={(e) => setForm({ ...form, model: e.target.value })} />
            </Field>
          </div>
          <div className="grid g2">
            <Field label="URL base" hint="Debe terminar en /v1 en la mayoría de servidores">
              <input value={form.base_url} onChange={(e) => setForm({ ...form, base_url: e.target.value })} />
            </Field>
            <Field label="Clave de API" hint={form.id ? "Vacío = conservar la actual" : "Los servidores locales suelen aceptar cualquier cosa"}>
              <input type="password" value={form.api_key} onChange={(e) => setForm({ ...form, api_key: e.target.value })} />
            </Field>
          </div>
          <Check label="Activo" checked={form.enabled} onChange={(enabled) => setForm({ ...form, enabled })} />
          <div className="row end" style={{ marginTop: 8 }}>
            <button onClick={() => setForm(null)}><X size={13} /> Cancelar</button>
            <button className="primary" disabled={!form.name.trim() || !form.base_url.trim() || busy === "save"} onClick={save}>
              {busy === "save" ? <Spinner /> : <CheckIcon size={13} />} Guardar
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

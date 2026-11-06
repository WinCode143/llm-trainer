import { useEffect, useRef, useState } from "react";
import { History, Pencil, Rewind, X } from "lucide-react";
import { api } from "../api";
import { Field, Spinner, useData } from "./ui";

export type Checkpoint = {
  checkpoint_id: string; next: string[]; next_labels: string[];
  done: string[]; vars: Record<string, any>; created_at: string | null;
};

/** Rebobinar una ejecución: volver a un punto, cambiar variables y seguir desde ahí.
 *  Lo que ya se calculó antes de ese punto no se repite, que es lo que hace útil iterar
 *  sobre el último prompt sin volver a pagar la búsqueda ni las llamadas anteriores. */
export function TimeTravel({ runId, onReplay }: {
  runId: number; onReplay: (checkpointId: string, edits: Record<string, string>) => void;
}) {
  const [data, reload] = useData<{ checkpoints: Checkpoint[] }>(`/api/flows/runs/${runId}/history`);
  const [editing, setEditing] = useState<string | null>(null);
  const [edits, setEdits] = useState<Record<string, string>>({});

  useEffect(() => { reload(); }, [runId]);

  // El primero es el estado final y el último el arranque: ni uno ni otro sirven para rebobinar
  const puntos = (data?.checkpoints ?? []).filter((c) => c.next.length && c.next[0] !== "__start__");
  if (!puntos.length) return null;

  const abrir = (c: Checkpoint) => {
    setEditing(c.checkpoint_id);
    setEdits(Object.fromEntries(Object.entries(c.vars)
      .filter(([, v]) => typeof v === "string")
      .map(([k, v]) => [k, String(v)])));
  };

  return (
    <div className="card">
      <div className="row between">
        <b><History size={14} /> Rebobinar</b>
        <span className="sub">{puntos.length} punto(s) de guardado</span>
      </div>
      <div className="sub" style={{ marginTop: 4 }}>
        Vuelve a cualquier paso y continúa desde ahí. Lo anterior no se repite, así que puedes probar
        otro texto sin pagar otra vez la búsqueda ni las llamadas al modelo que ya se hicieron.
      </div>
      <div className="stack" style={{ marginTop: 8, gap: 6 }}>
        {puntos.map((c) => (
          <div key={c.checkpoint_id} className="rewind">
            <div className="row between">
              <span>antes de <b>{c.next_labels.join(", ")}</b> <span className="sub">· {c.done.length} nodo(s) ya hechos</span></span>
              <div className="row" style={{ gap: 4 }}>
                <button className="sm" onClick={() => onReplay(c.checkpoint_id, {})}>
                  <Rewind size={12} /> Reanudar
                </button>
                <button className="sm ghost" title="Cambiar variables antes de continuar"
                  onClick={() => (editing === c.checkpoint_id ? setEditing(null) : abrir(c))}>
                  <Pencil size={12} />
                </button>
              </div>
            </div>
            {editing === c.checkpoint_id && (
              <div className="stack" style={{ marginTop: 8, gap: 6 }}>
                <div className="sub">
                  Cambia lo que quieras y continúa: se crea una rama nueva y la ejecución original se conserva.
                </div>
                {Object.keys(edits).length === 0 && <div className="sub">No hay variables de texto en este punto.</div>}
                {Object.entries(edits).map(([k, v]) => (
                  <Field key={k} label={k}>
                    <textarea rows={Math.min(6, Math.ceil(v.length / 70) + 1)} value={v}
                      onChange={(e) => setEdits({ ...edits, [k]: e.target.value })} />
                  </Field>
                ))}
                <div className="row end" style={{ gap: 6 }}>
                  <button className="sm" onClick={() => setEditing(null)}><X size={12} /> Cancelar</button>
                  <button className="sm primary" onClick={() => {
                    const c0 = puntos.find((p) => p.checkpoint_id === editing)!;
                    // Sólo viajan las variables que realmente cambiaste
                    const cambios = Object.fromEntries(Object.entries(edits).filter(([k, v]) => String(c0.vars[k] ?? "") !== v));
                    setEditing(null);
                    onReplay(c.checkpoint_id, cambios);
                  }}>Continuar con los cambios</button>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

/** Campo de texto que conoce las variables del flujo: se insertan con un clic o escribiendo «{». */
export function VarField({ value, onChange, vars, rows = 3, placeholder }: {
  value: string; onChange: (v: string) => void; vars: string[]; rows?: number; placeholder?: string;
}) {
  const ref = useRef<HTMLTextAreaElement>(null);
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState("");

  const insert = (name: string) => {
    const el = ref.current;
    const pos = el?.selectionStart ?? value.length;
    // Si el usuario venía escribiendo «{algo», se sustituye eso en vez de añadir otra llave
    const antes = value.slice(0, pos).replace(/\{[\w]*$/, "");
    onChange(`${antes}{${name}}${value.slice(pos)}`);
    setOpen(false);
    setFilter("");
    requestAnimationFrame(() => {
      el?.focus();
      const p = antes.length + name.length + 2;
      el?.setSelectionRange(p, p);
    });
  };

  const onKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "{") { setFilter(""); setOpen(true); }
    else if (e.key === "Escape") setOpen(false);
  };

  const visibles = vars.filter((v) => v.toLowerCase().includes(filter.toLowerCase()));

  return (
    <div className="varfield">
      <textarea ref={ref} rows={rows} value={value} placeholder={placeholder}
        onChange={(e) => {
          onChange(e.target.value);
          const hasta = e.target.value.slice(0, e.target.selectionStart);
          const m = hasta.match(/\{([\w]*)$/);   // ¿está escribiendo una variable ahora mismo?
          setOpen(!!m);
          setFilter(m?.[1] ?? "");
        }}
        onKeyDown={onKey} onBlur={() => setTimeout(() => setOpen(false), 150)} />
      {open && !!visibles.length && (
        <div className="varmenu">
          {visibles.slice(0, 8).map((v) => (
            <button key={v} onMouseDown={(e) => e.preventDefault()} onClick={() => insert(v)}>{v}</button>
          ))}
        </div>
      )}
      <div className="chips" style={{ marginTop: 4 }}>
        {vars.slice(0, 10).map((v) => (
          <button key={v} className="chip clickable" title={`Insertar {${v}}`} onClick={() => insert(v)}>{v}</button>
        ))}
      </div>
    </div>
  );
}

/** Aviso de que un flujo tiene problemas antes de ejecutarlo. */
export function Problems({ items }: { items: string[] }) {
  if (!items.length) return null;
  return (
    <div className="warn" style={{ marginTop: 8 }}>
      {items.map((w, i) => <div key={i}>⚠ {w}</div>)}
    </div>
  );
}

export const Loading = () => <div className="sub"><Spinner /> Cargando…</div>;
export { api };

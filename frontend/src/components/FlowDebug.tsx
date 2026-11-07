import { useEffect, useRef, useState } from "react";
import { History, Pencil, Rewind, X } from "lucide-react";
import { api } from "../api";
import { Field, Spinner, useData } from "./ui";

export type Checkpoint = {
  checkpoint_id: string; next: string[]; next_labels: string[];
  done: string[]; vars: Record<string, any>; created_at: string | null;
};

/** Rewind a run: go back to a point, change variables and continue from there.
  *  Whatever was computed before that point isn't repeated, which is what makes it useful to iterate
  *  on the last prompt without paying for the search or the earlier calls again. */
export function TimeTravel({ runId, onReplay }: {
  runId: number; onReplay: (checkpointId: string, edits: Record<string, string>) => void;
}) {
  const [data, reload] = useData<{ checkpoints: Checkpoint[] }>(`/api/flows/runs/${runId}/history`);
  const [editing, setEditing] = useState<string | null>(null);
  const [edits, setEdits] = useState<Record<string, string>>({});

  useEffect(() => { reload(); }, [runId]);

  // The first is the final state and the last one is the start: neither is useful for rewinding
  const points = (data?.checkpoints ?? []).filter((c) => c.next.length && c.next[0] !== "__start__");
  if (!points.length) return null;

  const openEditor = (c: Checkpoint) => {
    setEditing(c.checkpoint_id);
    setEdits(Object.fromEntries(Object.entries(c.vars)
      .filter(([, v]) => typeof v === "string")
      .map(([k, v]) => [k, String(v)])));
  };

  return (
    <div className="card">
      <div className="row between">
        <b><History size={14} /> Rewind</b>
        <span className="sub">{points.length} checkpoint(s)</span>
      </div>
      <div className="sub" style={{ marginTop: 4 }}>
        Go back to any step and continue from there. Earlier steps aren't repeated, so you can try
        different text without paying again for the search or the model calls that already ran.
      </div>
      <div className="stack" style={{ marginTop: 8, gap: 6 }}>
        {points.map((c) => (
          <div key={c.checkpoint_id} className="rewind">
            <div className="row between">
              <span>before <b>{c.next_labels.join(", ")}</b> <span className="sub">· {c.done.length} node(s) already done</span></span>
              <div className="row" style={{ gap: 4 }}>
                <button className="sm" onClick={() => onReplay(c.checkpoint_id, {})}>
                  <Rewind size={12} /> Resume
                </button>
                <button className="sm ghost" title="Change variables before continuing"
                  onClick={() => (editing === c.checkpoint_id ? setEditing(null) : openEditor(c))}>
                  <Pencil size={12} />
                </button>
              </div>
            </div>
            {editing === c.checkpoint_id && (
              <div className="stack" style={{ marginTop: 8, gap: 6 }}>
                <div className="sub">
                  Change whatever you want and continue: a new branch is created and the original run is kept.
                </div>
                {Object.keys(edits).length === 0 && <div className="sub">There are no text variables at this point.</div>}
                {Object.entries(edits).map(([k, v]) => (
                  <Field key={k} label={k}>
                    <textarea rows={Math.min(6, Math.ceil(v.length / 70) + 1)} value={v}
                      onChange={(e) => setEdits({ ...edits, [k]: e.target.value })} />
                  </Field>
                ))}
                <div className="row end" style={{ gap: 6 }}>
                  <button className="sm" onClick={() => setEditing(null)}><X size={12} /> Cancel</button>
                  <button className="sm primary" onClick={() => {
                    const c0 = points.find((p) => p.checkpoint_id === editing)!;
                    // Only the variables you actually changed are sent
                    const changes = Object.fromEntries(Object.entries(edits).filter(([k, v]) => String(c0.vars[k] ?? "") !== v));
                    setEditing(null);
                    onReplay(c.checkpoint_id, changes);
                  }}>Continue with the changes</button>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

/** A text field that knows the flow's variables: insert them with a click or by typing “{”. */
export function VarField({ value, onChange, vars, rows = 3, placeholder }: {
  value: string; onChange: (v: string) => void; vars: string[]; rows?: number; placeholder?: string;
}) {
  const ref = useRef<HTMLTextAreaElement>(null);
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState("");

  const insert = (name: string) => {
    const el = ref.current;
    const pos = el?.selectionStart ?? value.length;
    // If the user was typing “{something”, replace that instead of adding another brace
    const before = value.slice(0, pos).replace(/\{[\w]*$/, "");
    onChange(`${before}{${name}}${value.slice(pos)}`);
    setOpen(false);
    setFilter("");
    requestAnimationFrame(() => {
      el?.focus();
      const p = before.length + name.length + 2;
      el?.setSelectionRange(p, p);
    });
  };

  const onKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "{") { setFilter(""); setOpen(true); }
    else if (e.key === "Escape") setOpen(false);
  };

  const shown = vars.filter((v) => v.toLowerCase().includes(filter.toLowerCase()));

  return (
    <div className="varfield">
      <textarea ref={ref} rows={rows} value={value} placeholder={placeholder}
        onChange={(e) => {
          onChange(e.target.value);
          const upTo = e.target.value.slice(0, e.target.selectionStart);
          const m = upTo.match(/\{([\w]*)$/);   // is a variable being typed right now?
          setOpen(!!m);
          setFilter(m?.[1] ?? "");
        }}
        onKeyDown={onKey} onBlur={() => setTimeout(() => setOpen(false), 150)} />
      {open && !!shown.length && (
        <div className="varmenu">
          {shown.slice(0, 8).map((v) => (
            <button key={v} onMouseDown={(e) => e.preventDefault()} onClick={() => insert(v)}>{v}</button>
          ))}
        </div>
      )}
      <div className="chips" style={{ marginTop: 4 }}>
        {vars.slice(0, 10).map((v) => (
          <button key={v} className="chip clickable" title={`Insert {${v}}`} onClick={() => insert(v)}>{v}</button>
        ))}
      </div>
    </div>
  );
}

/** Warns that a flow has problems before running it. */
export function Problems({ items }: { items: string[] }) {
  if (!items.length) return null;
  return (
    <div className="warn" style={{ marginTop: 8 }}>
      {items.map((w, i) => <div key={i}>⚠ {w}</div>)}
    </div>
  );
}

export const Loading = () => <div className="sub"><Spinner /> Loading…</div>;
export { api };

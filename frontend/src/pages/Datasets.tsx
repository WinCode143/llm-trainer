import { useState } from "react";
import { ChevronLeft, ChevronRight, Plus, Scissors, Trash2, Upload } from "lucide-react";
import { api, type Collection, type Dataset, fmtDate } from "../api";
import { Adv, Field, Help, Modal, SimpleNote, useData, useDialog, useToast } from "../components/ui";
import { HelperPicker } from "../components/Helper";

const FORMAT_LABEL = { chat: "Chat / instructions", text: "Plain text", preference: "Preferences (DPO)" };

export default function Datasets() {
  const [list, reload] = useData<Dataset[]>("/api/datasets");
  const [sel, setSel] = useState<number | null>(null);
  const [showImport, setShowImport] = useState(false);
  const current = list?.find((d) => d.id === sel) ?? list?.[0];

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Datasets</h1>
          <div className="sub">Import, inspect and prepare training data</div>
        </div>
        <button className="primary" onClick={() => setShowImport(true)}><Plus size={15} /> New dataset</button>
      </div>
      <Help title="What is a dataset, and how do you prepare one?">
        <p>A dataset is the <b>examples the model learns from</b>: question and answer pairs with the style and knowledge you want it to imitate.</p>
        <ul>
          <li><b>Chat / instructions</b>: the usual choice to teach it how to answer. Accepted formats: Alpaca (<code>instruction</code>/<code>output</code>), ShareGPT, OpenAI (<code>messages</code>).</li>
          <li><b>Preferences</b> (<code>prompt</code>/<code>chosen</code>/<code>rejected</code>): for DPO, ORPO and KTO, which teach <i>which answer is better</i>.</li>
          <li><b>Plain text</b>: for continued pretraining on a domain.</li>
        </ul>
        <p style={{ marginTop: 8 }}>Where to get it: upload a file, import from Hugging Face, write it by hand, or <b>generate it</b> from your RAG documents or from a domain (“New dataset” → Synthetic). 50–100 good examples already make a difference; split into train/eval so you can measure.</p>
      </Help>
      <div className="split">
        <div className="card" style={{ padding: 8 }}>
          {!list?.length && <div className="empty">No datasets yet</div>}
          <table>
            <tbody>
              {list?.map((d) => (
                <tr key={d.id} className={`clickable ${current?.id === d.id ? "selected" : ""}`} onClick={() => setSel(d.id)}>
                  <td>
                    <div><b>{d.name}</b></div>
                    <div className="sub" style={{ fontSize: 12 }}>{d.num_rows.toLocaleString()} rows · {FORMAT_LABEL[d.format]}</div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {current ? <DatasetDetail key={current.id} ds={current} onChange={reload} /> : <div className="card empty">Create or import a dataset to start.</div>}
      </div>
      {showImport && <ImportModal onClose={() => setShowImport(false)} onDone={(d) => { reload(); setSel(d?.id ?? null); setShowImport(false); }} />}
    </>
  );
}

function DatasetDetail({ ds, onChange }: { ds: Dataset; onChange: () => void }) {
  const [page, setPage] = useState(0);
  const PAGE = 10;
  const [rows] = useData<{ total: number; rows: any[] }>(`/api/datasets/${ds.id}/rows?offset=${page * PAGE}&limit=${PAGE}&v=${ds.num_rows}`);
  const [stats] = useData<any>(`/api/datasets/${ds.id}/stats?v=${ds.num_rows}`);
  const { notify } = useToast();
  const { confirm, prompt } = useDialog();
  const [adding, setAdding] = useState(false);

  const remove = async () => {
    if (!(await confirm(`Delete the dataset “${ds.name}”?`, `${ds.num_rows} rows. This can't be undone.`, { label: "Delete" }))) return;
    await api.del(`/api/datasets/${ds.id}`);
    onChange();
  };
  const split = async () => {
    const r = await prompt("Split into train / eval", { message: "Share of rows that go to the evaluation set (0–1). Two new datasets will be created: -train and -eval.", value: "0.1" });
    if (!r) return;
    try {
      await api.post(`/api/datasets/${ds.id}/split`, { eval_ratio: parseFloat(r) });
      notify("Created the -train and -eval datasets");
      onChange();
    } catch (e: any) { notify(e.message, "err"); }
  };

  return (
    <div className="stack" style={{ gap: 16 }}>
      <div className="card">
        <div className="row between">
          <div>
            <h2 style={{ marginBottom: 2 }}>{ds.name}</h2>
            <div className="sub">{FORMAT_LABEL[ds.format]} · {ds.source} · {fmtDate(ds.created_at)}</div>
          </div>
          <div className="row">
            <button onClick={() => setAdding(true)}><Plus size={14} /> Add example</button>
            <button onClick={split}><Scissors size={14} /> Split train/eval</button>
            <button className="danger" onClick={remove}><Trash2 size={14} /></button>
          </div>
        </div>
        {stats && (
          <div className="grid g4" style={{ marginTop: 14 }}>
            <div className="stat"><div className="label">Rows</div><div className="value">{stats.rows.toLocaleString()}</div></div>
            <div className="stat"><div className="label">≈ Total tokens</div><div className="value">{stats.approx_tokens_total.toLocaleString()}</div></div>
            <div className="stat"><div className="label">Characters (mean / p95)</div><div className="value">{stats.chars_avg} / {stats.chars_p95}</div></div>
            <div className="stat"><div className="label">{stats.turns_avg ? "Turns per example" : "Maximum"}</div><div className="value">{stats.turns_avg ?? stats.chars_max}</div></div>
          </div>
        )}
      </div>

      <div className="card">
        <div className="row between" style={{ marginBottom: 10 }}>
          <h3 style={{ margin: 0 }}>Preview</h3>
          <div className="row">
            <button className="sm" disabled={page === 0} onClick={() => setPage(page - 1)}><ChevronLeft size={14} /></button>
            <span className="sub">{page * PAGE + 1}–{Math.min((page + 1) * PAGE, rows?.total ?? 0)} of {rows?.total ?? 0}</span>
            <button className="sm" disabled={(page + 1) * PAGE >= (rows?.total ?? 0)} onClick={() => setPage(page + 1)}><ChevronRight size={14} /></button>
          </div>
        </div>
        <div className="stack">
          {rows?.rows.map((r, i) => <RowView key={i} row={r} format={ds.format} />)}
        </div>
      </div>
      {adding && <AddRowModal ds={ds} onClose={() => setAdding(false)} onDone={() => { setAdding(false); onChange(); }} />}
    </div>
  );
}

function RowView({ row, format }: { row: any; format: Dataset["format"] }) {
  if (format === "chat")
    return (
      <div className="stack" style={{ gap: 6, borderBottom: "1px solid var(--border)", paddingBottom: 10 }}>
        {row.messages.map((m: any, i: number) => (
          <div key={i} className="row" style={{ alignItems: "flex-start", flexWrap: "nowrap" }}>
            <span className={`badge ${m.role === "assistant" ? "ok" : m.role === "system" ? "warn" : "accent"}`} style={{ minWidth: 72, textAlign: "center" }}>{m.role}</span>
            <div style={{ whiteSpace: "pre-wrap", flex: 1 }}>{m.content}</div>
          </div>
        ))}
      </div>
    );
  if (format === "preference")
    return (
      <div className="stack" style={{ gap: 6, borderBottom: "1px solid var(--border)", paddingBottom: 10 }}>
        <div><span className="badge accent">prompt</span> {row.prompt}</div>
        <div><span className="badge ok">chosen</span> {row.chosen}</div>
        <div><span className="badge err">rejected</span> {row.rejected}</div>
      </div>
    );
  return <div style={{ whiteSpace: "pre-wrap", borderBottom: "1px solid var(--border)", paddingBottom: 10 }}>{row.text}</div>;
}

function AddRowModal({ ds, onClose, onDone }: { ds: Dataset; onClose: () => void; onDone: () => void }) {
  const [a, setA] = useState("");
  const [b, setB] = useState("");
  const [c, setC] = useState("");
  const { notify } = useToast();
  const save = async () => {
    const row = ds.format === "chat" ? { instruction: a, output: b } : ds.format === "text" ? { text: a } : { prompt: a, chosen: b, rejected: c };
    try { await api.post(`/api/datasets/${ds.id}/append`, { rows: [row] }); onDone(); } catch (e: any) { notify(e.message, "err"); }
  };
  return (
    <Modal title="Add example" onClose={onClose}>
      <div className="stack">
        <Field label={ds.format === "chat" ? "User" : ds.format === "text" ? "Text" : "Prompt"}><textarea value={a} onChange={(e) => setA(e.target.value)} /></Field>
        {ds.format !== "text" && <Field label={ds.format === "chat" ? "Assistant" : "Preferred answer (chosen)"}><textarea value={b} onChange={(e) => setB(e.target.value)} /></Field>}
        {ds.format === "preference" && <Field label="Rejected answer (rejected)"><textarea value={c} onChange={(e) => setC(e.target.value)} /></Field>}
        <div className="row"><div className="spacer" /><button className="primary" onClick={save} disabled={!a}>Save</button></div>
      </div>
    </Modal>
  );
}

function ImportModal({ onClose, onDone }: { onClose: () => void; onDone: (d?: Dataset) => void }) {
  const [tab, setTab] = useState<"file" | "hf" | "manual" | "synth" | "template">("file");
  const [manualRows, setManualRows] = useState([{ a: "", b: "", c: "" }]);
  const [name, setName] = useState("");
  const [format, setFormat] = useState("");
  const [system, setSystem] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [repo, setRepo] = useState("");
  const [split, setSplit] = useState("train");
  const [subset, setSubset] = useState("");
  const [limit, setLimit] = useState("");
  const [busy, setBusy] = useState(false);
  const [cols] = useData<Collection[]>("/api/rag/collections");
  const [collId, setCollId] = useState<number | "">("");
  const [nChunks, setNChunks] = useState(20);
  const [pairs, setPairs] = useState(3);
  const [synthSrc, setSynthSrc] = useState<"docs" | "topics">("docs");
  const [domain, setDomain] = useState("");
  const [styles, setStyles] = useState<string[]>(["hecho", "explicacion"]);
  const [judgeMin, setJudgeMin] = useState(0);
  const [helper, setHelper] = useState<number | null>(null);
  const [template, setTemplate] = useState<string | null>(null);
  const [templates] = useData<any[]>("/api/datasets/templates");
  const { notify, track } = useToast();

  const submit = async () => {
    setBusy(true);
    try {
      if (tab === "file" && file) {
        const fd = new FormData();
        fd.append("file", file);
        fd.append("name", name || file.name.replace(/\.[^.]+$/, ""));
        if (format) fd.append("format", format);
        if (system) fd.append("system_prompt", system);
        onDone(await api.form<Dataset>("/api/datasets/upload", fd));
      } else if (tab === "hf") {
        const t = await api.post<any>("/api/datasets/import-hf", {
          repo, name: name || repo.split("/").pop(), split, subset: subset || null, limit: limit ? +limit : null,
          format: format || null, system_prompt: system || null,
        });
        track(t, (done) => onDone(done.result));
        onClose();
      } else if (tab === "manual") {
        const rows = manualRows.filter((r) => r.a.trim() && (format === "text" || r.b.trim())).map((r) =>
          format === "preference" ? { prompt: r.a, chosen: r.b, rejected: r.c }
            : format === "text" ? { text: r.a } : { instruction: r.a, output: r.b, ...(system ? { system } : {}) });
        if (!rows.length) throw new Error("Write at least one complete example");
        onDone(await api.post<Dataset>("/api/datasets/manual", { name, format: format || "chat", rows }));
      } else if (tab === "template") {
        if (!template) throw new Error("Choose a template");
        onDone(await api.post<Dataset>(`/api/datasets/templates/${template}?name=${encodeURIComponent(name)}`));
      } else if (tab === "synth") {
        const t = await api.post<any>("/api/datasets/synthesize", {
          source: synthSrc, collection_id: collId || null, domain: domain || null, name, num_chunks: nChunks, num_topics: nChunks,
          pairs_per_chunk: pairs, styles, judge_min_score: judgeMin, system_prompt: system || null, helper,
        });
        track(t, () => onDone());
        onClose();
      }
    } catch (e: any) { notify(e.message, "err"); } finally { setBusy(false); }
  };

  return (
    <Modal title="New dataset" onClose={onClose}>
      <div className="tabs">
        <button className={`tab ${tab === "file" ? "active" : ""}`} onClick={() => setTab("file")}>File</button>
        <button className={`tab ${tab === "hf" ? "active" : ""}`} onClick={() => setTab("hf")}>Hugging Face</button>
        <button className={`tab ${tab === "manual" ? "active" : ""}`} onClick={() => setTab("manual")}>By hand</button>
        <button className={`tab ${tab === "synth" ? "active" : ""}`} onClick={() => setTab("synth")}>Synthetic</button>
        <button className={`tab ${tab === "template" ? "active" : ""}`} onClick={() => setTab("template")}>Template</button>
      </div>
      <div className="stack">
        <Field label="Name"><input value={name} onChange={(e) => setName(e.target.value)} placeholder="my-dataset" /></Field>
        {tab === "template" && (
          <div className="stack">
            <div className="sub">
              Start from a use case. Each template creates a dataset with real, already written examples
              so you see the exact shape your data needs: then you edit them,
              add your own and train.
            </div>
            {!templates?.length && <div className="empty">Loading…</div>}
            {templates?.map((t) => (
              <label key={t.key} className={`option ${template === t.key ? "selected" : ""}`} onClick={() => { setTemplate(t.key); setName(t.key); }}>
                <input type="radio" checked={template === t.key} readOnly />
                <div>
                  <b>{t.nombre}</b> <span className="badge">{t.metodo.toUpperCase()}</span>
                  <div className="sub">{t.descripcion}</div>
                  <div className="sub" style={{ marginTop: 4 }}><b>How many:</b> {t.cuantos}</div>
                  <div className="sub"><b>Tip:</b> {t.consejo}</div>
                </div>
              </label>
            ))}
            <Field label="Dataset name"><input value={name} onChange={(e) => setName(e.target.value)} /></Field>
          </div>
        )}
        {tab === "manual" && (
          <div className="stack">
            <div className="sub">Write the examples one by one. You can add more later from the dataset's page (“Add example”).</div>
            {manualRows.map((r, i) => (
              <div key={i} className="source stack" style={{ gap: 6 }}>
                <div className="row between"><b style={{ fontSize: 12 }}>Example {i + 1}</b>
                  {manualRows.length > 1 && <button className="ghost sm danger" onClick={() => setManualRows(manualRows.filter((_, j) => j !== i))}>remove</button>}</div>
                <Field label={format === "preference" ? "Prompt" : format === "text" ? "Text" : "User asks"}>
                  <textarea rows={2} value={r.a} onChange={(e) => setManualRows(manualRows.map((x, j) => (j === i ? { ...x, a: e.target.value } : x)))} /></Field>
                {format !== "text" && (
                  <Field label={format === "preference" ? "Preferred answer" : "Answer it should learn"}>
                    <textarea rows={2} value={r.b} onChange={(e) => setManualRows(manualRows.map((x, j) => (j === i ? { ...x, b: e.target.value } : x)))} /></Field>
                )}
                {format === "preference" && (
                  <Field label="Rejected answer">
                    <textarea rows={2} value={r.c} onChange={(e) => setManualRows(manualRows.map((x, j) => (j === i ? { ...x, c: e.target.value } : x)))} /></Field>
                )}
              </div>
            ))}
            <button onClick={() => setManualRows([...manualRows, { a: "", b: "", c: "" }])}><Plus size={13} /> Add another example</button>
          </div>
        )}
        {tab === "file" && (
          <Field label="File" hint="JSONL, JSON, CSV, Parquet or TXT. Formats: Alpaca (instruction/output), ShareGPT (conversations), OpenAI (messages), DPO (prompt/chosen/rejected), text (text).">
            <input type="file" accept=".jsonl,.json,.csv,.parquet,.txt,.md" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          </Field>
        )}
        {tab === "hf" && (
          <>
            <Field label="Repository" hint="e.g. mlabonne/FineTome-100k, HuggingFaceH4/ultrafeedback_binarized"><input value={repo} onChange={(e) => setRepo(e.target.value)} placeholder="user/dataset" /></Field>
            <div className="grid g3">
              <Field label="Row limit"><input type="number" value={limit} onChange={(e) => setLimit(e.target.value)} placeholder="all" /></Field>
              <Adv>
                <Field label="Split"><input value={split} onChange={(e) => setSplit(e.target.value)} /></Field>
                <Field label="Subset (optional)"><input value={subset} onChange={(e) => setSubset(e.target.value)} /></Field>
              </Adv>
            </div>
          </>
        )}
        {tab === "synth" && (
          <>
            <div className="sub">Uses the loaded model to generate question/answer pairs. Semantic duplicates are removed and, optionally, a judge discards low-quality pairs.</div>
            <div className="row" style={{ gap: 16 }}>
              <label className="check"><input type="radio" checked={synthSrc === "docs"} onChange={() => setSynthSrc("docs")} /> From documents (RAG collection)</label>
              <label className="check"><input type="radio" checked={synthSrc === "topics"} onChange={() => setSynthSrc("topics")} /> From a domain (topic tree)</label>
            </div>
            {synthSrc === "docs" ? (
              <Field label="RAG collection">
                <select value={collId} onChange={(e) => setCollId(e.target.value ? +e.target.value : "")}>
                  <option value="">— choose —</option>
                  {cols?.map((c) => <option key={c.id} value={c.id}>{c.name} ({c.num_chunks} chunks)</option>)}
                </select>
              </Field>
            ) : (
              <Field label="Domain" hint="The model proposes subtopics and generates pairs for each one"><input value={domain} onChange={(e) => setDomain(e.target.value)} placeholder="Customer support for an online electronics store" /></Field>
            )}
            <div className="grid g3">
              <Field label={synthSrc === "docs" ? "Chunks to sample" : "Subtopics"}><input type="number" value={nChunks} onChange={(e) => setNChunks(+e.target.value)} /></Field>
              <Field label={synthSrc === "docs" ? "Pairs per chunk" : "Pairs per subtopic"}><input type="number" value={pairs} onChange={(e) => setPairs(+e.target.value)} /></Field>
              <Field label={`Judge filter: ${judgeMin ? `≥ ${judgeMin}/5` : "off"}`}><input type="range" min={0} max={5} step={1} value={judgeMin} onChange={(e) => setJudgeMin(+e.target.value)} /></Field>
              <HelperPicker value={helper} onChange={setHelper}
                hint="Generate with a bigger model than the one you'll train: that's how good synthetic data is made" />
            </div>
            <Adv>
              <Field label="Question types (they alternate for variety)">
                <div className="row" style={{ gap: 12 }}>
                  {[["hecho", "Facts"], ["explicacion", "Explanations"], ["comparacion", "Comparisons"], ["procedimiento", "Procedures"], ["aplicacion", "Practical cases"]].map(([k, l]) => (
                    <label key={k} className="check"><input type="checkbox" checked={styles.includes(k)} onChange={(e) => setStyles(e.target.checked ? [...styles, k] : styles.filter((x) => x !== k))} />{l}</label>
                  ))}
                </div>
              </Field>
            </Adv>
            <SimpleNote>Fact and explanation questions are generated, near-duplicates are removed and those the judge scores low are discarded.</SimpleNote>
          </>
        )}
        {tab !== "synth" && (
          <Field label="Format" hint={tab === "manual" ? "Sets which fields are asked for above" : "Detected automatically if left empty"}>
            <select value={format} onChange={(e) => setFormat(e.target.value)}>
              <option value="">{tab === "manual" ? "Chat / instructions" : "Auto-detect"}</option>
              <option value="chat">Chat / instructions</option>
              <option value="text">Plain text (continued pretraining)</option>
              <option value="preference">Preferences (DPO)</option>
            </select>
          </Field>
        )}
        <Field label="System prompt (optional)" hint="Added to every conversation that doesn't have one">
          <textarea value={system} onChange={(e) => setSystem(e.target.value)} rows={2} />
        </Field>
        <div className="row">
          <div className="spacer" />
          <button className="primary" disabled={busy || (tab === "file" ? !file : tab === "hf" ? !repo : tab === "manual" ? !name || !manualRows.some((r) => r.a.trim()) : !name || (synthSrc === "docs" ? !collId : !domain))} onClick={submit}>
            <Upload size={14} /> {tab === "synth" ? "Generate" : tab === "manual" ? "Create" : "Import"}
          </button>
        </div>
      </div>
    </Modal>
  );
}

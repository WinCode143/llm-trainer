import { useState } from "react";
import { Download, Eye, FileText, Globe, Image as ImageIcon, Layers, Pencil, Plus, Save, Search, Sparkles, Tag, Trash2, Upload, X } from "lucide-react";
import { api, type Collection, fmtDate } from "../api";
import { GraphView } from "../components/GraphView";
import { Adv, Check, Field, Help, Modal, SimpleNote, Spinner, useData, useDialog, useToast } from "../components/ui";
import { HelperPicker } from "../components/Helper";

const MODES: Record<string, string> = { hybrid: "Hybrid (BM25 + vectors)", vector: "Vectors only", bm25: "BM25 only (words)" };
const DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3";
type Options = { templates: Record<string, string>; query_modes: Record<string, string> };
/** Readable name of a document (URLs are shown decoded). */
const prettyDoc = (d: string) => { try { return d.startsWith("http") ? decodeURIComponent(d) : d; } catch { return d; } };
const allTags = (c: Collection) => [...new Set(Object.values(c.doc_meta ?? {}).flatMap((m) => m?.tags ?? []))].sort();

export default function Rag() {
  const [cols, reload] = useData<Collection[]>("/api/rag/collections");
  const [sel, setSel] = useState<number | null>(null);
  const [creating, setCreating] = useState(false);
  const cur = cols?.find((c) => c.id === sel) ?? cols?.[0];

  return (
    <>
      <div className="page-head">
        <div><h1>RAG</h1><div className="sub">Document collections with hybrid search, reranking and a relevance threshold</div></div>
        <button className="primary" onClick={() => setCreating(true)}><Plus size={14} /> New collection</button>
      </div>
      <Help title="What is RAG, and what does each tab do?">
        <p>RAG = the model <b>searches your documents before answering</b>. That way it answers with your information and cites where it found it, without any training.</p>
        <p style={{ marginTop: 6 }}>The process: documents are split into chunks → each chunk becomes a vector (embedding) → at question time the closest ones are found (by meaning and by words) → a <b>reranker</b> reorders them → the best ones are given to the model as context.</p>
        <ul>
          <li><b>Documents</b>: upload files or web pages, tag them and run the extras (questions per chunk, summaries, image descriptions).</li>
          <li><b>Test search</b>: see what gets retrieved and with what score, before asking the model.</li>
          <li><b>Graph</b>: entities and relations extracted from the documents (GraphRAG), for questions that span several chunks.</li>
          <li><b>Benchmark</b>: compares search modes with your own questions and tells you which one gets more right.</li>
          <li><b>Settings</b>: search mode, reranker, threshold, chunking template and embedding model.</li>
        </ul>
      </Help>
      {(cols?.length ?? 0) >= 2 && <CompararColecciones cols={cols ?? []} />}

      <div className="split">
        <div className="card" style={{ padding: 8 }}>
          {!cols?.length && <div className="empty">No collections</div>}
          <table><tbody>
            {cols?.map((c) => (
              <tr key={c.id} className={`clickable ${cur?.id === c.id ? "selected" : ""}`} onClick={() => setSel(c.id)}>
                <td><b>{c.name}</b><div className="sub" style={{ fontSize: 12 }}>{c.documents.length} docs · {c.num_chunks} chunks{c.reranker ? " · rerank" : ""}</div>
                  <div className="row" style={{ gap: 3, marginTop: 3 }}>
                    {c.features?.questions ? <span className="badge ok" title={`${c.features.questions} chunks with generated questions`}>questions</span> : null}
                    {c.features?.summaries ? <span className="badge ok" title={`${c.features.summaries} summaries`}>summaries</span> : null}
                    {c.features?.images ? <span className="badge ok" title={`${c.features.images} described images`}>images</span> : null}
                    {c.features?.graph ? <span className="badge ok" title="Knowledge graph built">graph</span> : null}
                  </div></td>
              </tr>
            ))}
          </tbody></table>
        </div>
        {cur ? <CollectionDetail key={cur.id} c={cur} onChange={reload} /> : <div className="card empty">Create a collection and upload PDF, DOCX, Markdown, TXT or HTML.</div>}
      </div>
      {creating && <CreateModal onClose={() => setCreating(false)} onDone={(c) => { setCreating(false); reload(); setSel(c.id); }} />}
    </>
  );
}

function CollectionDetail({ c, onChange }: { c: Collection; onChange: () => void }) {
  const [tab, setTab] = useState<"docs" | "search" | "graph" | "bench" | "settings">("docs");
  const { confirm } = useDialog();
  return (
    <div className="stack" style={{ gap: 16 }}>
      <div className="card">
        <div className="row between" style={{ flexWrap: "nowrap", alignItems: "flex-start" }}>
          <div style={{ minWidth: 0 }}>
            <h2 style={{ marginBottom: 2 }}>{c.name}</h2>
            <div className="sub">
              {MODES[c.search_mode]} · {c.reranker ? <>rerank <code>{c.reranker.split("/").pop()}</code></> : "no rerank"} · top-{c.top_k}
              {c.query_mode && c.query_mode !== "simple" && <> · query <b>{c.query_mode}</b></>}
              {c.min_score > 0 && <> · threshold {c.min_score}</>} · embeddings <code>{c.embed_model.split("/").pop()}</code> · {fmtDate(c.created_at)}
            </div>
          </div>
          <button className="danger" onClick={async () => (await confirm(`Delete the collection “${c.name}”?`, `${c.documents.length} documents and ${c.num_chunks} indexed chunks.`, { label: "Delete" })) && api.del(`/api/rag/collections/${c.id}`).then(onChange)}><Trash2 size={14} /></button>
        </div>
        <div className="tabs" style={{ marginTop: 12, marginBottom: 0 }}>
          {([["docs", "Documents"], ["search", "Test search"], ["graph", "Graph"], ["bench", "Benchmark"], ["settings", "Settings"]] as const).map(([k, l]) => (
            <button key={k} className={`tab ${tab === k ? "active" : ""}`} onClick={() => setTab(k)}>{l}</button>
          ))}
        </div>
      </div>
      {tab === "docs" && <Documents c={c} onChange={onChange} />}
      {tab === "search" && <SearchTester c={c} />}
      {tab === "graph" && <Graph c={c} />}
      {tab === "bench" && <Benchmark c={c} />}
      {tab === "settings" && <Settings c={c} onChange={onChange} />}
    </div>
  );
}

// ---------------- Documents + chunk viewer ----------------
function Documents({ c, onChange }: { c: Collection; onChange: () => void }) {
  const [drag, setDrag] = useState(false);
  const [viewing, setViewing] = useState<string | null>(null);
  const [opts] = useData<Options>("/api/rag/options");
  const [template, setTemplate] = useState(c.chunk_template || "general");
  const [tags, setTags] = useState("");
  const [urls, setUrls] = useState("");
  const [sitemap, setSitemap] = useState("");
  const [status] = useData<any>("/api/models/inference/status", 5000);
  const { track, notify } = useToast();
  const { confirm, prompt } = useDialog();
  const upload = async (files: FileList | File[]) => {
    const fd = new FormData();
    for (const f of Array.from(files)) fd.append("files", f);
    fd.append("template", template);
    if (tags.trim()) fd.append("tags", tags);
    try { track(await api.form(`/api/rag/collections/${c.id}/documents`, fd), onChange); } catch (e: any) { notify(e.message, "err"); }
  };
  const addUrls = async () => {
    try {
      const t = await api.post<any>(`/api/rag/collections/${c.id}/urls`, {
        urls: urls.split(/\s+/).filter(Boolean), sitemap: sitemap || null, template, tags: tags.split(",").map((x) => x.trim()).filter(Boolean),
      });
      track(t, (done) => done.result?.ingest_task && track({ ...t, id: done.result.ingest_task, title: `Index pages into ${c.name}` }, onChange));
      setUrls(""); setSitemap("");
    } catch (e: any) { notify(e.message, "err"); }
  };
  const run = async (what: "enrich" | "summaries" | "images") => {
    if (!status?.loaded) return notify("Load a model first (Playground or Models): it's used to generate the text", "err");
    const ok = await confirm(
      what === "enrich" ? "Generate questions per chunk" : what === "images" ? "Describe PDF images" : "Build hierarchical summaries",
      what === "images"
        ? `The images of the indexed PDFs are extracted and the loaded vision model (${status.loaded.name}) describes them. Each description is indexed as a searchable chunk and the image is shown as a source. The PDF must have been uploaded with this version (the original is kept).`
        : what === "enrich"
        ? `The loaded model (${status.loaded.name}) will write 3 questions and keywords for each chunk not yet enriched. They're indexed as their own vectors and in BM25, and they serve as pairs to fine-tune embeddings. Takes ~2–6 s per chunk (${c.num_chunks} in total). Measure the effect with Benchmark: with small models (<7B) search may not improve.`
        : `The loaded model (${status.loaded.name}) will summarize groups of chunks and each whole document. This lets it answer global questions (“what is … about?”, “summarize…”).`,
      { label: "Start", danger: false });
    if (!ok) return;
    try { track(await api.post(`/api/rag/collections/${c.id}/${what}`, {}), onChange); } catch (e: any) { notify(e.message, "err"); }
  };
  const editTags = async (doc: string) => {
    const cur = c.doc_meta?.[doc]?.tags ?? [];
    const v = await prompt(`Tags for “${doc}”`, { message: "Comma-separated. They're used to filter search (Playground, API, MCP).", value: cur.join(", ") });
    if (v === null) return;
    await api.patch(`/api/rag/collections/${c.id}/documents`, { filename: doc, tags: v.split(",").map((x) => x.trim()).filter(Boolean) });
    onChange();
  };
  return (
    <div className="card">
      <label
        className="card empty" style={{ borderStyle: "dashed", borderColor: drag ? "var(--accent)" : undefined, cursor: "pointer", display: "block" }}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); upload(e.dataTransfer.files); }}>
        <Upload size={18} /><div>Drop documents or click — PDF (with OCR if scanned), DOCX, PPTX, XLSX, MD, TXT, HTML, CSV</div>
        <div className="sub" style={{ fontSize: 12 }}>They're split by section (headings) and each chunk is indexed with its “document › section” path</div>
        <input type="file" multiple hidden accept=".pdf,.docx,.pptx,.xlsx,.md,.txt,.html,.htm,.csv,.json" onChange={(e) => e.target.files && upload(e.target.files)} />
      </label>
      <div className="grid g2" style={{ marginTop: 10 }}>
        <Field label="Chunking template" hint={opts?.templates[template]}>
          <select value={template} onChange={(e) => setTemplate(e.target.value)}>
            {Object.entries(opts?.templates ?? { general: "" }).map(([k, d]) => <option key={k} value={k}>{k} — {d}</option>)}
          </select>
        </Field>
        <Field label="Tags for what you upload" hint="Comma-separated (e.g. “legal, 2025”)"><input value={tags} onChange={(e) => setTags(e.target.value)} /></Field>
      </div>
      <details style={{ marginTop: 10 }}>
        <summary style={{ cursor: "pointer" }}><Globe size={13} /> <b>Add web pages</b> <span className="sub">— single URLs or a whole sitemap</span></summary>
        <div className="stack" style={{ marginTop: 8 }}>
          <textarea rows={3} value={urls} onChange={(e) => setUrls(e.target.value)} placeholder={"https://example.com/page-1\nhttps://example.com/manual.pdf"} />
          <div className="row" style={{ flexWrap: "nowrap" }}>
            <input value={sitemap} onChange={(e) => setSitemap(e.target.value)} placeholder="https://example.com/sitemap.xml (optional, up to 30 pages)" />
            <button className="primary" disabled={!urls.trim() && !sitemap.trim()} onClick={addUrls}>Download and index</button>
          </div>
        </div>
      </details>
      <div className="row" style={{ marginTop: 12 }}>
        <button onClick={() => run("enrich")} title="The LLM generates questions per chunk"><Sparkles size={14} /> Generate questions per chunk</button>
        <button onClick={() => run("summaries")} title="Summaries per group and per document (RAPTOR-style)"><Layers size={14} /> Build hierarchical summaries</button>
        <button onClick={() => run("images")} title="Needs a vision model loaded (Qwen-VL…)" disabled={!status?.loaded?.vision}><ImageIcon size={14} /> Describe PDF images</button>
        <span className="sub" style={{ fontSize: 12 }}>{status?.loaded ? `will use ${status.loaded.name}${status.loaded.vision ? " (with vision)" : ""}` : "they need a loaded model"}</span>
      </div>
      {c.documents.length > 0 && (
        <table style={{ marginTop: 10 }}><tbody>
          {c.documents.map((d) => (
            <tr key={d}><td><FileText size={13} /> <span title={d}>{prettyDoc(d)}</span>
                <div className="row" style={{ gap: 4, marginTop: 2 }}>
                  {c.doc_meta?.[d]?.template && c.doc_meta[d].template !== "general" && <span className="badge">{c.doc_meta[d].template}</span>}
                  {(c.doc_meta?.[d]?.tags ?? []).map((t) => <span key={t} className="badge accent">{t}</span>)}
                  {c.originals && c.originals[d] === false && (
                    <span className="badge" title="The original file isn't kept: it can be searched and answered from as usual, but not re-chunked with another template, re-parsed or have its images extracted. Upload it again if you need that.">no original</span>
                  )}
                </div></td>
              <td style={{ textAlign: "right", whiteSpace: "nowrap" }}>
                <button className="sm ghost" onClick={() => editTags(d)} title="Tags"><Tag size={13} /></button>
                <button className="sm ghost" onClick={() => setViewing(d)}><Eye size={13} /> Chunks</button>
                <button className="sm ghost danger" onClick={async () => (await confirm(`Remove “${d}” from the collection?`, undefined, { label: "Remove" })) && api.del(`/api/rag/collections/${c.id}/documents?filename=${encodeURIComponent(d)}`).then(onChange)}><Trash2 size={13} /></button>
              </td></tr>
          ))}
        </tbody></table>
      )}
      {viewing && <ChunkViewer c={c} source={viewing} onClose={() => { setViewing(null); onChange(); }} />}
    </div>
  );
}

function ChunkViewer({ c, source, onClose }: { c: Collection; source: string; onClose: () => void }) {
  const [page, setPage] = useState(0);
  const [data, reload] = useData<{ total: number; chunks: any[] }>(`/api/rag/collections/${c.id}/chunks?source=${encodeURIComponent(source)}&offset=${page * 20}&limit=20`);
  const [editing, setEditing] = useState<{ id: string; text: string } | null>(null);
  const { notify } = useToast();
  const save = async () => {
    try { await api.put(`/api/rag/collections/${c.id}/chunks/${editing!.id}`, { text: editing!.text }); setEditing(null); reload(); notify("Chunk updated and re-indexed"); } catch (e: any) { notify(e.message, "err"); }
  };
  return (
    <Modal title={`${prettyDoc(source)} — ${data?.total ?? "…"} chunks`} onClose={onClose} wide>
      <div className="stack">
        {data?.chunks.map((ch) => (
          <div key={ch.id} className="source">
            <div className="row between">
              <span className="sub" style={{ fontSize: 12 }}>{ch.level ? <span className="badge warn">summary L{ch.level}</span> : `#${ch.chunk}`}{ch.page ? ` · p. ${ch.page}` : ""}{ch.section ? ` · ${ch.section}` : ""} · {ch.text.length} chars{ch.edited ? " · edited" : ""}</span>
              <div className="row" style={{ gap: 2 }}>
                {editing?.id === ch.id
                  ? <><button className="sm primary" onClick={save}><Save size={12} /></button><button className="sm ghost" onClick={() => setEditing(null)}><X size={12} /></button></>
                  : <button className="sm ghost" onClick={() => setEditing({ id: ch.id, text: ch.text })}><Pencil size={12} /></button>}
                <button className="sm ghost danger" onClick={() => api.del(`/api/rag/collections/${c.id}/chunks/${ch.id}`).then(reload)}><Trash2 size={12} /></button>
              </div>
            </div>
            {editing?.id === ch.id
              ? <textarea rows={8} value={editing?.text ?? ""} onChange={(e) => setEditing({ id: ch.id, text: e.target.value })} style={{ marginTop: 6 }} />
              : <div style={{ whiteSpace: "pre-wrap", fontSize: 13, marginTop: 4 }}>{ch.text}</div>}
            {ch.questions && <div className="sub" style={{ fontSize: 12, marginTop: 6, borderTop: "1px dashed var(--border)", paddingTop: 4 }}>
              <Sparkles size={11} /> {ch.questions.split("\n").join(" · ")}{ch.keywords ? <> · <i>{ch.keywords}</i></> : null}</div>}
          </div>
        ))}
        <div className="row" style={{ justifyContent: "center" }}>
          <button className="sm" disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</button>
          <span className="sub">{page * 20 + 1}–{Math.min((page + 1) * 20, data?.total ?? 0)}</span>
          <button className="sm" disabled={(page + 1) * 20 >= (data?.total ?? 0)} onClick={() => setPage(page + 1)}>Next</button>
        </div>
      </div>
    </Modal>
  );
}

// ---------------- Search tester ----------------
function SearchTester({ c }: { c: Collection }) {
  const [q, setQ] = useState("");
  const [k, setK] = useState(c.top_k);
  const [mode, setMode] = useState<string>(c.search_mode);
  const [rerank, setRerank] = useState(!!c.reranker);
  const [qmode, setQmode] = useState(c.query_mode || "simple");
  const [ftags, setFtags] = useState<string[]>([]);
  const [opts] = useData<Options>("/api/rag/options");
  const [res, setRes] = useState<{ hits: any[]; ms: number } | null>(null);
  const [busy, setBusy] = useState(false);
  const { notify } = useToast();
  const search = async () => {
    setBusy(true);
    try {
      setRes(await api.post(`/api/rag/collections/${c.id}/search`, { query: q, k, mode, reranker: rerank ? c.reranker || DEFAULT_RERANKER : "", min_score: 0, query_mode: qmode, tags: ftags.length ? ftags : null }));
    } catch (e: any) { notify(e.message, "err"); } finally { setBusy(false); }
  };
  return (
    <div className="card stack">
      <div className="row" style={{ flexWrap: "nowrap" }}>
        <input value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && search()} placeholder="Type a question…" />
        <button className="primary" onClick={search} disabled={!q || busy}>{busy ? <Spinner /> : <Search size={14} />}</button>
      </div>
      <div className="row" style={{ gap: 14 }}>
        <select value={mode} onChange={(e) => setMode(e.target.value)} style={{ width: "auto" }}>
          {Object.entries(MODES).map(([k, l]) => <option key={k} value={k}>{l}</option>)}
        </select>
        <label className="check"><input type="checkbox" checked={rerank} onChange={(e) => setRerank(e.target.checked)} /> Reranking</label>
        <select value={qmode} onChange={(e) => setQmode(e.target.value)} style={{ width: "auto" }} title={opts?.query_modes[qmode]}>
          {Object.keys(opts?.query_modes ?? { simple: "" }).map((m) => <option key={m} value={m}>query: {m}</option>)}
        </select>
        <label className="check">top-k <input type="number" value={k} onChange={(e) => setK(+e.target.value)} style={{ width: 60 }} /></label>
        {res && <span className="sub">{res.hits.length} results · {res.ms} ms</span>}
      </div>
      {allTags(c).length > 0 && (
        <div className="row" style={{ gap: 6 }}><span className="sub">Filter:</span>
          {allTags(c).map((t) => <button key={t} className={`sm ${ftags.includes(t) ? "primary" : ""}`} onClick={() => setFtags(ftags.includes(t) ? ftags.filter((x) => x !== t) : [...ftags, t])}><Tag size={11} /> {t}</button>)}
        </div>
      )}
      {qmode !== "simple" && <div className="sub" style={{ fontSize: 12 }}>{opts?.query_modes[qmode]} — needs a loaded model.</div>}
      {(res?.hits[0]?.rewritten_queries || res?.hits[0]?.agent_queries) && (
        <div className="source sub" style={{ fontSize: 12 }}><b>{res!.hits[0].agent_queries ? "Agent searches" : "Generated queries"}:</b>
          <ol style={{ margin: "4px 0 0", paddingLeft: 18 }}>{(res!.hits[0].agent_queries ?? res!.hits[0].rewritten_queries).map((x: string, i: number) => <li key={i}>{x}</li>)}</ol></div>
      )}
      {res && !res.hits.length && <div className="empty">No results</div>}
      {res?.hits.map((h, i) => {
        const pass = h.score >= (c.min_score || 0);
        return (
          <div key={h.id} className="source" style={{ opacity: pass ? 1 : 0.55 }}>
            <div className="row between">
              <b>[{i + 1}] {prettyDoc(h.source)}{h.page ? ` · p. ${h.page}` : ""}</b>
              <div className="row" style={{ gap: 4 }}>
                {h.rerank_score != null && <span className="badge ok" title="Reranker score (0–1): what the threshold uses">rerank {h.rerank_score.toFixed(3)}</span>}
                <span className="badge accent" title="Cosine similarity with the question">cos {h.vector_score?.toFixed(3)}</span>
                {h.bm25_rank && <span className="badge" title="Position in BM25 (words)">bm25 #{h.bm25_rank}</span>}
                {h.vector_rank && <span className="badge" title="Position by vectors">vec #{h.vector_rank}</span>}
                {!pass && <span className="badge err" title={`Below the ${c.min_score} threshold`}>below threshold</span>}
              </div>
            </div>
            {h.section && <div className="sub" style={{ fontSize: 12 }}>{h.section}</div>}
            {h.image && <img src={`/api/rag/collections/${c.id}/images/${h.image}`} alt="" style={{ maxHeight: 180, borderRadius: 8, marginTop: 6 }} />}
            <div style={{ whiteSpace: "pre-wrap", fontSize: 13, marginTop: 4 }}>{h.text}</div>
          </div>
        );
      })}
    </div>
  );
}

// ---------------- Retrieval benchmark ----------------
function parseQs(text: string) {
  return text.split("\n").filter((l) => l.trim()).map((l) => {
    const [question, kw] = l.split("|").map((s) => s.trim());
    return { question, keywords: kw ? kw.split(",").map((s) => s.trim()).filter(Boolean) : [] };
  });
}

function Benchmark({ c }: { c: Collection }) {
  const [text, setText] = useState("");
  const [k, setK] = useState(c.top_k);
  const [res, setRes] = useState<any[] | null>(null);
  const [busy, setBusy] = useState(false);
  const { notify } = useToast();
  const qs = parseQs(text);
  const run = async () => {
    setBusy(true);
    try {
      const t = await api.post<any>(`/api/rag/collections/${c.id}/benchmark`, { questions: qs, k });
      for (;;) {
        const s = await api.get<any>(`/api/tasks/${t.id}`);
        if (s.status === "completed") { setRes(s.result); break; }
        if (s.status === "failed") throw new Error(s.error?.split("\n")[0]);
        await new Promise((r) => setTimeout(r, 1000));
      }
    } catch (e: any) { notify(e.message, "err"); } finally { setBusy(false); }
  };
  const best = res ? Math.max(...res.map((r) => r.mrr)) : 0;
  return (
    <div className="card stack">
      <div className="sub">Compares vectors, BM25, hybrid and hybrid + reranking on your own questions. A hit = some chunk in the top-k contains all the keywords.</div>
      <Field label={`Questions (${qs.length})`} hint="One per line: question | keyword 1, keyword 2">
        <textarea rows={7} className="mono" value={text} onChange={(e) => setText(e.target.value)}
          placeholder={"Which volcano destroyed Pompeii? | Vesuvius\nHow many vacation days are there? | 23"} />
      </Field>
      <div className="row">
        <label className="check">top-k <input type="number" value={k} onChange={(e) => setK(+e.target.value)} style={{ width: 60 }} /></label>
        <div className="spacer" />
        <button className="primary" disabled={busy || !qs.length || qs.some((q) => !q.keywords.length)} onClick={run}>{busy ? <Spinner /> : null} Run benchmark</button>
      </div>
      {qs.some((q) => !q.keywords.length) && text && <div className="sub">Each line needs at least one keyword after “|”.</div>}
      {res && (
        <>
          <table>
            <thead><tr><th>Configuration</th><th title="Share of questions with the answer in the top-k">Hit@{k}</th><th title="Mean Reciprocal Rank: 1 = the answer is always in 1st place">MRR</th><th>Mean latency</th></tr></thead>
            <tbody>{res.map((r) => (
              <tr key={r.label} className={r.mrr === best ? "selected" : ""}><td><b>{r.label}</b>{r.mrr === best && " ★"}</td>
                <td>{Math.round(r.hit_rate * 100)}%</td><td>{r.mrr.toFixed(3)}</td><td>{r.latency_ms} ms</td></tr>
            ))}</tbody>
          </table>
          <h3 style={{ marginTop: 8 }}>Position of the answer per question</h3>
          <table>
            <thead><tr><th>Question</th>{res.map((r) => <th key={r.label} style={{ textAlign: "center" }}>{r.label}</th>)}</tr></thead>
            <tbody>{res[0].per_question.map((pq: any, i: number) => (
              <tr key={i}><td>{pq.question}</td>
                {res.map((r) => { const rank = r.per_question[i].rank; return (
                  <td key={r.label} style={{ textAlign: "center", color: rank === 1 ? "var(--ok)" : rank ? "var(--warn)" : "var(--err)" }}>{rank ? `#${rank}` : "✗"}</td>); })}
              </tr>
            ))}</tbody>
          </table>
        </>
      )}
    </div>
  );
}

// ---------------- Settings ----------------
function Settings({ c, onChange }: { c: Collection; onChange: () => void }) {
  const [rerankers] = useData<any[]>("/api/rag/rerankers");
  const [opts] = useData<Options>("/api/rag/options");
  const [s, setS] = useState<{ search_mode: string; reranker: string; top_k: number; min_score: number; fallback: string; chunk_template: string; query_mode: string; parser: string }>(
    { search_mode: c.search_mode, reranker: c.reranker ?? "", top_k: c.top_k, min_score: c.min_score, fallback: c.fallback, chunk_template: c.chunk_template || "general", query_mode: c.query_mode || "simple", parser: (c as any).parser || "auto" });
  const { notify } = useToast();
  const save = async () => {
    try { await api.patch(`/api/rag/collections/${c.id}`, s); notify("Settings saved"); onChange(); } catch (e: any) { notify(e.message, "err"); }
  };
  return (
    <div className="card stack">
      <div className="grid g2">
        <Adv><Field label="Search mode" hint="Hybrid combines meaning (vectors) and exact words (BM25): better with proper names, codes and figures">
          <select value={s.search_mode} onChange={(e) => setS({ ...s, search_mode: e.target.value })}>
            {Object.entries(MODES).map(([k, l]) => <option key={k} value={k}>{l}</option>)}
          </select>
        </Field>
        <Field label="Reranker" hint="Reorders ~20 candidates with a cross-encoder. More precise (+~300 ms) and gives reliable scores for the threshold">
          <select value={s.reranker} onChange={(e) => setS({ ...s, reranker: e.target.value, min_score: e.target.value && !s.min_score ? 0.02 : e.target.value ? s.min_score : 0 })}>
            <option value="">Sin reranking</option>
            {rerankers?.map((r) => <option key={r.name} value={r.name}>{r.name} — {r.note}</option>)}
          </select>
        </Field>
        </Adv>
        <Field label="Chunks per answer (top-k)"><input type="number" min={1} max={20} value={s.top_k} onChange={(e) => setS({ ...s, top_k: +e.target.value })} /></Field>
        <Adv><Field label={`Relevance threshold: ${s.min_score}`} hint={s.reranker ? "On the reranker score. 0.02 drops the irrelevant without losing paraphrases" : "⚠ Without a reranker the threshold applies to cosine, which barely discriminates (almost everything scores 0.7–0.9)"}>
          <input type="range" min={0} max={s.reranker ? 0.5 : 1} step={0.01} value={s.min_score} onChange={(e) => setS({ ...s, min_score: +e.target.value })} />
        </Field>
        <Field label="Query mode" hint={opts?.query_modes[s.query_mode]}>
          <select value={s.query_mode} onChange={(e) => setS({ ...s, query_mode: e.target.value })}>
            {Object.keys(opts?.query_modes ?? { simple: "" }).map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        </Field>
        <Field label="Default chunking template" hint="For new documents; already indexed ones don't change">
          <select value={s.chunk_template} onChange={(e) => setS({ ...s, chunk_template: e.target.value })}>
            {Object.entries(opts?.templates ?? { general: "" }).map(([k, d]) => <option key={k} value={k}>{k} — {d}</option>)}
          </select>
        </Field>
        <Field label="Document parser" hint="Docling analyzes layout and table structure with vision models (slower, better with complex PDFs). PPTX/XLSX always use Docling.">
          <select value={s.parser} onChange={(e) => setS({ ...s, parser: e.target.value })}>
            <option value="auto">Fast (pymupdf4llm + Tesseract OCR)</option>
            <option value="docling">Docling (layout and complex tables)</option>
          </select>
        </Field>
        </Adv>
        <Field label="If no chunk passes the threshold" hint="Keeps the model from getting stuck (or making things up) with irrelevant context">
          <select value={s.fallback} onChange={(e) => setS({ ...s, fallback: e.target.value })}>
            <option value="no_context">Answer without RAG (from its own knowledge)</option>
            <option value="refuse">Say it's not in the documents</option>
          </select>
        </Field>
      </div>
      <SimpleNote>{s.search_mode} search{s.reranker ? " with a reranker" : ""}, query mode “{s.query_mode}” and threshold {s.min_score}: the values that did best in our benchmark (1.00 hit rate, 0.95 MRR).</SimpleNote>
      <div className="sub" style={{ fontSize: 12 }}>The chunk size ({c.chunk_size}/{c.chunk_overlap}) can't be changed without uploading the documents again.</div>
      <Reindex c={c} onChange={onChange} />
      <div className="row"><div className="spacer" /><button className="primary" onClick={save}><Save size={14} /> Save</button></div>
      <ChunkPreview size={c.chunk_size} overlap={c.chunk_overlap} />
    </div>
  );
}

function ChunkPreview({ size: s0, overlap: o0 }: { size: number; overlap: number }) {
  const [text, setText] = useState("");
  const [size, setSize] = useState(s0);
  const [overlap, setOverlap] = useState(o0);
  const [chunks, setChunks] = useState<string[] | null>(null);
  return (
    <details>
      <summary style={{ cursor: "pointer" }}><b>Chunking lab</b> <span className="sub">— see how a text gets split</span></summary>
      <div className="stack" style={{ marginTop: 12 }}>
        <textarea rows={5} value={text} onChange={(e) => setText(e.target.value)} placeholder="Paste text here…" />
        <div className="row">
          <Field label="Size"><input type="number" value={size} onChange={(e) => setSize(+e.target.value)} /></Field>
          <Field label="Overlap"><input type="number" value={overlap} onChange={(e) => setOverlap(+e.target.value)} /></Field>
          <button style={{ alignSelf: "flex-end" }} onClick={async () => setChunks(await api.post("/api/rag/chunk-preview", { text, chunk_size: size, chunk_overlap: overlap }))}>Split</button>
        </div>
        {chunks?.map((c, i) => <div key={i} className="source"><span className="badge">{i + 1} · {c.length} car.</span><div style={{ whiteSpace: "pre-wrap", marginTop: 4 }}>{c}</div></div>)}
      </div>
    </details>
  );
}

function CreateModal({ onClose, onDone }: { onClose: () => void; onDone: (c: Collection) => void }) {
  const [rerankers] = useData<any[]>("/api/rag/rerankers");
  const [name, setName] = useState("");
  const [model, setModel] = useState("");
  const [size, setSize] = useState(800);
  const [overlap, setOverlap] = useState(120);
  const [reranker, setReranker] = useState(DEFAULT_RERANKER);
  const { notify } = useToast();
  const go = async () => {
    try {
      onDone(await api.post("/api/rag/collections", {
        name, embed_model: model || null, chunk_size: size, chunk_overlap: overlap, search_mode: "hybrid",
        reranker: reranker || null, min_score: reranker ? 0.02 : 0,
      }));
    } catch (e: any) { notify(e.message, "err"); }
  };
  return (
    <Modal title="New collection" onClose={onClose}>
      <div className="stack">
        <Field label="Name"><input value={name} onChange={(e) => setName(e.target.value)} /></Field>
        <Field label="Embedding model" hint="You can change it later, but that means re-indexing the whole collection">
          <EmbedPicker value={model} onChange={setModel} />
        </Field>
        <Adv>
          <Field label="Reranker" hint="Recommended: it improves precision a lot and lets you filter out irrelevant context">
            <select value={reranker} onChange={(e) => setReranker(e.target.value)}>
              <option value="">Sin reranking</option>
              {rerankers?.map((r) => <option key={r.name} value={r.name}>{r.name} — {r.note}</option>)}
            </select>
          </Field>
          <div className="grid g2">
            <Field label="Chunk size (characters)"><input type="number" value={size} onChange={(e) => setSize(+e.target.value)} /></Field>
            <Field label="Overlap (characters)"><input type="number" value={overlap} onChange={(e) => setOverlap(+e.target.value)} /></Field>
          </div>
        </Adv>
        <SimpleNote>It's created with hybrid search, a reranker and 800-character chunks, which is what worked best for us. You can change everything later in Settings.</SimpleNote>
        <div className="row"><div className="spacer" /><button className="primary" disabled={!name} onClick={go}>Create</button></div>
      </div>
    </Modal>
  );
}

function Reindex({ c, onChange }: { c: Collection; onChange: () => void }) {
  const [m, setM] = useState(c.embed_model);
  const { track, notify } = useToast();
  const go = async () => {
    try { track(await api.post(`/api/rag/collections/${c.id}/reindex`, { embed_model: m }), onChange); } catch (e: any) { notify(e.message, "err"); }
  };
  return (
    <div className="row source" style={{ padding: "8px 10px", flexWrap: "nowrap" }}>
      <span className="sub" style={{ whiteSpace: "nowrap" }}>Embedding model:</span>
      <EmbedPicker value={m} onChange={setM} current={c.embed_model} />
      <button className="sm" disabled={m === c.embed_model} onClick={go}>Re-index</button>
    </div>
  );
}


/** Embedding model picker that can download from the catalog.
  *  Changing the model forces a re-index: old vectors aren't comparable with new ones. */
function EmbedPicker({ value, onChange, current }: { value: string; onChange: (v: string) => void; current?: string }) {
  const [models, reload] = useData<any[]>("/api/rag/embed-models");
  const { track, notify } = useToast();
  const sel = models?.find((m) => m.name === value);
  const pending = sel && !sel.downloaded;
  const download = async () => {
    try { track(await api.post(`/api/rag/embed-models/download`, { name: value }), () => reload()); }
    catch (e: any) { notify(e.message, "err"); }
  };
  return (
    <div className="stack" style={{ gap: 4 }}>
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {current && !models?.some((x) => x.name === current) && <option value={current}>{current}</option>}
        {!current && <option value="">Default (multilingual-e5-small)</option>}
        {models?.map((x) => (
          <option key={x.name} value={x.name}>
            {x.tuned ? "★ " : x.downloaded ? "" : "⬇ "}{x.name.split("/").slice(-1)[0]}
            {x.dim ? ` · ${x.dim}d` : ""}{x.params ? ` · ${x.params}` : ""}{x.note ? ` — ${x.note}` : ""}
          </option>
        ))}
      </select>
      {pending && (
        <div className="row" style={{ gap: 6, alignItems: "center" }}>
          <span className="sub">Not downloaded yet.</span>
          <button className="sm" onClick={download}><Download size={12} /> Download now</button>
        </div>
      )}
      {sel?.dim && <div className="sub">{sel.dim}-dimension vectors. The ones marked ⬇ download automatically the first time they're used.</div>}
    </div>
  );
}

// ---------------- Knowledge graph (GraphRAG) ----------------
function Graph({ c }: { c: Collection }) {
  const [g, reload] = useData<any>(`/api/rag/collections/${c.id}/graph?limit=250`);
  const [status] = useData<any>("/api/models/inference/status", 5000);
  const [q, setQ] = useState("");
  const [res, setRes] = useState<any>(null);
  const [maxChunks, setMaxChunks] = useState(60);
  const [src, setSrc] = useState("");
  const { track, notify } = useToast();
  const { confirm } = useDialog();
  const [helper, setHelper] = useState<number | null>(null);

  const build = async () => {
    if (!status?.loaded && !helper) return notify("Load a model or pick a helper: an LLM extracts the graph", "err");
    const ok = await confirm("Build the knowledge graph",
      `The model (${helper ? "external helper" : status.loaded.name}) will extract entities and relations from ${maxChunks || "all the"} chunks${src ? ` of ${src}` : ""}, and summarize the communities it finds. Takes ~5-10 s per chunk. Build it again to redo it.`,
      { label: "Build", danger: false });
    if (!ok) return;
    try { track(await api.post(`/api/rag/collections/${c.id}/graph`, { max_chunks: maxChunks || null, sources: src ? [src] : null, helper }), reload); } catch (e: any) { notify(e.message, "err"); }
  };
  const search = async () => {
    try { setRes(await api.get<any>(`/api/rag/collections/${c.id}/graph/query?q=${encodeURIComponent(q)}`)); } catch (e: any) { notify(e.message, "err"); }
  };

  return (
    <div className="card stack">
      <div className="sub">
        The graph belongs <b>only to “{c.name}”</b>: each collection has its own and they don't mix.
        {g?.stats && <> It was built from <b>{g.stats.chunks ?? "?"} of the {c.num_chunks} chunks</b> in
        this collection{(g.stats.chunks ?? 0) < c.num_chunks ? " — the rest isn't in the graph; raise “Max chunks” and rebuild if you want to cover them" : ""}.</>}
      </div>
      <div className="sub">The LLM extracts <b>entities and relations</b> from each chunk and joins them into a graph. It answers relational or multi-hop questions better (“how are X and Y related?”, “who took part in…?”), where the information is spread across chunks. To use it when answering, set the query mode to <b>graph</b> (Settings tab).</div>
      <div className="row">
        <Field label="Max chunks" hint="0 = all"><input type="number" value={maxChunks} onChange={(e) => setMaxChunks(+e.target.value)} style={{ width: 90 }} /></Field>
        <Field label="Only one document (optional)">
          <select value={src} onChange={(e) => setSrc(e.target.value)}><option value="">all</option>{c.documents.map((d) => <option key={d} value={d}>{prettyDoc(d)}</option>)}</select>
        </Field>
        <HelperPicker value={helper} onChange={setHelper}
          hint="Extracting entities is where a small model makes the most mistakes" />
        <button className="primary" style={{ alignSelf: "flex-end" }} onClick={build}>{g?.stats ? "Rebuild" : "Build"} graph</button>
      </div>
      {!g?.stats ? <div className="empty">There's no graph for this collection yet.</div> : (
        <>
          <div className="row" style={{ gap: 16 }}>
            <span className="sub">{g.stats.entities} entities · {g.stats.edges} relations · {g.stats.communities} topics · {g.stats.chunks} chunks analyzed{g.stats.failed ? ` · ${g.stats.failed} extraction failures` : ""}</span>
          </div>
          <div className="row" style={{ flexWrap: "nowrap" }}>
            <input value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && search()} placeholder="Try a question against the graph…" />
            <button className="primary" onClick={search} disabled={!q}><Search size={14} /></button>
          </div>
          {res?.context && (
            <div className="source" style={{ fontSize: 13 }}>
              <div className="sub" style={{ fontSize: 12 }}>Seed entities: <b>{res.seeds.join(", ")}</b> · neighbours: {res.neighbors.slice(0, 6).join(", ")} · {res.relations} relations{res.communities?.length ? ` · topics: ${res.communities.join(", ")}` : ""}</div>
              <div style={{ whiteSpace: "pre-wrap", marginTop: 6 }}>{res.context}</div>
            </div>
          )}
          {res && !res.context && <div className="sub">No entity in the graph matches that question.</div>}
          <div style={{ border: "1px solid var(--border)", borderRadius: 8, overflow: "hidden" }}>
            <GraphView nodes={g.nodes} links={g.links} highlight={res?.seeds} />
          </div>
          <div className="sub" style={{ fontSize: 12 }}>Size = number of connections · colour = topic (community) · hover to see the description and neighbours · red = entities in your query</div>
          {g.communities?.length > 0 && (
            <details>
              <summary style={{ cursor: "pointer" }}><b>Detected topics ({g.communities.length})</b></summary>
              <div className="stack" style={{ marginTop: 8 }}>
                {g.communities.map((cm: any) => (
                  <div key={cm.id} className="source"><b>{cm.title}</b> <span className="badge">{cm.size} entities</span>
                    {cm.summary && <div className="sub" style={{ marginTop: 4 }}>{cm.summary}</div>}</div>
                ))}
              </div>
            </details>
          )}
        </>
      )}
    </div>
  );
}

/** The same questions against several collections.
  *  It compares RETRIEVAL only: the model isn't involved, so it really measures whether changing the
  *  chunking, the reranker or the embeddings helped, without the answer muddying it. */
function CompararColecciones({ cols }: { cols: Collection[] }) {
  const [expanded, setExpanded] = useState(false);
  const [sel, setSel] = useState<number[]>([]);
  const [txt, setTxt] = useState("");
  const [res, setRes] = useState<any>(null);
  const { track, notify } = useToast();
  const questions = txt.split("\n").map((l) => l.trim()).filter(Boolean);

  const go = async () => {
    try {
      const t = await api.post<any>("/api/rag/compare", { collection_ids: sel, questions: questions });
      track(t, (done) => setRes(done.result));
    } catch (e: any) { notify(e.message, "err"); }
  };

  if (!expanded) {
    return (
      <div className="card" style={{ padding: "8px 12px" }}>
        <button className="ghost sm" onClick={() => setExpanded(true)}>
          <Search size={13} /> Compare collections with the same questions
        </button>
      </div>
    );
  }
  return (
    <div className="card stack">
      <div className="row between">
        <b><Search size={14} /> Compare retrieval</b>
        <button className="sm ghost" onClick={() => setExpanded(false)}>Close</button>
      </div>
      <div className="sub">
        Runs the same questions against several collections and shows what each one retrieves. The
        model isn't involved: it measures search, which is what you change when you touch the chunking, the
        reranker or the embeddings.
      </div>
      <Field label="Collections">
        <div className="row" style={{ gap: 14, flexWrap: "wrap" }}>
          {cols.map((c) => (
            <Check key={c.id} label={`${c.name} (${c.num_chunks})`} checked={sel.includes(c.id)}
              onChange={(v) => setSel(v ? [...sel, c.id] : sel.filter((x) => x !== c.id))} />
          ))}
        </div>
      </Field>
      <Field label={`Questions (${questions.length})`} hint="One per line">
        <textarea rows={4} value={txt} onChange={(e) => setTxt(e.target.value)}
          placeholder={"What is the return period?\nWhat warranty do the products have?"} />
      </Field>
      <div className="row end">
        <button className="primary" disabled={sel.length < 2 || !questions.length} onClick={go}>
          Compare {sel.length} collections
        </button>
      </div>
      {res && (
        <table style={{ marginTop: 8 }}>
          <thead><tr><th>Collection</th><th>Settings</th><th>Chunks</th><th>No results</th><th>Time</th></tr></thead>
          <tbody>
            {res.colecciones.map((c: any) => (
              <tr key={c.id}>
                <td><b>{c.coleccion}</b></td>
                <td className="sub">{c.ajustes.busqueda}{c.ajustes.reranker ? " + rerank" : ""} · k={c.ajustes.top_k} · chunking {c.ajustes.troceo}</td>
                <td>{c.fragmentos_media}</td>
                <td style={{ color: c.sin_resultados ? "var(--err)" : undefined }}>{c.sin_resultados}</td>
                <td>{c.segundos_media}s</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

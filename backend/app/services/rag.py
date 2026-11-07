"""RAG: structured extraction, chunking by section, hybrid search (BM25 + vectors), reranking and a threshold.

Query pipeline:
    1. candidates by vectors (Chroma, cosine) and/or BM25 (in-memory index)
    2. fusion with Reciprocal Rank Fusion (hybrid mode)
    3. optional reranking with a cross-encoder (score 0–1)
    4. min_score threshold → if nothing passes, fallback (no context, or refuse)
"""
import hashlib
import io
import math
import re
import threading
import time
import unicodedata
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

import numpy as np

from ..config import settings

_client = None
_models: dict[str, object] = {}
_model_lock = threading.Lock()

DEFAULT_RAG_SYSTEM = (
    "You are an assistant that answers using ONLY the provided context. "
    "If the answer isn't in the context, say so clearly. Cite the sources with [n]."
)
REFUSE_TEXT = ("No relevant information was found in the documents for this question. "
               "Answer saying that it doesn't appear in the available documents, without making anything up.")
RRF_K = 60


def client():
    global _client
    if _client is None:
        import chromadb
        _client = chromadb.PersistentClient(path=str(settings.rag_dir / "chroma"))
    return _client


def chroma_name(collection_id: int) -> str:
    return f"llmt_{collection_id}"


def _col(collection_id: int):
    return client().get_or_create_collection(chroma_name(collection_id), metadata={"hnsw:space": "cosine"})


def _qcol(collection_id: int):
    """Auxiliary collection with one vector per generated question (metadata parent = the chunk's id)."""
    return client().get_or_create_collection(chroma_name(collection_id) + "_q", metadata={"hnsw:space": "cosine"})


def _index_questions(collection_id: int, embed_model: str, chunk_id: str, meta: dict) -> None:
    qc = _qcol(collection_id)
    qc.delete(where={"parent": chunk_id})
    qs = [q for q in (meta.get("questions") or "").splitlines() if q.strip()]
    if qs:
        qc.add(ids=[f"{chunk_id}::{i}" for i in range(len(qs))], documents=qs,
               metadatas=[{"parent": chunk_id, "source": meta.get("source", "")} for _ in qs],
               embeddings=embed(embed_model, qs, "query"))


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def embedder(name: str):
    with _model_lock:
        if name not in _models:
            from sentence_transformers import SentenceTransformer
            _models[name] = SentenceTransformer(name, device=_device())
        return _models[name]


def reranker_model(name: str):
    key = f"rerank:{name}"
    with _model_lock:
        if key not in _models:
            from sentence_transformers import CrossEncoder
            _models[key] = CrossEncoder(name, device=_device(), max_length=512)
        return _models[key]


def loaded_model_names() -> list[str]:
    """Which embedders and rerankers are in memory right now."""
    with _model_lock:
        return sorted(k.replace("rr:", "reranker ") for k in _models)


def unload_models() -> float:
    """Frees embedders and rerankers from VRAM.

    `clear()` only drops the dictionary's reference: torch modules have cycles, so
    without running the garbage collector before emptying the cache the tensors stay alive and VRAM doesn't go down.
    Returns the GB freed so it can be checked."""
    import gc

    def used():
        try:
            import torch
            return torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0
        except Exception:  # noqa: BLE001
            return 0.0

    before = used()
    with _model_lock:
        _models.clear()
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:  # noqa: BLE001
        pass
    return round(max(0.0, before - used()), 2)


def warmup() -> None:
    """Preloads the default embedder and the rerankers/embedders the collections use."""
    from sqlmodel import Session, select

    from ..db import RagCollection, engine
    t0 = time.time()
    with Session(engine) as s:
        cols = s.exec(select(RagCollection)).all()
    names = {settings.default_embed_model} | {c.embed_model for c in cols}
    for n in names:
        embed(n, ["warm-up"], "query")
    for r in {c.reranker for c in cols if c.reranker}:
        reranker_model(r).predict([("a", "b")], show_progress_bar=False)
    print(f"[rag] models preloaded in {time.time() - t0:.1f}s: {sorted(names)} + rerankers")


def _prefix(name: str, kind: str) -> str:
    # E5 models expect "query: " / "passage: " prefixes
    if "e5" in name.lower():
        return "query: " if kind == "query" else "passage: "
    return ""


def embed(name: str, texts: list[str], kind: str = "passage") -> list[list[float]]:
    p = _prefix(name, kind)
    vecs = embedder(name).encode([p + t for t in texts], batch_size=32, normalize_embeddings=True,
                                 show_progress_bar=False)
    return vecs.tolist()


# What used to live here now has its own modules. It's re-exported so code that already imported
# `rag.chunk_text`, `rag.extract` or `rag.fetch_url` keeps working: the split is internal.
from .rag_chunking import (  # noqa: E402,F401
    BM25, bm25_index, chunk_text, contextual, invalidate, make_chunks, normalize_tokens, split_sections,
)
from .rag_extract import _html, extract, fetch_url, sitemap_urls, url_ext  # noqa: E402,F401

# ---------------- ingestion and management ----------------
def files_dir(collection_id: int) -> Path:
    d = settings.rag_dir / "files" / str(collection_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def stored_path(collection_id: int, filename: str, ext: str | None = None) -> Path:
    """Path where the original is stored (a stable name derived from the document's name)."""
    e = ext or Path(filename.split("?")[0]).suffix.lower() or ".bin"
    return files_dir(collection_id) / (hashlib.sha1(filename.encode()).hexdigest()[:16] + e)


def ingest(collection_id: int, embed_model: str, filename: str, raw: bytes, size: int, overlap: int,
           progress=None, template: str = "general", ext: str | None = None, parser: str = "auto") -> int:
    col = _col(collection_id)
    # Keep the original so images can be extracted later
    try:
        stored_path(collection_id, filename, ext).write_bytes(raw)
    except OSError as e:
        # Without the original it can't be re-chunked or have its images extracted later. Ingestion isn't
        # aborted for this, but it has to be visible: it used to be a print among thousands of log lines.
        print(f"[rag] WARNING: couldn't save the original of {filename}: {e}. "
              "That document can't be re-chunked or re-parsed.")
    doc_hash = hashlib.sha1(filename.encode() + raw).hexdigest()[:10]
    chunks = make_chunks(filename, extract(filename, raw, ext, parser), size, overlap, template)
    if not chunks:
        raise ValueError(f"Couldn't extract text from {filename} (a scanned PDF with no text layer?)")
    col.delete(where={"source": filename})  # replaces earlier versions of the document
    _qcol(collection_id).delete(where={"source": filename})
    bs = 64
    for start in range(0, len(chunks), bs):
        batch = chunks[start:start + bs]
        col.add(
            ids=[f"{doc_hash}-{start + j}" for j in range(len(batch))],
            documents=[c for c, _ in batch],
            metadatas=[m for _, m in batch],
            embeddings=embed(embed_model, [contextual(c, m) for c, m in batch], "passage"),
        )
        if progress:
            progress(min(1.0, (start + bs) / len(chunks)))
    invalidate(collection_id)
    return len(chunks)


def reindex(coll, new_model: str, task) -> dict:
    """Recomputes the embeddings with another model. All of them are computed first (a cancellable phase with no side effects)
    and only written at the end: that way cancelling never leaves the collection halfway between two models."""
    col = _col(coll.id)
    got = col.get(include=["documents", "metadatas"])
    ids, docs, metas = got["ids"], got["documents"], got["metadatas"]
    bs = 64
    vectors: list[list[float]] = []
    for i in range(0, len(ids), bs):
        if task.cancel_requested:
            return {"cancelled": True, "chunks": 0}
        task.update(0.9 * i / max(1, len(ids)), f"Computing vectors {i}/{len(ids)}")
        vectors += embed(new_model, [contextual(d, m) for d, m in zip(docs[i:i + bs], metas[i:i + bs])])
    task.update(0.92, "Saving…")
    for i in range(0, len(ids), bs):
        col.update(ids=ids[i:i + bs], embeddings=vectors[i:i + bs])
    try:
        client().delete_collection(chroma_name(coll.id) + "_q")
    except Exception:  # noqa: BLE001
        pass
    for n, (cid, m) in enumerate(zip(ids, metas)):
        if m.get("questions"):
            task.update(0.92 + 0.08 * n / max(1, len(ids)), "Re-indexing generated questions…")
            _index_questions(coll.id, new_model, cid, m)
    invalidate(coll.id)
    return {"cancelled": False, "chunks": len(ids)}


def remove_document(collection_id: int, filename: str) -> None:
    _col(collection_id).delete(where={"source": filename})
    _qcol(collection_id).delete(where={"source": filename})
    invalidate(collection_id)


def list_chunks(collection_id: int, source: str | None, offset: int, limit: int) -> dict:
    got = _col(collection_id).get(where={"source": source} if source else None, include=["documents", "metadatas"])
    rows = sorted(({"id": i, "text": d, **m} for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])),
                  key=lambda r: (r.get("source", ""), r.get("chunk", 0)))
    return {"total": len(rows), "chunks": rows[offset:offset + limit]}


def update_chunk(collection_id: int, embed_model: str, chunk_id: str, text: str) -> dict:
    col = _col(collection_id)
    got = col.get(ids=[chunk_id], include=["metadatas"])
    if not got["ids"]:
        raise KeyError(chunk_id)
    meta = {**got["metadatas"][0], "edited": True}
    col.update(ids=[chunk_id], documents=[text], metadatas=[meta],
               embeddings=embed(embed_model, [contextual(text, meta)], "passage"))
    invalidate(collection_id)
    return {"id": chunk_id, "text": text, **meta}


def delete_chunk(collection_id: int, chunk_id: str) -> None:
    _col(collection_id).delete(ids=[chunk_id])
    _qcol(collection_id).delete(where={"parent": chunk_id})
    invalidate(collection_id)


def features(collection_id: int) -> dict:
    """Which extras the collection has (to show in the interface)."""
    try:
        col = _col(collection_id)
        got = col.get(include=["metadatas"])
        metas = got["metadatas"]
        return {"questions": sum(1 for m in metas if m.get("questions")), "summaries": sum(1 for m in metas if m.get("level")),
                "images": sum(1 for m in metas if m.get("kind") == "image"), "chunks": len(metas)}
    except Exception as e:  # noqa: BLE001
        print(f"[rag] couldn't read the extras of collection {collection_id}: {e}")
        return {"questions": 0, "summaries": 0, "images": 0, "chunks": 0}


def rebuild_text(collection_id: int, filename: str) -> dict:
    """Rebuilds an approximate .md by joining a document's indexed chunks.

    It isn't the original: it's the ALREADY PARSED and chunked text, glued back together. It's useful to
    re-chunk it with another template, which is the usual reason to want the original, but not to
    switch parsers or extract images, because that needs the real file.
    The overlap between chunks is trimmed by comparing the end of one with the start of the next.
    """
    col = _col(collection_id)
    got = col.get(where={"source": filename}, include=["documents", "metadatas"])
    pieces = [(m.get("chunk", i), d) for i, (d, m) in enumerate(zip(got["documents"], got["metadatas"]))
              if not m.get("level") and m.get("kind") != "image"]
    if not pieces:
        raise RuntimeError(f"There are no indexed chunks of {filename}")
    pieces.sort(key=lambda t: t[0])

    parts: list[str] = []
    for _, piece in pieces:
        if parts:
            prev = parts[-1]
            # Find the longest overlap between the end of the previous one and the start of this one
            max_len = min(len(prev), len(piece), 1000)
            cut = next((n for n in range(max_len, 30, -1) if prev.endswith(piece[:n])), 0)
            piece = piece[cut:]
        parts.append(piece)
    content = "\n\n".join(p.strip() for p in parts if p.strip())

    destination = stored_path(collection_id, filename, ".md")
    destination.write_text(f"<!-- Rebuilt from the indexed chunks; this isn't the original file -->\n\n"
                       + content, encoding="utf-8")
    return {"archivo": destination.name, "fragmentos": len(pieces), "caracteres": len(content)}


def stored_originals(collection_id: int, documents: list[str]) -> dict[str, bool]:
    """Which documents keep their original file.

    Only with the original can you re-chunk with another template, switch parsers or extract
    images. Documents ingested before originals were kept don't have it, and without
    this there'd be no way to tell which ones."""
    import hashlib as _h
    folder = files_dir(collection_id)
    out = {}
    for d in documents or []:
        try:
            # Look up by hash with any extension: a rebuilt text is saved as
            # .md even if the document was .html, and comparing the extension didn't find it.
            key = _h.sha1(d.encode()).hexdigest()[:16]
            out[d] = any(folder.glob(f"{key}.*"))
        except OSError:
            out[d] = False
    return out


def count(collection_id: int) -> int:
    try:
        return client().get_collection(chroma_name(collection_id)).count()
    except Exception:  # noqa: BLE001
        return 0


def drop(collection_id: int) -> None:
    invalidate(collection_id)
    from . import graph
    graph.drop(collection_id)
    for name in (chroma_name(collection_id), chroma_name(collection_id) + "_q"):
        try:
            client().delete_collection(name)
        except Exception:  # noqa: BLE001
            pass


# ---------------- search ----------------
QUERY_MODES = {
    "simple": "The question as is",
    "multi": "The LLM writes 3 rephrasings and the results are fused (better recall)",
    "hyde": "The LLM writes a hypothetical answer and searches with it (HyDE)",
    "agentic": "Iterative search: the LLM decides whether the context is enough or asks for another search (up to 3 rounds)",
    "graph": "GraphRAG: adds entities, relations and topics from the knowledge graph (it has to be built first)",
}


def allowed_sources(coll, sources: list[str] | None = None, tags: list[str] | None = None) -> set[str] | None:
    """Documents allowed by an explicit filter and/or tags (None = no filter)."""
    allowed = set(sources) if sources else None
    if tags:
        meta = coll.doc_meta or {}
        tagged = {doc for doc, m in meta.items() if set(tags) & set((m or {}).get("tags", []))}
        allowed = tagged if allowed is None else allowed & tagged
    return allowed


def _llm(prompt: str, max_tokens: int = 256, helper: int | None = None) -> str | None:
    """Auxiliary call (rephrase, HyDE, summarize, extract entities…). None if it couldn't generate.

    Uses the given helper model; if there isn't one, the model loaded in VRAM."""
    from . import providers
    try:
        return providers.complete(helper, prompt, max_tokens=max_tokens, temperature=0.3)
    except Exception as e:  # noqa: BLE001
        print("[rag] auxiliary LLM call failed:", e)
        return None


def _candidates(coll, col, text_query: str, vec_query: str, n_cand: int, mode: str,
                allowed: set[str] | None) -> dict[str, dict]:
    ranks: dict[str, dict] = {}
    if mode in ("vector", "hybrid"):
        qv = embed(coll.embed_model, [vec_query], "query" if vec_query == text_query else "passage")[0]
        where = {"source": {"$in": sorted(allowed)}} if allowed else None
        res = col.query(query_embeddings=[qv], n_results=n_cand, include=["distances"], where=where)
        for r, cid in enumerate(res["ids"][0]):
            ranks.setdefault(cid, {})["vector_rank"] = r + 1
        # Generated questions (multi-vector): a third independent signal for the RRF fusion
        qc = _qcol(coll.id)
        if qc.count():
            qres = qc.query(query_embeddings=[qv], n_results=min(qc.count(), n_cand * 2), include=["metadatas"], where=where)
            r = 0
            for m in qres["metadatas"][0]:
                if "question_rank" in ranks.get(m["parent"], {}):
                    continue
                r += 1
                ranks.setdefault(m["parent"], {})["question_rank"] = r
                if r >= n_cand:
                    break
    if mode in ("bm25", "hybrid"):
        for r, (cid, sc) in enumerate(bm25_index(coll.id).search(text_query, n_cand, allowed)):
            ranks.setdefault(cid, {})["bm25_rank"] = r + 1
            ranks[cid]["bm25_score"] = round(sc, 3)
    return ranks


def search(coll, query: str, k: int | None = None, mode: str | None = None, reranker: str | None = "__coll__",
           min_score: float | None = None, with_debug: bool = False, sources: list[str] | None = None,
           tags: list[str] | None = None, query_mode: str | None = None) -> list[dict]:
    """Search with the collection's settings; any parameter can be overridden."""
    k = k or coll.top_k or 4
    mode = mode or coll.search_mode or "hybrid"
    reranker = coll.reranker if reranker == "__coll__" else reranker
    min_score = coll.min_score if min_score is None else min_score
    query_mode = query_mode or getattr(coll, "query_mode", None) or "simple"
    allowed = allowed_sources(coll, sources, tags)
    if allowed is not None and not allowed:
        return []
    col = _col(coll.id)
    total = col.count()
    if total == 0:
        return []
    n_cand = min(total, max(k * 5, 20)) if (reranker or mode == "hybrid") else min(total, k)

    # Query variants: (text for BM25, text for vectors)
    variants = [(query, query)]
    rewritten: list[str] = []
    if query_mode == "multi":
        out = _llm("Rephrase the following question in 3 different ways (synonyms, more specific, more general). "
                   f"One per line, without numbering or explaining.\n\nQuestion: {query}", 160)
        rewritten = [l.strip(" -•\t0123456789.") for l in (out or "").splitlines() if len(l.strip()) > 5][:3]
        variants += [(r, r) for r in rewritten]
    elif query_mode == "hyde":
        out = _llm(f"Write a short paragraph that answers this question the way a reference document would "
                   f"(even if you're not sure):\n\n{query}", 180)
        if out:
            rewritten = [out]
            variants = [(query, query), (query, out)]

    ranks: dict[str, dict] = {}
    for tq, vq in variants:
        for cid, d in _candidates(coll, col, tq, vq, n_cand, mode, allowed).items():
            agg = ranks.setdefault(cid, {"rrf": 0.0})
            agg["rrf"] += sum(1 / (RRF_K + d[key]) for key in ("vector_rank", "bm25_rank", "question_rank") if key in d)
            for key in ("vector_rank", "bm25_rank", "question_rank", "bm25_score"):
                if key in d and (key not in agg or (key.endswith("rank") and d[key] < agg[key])):
                    agg[key] = d[key]
    if not ranks:
        return []
    cand_ids = sorted(ranks, key=lambda c: -ranks[c]["rrf"])[:n_cand]

    qvec = embed(coll.embed_model, [query], "query")[0]
    got = col.get(ids=cand_ids, include=["documents", "metadatas", "embeddings"])
    by_id = {i: (d, m, e) for i, d, m, e in zip(got["ids"], got["documents"], got["metadatas"], got["embeddings"])}
    hits = []
    for cid in cand_ids:
        doc, meta, emb = by_id[cid]
        h = {"id": cid, "text": doc, **meta, **ranks[cid]}
        h["vector_score"] = round(float(np.dot(qvec, emb)), 4)  # similarity to the ORIGINAL question
        hits.append(h)

    if reranker:
        scores = reranker_model(reranker).predict([(query, contextual(h["text"], h)) for h in hits],
                                                  show_progress_bar=False)
        scores = np.asarray(scores, dtype=float)
        if scores.min() < 0 or scores.max() > 1:  # logits → probability
            scores = 1 / (1 + np.exp(-scores))
        for h, sc in zip(hits, scores):
            h["rerank_score"] = round(float(sc), 4)
            h["score"] = h["rerank_score"]
        hits.sort(key=lambda h: -h["score"])
    else:
        for h in hits:
            h["score"] = h["vector_score"] if mode != "bm25" else h.get("bm25_score", 0)
        if mode == "vector":
            hits.sort(key=lambda h: -h["score"])
    hits = [h for h in hits if h["score"] >= (min_score or 0)][:k]
    for h in hits:
        if rewritten:
            h["rewritten_queries"] = rewritten
        if not with_debug:
            h.pop("rrf", None)
    return hits


AGENT_PROMPT = """You have to answer: “{q}”
Chunks retrieved so far:
{ctx}

Do they have enough information to answer? Reply with JSON ONLY:
{{"enough": true}}  or  {{"enough": false, "new_query": "<a different query to find what's missing>"}}"""


def agentic_search(coll, question: str, k: int | None = None, max_rounds: int = 3, **kw) -> list[dict]:
    """Agentic RAG: search, the LLM judges whether it's enough and, if not, proposes another query (up to max_rounds)."""
    import json as _json
    k = k or coll.top_k or 4
    seen: dict[str, dict] = {}
    queries = [question]
    for rnd in range(max_rounds):
        for h in search(coll, queries[-1], k=k, query_mode="simple", **kw):
            if h["id"] not in seen:
                seen[h["id"]] = {**h, "round": rnd + 1, "query": queries[-1]}
        if rnd == max_rounds - 1:
            break
        ctx = "\n".join(f"- {h['text'][:300]}" for h in list(seen.values())[:8]) or "(none)"
        out = _llm(AGENT_PROMPT.format(q=question, ctx=ctx), 120)
        m = re.search(r"\{.*\}", out or "", re.S)
        try:
            d = _json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            break
        nq = (d.get("new_query") or "").strip()
        if d.get("enough") or not nq or nq in queries:
            break
        queries.append(nq)
    hits = sorted(seen.values(), key=lambda h: -h["score"])[: k + 2]
    for h in hits:
        h["agent_queries"] = queries
    return hits


def graph_search(coll, question: str, k: int | None = None, **kw) -> list[dict]:
    """GraphRAG: normal search + graph context (entities/relations/topics) + the chunks where they appear."""
    from . import graph
    hits = search(coll, question, k=k, query_mode="simple", **kw)
    g = graph.query(coll, question)
    if not g:
        return hits
    have = {h["id"] for h in hits}
    extra_ids = [c for c in g["chunk_ids"] if c not in have][:2]
    if extra_ids:
        got = _col(coll.id).get(ids=extra_ids, include=["documents", "metadatas"])
        for i, d, m in zip(got["ids"], got["documents"], got["metadatas"]):
            hits.append({"id": i, "text": d, **m, "score": 0.5, "via_graph": True})
    graph_hit = {"id": "graph", "text": g["context"], "source": "Knowledge graph", "score": 1.0,
                 "section": ", ".join(g["seeds"][:4]), "graph": {k2: g[k2] for k2 in ("seeds", "neighbors", "relations", "communities")}}
    return [graph_hit] + hits


def retrieve(coll, question: str, **kw) -> list[dict]:
    """Retrieves according to the collection's query mode (agentic, graph or normal search)."""
    qm = kw.pop("query_mode", None) or getattr(coll, "query_mode", "simple")
    if qm == "agentic":
        return agentic_search(coll, question, **kw)
    if qm == "graph":
        return graph_search(coll, question, **kw)
    return search(coll, question, query_mode=qm, **kw)


def prepare(coll, question: str, history: list[dict] | None = None, system_prompt: str | None = None,
            rag_system_prompt: str | None = None, k: int | None = None, min_score: float | None = None,
            sources: list[str] | None = None, tags: list[str] | None = None
            ) -> tuple[list[dict], list[dict], str]:
    """Builds the messages for the LLM. Returns (messages, chunks, mode) with mode: rag | no_context | refuse."""
    hits = retrieve(coll, question, k=k, min_score=min_score, sources=sources, tags=tags)
    msgs, mode = messages_for(coll, question, hits, history, system_prompt, rag_system_prompt)
    return msgs, hits, mode


def context_overflow(ctx: str, question: str = "") -> str | None:
    """Will the context not fit in the loaded model's window?

    When it doesn't fit, llama.cpp and transformers silently cut it from the start: the model
    answers with half the chunks and it looks like RAG didn't find the information, when
    it actually found it and lost it on the way. Warning about it is the difference between debugging the
    chunking and going crazy over the search."""
    from .inference import manager

    cur = manager.current
    if not cur:
        return None
    window = getattr(cur, "n_ctx", 0) or 8192
    # ~3.5 characters per token, and room is kept for the answer
    tokens = int((len(ctx) + len(question)) / 3.5)
    if tokens > window * 0.75:
        return (f"The context takes ~{tokens} tokens and the model's window is {window}: "
                f"it will be cut. Lower “Chunks per answer” or the chunk size.")
    return None


def messages_for(coll, question: str, hits: list[dict], history: list[dict] | None = None,
                 system_prompt: str | None = None, rag_system_prompt: str | None = None) -> tuple[list[dict], str]:
    """Messages from already retrieved chunks, applying the collection's fallback if there are none."""
    if hits:
        return build_messages(question, hits, rag_system_prompt, history), "rag"
    if coll.fallback == "refuse":
        return [{"role": "system", "content": REFUSE_TEXT}] + (history or []) + [{"role": "user", "content": question}], "refuse"
    msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + (history or []) + \
           [{"role": "user", "content": question}]
    return msgs, "no_context"


def build_messages(question: str, hits: list[dict], system_prompt: str | None = None,
                   history: list[dict] | None = None) -> list[dict]:
    def label(h):
        parts = [h.get("source") or ""]
        if h.get("page"):
            parts.append(f"p. {h['page']}")
        if h.get("section"):
            parts.append(h["section"])
        return " · ".join(p for p in parts if p)

    ctx = "\n\n".join(f"[{i + 1}] ({label(h)})\n{h['text']}" for i, h in enumerate(hits))
    warning = context_overflow(ctx, question)
    if warning:
        print(f"[rag] {warning}")
    msgs = [{"role": "system", "content": system_prompt or DEFAULT_RAG_SYSTEM}]
    msgs += history or []
    msgs.append({"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"})
    return msgs


def sample_chunks(collection_id: int, n: int) -> list[dict]:
    col = _col(collection_id)
    total = col.count()
    if total == 0:
        return []
    import random
    got = col.get(include=["documents", "metadatas"])
    idx = random.sample(range(total), min(n, total))
    return [{"text": got["documents"][i], **got["metadatas"][i]} for i in idx]


# ---------------- retrieval benchmark ----------------
def _is_hit(q: dict, chunk: str, expected_vec) -> bool:
    kws = q.get("keywords") or []
    if kws:
        norm = " ".join(normalize_tokens(chunk))
        return all(" ".join(normalize_tokens(k)) in norm for k in kws)
    if expected_vec is not None:
        v = np.array(embed(settings.default_embed_model, [chunk], "passage")[0])
        return float(v @ expected_vec) >= 0.85
    return False


def benchmark(coll, questions: list[dict], k: int, configs: list[dict]) -> list[dict]:
    """Compares search configurations: hit rate@k and MRR using keywords (or the reference)."""
    exp_vecs = {}
    for i, q in enumerate(questions):
        if not q.get("keywords") and q.get("expected"):
            exp_vecs[i] = np.array(embed(settings.default_embed_model, [q["expected"]], "passage")[0])
    out = []
    for cfg in configs:
        hits_n, rr, lat, per_q = 0, 0.0, 0.0, []
        for i, q in enumerate(questions):
            t0 = time.time()
            res = search(coll, q["question"], k=k, mode=cfg["mode"], reranker=cfg.get("reranker"), min_score=0)
            lat += time.time() - t0
            rank = next((j + 1 for j, h in enumerate(res) if _is_hit(q, h["text"], exp_vecs.get(i))), None)
            if rank:
                hits_n += 1
                rr += 1 / rank
            per_q.append({"question": q["question"], "rank": rank,
                          "top": [{"source": h.get("source"), "section": h.get("section"), "score": h["score"],
                                   "text": h["text"][:200]} for h in res[:3]]})
        n = max(1, len(questions))
        out.append({"label": cfg["label"], "mode": cfg["mode"], "reranker": cfg.get("reranker"),
                    "hit_rate": round(hits_n / n, 3), "mrr": round(rr / n, 3), "latency_ms": round(lat / n * 1000),
                    "per_question": per_q})
    return out


# ---------------- enrichment: questions per chunk ----------------
ENRICH_PROMPT = """Read the CHUNK and write {n} different, specific questions that it answers,
the way a user would ask them. One per line, without numbering. Then one line starting with
"KEYWORDS:" with 3-6 comma-separated keywords.

CHUNK:
{text}"""


def enrich(coll, task, n_questions: int = 3, sources: list[str] | None = None, overwrite: bool = False) -> dict:
    """Generates questions and keywords per chunk with the loaded LLM and re-indexes (improves recall)."""
    col = _col(coll.id)
    got = col.get(where={"source": {"$in": sources}} if sources else None, include=["documents", "metadatas"])
    todo = [(i, d, m) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])
            if overwrite or not m.get("questions")]
    done = failed = 0
    for n, (cid, doc, meta) in enumerate(todo):
        if task.cancel_requested:
            break
        task.update(n / max(1, len(todo)), f"Chunk {n + 1}/{len(todo)} · {meta.get('source')}")
        out = _llm(ENRICH_PROMPT.format(n=n_questions, text=doc[:2500]), 220)
        if not out:
            failed += 1
            if n == 0:
                raise RuntimeError("Load a model (Playground or Models) to generate the questions")
            continue
        lines = [l.strip(" -•\t") for l in out.splitlines() if l.strip()]
        kws = next((l.split(":", 1)[1].strip() for l in lines if l.upper().startswith(("KEYWORDS", "CLAVES"))), "")
        qs = [re.sub(r"^\d+[.)]\s*", "", l) for l in lines if not l.upper().startswith(("KEYWORDS", "CLAVES")) and len(l) > 8][:n_questions]
        meta = {**meta, "questions": "\n".join(qs), "keywords": kws}
        col.update(ids=[cid], metadatas=[meta])
        _index_questions(coll.id, coll.embed_model, cid, meta)
        done += 1
    invalidate(coll.id)
    return {"enriched": done, "failed": failed, "pending": len(todo) - done - failed}


# ---------------- hierarchical summaries (RAPTOR-style) ----------------
SUMMARY_PROMPT = """Summarize the following text in one paragraph of 4-6 sentences, keeping important names, figures and
conclusions. Only the summary.

TEXT:
{text}"""


def build_summaries(coll, task, group: int = 6, sources: list[str] | None = None) -> dict:
    """Creates summary chunks for groups of consecutive chunks (level 1) and a summary of the document
    (level 2). They serve global questions ("what is … about?", "summarize…") that no single chunk answers."""
    col = _col(coll.id)
    got = col.get(where={"source": {"$in": sources}} if sources else None, include=["documents", "metadatas"])
    by_doc: dict[str, list[tuple[int, str]]] = {}
    for d, m in zip(got["documents"], got["metadatas"]):
        if m.get("level"):
            continue
        by_doc.setdefault(m.get("source", ""), []).append((m.get("chunk", 0), d))
    total = sum(math.ceil(len(v) / group) + 1 for v in by_doc.values())
    done, created = 0, 0
    for src, items in by_doc.items():
        col.delete(where={"$and": [{"source": src}, {"level": {"$gte": 1}}]})
        items.sort()
        level1 = []
        for gi in range(0, len(items), group):
            task.update(done / max(1, total), f"Summarizing {src} ({gi // group + 1})")
            text = "\n\n".join(t for _, t in items[gi:gi + group])[:9000]
            summ = _llm(SUMMARY_PROMPT.format(text=text), 320)
            done += 1
            if not summ:
                raise RuntimeError("Load a model to generate the summaries")
            level1.append(summ)
        top = _llm(SUMMARY_PROMPT.format(text="\n\n".join(level1)[:9000]), 400) if len(level1) > 1 else None
        done += 1
        rows = [(t, {"source": src, "section": f"Summary part {i + 1}", "level": 1, "chunk": 10000 + i})
                for i, t in enumerate(level1)]
        if top:
            rows.append((top, {"source": src, "section": "Document summary", "level": 2, "chunk": 20000}))
        h = hashlib.sha1(src.encode()).hexdigest()[:8]
        col.add(ids=[f"sum-{h}-{m['level']}-{m['chunk']}" for _, m in rows], documents=[t for t, _ in rows],
                metadatas=[m for _, m in rows], embeddings=embed(coll.embed_model, [contextual(t, m) for t, m in rows]))
        created += len(rows)
    invalidate(coll.id)
    return {"summaries": created, "documents": len(by_doc)}


# ---------------- document images (multimodal) ----------------
CAPTION_PROMPT = ("Describe this image from a document in 2-4 sentences. If it's a chart, table or diagram, "
                  "explain what it shows and include the readable data or labels. Be specific, with no preamble.")


def images_dir(collection_id: int) -> Path:
    d = settings.rag_dir / "images" / str(collection_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def extract_images(pdf_bytes: bytes, min_px: int = 160) -> list[tuple[bytes, dict]]:
    """Images embedded in a PDF (small icons and ornaments are discarded)."""
    import pymupdf
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    out: list[tuple[bytes, dict]] = []
    seen: set[int] = set()
    for pno, page in enumerate(doc, start=1):
        for info in page.get_images(full=True):
            xref = info[0]
            if xref in seen:
                continue
            seen.add(xref)
            try:
                pix = pymupdf.Pixmap(doc, xref)
                if pix.width < min_px or pix.height < min_px:
                    continue
                if pix.n - pix.alpha >= 4:  # CMYK → RGB
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                out.append((pix.tobytes("png"), {"page": pno, "width": pix.width, "height": pix.height}))
            except Exception as e:  # noqa: BLE001
                print("[rag] image can't be extracted:", e)
    return out


def caption_images(coll, task, sources: list[str] | None = None, min_px: int = 160, max_images: int | None = None) -> dict:
    """Extracts the images of the indexed PDFs and describes them with the loaded VLM; each description is a searchable
    chunk that links to the image (it's shown as a source)."""
    from .inference import GenParams, manager
    cur = manager.current
    if not cur or getattr(cur, "processor", None) is None:
        raise RuntimeError("Load a vision model (e.g. Qwen/Qwen2.5-VL-3B-Instruct) to describe images")
    col = _col(coll.id)
    docs = [d for d in (coll.documents or []) if (sources is None or d in sources)]
    jobs: list[tuple[str, bytes, dict]] = []
    for d in docs:
        p = stored_path(coll.id, d)
        if not p.exists() or p.suffix.lower() != ".pdf":
            continue
        for img, meta in extract_images(p.read_bytes(), min_px):
            jobs.append((d, img, meta))
    if not jobs:
        raise RuntimeError("No images were found in the indexed PDFs (did you upload them before this version? "
                           "upload them again to keep the original)")
    if max_images:
        jobs = jobs[:max_images]
    added = 0
    for i, (doc, img, meta) in enumerate(jobs):
        if task.cancel_requested:
            break
        task.update(i / len(jobs), f"Image {i + 1}/{len(jobs)} · {doc}")
        name = f"{hashlib.sha1(img).hexdigest()[:12]}.png"
        path = images_dir(coll.id) / name
        path.write_bytes(img)
        msgs = [{"role": "user", "content": [{"type": "image", "url": str(path)}, {"type": "text", "text": CAPTION_PROMPT}]}]
        caption = "".join(manager.stream_chat(msgs, GenParams(max_new_tokens=220, temperature=0.2, enable_thinking=False))).strip()
        caption = re.sub(r"<think>.*?</think>", "", caption, flags=re.S).strip()
        if not caption:
            continue
        m = {"source": doc, "page": meta["page"], "section": f"Image (p. {meta['page']})", "image": name,
             "chunk": 30000 + i, "kind": "image"}
        text = f"[Image in {Path(doc).name}, page {meta['page']}] {caption}"
        col.add(ids=[f"img-{name}"], documents=[text], metadatas=[m], embeddings=embed(coll.embed_model, [contextual(text, m)]))
        added += 1
    invalidate(coll.id)
    return {"images": len(jobs), "described": added}

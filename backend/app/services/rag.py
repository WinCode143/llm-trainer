"""RAG: extracción estructurada, chunking por secciones, búsqueda híbrida (BM25 + vectores), reranking y umbral.

Pipeline de consulta:
  1. candidatos por vectores (Chroma, coseno) y/o BM25 (índice en memoria)
  2. fusión con Reciprocal Rank Fusion (modo hybrid)
  3. reranking opcional con cross-encoder (score 0–1)
  4. umbral min_score → si nada lo supera, fallback (sin contexto o negarse)
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
    "Eres un asistente que responde usando EXCLUSIVAMENTE el contexto proporcionado. "
    "Si la respuesta no está en el contexto, dilo claramente. Cita las fuentes con [n]."
)
REFUSE_TEXT = ("No se encontró información relevante en los documentos para esta pregunta. "
               "Responde indicando que no aparece en los documentos disponibles, sin inventar.")
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
    """Colección auxiliar con un vector por pregunta generada (metadato parent = id del fragmento)."""
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
    """Qué embedders y rerankers hay ahora mismo en memoria."""
    with _model_lock:
        return sorted(k.replace("rr:", "reranker ") for k in _models)


def unload_models() -> float:
    """Libera embedders y rerankers de la VRAM.

    El `clear()` sólo suelta la referencia del diccionario: los módulos de torch tienen ciclos, así
    que sin pasar el recolector antes de vaciar la caché los tensores siguen vivos y la VRAM no baja.
    Devuelve los GB liberados para poder comprobarlo."""
    import gc

    def usada():
        try:
            import torch
            return torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0
        except Exception:  # noqa: BLE001
            return 0.0

    antes = usada()
    with _model_lock:
        _models.clear()
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:  # noqa: BLE001
        pass
    return round(max(0.0, antes - usada()), 2)


def warmup() -> None:
    """Precarga el embedder por defecto y los rerankers/embedders que usan las colecciones."""
    from sqlmodel import Session, select

    from ..db import RagCollection, engine
    t0 = time.time()
    with Session(engine) as s:
        cols = s.exec(select(RagCollection)).all()
    names = {settings.default_embed_model} | {c.embed_model for c in cols}
    for n in names:
        embed(n, ["calentamiento"], "query")
    for r in {c.reranker for c in cols if c.reranker}:
        reranker_model(r).predict([("a", "b")], show_progress_bar=False)
    print(f"[rag] modelos precargados en {time.time() - t0:.1f}s: {sorted(names)} + rerankers")


def _prefix(name: str, kind: str) -> str:
    # Los modelos E5 esperan prefijos "query: " / "passage: "
    if "e5" in name.lower():
        return "query: " if kind == "query" else "passage: "
    return ""


def embed(name: str, texts: list[str], kind: str = "passage") -> list[list[float]]:
    p = _prefix(name, kind)
    vecs = embedder(name).encode([p + t for t in texts], batch_size=32, normalize_embeddings=True,
                                 show_progress_bar=False)
    return vecs.tolist()


# Lo que vivía aquí y ahora está en módulos propios. Se reexporta para no romper a quien ya
# importaba `rag.chunk_text`, `rag.extract` o `rag.fetch_url`: la partición es interna.
from .rag_chunking import (  # noqa: E402,F401
    BM25, bm25_index, chunk_text, contextual, invalidate, make_chunks, normalize_tokens, split_sections,
)
from .rag_extract import _html, extract, fetch_url, sitemap_urls, url_ext  # noqa: E402,F401

# ---------------- ingesta y gestión ----------------
def files_dir(collection_id: int) -> Path:
    d = settings.rag_dir / "files" / str(collection_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def stored_path(collection_id: int, filename: str, ext: str | None = None) -> Path:
    """Ruta donde se guarda el original (nombre estable a partir del nombre del documento)."""
    e = ext or Path(filename.split("?")[0]).suffix.lower() or ".bin"
    return files_dir(collection_id) / (hashlib.sha1(filename.encode()).hexdigest()[:16] + e)


def ingest(collection_id: int, embed_model: str, filename: str, raw: bytes, size: int, overlap: int,
           progress=None, template: str = "general", ext: str | None = None, parser: str = "auto") -> int:
    col = _col(collection_id)
    # Guardamos el original para poder extraer imágenes después
    try:
        stored_path(collection_id, filename, ext).write_bytes(raw)
    except OSError as e:
        # Sin el original no se puede volver a trocear ni extraer imágenes después. No se aborta la
        # ingesta por esto, pero tiene que verse: antes era un print entre miles de líneas de log.
        print(f"[rag] AVISO: no se pudo guardar el original de {filename}: {e}. "
              "Ese documento no se podrá re-trocear ni re-parsear.")
    doc_hash = hashlib.sha1(filename.encode() + raw).hexdigest()[:10]
    chunks = make_chunks(filename, extract(filename, raw, ext, parser), size, overlap, template)
    if not chunks:
        raise ValueError(f"No se pudo extraer texto de {filename} (¿PDF escaneado sin capa de texto?)")
    col.delete(where={"source": filename})  # reemplaza versiones anteriores del documento
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
    """Recalcula los embeddings con otro modelo. Primero se calculan todos (fase cancelable sin efectos) y
    sólo al final se escriben: así cancelar nunca deja la colección a medias entre dos modelos."""
    col = _col(coll.id)
    got = col.get(include=["documents", "metadatas"])
    ids, docs, metas = got["ids"], got["documents"], got["metadatas"]
    bs = 64
    vectors: list[list[float]] = []
    for i in range(0, len(ids), bs):
        if task.cancel_requested:
            return {"cancelled": True, "chunks": 0}
        task.update(0.9 * i / max(1, len(ids)), f"Calculando vectores {i}/{len(ids)}")
        vectors += embed(new_model, [contextual(d, m) for d, m in zip(docs[i:i + bs], metas[i:i + bs])])
    task.update(0.92, "Guardando…")
    for i in range(0, len(ids), bs):
        col.update(ids=ids[i:i + bs], embeddings=vectors[i:i + bs])
    try:
        client().delete_collection(chroma_name(coll.id) + "_q")
    except Exception:  # noqa: BLE001
        pass
    for n, (cid, m) in enumerate(zip(ids, metas)):
        if m.get("questions"):
            task.update(0.92 + 0.08 * n / max(1, len(ids)), "Re-indexando preguntas generadas…")
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
    """Qué extras tiene la colección (para mostrarlo en la interfaz)."""
    try:
        col = _col(collection_id)
        got = col.get(include=["metadatas"])
        metas = got["metadatas"]
        return {"questions": sum(1 for m in metas if m.get("questions")), "summaries": sum(1 for m in metas if m.get("level")),
                "images": sum(1 for m in metas if m.get("kind") == "image"), "chunks": len(metas)}
    except Exception as e:  # noqa: BLE001
        print(f"[rag] no se pudieron leer los extras de la colección {collection_id}: {e}")
        return {"questions": 0, "summaries": 0, "images": 0, "chunks": 0}


def reconstruir_texto(collection_id: int, filename: str) -> dict:
    """Rehace un .md aproximado juntando los fragmentos indexados de un documento.

    No es el original: es el texto YA PARSEADO y troceado, vuelto a pegar. Sirve para volver a
    trocearlo con otra plantilla, que es el motivo habitual para querer el original, pero no para
    cambiar de parser ni extraer imágenes, porque eso necesita el archivo de verdad.
    El solape entre fragmentos se recorta comparando el final de uno con el principio del siguiente.
    """
    col = _col(collection_id)
    got = col.get(where={"source": filename}, include=["documents", "metadatas"])
    trozos = [(m.get("chunk", i), d) for i, (d, m) in enumerate(zip(got["documents"], got["metadatas"]))
              if not m.get("level") and m.get("kind") != "image"]
    if not trozos:
        raise RuntimeError(f"No hay fragmentos indexados de {filename}")
    trozos.sort(key=lambda t: t[0])

    partes: list[str] = []
    for _, texto in trozos:
        if partes:
            previo = partes[-1]
            # Busca el solape más largo entre el final del anterior y el principio de éste
            maximo = min(len(previo), len(texto), 1000)
            corte = next((n for n in range(maximo, 30, -1) if previo.endswith(texto[:n])), 0)
            texto = texto[corte:]
        partes.append(texto)
    contenido = "\n\n".join(p.strip() for p in partes if p.strip())

    destino = stored_path(collection_id, filename, ".md")
    destino.write_text(f"<!-- Reconstruido desde los fragmentos indexados, no es el archivo original -->\n\n"
                       + contenido, encoding="utf-8")
    return {"archivo": destino.name, "fragmentos": len(trozos), "caracteres": len(contenido)}


def stored_originals(collection_id: int, documents: list[str]) -> dict[str, bool]:
    """Qué documentos conservan su archivo original.

    Sólo con el original se puede volver a trocear con otra plantilla, cambiar de parser o extraer
    imágenes. Los documentos ingeridos antes de que se guardaran los originales no lo tienen, y sin
    esto no habría forma de saber cuáles."""
    import hashlib as _h
    carpeta = files_dir(collection_id)
    out = {}
    for d in documents or []:
        try:
            # Se busca por el hash con cualquier extensión: un texto reconstruido se guarda como
            # .md aunque el documento fuera .html, y comparando la extensión no se encontraba.
            clave = _h.sha1(d.encode()).hexdigest()[:16]
            out[d] = any(carpeta.glob(f"{clave}.*"))
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


# ---------------- búsqueda ----------------
QUERY_MODES = {
    "simple": "La pregunta tal cual",
    "multi": "El LLM genera 3 reformulaciones y se fusionan los resultados (mejor recall)",
    "hyde": "El LLM escribe una respuesta hipotética y se busca por ella (HyDE)",
    "agentic": "Búsqueda iterativa: el LLM decide si el contexto basta o pide otra búsqueda (hasta 3 rondas)",
    "graph": "GraphRAG: añade entidades, relaciones y temas del grafo de conocimiento (hay que construirlo antes)",
}


def allowed_sources(coll, sources: list[str] | None = None, tags: list[str] | None = None) -> set[str] | None:
    """Documentos permitidos según filtro explícito y/o etiquetas (None = sin filtro)."""
    allowed = set(sources) if sources else None
    if tags:
        meta = coll.doc_meta or {}
        tagged = {doc for doc, m in meta.items() if set(tags) & set((m or {}).get("tags", []))}
        allowed = tagged if allowed is None else allowed & tagged
    return allowed


def _llm(prompt: str, max_tokens: int = 256, helper: int | None = None) -> str | None:
    """Llamada auxiliar (reformular, HyDE, resumir, extraer entidades…). None si no se pudo generar.

    Usa el modelo ayudante indicado; si no hay, el modelo cargado en VRAM."""
    from . import providers
    try:
        return providers.complete(helper, prompt, max_tokens=max_tokens, temperature=0.3)
    except Exception as e:  # noqa: BLE001
        print("[rag] llamada auxiliar al LLM falló:", e)
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
        # Preguntas generadas (multi-vector): tercera señal independiente para la fusión RRF
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
    """Búsqueda con la configuración de la colección; cualquier parámetro puede sobrescribirse."""
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

    # Variantes de la consulta: (texto para BM25, texto para vectores)
    variants = [(query, query)]
    rewritten: list[str] = []
    if query_mode == "multi":
        out = _llm("Reformula la siguiente pregunta de 3 maneras distintas (sinónimos, más específica, más general). "
                   f"Una por línea, sin numerar ni explicar.\n\nPregunta: {query}", 160)
        rewritten = [l.strip(" -•\t0123456789.") for l in (out or "").splitlines() if len(l.strip()) > 5][:3]
        variants += [(r, r) for r in rewritten]
    elif query_mode == "hyde":
        out = _llm(f"Escribe un párrafo breve que responda a esta pregunta como lo haría un documento de referencia "
                   f"(aunque no estés seguro):\n\n{query}", 180)
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
        h["vector_score"] = round(float(np.dot(qvec, emb)), 4)  # similitud con la pregunta ORIGINAL
        hits.append(h)

    if reranker:
        scores = reranker_model(reranker).predict([(query, contextual(h["text"], h)) for h in hits],
                                                  show_progress_bar=False)
        scores = np.asarray(scores, dtype=float)
        if scores.min() < 0 or scores.max() > 1:  # logits → probabilidad
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


AGENT_PROMPT = """Tienes que responder: «{q}»
Fragmentos recuperados hasta ahora:
{ctx}

¿Tienen información suficiente para responder? Responde SOLO con JSON:
{{"suficiente": true}}  o  {{"suficiente": false, "nueva_busqueda": "<consulta distinta para encontrar lo que falta>"}}"""


def agentic_search(coll, question: str, k: int | None = None, max_rounds: int = 3, **kw) -> list[dict]:
    """RAG agéntico: busca, el LLM juzga si basta y, si no, propone otra consulta (hasta max_rounds)."""
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
        ctx = "\n".join(f"- {h['text'][:300]}" for h in list(seen.values())[:8]) or "(ninguno)"
        out = _llm(AGENT_PROMPT.format(q=question, ctx=ctx), 120)
        m = re.search(r"\{.*\}", out or "", re.S)
        try:
            d = _json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            break
        nq = (d.get("nueva_busqueda") or "").strip()
        if d.get("suficiente") or not nq or nq in queries:
            break
        queries.append(nq)
    hits = sorted(seen.values(), key=lambda h: -h["score"])[: k + 2]
    for h in hits:
        h["agent_queries"] = queries
    return hits


def graph_search(coll, question: str, k: int | None = None, **kw) -> list[dict]:
    """GraphRAG: búsqueda normal + contexto del grafo (entidades/relaciones/temas) + fragmentos donde aparecen."""
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
    graph_hit = {"id": "graph", "text": g["context"], "source": "Grafo de conocimiento", "score": 1.0,
                 "section": ", ".join(g["seeds"][:4]), "graph": {k2: g[k2] for k2 in ("seeds", "neighbors", "relations", "communities")}}
    return [graph_hit] + hits


def retrieve(coll, question: str, **kw) -> list[dict]:
    """Recupera según el modo de consulta de la colección (agéntico, grafo o búsqueda normal)."""
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
    """Construye los mensajes para el LLM. Devuelve (mensajes, fragmentos, modo) con modo: rag | no_context | refuse."""
    hits = retrieve(coll, question, k=k, min_score=min_score, sources=sources, tags=tags)
    msgs, mode = messages_for(coll, question, hits, history, system_prompt, rag_system_prompt)
    return msgs, hits, mode


def contexto_excedido(ctx: str, pregunta: str = "") -> str | None:
    """¿El contexto no va a caber en la ventana del modelo cargado?

    Cuando no cabe, llama.cpp y transformers lo recortan por el principio sin decir nada: el modelo
    responde con la mitad de los fragmentos y parece que el RAG no encontró la información, cuando
    en realidad la encontró y se tiró por el camino. Avisar es la diferencia entre depurar el
    troceo o volverse loco con la búsqueda."""
    from .inference import manager

    cur = manager.current
    if not cur:
        return None
    ventana = getattr(cur, "n_ctx", 0) or 8192
    # ~3,5 caracteres por token en español, y se reserva sitio para la respuesta
    tokens = int((len(ctx) + len(pregunta)) / 3.5)
    if tokens > ventana * 0.75:
        return (f"El contexto ocupa ~{tokens} tokens y la ventana del modelo es {ventana}: "
                f"se recortará. Baja «Fragmentos por respuesta» o el tamaño de troceo.")
    return None


def messages_for(coll, question: str, hits: list[dict], history: list[dict] | None = None,
                 system_prompt: str | None = None, rag_system_prompt: str | None = None) -> tuple[list[dict], str]:
    """Mensajes a partir de fragmentos ya recuperados, aplicando el fallback de la colección si no hay ninguno."""
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
            parts.append(f"pág. {h['page']}")
        if h.get("section"):
            parts.append(h["section"])
        return " · ".join(p for p in parts if p)

    ctx = "\n\n".join(f"[{i + 1}] ({label(h)})\n{h['text']}" for i, h in enumerate(hits))
    aviso = contexto_excedido(ctx, question)
    if aviso:
        print(f"[rag] {aviso}")
    msgs = [{"role": "system", "content": system_prompt or DEFAULT_RAG_SYSTEM}]
    msgs += history or []
    msgs.append({"role": "user", "content": f"Contexto:\n{ctx}\n\nPregunta: {question}"})
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


# ---------------- benchmark de recuperación ----------------
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
    """Compara configuraciones de búsqueda: hit rate@k y MRR usando palabras clave (o la referencia)."""
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


# ---------------- enriquecimiento: preguntas por fragmento ----------------
ENRICH_PROMPT = """Lee el FRAGMENTO y escribe {n} preguntas distintas y concretas que se respondan con él,
tal como las haría un usuario. Una por línea, sin numerar. Después una línea que empiece por
"CLAVES:" con 3-6 palabras clave separadas por comas.

FRAGMENTO:
{text}"""


def enrich(coll, task, n_questions: int = 3, sources: list[str] | None = None, overwrite: bool = False) -> dict:
    """Genera con el LLM cargado preguntas y palabras clave por fragmento y re-indexa (mejora el recall)."""
    col = _col(coll.id)
    got = col.get(where={"source": {"$in": sources}} if sources else None, include=["documents", "metadatas"])
    todo = [(i, d, m) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])
            if overwrite or not m.get("questions")]
    done = failed = 0
    for n, (cid, doc, meta) in enumerate(todo):
        if task.cancel_requested:
            break
        task.update(n / max(1, len(todo)), f"Fragmento {n + 1}/{len(todo)} · {meta.get('source')}")
        out = _llm(ENRICH_PROMPT.format(n=n_questions, text=doc[:2500]), 220)
        if not out:
            failed += 1
            if n == 0:
                raise RuntimeError("Carga un modelo (Playground o Modelos) para generar las preguntas")
            continue
        lines = [l.strip(" -•\t") for l in out.splitlines() if l.strip()]
        kws = next((l.split(":", 1)[1].strip() for l in lines if l.upper().startswith("CLAVES")), "")
        qs = [re.sub(r"^\d+[.)]\s*", "", l) for l in lines if not l.upper().startswith("CLAVES") and len(l) > 8][:n_questions]
        meta = {**meta, "questions": "\n".join(qs), "keywords": kws}
        col.update(ids=[cid], metadatas=[meta])
        _index_questions(coll.id, coll.embed_model, cid, meta)
        done += 1
    invalidate(coll.id)
    return {"enriched": done, "failed": failed, "pending": len(todo) - done - failed}


# ---------------- resúmenes jerárquicos (tipo RAPTOR) ----------------
SUMMARY_PROMPT = """Resume el siguiente texto en un párrafo de 4-6 frases, conservando nombres, cifras y
conclusiones importantes. Sólo el resumen.

TEXTO:
{text}"""


def build_summaries(coll, task, group: int = 6, sources: list[str] | None = None) -> dict:
    """Crea fragmentos-resumen por grupos de fragmentos consecutivos (nivel 1) y un resumen del documento
    (nivel 2). Sirven para preguntas globales ("¿de qué trata…?", "resume…") que ningún fragmento responde solo."""
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
            task.update(done / max(1, total), f"Resumiendo {src} ({gi // group + 1})")
            text = "\n\n".join(t for _, t in items[gi:gi + group])[:9000]
            summ = _llm(SUMMARY_PROMPT.format(text=text), 320)
            done += 1
            if not summ:
                raise RuntimeError("Carga un modelo para generar los resúmenes")
            level1.append(summ)
        top = _llm(SUMMARY_PROMPT.format(text="\n\n".join(level1)[:9000]), 400) if len(level1) > 1 else None
        done += 1
        rows = [(t, {"source": src, "section": f"Resumen parte {i + 1}", "level": 1, "chunk": 10000 + i})
                for i, t in enumerate(level1)]
        if top:
            rows.append((top, {"source": src, "section": "Resumen del documento", "level": 2, "chunk": 20000}))
        h = hashlib.sha1(src.encode()).hexdigest()[:8]
        col.add(ids=[f"sum-{h}-{m['level']}-{m['chunk']}" for _, m in rows], documents=[t for t, _ in rows],
                metadatas=[m for _, m in rows], embeddings=embed(coll.embed_model, [contextual(t, m) for t, m in rows]))
        created += len(rows)
    invalidate(coll.id)
    return {"summaries": created, "documents": len(by_doc)}


# ---------------- imágenes de los documentos (multimodal) ----------------
CAPTION_PROMPT = ("Describe esta imagen de un documento en 2-4 frases, en español. Si es un gráfico, tabla o diagrama, "
                  "explica qué representa e incluye los datos o etiquetas legibles. Sé concreto, sin preámbulos.")


def images_dir(collection_id: int) -> Path:
    d = settings.rag_dir / "images" / str(collection_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def extract_images(pdf_bytes: bytes, min_px: int = 160) -> list[tuple[bytes, dict]]:
    """Imágenes embebidas en un PDF (descarta iconos y adornos pequeños)."""
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
                print("[rag] imagen no extraíble:", e)
    return out


def caption_images(coll, task, sources: list[str] | None = None, min_px: int = 160, max_images: int | None = None) -> dict:
    """Extrae las imágenes de los PDF indexados y las describe con el VLM cargado; cada descripción es un fragmento
    buscable que enlaza a la imagen (se muestra como fuente)."""
    from .inference import GenParams, manager
    cur = manager.current
    if not cur or getattr(cur, "processor", None) is None:
        raise RuntimeError("Carga un modelo de visión (p. ej. Qwen/Qwen2.5-VL-3B-Instruct) para describir imágenes")
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
        raise RuntimeError("No se encontraron imágenes en los PDF indexados (¿los subiste antes de esta versión? "
                           "vuelve a subirlos para guardar el original)")
    if max_images:
        jobs = jobs[:max_images]
    added = 0
    for i, (doc, img, meta) in enumerate(jobs):
        if task.cancel_requested:
            break
        task.update(i / len(jobs), f"Imagen {i + 1}/{len(jobs)} · {doc}")
        name = f"{hashlib.sha1(img).hexdigest()[:12]}.png"
        path = images_dir(coll.id) / name
        path.write_bytes(img)
        msgs = [{"role": "user", "content": [{"type": "image", "url": str(path)}, {"type": "text", "text": CAPTION_PROMPT}]}]
        caption = "".join(manager.stream_chat(msgs, GenParams(max_new_tokens=220, temperature=0.2, enable_thinking=False))).strip()
        caption = re.sub(r"<think>.*?</think>", "", caption, flags=re.S).strip()
        if not caption:
            continue
        m = {"source": doc, "page": meta["page"], "section": f"Imagen (pág. {meta['page']})", "image": name,
             "chunk": 30000 + i, "kind": "image"}
        text = f"[Imagen en {Path(doc).name}, página {meta['page']}] {caption}"
        col.add(ids=[f"img-{name}"], documents=[text], metadatas=[m], embeddings=embed(coll.embed_model, [contextual(text, m)]))
        added += 1
    invalidate(coll.id)
    return {"images": len(jobs), "described": added}

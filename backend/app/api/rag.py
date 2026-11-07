import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlmodel import Session, select

from ..config import settings
from ..db import RagCollection, engine, get_session
from ..services import rag as svc
from ..services import tasks

router = APIRouter(prefix="/api/rag", tags=["rag"])

EMBED_MODELS = [
    # dim = vector size; more dimensions means more precision and more disk/RAM.
    {"name": "intfloat/multilingual-e5-small", "note": "Multilingual, light and fast", "params": "118M", "dim": 384},
    {"name": "intfloat/multilingual-e5-base", "note": "Multilingual, balanced quality/speed", "params": "278M", "dim": 768},
    {"name": "intfloat/multilingual-e5-large", "note": "Multilingual, best quality", "params": "560M", "dim": 1024},
    {"name": "BAAI/bge-m3", "note": "Multilingual SOTA, 8K context per chunk", "params": "568M", "dim": 1024},
    {"name": "jinaai/jina-embeddings-v3", "note": "Multilingual, 8K context, very good at retrieval", "params": "572M", "dim": 1024},
    {"name": "Snowflake/snowflake-arctic-embed-l-v2.0", "note": "Multilingual, strong on long documents", "params": "568M", "dim": 1024},
    {"name": "Alibaba-NLP/gte-multilingual-base", "note": "Multilingual, 8K context, light for what it delivers", "params": "305M", "dim": 768},
    {"name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "note": "Multilingual, the lightest", "params": "118M", "dim": 384},
    {"name": "jinaai/jina-embeddings-v2-base-es", "note": "Bilingual Spanish/English, built for Spanish", "params": "161M", "dim": 768},
    {"name": "sentence-transformers/all-MiniLM-L6-v2", "note": "English only, very fast", "params": "23M", "dim": 384},
    {"name": "sentence-transformers/all-mpnet-base-v2", "note": "English only, the classic baseline", "params": "110M", "dim": 768},
    {"name": "nomic-ai/nomic-embed-text-v1.5", "note": "English, 8K context, truncatable dimension", "params": "137M", "dim": 768},
]
RERANKERS = [
    {"name": "BAAI/bge-reranker-v2-m3", "note": "Multilingual, the most precise (568M, ~1 GB VRAM)"},
    {"name": "BAAI/bge-reranker-base", "note": "English/Chinese, light (278M)"},
    {"name": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "note": "Multilingual, very fast (118M)"},
]


@router.get("/rerankers")
def rerankers():
    return RERANKERS


@router.get("/options")
def options():
    """Available chunking templates and query modes."""
    return {"templates": svc.TEMPLATES, "query_modes": svc.QUERY_MODES}


def _is_downloaded(repo: str) -> bool:
    """Is it already in the Hugging Face cache? (or is it the local path of a fine-tuned model)"""
    from pathlib import Path
    if Path(repo).exists():
        return True
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo, local_files_only=True)
        return True
    except Exception:  # noqa: BLE001
        return False


@router.get("/embed-models")
def embed_models(s: Session = Depends(get_session)):
    """Catalog of embedding models, with your fine-tuned ones first and marking which are already downloaded."""
    from ..db import Model
    tuned = [{"name": m.path, "note": f"Fine-tuned: {m.name} (base {m.base_model})", "tuned": True, "downloaded": True}
             for m in s.exec(select(Model).where(Model.kind == "embedding")).all()]
    catalog = [{**m, "downloaded": _is_downloaded(m["name"])} for m in EMBED_MODELS]
    return tuned + catalog


class EmbedDownloadBody(BaseModel):
    name: str


@router.post("/embed-models/download")
def download_embed_model(body: EmbedDownloadBody):
    """Downloads the model to the cache so the first ingestion doesn't have to wait."""
    if not any(m["name"] == body.name for m in EMBED_MODELS):
        raise HTTPException(400, "That model isn't in the catalog")

    def work(task):
        from sentence_transformers import SentenceTransformer
        task.update(message=f"Downloading {body.name}…")
        m = SentenceTransformer(body.name, device="cpu")  # downloads it and checks that it loads
        dim = m.get_sentence_embedding_dimension()
        del m
        return {"name": body.name, "dim": dim}

    return tasks.submit("download", f"Embeddings {body.name.split('/')[-1]}", work).to_dict()


class CompareBody(BaseModel):
    collection_ids: list[int]
    questions: list[str]
    k: int | None = None


@router.post("/compare")
def compare_collections(body: CompareBody, s: Session = Depends(get_session)):
    """The same questions against several collections, to see which one retrieves better.

    It compares RETRIEVAL only, not the answer: the model isn't involved, so it measures what you
    actually want to measure when you change the chunking, the reranker or the embedding model."""
    import time

    if len(body.collection_ids) < 2:
        raise HTTPException(400, "Choose at least two collections")
    if not body.questions:
        raise HTTPException(400, "Write at least one question")

    cols = [_get(s, cid) for cid in body.collection_ids]

    def work(task):
        out = []
        total = len(cols) * len(body.questions)
        done_n = 0
        for c in cols:
            rows, t_total = [], 0.0
            for q in body.questions:
                task.update(done_n / total, f"{c.name}: {q[:40]}")
                t0 = time.time()
                hits = svc.retrieve(c, q, k=body.k or c.top_k or 4)
                secs = time.time() - t0
                t_total += secs
                rows.append({
                    "pregunta": q, "n": len(hits), "segundos": round(secs, 2),
                    "fuentes": sorted({h.get("source") for h in hits if h.get("source")}),
                    "mejor": (hits[0]["text"][:300] if hits else None),
                    "score": round(hits[0].get("score", 0), 3) if hits else None,
                })
                done_n += 1
            empty_n = sum(1 for f in rows if not f["n"])
            out.append({
                "coleccion": c.name, "id": c.id,
                "ajustes": {"busqueda": c.search_mode, "reranker": c.reranker, "top_k": c.top_k,
                            "umbral": c.min_score, "embeddings": c.embed_model, "troceo": c.chunk_size},
                "sin_resultados": empty_n,
                "segundos_media": round(t_total / max(1, len(body.questions)), 2),
                "fragmentos_media": round(sum(f["n"] for f in rows) / max(1, len(rows)), 1),
                "filas": rows,
            })
        return {"colecciones": out, "preguntas": body.questions}

    return tasks.submit("rag-compare", f"Compare {len(cols)} collections", work).to_dict()


class ReindexBody(BaseModel):
    embed_model: str


@router.post("/collections/{cid}/reindex")
def reindex(cid: int, body: ReindexBody, s: Session = Depends(get_session)):
    """Changes the collection's embedding model and recomputes all the vectors."""
    c = _get(s, cid)
    s.expunge(c)

    def work(task):
        r = svc.reindex(c, body.embed_model, task)
        if r["cancelled"]:
            return {"cancelled": True, "message": "Cancelled before writing: the collection hasn't changed"}
        with Session(engine) as s2:  # the model only changes if it completed
            cc = s2.get(RagCollection, cid)
            cc.embed_model = body.embed_model
            s2.add(cc)
            s2.commit()
        return {"chunks": r["chunks"], "embed_model": body.embed_model}

    return tasks.submit("rag-reindex", f"Re-index {c.name}", work).to_dict()


@router.get("/collections")
def list_collections(s: Session = Depends(get_session)):
    """Collections with a summary of the extras they have (questions, summaries, graph, images)."""
    from ..services import graph as graph_svc
    out = []
    for c in s.exec(select(RagCollection).order_by(RagCollection.created_at.desc())).all():
        c.num_chunks = svc.count(c.id)
        d = c.model_dump(mode="json")
        d["features"] = svc.features(c.id) | {"graph": graph_svc.exists(c.id)}
        d["originals"] = svc.stored_originals(c.id, c.documents)
        out.append(d)
    return out


class CollBody(BaseModel):
    name: str
    embed_model: str | None = None
    chunk_size: int = 800
    chunk_overlap: int = 120
    search_mode: str = "hybrid"
    reranker: str | None = None
    top_k: int = 4
    min_score: float = 0.0
    fallback: str = "no_context"
    chunk_template: str = "general"
    query_mode: str = "simple"
    parser: str = "auto"


@router.post("/collections")
def create(body: CollBody, s: Session = Depends(get_session)):
    if s.exec(select(RagCollection).where(RagCollection.name == body.name)).first():
        raise HTTPException(409, "There's already a collection with that name")
    c = RagCollection(**{**body.model_dump(), "embed_model": body.embed_model or settings.default_embed_model})
    s.add(c)
    s.commit()
    s.refresh(c)
    return c


def _get(s, cid) -> RagCollection:
    c = s.get(RagCollection, cid)
    if not c:
        raise HTTPException(404, "Collection not found")
    return c


class CollSettings(BaseModel):
    search_mode: str | None = None
    reranker: str | None = None
    top_k: int | None = None
    min_score: float | None = None
    fallback: str | None = None
    chunk_template: str | None = None
    query_mode: str | None = None
    parser: str | None = None


@router.patch("/collections/{cid}")
def update_settings(cid: int, body: CollSettings, s: Session = Depends(get_session)):
    """Search settings (no re-indexing needed). reranker="" turns it off."""
    c = _get(s, cid)
    for k, v in body.model_dump(exclude_unset=True).items():
        setattr(c, k, (v or None) if k == "reranker" else v)
    s.add(c)
    s.commit()
    s.refresh(c)
    return c


def _ingest_task(c: RagCollection, items: list[tuple[str, bytes, str | None]], template: str, tags: list[str],
                 title: str, extra_meta: dict | None = None) -> dict:
    """items: [(document name, bytes, forced extension)]. Updates documents and tags when done."""
    coll_id, model, size, overlap, parser = c.id, c.embed_model, c.chunk_size, c.chunk_overlap, c.parser or "auto"

    def work(task):
        added, errors = {}, {}
        for i, (fname, raw, ext) in enumerate(items):
            task.update(i / len(items), f"Indexing {fname}")
            try:
                added[fname] = svc.ingest(coll_id, model, fname, raw, size, overlap, template=template, ext=ext, parser=parser,
                                          progress=lambda p: task.update((i + p) / len(items)))
            except Exception as e:  # noqa: BLE001  — one bad document doesn't abort the batch
                errors[fname] = str(e)
        with Session(engine) as s2:
            cc = s2.get(RagCollection, coll_id)
            cc.documents = sorted(set(cc.documents) | set(added))
            meta = dict(cc.doc_meta or {})
            for f in added:
                meta[f] = {**meta.get(f, {}), "tags": sorted(set(meta.get(f, {}).get("tags", [])) | set(tags)),
                           "template": template, **(extra_meta or {})}
            cc.doc_meta = meta
            cc.num_chunks = svc.count(coll_id)
            s2.add(cc)
            s2.commit()
        if errors and not added:
            raise RuntimeError("; ".join(f"{k}: {v}" for k, v in errors.items())[:500])
        return {"added": added, "errors": errors}

    return tasks.submit("rag-ingest", title, work).to_dict()


@router.post("/collections/{cid}/documents")
async def upload(cid: int, files: list[UploadFile] = File(...), template: str | None = Form(None),
                 tags: str | None = Form(None), s: Session = Depends(get_session)):
    c = _get(s, cid)
    items = [(f.filename, await f.read(), None) for f in files]
    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
    return _ingest_task(c, items, template or c.chunk_template or "general", tag_list,
                        f"Index {len(items)} document(s) into {c.name}")


class UrlsBody(BaseModel):
    urls: list[str] = []
    sitemap: str | None = None
    max_pages: int = 30
    tags: list[str] = []
    template: str | None = None


@router.post("/collections/{cid}/urls")
def add_urls(cid: int, body: UrlsBody, s: Session = Depends(get_session)):
    """Downloads web pages (or all those in a sitemap) and indexes them. The URL becomes a citable source."""
    c = _get(s, cid)
    s.expunge(c)

    def work(task):
        urls = list(body.urls)
        if body.sitemap:
            task.update(message="Reading the sitemap…")
            urls += svc.sitemap_urls(body.sitemap, body.max_pages)
        urls = list(dict.fromkeys(u.strip() for u in urls if u.strip()))[: body.max_pages]
        items, errors = [], {}
        for i, u in enumerate(urls):
            task.update(i / max(1, len(urls)) * 0.5, f"Downloading {u}")
            try:
                raw = svc.fetch_url(u)
                items.append((u, raw, svc.url_ext(u, raw)))
            except Exception as e:  # noqa: BLE001
                errors[u] = str(e)
        if not items:
            raise RuntimeError("Couldn't download any URL: " + "; ".join(f"{k}: {v}" for k, v in errors.items())[:400])
        # Indexing runs in another task (with its own progress)
        t = _ingest_task(c, items, body.template or c.chunk_template or "general", body.tags,
                         f"Index {len(items)} web page(s) into {c.name}")
        return {"downloaded": len(items), "download_errors": errors, "ingest_task": t["id"]}

    return tasks.submit("rag-web", f"Download {len(body.urls) or 'sitemap'} URL(s)", work).to_dict()


class DocMeta(BaseModel):
    filename: str
    tags: list[str]


@router.patch("/collections/{cid}/documents")
def set_doc_meta(cid: int, body: DocMeta, s: Session = Depends(get_session)):
    c = _get(s, cid)
    meta = dict(c.doc_meta or {})
    meta[body.filename] = {**meta.get(body.filename, {}), "tags": sorted({t.strip() for t in body.tags if t.strip()})}
    c.doc_meta = meta
    s.add(c)
    s.commit()
    s.refresh(c)
    return c


class EnrichBody(BaseModel):
    n_questions: int = 3
    sources: list[str] | None = None
    overwrite: bool = False


@router.post("/collections/{cid}/enrich")
def enrich(cid: int, body: EnrichBody, s: Session = Depends(get_session)):
    """Generates questions/keywords per chunk with the loaded model (improves recall)."""
    c = _get(s, cid)
    s.expunge(c)
    return tasks.submit("rag-enrich", f"Enrich {c.name}",
                        lambda task: svc.enrich(c, task, body.n_questions, body.sources, body.overwrite)).to_dict()


class ImagesBody(BaseModel):
    sources: list[str] | None = None
    min_px: int = 160
    max_images: int | None = None


@router.post("/collections/{cid}/images")
def describe_images(cid: int, body: ImagesBody, s: Session = Depends(get_session)):
    """Extracts the images of the PDFs and describes them with the loaded vision model (they become searchable)."""
    c = _get(s, cid)
    s.expunge(c)
    return tasks.submit("rag-images", f"Describe images in {c.name}",
                        lambda task: svc.caption_images(c, task, body.sources, body.min_px, body.max_images)).to_dict()


@router.get("/collections/{cid}/images/{name}")
def get_image(cid: int, name: str):
    """Serves an extracted image (shown as a source in the chat)."""
    from fastapi.responses import FileResponse
    p = svc.images_dir(cid) / Path(name).name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png")


class GraphBody(BaseModel):
    sources: list[str] | None = None
    max_chunks: int | None = None
    max_entities: int = 8
    helper: int | None = None
    summarize_communities: bool = True


@router.post("/collections/{cid}/graph")
def build_graph(cid: int, body: GraphBody, s: Session = Depends(get_session)):
    """Builds the knowledge graph (entities and relations) with the loaded model."""
    from ..services import graph
    c = _get(s, cid)
    s.expunge(c)
    return tasks.submit("rag-graph", f"Graph for {c.name}", lambda task: graph.build(
        c, task, body.sources, body.max_chunks, body.max_entities, body.summarize_communities,
        body.helper)).to_dict()


@router.get("/collections/{cid}/graph")
def get_graph(cid: int, limit: int = 250, s: Session = Depends(get_session)):
    from ..services import graph
    _get(s, cid)
    return graph.view(cid, limit) or {"nodes": [], "links": [], "communities": [], "stats": None}


@router.get("/collections/{cid}/graph/query")
def graph_query(cid: int, q: str, s: Session = Depends(get_session)):
    from ..services import graph
    return graph.query(_get(s, cid), q) or {"context": None}


class SummariesBody(BaseModel):
    sources: list[str] | None = None
    group: int = 6


@router.post("/collections/{cid}/summaries")
def summaries(cid: int, body: SummariesBody, s: Session = Depends(get_session)):
    """Hierarchical summaries (RAPTOR-style) for global questions about documents."""
    c = _get(s, cid)
    s.expunge(c)

    def work(task):
        r = svc.build_summaries(c, task, body.group, body.sources)
        with Session(engine) as s2:
            cc = s2.get(RagCollection, cid)
            cc.num_chunks = svc.count(cid)
            s2.add(cc)
            s2.commit()
        return r

    return tasks.submit("rag-summaries", f"Summaries for {c.name}", work).to_dict()


@router.post("/collections/{cid}/documents/rebuild")
def rebuild(cid: int, filename: str, s: Session = Depends(get_session)):
    """Rebuilds an approximate text of the document from its indexed chunks."""
    _get(s, cid)
    try:
        return svc.rebuild_text(cid, filename)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, str(e)) from e


@router.delete("/collections/{cid}/documents")
def remove_doc(cid: int, filename: str, s: Session = Depends(get_session)):
    c = _get(s, cid)
    svc.remove_document(cid, filename)
    c.documents = [d for d in c.documents if d != filename]
    c.doc_meta = {k: v for k, v in (c.doc_meta or {}).items() if k != filename}
    c.num_chunks = svc.count(cid)
    s.add(c)
    s.commit()
    return c


@router.delete("/collections/{cid}")
def delete(cid: int, s: Session = Depends(get_session)):
    c = _get(s, cid)
    svc.drop(cid)
    s.delete(c)
    s.commit()
    return {"ok": True}


class SearchBody(BaseModel):
    query: str
    k: int | None = None
    mode: str | None = None
    reranker: str | None = "__coll__"
    min_score: float | None = None
    sources: list[str] | None = None
    tags: list[str] | None = None
    query_mode: str | None = None


@router.post("/collections/{cid}/search")
def search(cid: int, body: SearchBody, s: Session = Depends(get_session)):
    c = _get(s, cid)
    t0 = time.time()
    # "__coll__" = use the collection's reranker; "" or null = no reranker
    rr = body.reranker if body.reranker == "__coll__" else (body.reranker or None)
    kw = dict(k=body.k, mode=body.mode, reranker=rr, min_score=body.min_score, sources=body.sources, tags=body.tags)
    hits = svc.retrieve(c, body.query, query_mode=body.query_mode, **kw)
    return {"hits": hits, "ms": round((time.time() - t0) * 1000)}


@router.get("/collections/{cid}/chunks")
def chunks(cid: int, source: str | None = None, offset: int = 0, limit: int = 50, s: Session = Depends(get_session)):
    _get(s, cid)
    return svc.list_chunks(cid, source, offset, limit)


class ChunkEdit(BaseModel):
    text: str


@router.put("/collections/{cid}/chunks/{chunk_id}")
def edit_chunk(cid: int, chunk_id: str, body: ChunkEdit, s: Session = Depends(get_session)):
    c = _get(s, cid)
    try:
        return svc.update_chunk(cid, c.embed_model, chunk_id, body.text)
    except KeyError:
        raise HTTPException(404, "Chunk not found")


@router.delete("/collections/{cid}/chunks/{chunk_id}")
def del_chunk(cid: int, chunk_id: str, s: Session = Depends(get_session)):
    c = _get(s, cid)
    svc.delete_chunk(cid, chunk_id)
    c.num_chunks = svc.count(cid)
    s.add(c)
    s.commit()
    return {"ok": True}


class BenchQuestion(BaseModel):
    question: str
    expected: str | None = None
    keywords: list[str] = []


class BenchBody(BaseModel):
    questions: list[BenchQuestion]
    k: int = 4
    reranker: str | None = None  # for the "+ rerank" config; defaults to the collection's or bge-reranker-v2-m3


@router.post("/collections/{cid}/benchmark")
def benchmark(cid: int, body: BenchBody, s: Session = Depends(get_session)):
    """Compares vector / BM25 / hybrid / hybrid+rerank on questions with keywords or a reference."""
    c = _get(s, cid)
    rr = body.reranker or c.reranker or RERANKERS[0]["name"]
    configs = [{"label": "Vectors", "mode": "vector"}, {"label": "BM25", "mode": "bm25"},
               {"label": "Hybrid", "mode": "hybrid"}, {"label": "Hybrid + rerank", "mode": "hybrid", "reranker": rr}]
    qs = [q.model_dump() for q in body.questions]
    s.expunge(c)

    def work(task):
        task.update(message="Evaluating retrieval…")
        return svc.benchmark(c, qs, body.k, configs)

    return tasks.submit("rag-bench", f"Retrieval benchmark ({c.name})", work).to_dict()


class ChunkPreview(BaseModel):
    text: str
    chunk_size: int = 800
    chunk_overlap: int = 120


@router.post("/chunk-preview")
def chunk_preview(body: ChunkPreview):
    return svc.chunk_text(body.text, body.chunk_size, body.chunk_overlap)

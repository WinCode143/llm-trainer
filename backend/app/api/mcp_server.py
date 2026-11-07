"""MCP (Model Context Protocol) server: exposes LLM Trainer's RAG collections and models
to external agents (Claude Code/Desktop, Cursor, Open WebUI…) at http://<host>:<port>/mcp/

The tools do blocking work (embeddings, GPU), so they run in a separate thread.
"""
import anyio
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from sqlmodel import Session, select

from ..db import Model, RagCollection, engine
from ..services import rag
from ..services.evaluation import strip_think
from ..services.inference import GenParams, manager

server = MCPServer(
    name="llm-trainer",
    title="LLM Trainer",
    instructions=("Access to LLM Trainer's document collections (RAG) and local models. "
                  "Use list_collections to see what's there, search_documents to retrieve chunks with their source "
                  "and ask to get an answer written by the local model with citations. You can also add "
                  "documents (add_document with text, ingest_url with a page), create training datasets "
                  "(create_dataset with rows, import_hf_dataset from Hugging Face) and run saved flows."),
)


def _collection(name: str) -> RagCollection:
    with Session(engine, expire_on_commit=False) as s:
        c = s.exec(select(RagCollection).where(RagCollection.name == name)).first()
        if not c and name.isdigit():
            c = s.get(RagCollection, int(name))
    if not c:
        raise ToolError(f"Collection '{name}' not found. Use list_collections.")
    return c


@server.tool(description="Lists the available RAG collections with their documents and tags.")
async def list_collections() -> list[dict]:
    with Session(engine) as s:
        cols = s.exec(select(RagCollection)).all()
        return [{"name": c.name, "documents": c.documents, "chunks": rag.count(c.id),
                 "tags": sorted({t for m in (c.doc_meta or {}).values() for t in (m or {}).get("tags", [])}),
                 "search": f"{c.search_mode}{' + rerank' if c.reranker else ''}"} for c in cols]


@server.tool(description="Searches a collection for relevant chunks (hybrid search + rerank). "
                         "Returns the text, the source document, page/section and a relevance score (0-1).")
async def search_documents(collection: str, query: str, k: int = 5, tags: list[str] | None = None) -> list[dict]:
    c = _collection(collection)
    hits = await anyio.to_thread.run_sync(lambda: rag.retrieve(c, query, k=k, tags=tags or None))
    return [{"text": h["text"], "source": h.get("source"), "page": h.get("page"), "section": h.get("section"),
             "score": h["score"]} for h in hits]


@server.tool(description="Answers a question using the collection as context and the loaded local model. "
                         "Includes the cited sources. Requires a model loaded in LLM Trainer.")
async def ask(collection: str, question: str, max_tokens: int = 512) -> dict:
    c = _collection(collection)
    if not manager.current:
        raise ToolError("No model is loaded in LLM Trainer (load one in Playground or Models).")

    def run():
        msgs, hits, mode = rag.prepare(c, question)
        text = "".join(manager.stream_chat(msgs, GenParams(max_new_tokens=max_tokens, temperature=0.2,
                                                           enable_thinking=False)))
        return {"answer": strip_think(text), "model": manager.current.name, "rag_mode": mode,
                "sources": [{"n": i + 1, "source": h.get("source"), "page": h.get("page"), "score": h["score"]}
                            for i, h in enumerate(hits)]}

    return await anyio.to_thread.run_sync(run)


@server.tool(description="Lists the registered models (base, LoRA, merged, GGUF) and says which one is loaded.")
async def list_models() -> list[dict]:
    with Session(engine) as s:
        rows = s.exec(select(Model)).all()
    loaded = manager.current.name if manager.current else None
    return [{"name": m.name, "kind": m.kind, "base_model": m.base_model, "loaded": m.name == loaded} for m in rows]


# ------------------------------------------------------------------ writing
# These tools modify the application. They're explicit and limited: an external agent can
# ingest documents, create datasets and run flows, but not delete anything or touch the models.


@server.tool(description="Ingests a document into a RAG collection from its text. "
                         "Use it to add notes, transcripts or any text you want to be able to query later.")
async def add_document(collection: str, filename: str, text: str) -> dict:
    c = _collection(collection)

    def run():
        name = filename if filename.endswith((".md", ".txt")) else f"{filename}.md"
        chunks = rag.ingest(c.id, c.embed_model, name, text.encode("utf-8"), c.chunk_size, c.chunk_overlap,
                            template=c.chunk_template)
        with Session(engine) as s:
            col = s.get(RagCollection, c.id)
            col.num_chunks = (col.num_chunks or 0) + chunks
            if name not in (col.documents or []):
                col.documents = [*(col.documents or []), name]
            s.add(col)
            s.commit()
        return {"document": name, "chunks": chunks, "collection": c.name}

    return await anyio.to_thread.run_sync(run)


@server.tool(description="Creates a training dataset from question/answer pairs. "
                         "Each row's format: {'instruction': …, 'output': …}. It's ready to train in the app.")
async def create_dataset(name: str, rows: list[dict]) -> dict:
    from ..db import Dataset
    from ..services import datasets as ds_service

    def run():
        clean = [r for r in rows if r.get("instruction") and r.get("output")]
        if not clean:
            raise ToolError("No row has 'instruction' and 'output'")
        with Session(engine) as s:
            if s.exec(select(Dataset).where(Dataset.name == name)).first():
                raise ToolError(f"There's already a dataset called '{name}'. Pick another name.")
            fmt, rows = ds_service.normalize(clean, "chat")
            path = ds_service.write_jsonl(name, rows)
            d = Dataset(name=name, format=fmt, path=str(path), num_rows=len(rows), source="mcp")
            s.add(d)
            s.commit()
            s.refresh(d)
            return {"id": d.id, "name": d.name, "rows": d.num_rows, "format": d.format}

    return await anyio.to_thread.run_sync(run)


@server.tool(description="Downloads a web page (or a linked PDF) and indexes it into a RAG collection, "
                         "with the URL as a citable source. Use it after finding an article, some documentation or a paper.")
async def ingest_url(collection: str, url: str) -> dict:
    c = _collection(collection)
    if not url.startswith(("http://", "https://")):
        raise ToolError("The URL has to start with http:// or https://")

    def run():
        try:
            raw = rag.fetch_url(url)
        except Exception as e:  # noqa: BLE001
            raise ToolError(f"Couldn't download {url}: {e}") from e
        chunks = rag.ingest(c.id, c.embed_model, url, raw, c.chunk_size, c.chunk_overlap,
                            template=c.chunk_template, ext=rag.url_ext(url, raw), parser=c.parser or "auto")
        with Session(engine) as s:
            col = s.get(RagCollection, c.id)
            col.documents = sorted(set(col.documents or []) | {url})
            col.num_chunks = rag.count(c.id)
            s.add(col)
            s.commit()
        return {"document": url, "chunks": chunks, "collection": c.name}

    return await anyio.to_thread.run_sync(run)


def inbox_dir():
    from ..config import settings
    d = (settings.data_dir / "inbox").resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


@server.tool(description="Indexes a file from the app's inbox folder (data/inbox) into a RAG collection: "
                         "PDF, Word, Markdown, CSV, HTML… The path is relative to that folder.")
async def ingest_file(collection: str, path: str) -> dict:
    c = _collection(collection)
    base = inbox_dir()
    f = (base / path.lstrip("/")).resolve()
    # Only that folder: an external agent mustn't be able to put any file on the disk into a collection
    if base not in f.parents or not f.is_file():
        raise ToolError(f"'{path}' isn't a file inside data/inbox")

    def run():
        chunks = rag.ingest(c.id, c.embed_model, f.name, f.read_bytes(), c.chunk_size, c.chunk_overlap,
                            template=c.chunk_template, parser=c.parser or "auto")
        with Session(engine) as s:
            col = s.get(RagCollection, c.id)
            already = f.name in (col.documents or [])
            col.documents = sorted(set(col.documents or []) | {f.name})
            col.num_chunks = rag.count(c.id)
            s.add(col)
            s.commit()
        # Re-indexing replaces the previous version (it doesn't duplicate); it's flagged so the agent knows what was new
        return {"document": f.name, "chunks": chunks, "collection": c.name, "already_indexed": already}

    return await anyio.to_thread.run_sync(run)


@server.tool(description="Imports a dataset from the Hugging Face Hub (e.g. 'tatsu-lab/alpaca') as a training "
                         "dataset in the app. It detects the format (instruction/answer, chat, preferences). "
                         "Look up the exact repository name first; limit caps the rows (500 by default).")
async def import_hf_dataset(repo: str, name: str = "", split: str = "train", subset: str = "",
                            limit: int = 500) -> dict:
    from ..db import Dataset
    from ..services import datasets as ds_service

    name_ = (name or repo.split("/")[-1]).strip()
    limit = max(1, min(int(limit or 500), 20000))

    def run():
        with Session(engine) as s:
            if s.exec(select(Dataset).where(Dataset.name == name_)).first():
                raise ToolError(f"There's already a dataset called '{name_}'. Pass another name.")
        try:
            fmt, rows = ds_service.import_hf(repo, split, subset or None, limit, None, None)
        except Exception as e:  # noqa: BLE001
            raise ToolError(f"Couldn't import {repo}: {e}") from e
        with Session(engine) as s:
            path = ds_service.write_jsonl(name_, rows)
            d = Dataset(name=name_, format=fmt, path=str(path), num_rows=len(rows), source=f"hf:{repo}")
            s.add(d)
            s.commit()
            s.refresh(d)
            out = {"id": d.id, "name": d.name, "rows": d.num_rows, "format": d.format,
                   "example": {k: str(v)[:300] for k, v in (rows[0] if rows else {}).items()}}
            if fmt == "text":
                # Running text: good for pretraining, not for teaching it to answer instructions
                out["warning"] = ("It has no question/answer pairs: it's running text. It's for continued "
                                "pretraining, not for an assistant. Another dataset is probably a better fit.")
            return out

    return await anyio.to_thread.run_sync(run)


@server.tool(description="Lists the app's training datasets with their format and number of rows.")
async def list_datasets() -> list[dict]:
    from ..db import Dataset
    with Session(engine) as s:
        return [{"name": d.name, "format": d.format, "rows": d.num_rows, "source": d.source}
                for d in s.exec(select(Dataset)).all()]


@server.tool(description="Lists the saved flows (multi-step processes defined in the app).")
async def list_flows() -> list[dict]:
    from ..db import Flow
    with Session(engine) as s:
        return [{"name": f.name, "description": f.description, "nodes": len(f.spec.get("nodes", []))}
                for f in s.exec(select(Flow)).all()]


@server.tool(description="Runs a saved flow with an input and returns its final result. "
                         "If the flow has an approval node, it will stop and has to be continued from the app.")
async def run_flow(name: str, input: str) -> dict:
    from ..db import Flow
    from ..services import flows as flows_svc

    with Session(engine, expire_on_commit=False) as s:
        f = s.exec(select(Flow).where(Flow.name == name)).first()
    if not f:
        raise ToolError(f"Flow '{name}' not found. Use list_flows.")

    def run():
        steps, output, waiting = [], "", False
        for ev in flows_svc.run(f.spec, {"_input": input, "input": input}):
            if ev["type"] == "node":
                steps.append(ev["label"])
            elif ev["type"] == "interrupt":
                waiting = True
            elif ev["type"] == "done":
                output = ev.get("vars", {}).get("_output", "")
            elif ev["type"] == "error":
                raise ToolError(ev["message"])
        return {"result": output, "steps": steps,
                "waiting_for_approval": waiting} if not waiting else {
            "steps": steps, "waiting_for_approval": True,
            "warning": "The flow stopped at an approval node; continue it from the app's Flows screen."}

    return await anyio.to_thread.run_sync(run)


# ASGI sub-application; mounted at /mcp and its session manager starts in FastAPI's lifespan
app = server.streamable_http_app(streamable_http_path="/")

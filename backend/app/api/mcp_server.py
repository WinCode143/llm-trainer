"""Servidor MCP (Model Context Protocol): expone las colecciones RAG y los modelos de LLM Trainer
a agentes externos (Claude Code/Desktop, Cursor, Open WebUI…) en http://<host>:<puerto>/mcp/

Las herramientas hacen trabajo bloqueante (embeddings, GPU), así que se ejecutan en un hilo aparte.
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
    instructions=("Acceso a las colecciones de documentos (RAG) y a los modelos locales de LLM Trainer. "
                  "Usa list_collections para ver qué hay, search_documents para recuperar fragmentos con su fuente "
                  "y ask para obtener una respuesta redactada por el modelo local con citas. También puedes añadir "
                  "documentos (add_document con texto, ingest_url con una página), crear datasets de entrenamiento "
                  "(create_dataset con filas, import_hf_dataset desde Hugging Face) y ejecutar flujos guardados."),
)


def _collection(name: str) -> RagCollection:
    with Session(engine, expire_on_commit=False) as s:
        c = s.exec(select(RagCollection).where(RagCollection.name == name)).first()
        if not c and name.isdigit():
            c = s.get(RagCollection, int(name))
    if not c:
        raise ToolError(f"Colección '{name}' no encontrada. Usa list_collections.")
    return c


@server.tool(description="Lista las colecciones RAG disponibles con sus documentos y etiquetas.")
async def list_collections() -> list[dict]:
    with Session(engine) as s:
        cols = s.exec(select(RagCollection)).all()
        return [{"name": c.name, "documents": c.documents, "chunks": rag.count(c.id),
                 "tags": sorted({t for m in (c.doc_meta or {}).values() for t in (m or {}).get("tags", [])}),
                 "search": f"{c.search_mode}{' + rerank' if c.reranker else ''}"} for c in cols]


@server.tool(description="Busca fragmentos relevantes en una colección (búsqueda híbrida + rerank). "
                         "Devuelve texto, documento de origen, página/sección y score de relevancia (0-1).")
async def search_documents(collection: str, query: str, k: int = 5, tags: list[str] | None = None) -> list[dict]:
    c = _collection(collection)
    hits = await anyio.to_thread.run_sync(lambda: rag.retrieve(c, query, k=k, tags=tags or None))
    return [{"text": h["text"], "source": h.get("source"), "page": h.get("page"), "section": h.get("section"),
             "score": h["score"]} for h in hits]


@server.tool(description="Responde a una pregunta usando la colección como contexto y el modelo local cargado. "
                         "Incluye las fuentes citadas. Requiere un modelo cargado en LLM Trainer.")
async def ask(collection: str, question: str, max_tokens: int = 512) -> dict:
    c = _collection(collection)
    if not manager.current:
        raise ToolError("No hay ningún modelo cargado en LLM Trainer (cárgalo en Playground o Modelos).")

    def run():
        msgs, hits, mode = rag.prepare(c, question)
        text = "".join(manager.stream_chat(msgs, GenParams(max_new_tokens=max_tokens, temperature=0.2,
                                                           enable_thinking=False)))
        return {"answer": strip_think(text), "model": manager.current.name, "rag_mode": mode,
                "sources": [{"n": i + 1, "source": h.get("source"), "page": h.get("page"), "score": h["score"]}
                            for i, h in enumerate(hits)]}

    return await anyio.to_thread.run_sync(run)


@server.tool(description="Lista los modelos registrados (base, LoRA, fusionados, GGUF) e indica cuál está cargado.")
async def list_models() -> list[dict]:
    with Session(engine) as s:
        rows = s.exec(select(Model)).all()
    loaded = manager.current.name if manager.current else None
    return [{"name": m.name, "kind": m.kind, "base_model": m.base_model, "loaded": m.name == loaded} for m in rows]


# ------------------------------------------------------------------ escritura
# Estas herramientas modifican la aplicación. Se dejan explícitas y acotadas: un agente externo puede
# ingerir documentos, crear datasets y lanzar flujos, pero no borrar nada ni tocar los modelos.


@server.tool(description="Ingiere un documento en una colección RAG a partir de su texto. "
                         "Úsalo para añadir notas, transcripciones o cualquier texto que quieras poder consultar después.")
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
        return {"documento": name, "fragmentos": chunks, "coleccion": c.name}

    return await anyio.to_thread.run_sync(run)


@server.tool(description="Crea un dataset de entrenamiento a partir de pares pregunta/respuesta. "
                         "Formato de cada fila: {'instruction': …, 'output': …}. Queda listo para entrenar en la app.")
async def create_dataset(name: str, rows: list[dict]) -> dict:
    from ..db import Dataset
    from ..services import datasets as ds_service

    def run():
        limpio = [r for r in rows if r.get("instruction") and r.get("output")]
        if not limpio:
            raise ToolError("Ninguna fila tiene 'instruction' y 'output'")
        with Session(engine) as s:
            if s.exec(select(Dataset).where(Dataset.name == name)).first():
                raise ToolError(f"Ya existe un dataset llamado '{name}'. Elige otro nombre.")
            fmt, filas = ds_service.normalize(limpio, "chat")
            path = ds_service.write_jsonl(name, filas)
            d = Dataset(name=name, format=fmt, path=str(path), num_rows=len(filas), source="mcp")
            s.add(d)
            s.commit()
            s.refresh(d)
            return {"id": d.id, "nombre": d.name, "filas": d.num_rows, "formato": d.format}

    return await anyio.to_thread.run_sync(run)


@server.tool(description="Descarga una página web (o un PDF enlazado) y la indexa en una colección RAG, "
                         "con la URL como fuente citable. Úsalo tras encontrar un artículo, una documentación o un paper.")
async def ingest_url(collection: str, url: str) -> dict:
    c = _collection(collection)
    if not url.startswith(("http://", "https://")):
        raise ToolError("La URL tiene que empezar por http:// o https://")

    def run():
        try:
            raw = rag.fetch_url(url)
        except Exception as e:  # noqa: BLE001
            raise ToolError(f"No se pudo descargar {url}: {e}") from e
        chunks = rag.ingest(c.id, c.embed_model, url, raw, c.chunk_size, c.chunk_overlap,
                            template=c.chunk_template, ext=rag.url_ext(url, raw), parser=c.parser or "auto")
        with Session(engine) as s:
            col = s.get(RagCollection, c.id)
            col.documents = sorted(set(col.documents or []) | {url})
            col.num_chunks = rag.count(c.id)
            s.add(col)
            s.commit()
        return {"documento": url, "fragmentos": chunks, "coleccion": c.name}

    return await anyio.to_thread.run_sync(run)


def carpeta_entrada():
    from ..config import settings
    d = (settings.data_dir / "entrada").resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


@server.tool(description="Indexa en una colección RAG un archivo de la carpeta de entrada de la app (data/entrada): "
                         "PDF, Word, Markdown, CSV, HTML… La ruta es relativa a esa carpeta.")
async def ingest_file(collection: str, path: str) -> dict:
    c = _collection(collection)
    base = carpeta_entrada()
    f = (base / path.lstrip("/")).resolve()
    # Sólo esa carpeta: un agente externo no debe poder meter cualquier archivo del disco en una colección
    if base not in f.parents or not f.is_file():
        raise ToolError(f"'{path}' no es un archivo dentro de data/entrada")

    def run():
        chunks = rag.ingest(c.id, c.embed_model, f.name, f.read_bytes(), c.chunk_size, c.chunk_overlap,
                            template=c.chunk_template, parser=c.parser or "auto")
        with Session(engine) as s:
            col = s.get(RagCollection, c.id)
            ya = f.name in (col.documents or [])
            col.documents = sorted(set(col.documents or []) | {f.name})
            col.num_chunks = rag.count(c.id)
            s.add(col)
            s.commit()
        # Re-indexar reemplaza la versión anterior (no duplica); se avisa para que el agente sepa qué era nuevo
        return {"documento": f.name, "fragmentos": chunks, "coleccion": c.name, "ya_estaba": ya}

    return await anyio.to_thread.run_sync(run)


@server.tool(description="Importa un dataset del Hub de Hugging Face (p. ej. 'tatsu-lab/alpaca') como dataset de "
                         "entrenamiento de la app. Detecta el formato (instrucción/respuesta, chat, preferencias). "
                         "Busca antes el nombre exacto del repositorio; limit acota las filas (por defecto 500).")
async def import_hf_dataset(repo: str, name: str = "", split: str = "train", subset: str = "",
                            limit: int = 500) -> dict:
    from ..db import Dataset
    from ..services import datasets as ds_service

    nombre = (name or repo.split("/")[-1]).strip()
    limit = max(1, min(int(limit or 500), 20000))

    def run():
        with Session(engine) as s:
            if s.exec(select(Dataset).where(Dataset.name == nombre)).first():
                raise ToolError(f"Ya existe un dataset llamado '{nombre}'. Pasa otro name.")
        try:
            fmt, filas = ds_service.import_hf(repo, split, subset or None, limit, None, None)
        except Exception as e:  # noqa: BLE001
            raise ToolError(f"No se pudo importar {repo}: {e}") from e
        with Session(engine) as s:
            path = ds_service.write_jsonl(nombre, filas)
            d = Dataset(name=nombre, format=fmt, path=str(path), num_rows=len(filas), source=f"hf:{repo}")
            s.add(d)
            s.commit()
            s.refresh(d)
            out = {"id": d.id, "nombre": d.name, "filas": d.num_rows, "formato": d.format,
                   "ejemplo": {k: str(v)[:300] for k, v in (filas[0] if filas else {}).items()}}
            if fmt == "text":
                # Texto corrido: sirve para preentrenar, no para enseñar a responder instrucciones
                out["aviso"] = ("No tiene pares pregunta/respuesta: es texto corrido. Sirve para preentrenamiento "
                                "continuado, no para un asistente. Probablemente convenga otro dataset.")
            return out

    return await anyio.to_thread.run_sync(run)


@server.tool(description="Lista los datasets de entrenamiento de la app con su formato y número de filas.")
async def list_datasets() -> list[dict]:
    from ..db import Dataset
    with Session(engine) as s:
        return [{"nombre": d.name, "formato": d.format, "filas": d.num_rows, "origen": d.source}
                for d in s.exec(select(Dataset)).all()]


@server.tool(description="Lista los flujos guardados (procesos de varios pasos definidos en la app).")
async def list_flows() -> list[dict]:
    from ..db import Flow
    with Session(engine) as s:
        return [{"nombre": f.name, "descripcion": f.description, "nodos": len(f.spec.get("nodes", []))}
                for f in s.exec(select(Flow)).all()]


@server.tool(description="Ejecuta un flujo guardado con una entrada y devuelve su resultado final. "
                         "Si el flujo tiene un nodo de aprobación, se detendrá y habrá que continuarlo desde la app.")
async def run_flow(name: str, input: str) -> dict:
    from ..db import Flow
    from ..services import flows as flows_svc

    with Session(engine, expire_on_commit=False) as s:
        f = s.exec(select(Flow).where(Flow.name == name)).first()
    if not f:
        raise ToolError(f"Flujo '{name}' no encontrado. Usa list_flows.")

    def run():
        pasos, salida, esperando = [], "", False
        for ev in flows_svc.run(f.spec, {"_entrada": input, "entrada": input}):
            if ev["type"] == "node":
                pasos.append(ev["label"])
            elif ev["type"] == "interrupt":
                esperando = True
            elif ev["type"] == "done":
                salida = ev.get("vars", {}).get("_salida", "")
            elif ev["type"] == "error":
                raise ToolError(ev["message"])
        return {"resultado": salida, "pasos": pasos,
                "pendiente_de_aprobacion": esperando} if not esperando else {
            "pasos": pasos, "pendiente_de_aprobacion": True,
            "aviso": "El flujo se detuvo en un nodo de aprobación; continúalo desde la pantalla Flujos de la app."}

    return await anyio.to_thread.run_sync(run)


# Sub-aplicación ASGI; se monta en /mcp y su session manager arranca en el lifespan de FastAPI
app = server.streamable_http_app(streamable_http_path="/")

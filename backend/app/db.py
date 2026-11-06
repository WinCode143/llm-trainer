from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, Column
from sqlmodel import Field, Session, SQLModel, create_engine, select

from .config import settings

engine = create_engine(f"sqlite:///{settings.db_path}", connect_args={"check_same_thread": False})


def now() -> datetime:
    return datetime.now(timezone.utc)


class Dataset(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    # chat | text | preference
    format: str
    path: str
    num_rows: int = 0
    source: str = ""
    created_at: datetime = Field(default_factory=now)


class Model(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    # hf (repo o carpeta local) | lora | merged | gguf
    kind: str
    path: str
    base_model: str | None = None
    job_id: int | None = None
    created_at: datetime = Field(default_factory=now)


class Job(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    # train | export
    type: str
    # queued | running | completed | failed | cancelled
    status: str = "queued"
    config: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    result: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    error: str | None = None
    pid: int | None = None
    created_at: datetime = Field(default_factory=now)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class RagCollection(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    embed_model: str
    chunk_size: int = 800
    chunk_overlap: int = 120
    # vector | bm25 | hybrid (BM25 + vectores fusionados con RRF)
    search_mode: str = "hybrid"
    # Cross-encoder para reordenar candidatos (None = sin reranking)
    reranker: str | None = None
    top_k: int = 4
    # Umbral sobre el score final (rerank 0–1 si hay reranker; si no, similitud coseno)
    min_score: float = 0.0
    # Si ningún fragmento supera el umbral: no_context (responder sin RAG) | refuse (decir que no está en los documentos)
    fallback: str = "no_context"
    # general | qa | table | laws | page | one  (cómo se trocean los documentos nuevos)
    chunk_template: str = "general"
    # auto (pymupdf4llm/python-docx, rápido) | docling (layout y tablas con modelos de visión; PPTX/XLSX)
    parser: str = "auto"
    # simple | multi (varias reformulaciones) | hyde (respuesta hipotética) | agentic (búsqueda iterativa)
    query_mode: str = "simple"
    # {archivo: {"tags": [...], "url": "..."}} para filtrar la búsqueda por documento/etiqueta
    doc_meta: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    num_chunks: int = 0
    documents: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=now)


class QuestionSet(SQLModel, table=True):
    """Conjunto de preguntas de evaluación, guardado para reutilizarlo.

    Antes las preguntas vivían dentro de cada ejecución: escribir veinte a mano y volver a usarlas
    obligaba a copiarlas de un run anterior. Guardarlas aparte es lo que permite comparar modelos
    distintos, en momentos distintos, contra la misma vara de medir."""
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    description: str = ""
    # [{question, expected, keywords: []}]
    questions: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    source: str = "manual"     # manual | generado | importado
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)


class EvalRun(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    status: str = "queued"
    config: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    # [{model_id, model_name, question, expected, answer, similarity, contains, latency_s, rating}]
    results: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    summary: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    error: str | None = None
    created_at: datetime = Field(default_factory=now)


class Agent(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    description: str = ""
    model_id: int | None = None  # modelo preferido (se carga al ejecutar si hace falta)
    helper: int | None = None    # modelo ayudante externo; si está puesto, manda sobre model_id
    system_prompt: str = ""
    # [{"id": "web_search", "config": {...}}]
    tools: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    max_steps: int = 5
    max_tokens: int = 700
    temperature: float = 0.3
    created_at: datetime = Field(default_factory=now)


class Provider(SQLModel, table=True):
    """Modelo ayudante externo: un endpoint al que delegar generaciones auxiliares."""
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    kind: str = "openai"  # openai (cualquier servidor compatible) | anthropic
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    enabled: bool = True
    note: str = ""
    created_at: datetime = Field(default_factory=now)


class McpServer(SQLModel, table=True):
    """Servidor MCP externo registrado una vez y reutilizable por todos los agentes y flujos.

    Dos transportes: «http» (un servidor ya levantado en una URL) y «stdio» (un programa que la app
    lanza y con el que habla por la entrada/salida estándar, que es como se distribuyen casi todos
    los servidores MCP conocidos: npx @modelcontextprotocol/server-…, uvx mcp-server-…)."""
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    transport: str = "stdio"          # stdio | http
    url: str = ""
    command: str = ""                 # p. ej. npx
    args: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    env: dict[str, str] = Field(default_factory=dict, sa_column=Column(JSON))  # tokens, en claro
    enabled: bool = True
    note: str = ""
    created_at: datetime = Field(default_factory=now)


class Trigger(SQLModel, table=True):
    """Algo que lanza un flujo o un agente sin que tengas que pulsar un botón en la app.

    kind: webhook (una URL secreta), telegram (un bot que escucha mensajes) o schedule (una hora).
    Un flujo que nadie puede disparar es un prototipo: esto es lo que lo conecta con el mundo."""
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    kind: str = "webhook"                 # webhook | telegram | schedule
    target_kind: str = "flow"             # flow | agent
    target_id: int = 0
    # webhook: {token} · telegram: {bot_token, allowed_chat_ids} · schedule: {cada_min | diario, input, avisar_a}
    config: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    enabled: bool = True
    last_runs: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))  # últimas 20
    created_at: datetime = Field(default_factory=now)


class AgentEvalRun(SQLModel, table=True):
    """Evaluación de un agente: ¿elige la herramienta correcta y responde bien?

    Evaluación mide modelos; esto mide agentes, donde la pregunta es otra. Con modelos pequeños lo
    que falla no es tanto el texto como la decisión de qué herramienta usar, y eso sin medirlo sólo
    se puede juzgar a ojo."""
    id: int | None = Field(default=None, primary_key=True)
    agent_id: int
    name: str = ""
    # [{question, tool, keywords}] · tool = nombre esperado o "ninguna"
    cases: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    results: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    summary: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    status: str = "running"
    created_at: datetime = Field(default_factory=now)


class Flow(SQLModel, table=True):
    """Flujo visual: nodos y conexiones que se compilan a un grafo ejecutable."""
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    description: str = ""
    # {"nodes": [{id, type, label, config, x, y}], "edges": [{from, to, when}]}
    spec: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)


class FlowRun(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    flow_id: int = Field(index=True)
    # running | waiting (aprobación) | completed | failed | cancelled
    status: str = "running"
    thread_id: str = ""
    inputs: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    state: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    trace: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    interrupt: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    error: str | None = None
    created_at: datetime = Field(default_factory=now)
    finished_at: datetime | None = None


class Conversation(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    title: str = "Nueva conversación"
    # [{role, content, sources?, stats?, ragMode?}]
    messages: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    # ajustes del playground usados en esta conversación (modelo, rag, params, system)
    settings: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)


def _migrate() -> None:
    """Añade columnas nuevas a tablas existentes (SQLite no lo hace solo con create_all)."""
    from sqlalchemy import inspect, text

    insp = inspect(engine)
    with engine.begin() as conn:
        for table in SQLModel.metadata.sorted_tables:
            if not insp.has_table(table.name):
                continue
            existing = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name in existing:
                    continue
                ddl = f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {col.type.compile(engine.dialect)}'
                default = getattr(col.default, "arg", None)
                if default is not None and not callable(default):
                    ddl += f" DEFAULT {default!r}" if not isinstance(default, str) else f" DEFAULT '{default}'"
                conn.execute(text(ddl))


def init_db() -> None:
    SQLModel.metadata.create_all(engine)
    _migrate()


def get_session():
    with Session(engine) as session:
        yield session


__all__ = ["Dataset", "Model", "Job", "RagCollection", "EvalRun", "Conversation", "Flow", "FlowRun", "Provider", "QuestionSet", "McpServer", "Trigger", "AgentEvalRun", "engine", "init_db", "get_session", "Session", "select", "now"]

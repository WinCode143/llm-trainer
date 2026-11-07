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
    # hf (repo or local folder) | lora | merged | gguf
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
    # vector | bm25 | hybrid (BM25 + vectors fused with RRF)
    search_mode: str = "hybrid"
    # Cross-encoder to reorder candidates (None = no reranking)
    reranker: str | None = None
    top_k: int = 4
    # Threshold on the final score (rerank 0–1 if there's a reranker; otherwise cosine similarity)
    min_score: float = 0.0
    # If no chunk passes the threshold: no_context (answer without RAG) | refuse (say it's not in the documents)
    fallback: str = "no_context"
    # general | qa | table | laws | page | one  (how new documents are chunked)
    chunk_template: str = "general"
    # auto (pymupdf4llm/python-docx, fast) | docling (layout and tables with vision models; PPTX/XLSX)
    parser: str = "auto"
    # simple | multi (several rephrasings) | hyde (hypothetical answer) | agentic (iterative search)
    query_mode: str = "simple"
    # {file: {"tags": [...], "url": "..."}} to filter search by document/tag
    doc_meta: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    num_chunks: int = 0
    documents: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=now)


class QuestionSet(SQLModel, table=True):
    """A set of evaluation questions, saved for reuse.

    Questions used to live inside each run: writing twenty by hand and using them again
    meant copying them from an earlier run. Saving them separately is what lets you compare different
    models, at different times, against the same yardstick."""
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    description: str = ""
    # [{question, expected, keywords: []}]
    questions: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    source: str = "manual"     # manual | generated | imported
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
    model_id: int | None = None  # preferred model (loaded when running, if needed)
    helper: int | None = None    # external helper model; when set, it takes precedence over model_id
    system_prompt: str = ""
    # [{"id": "web_search", "config": {...}}]
    tools: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    max_steps: int = 5
    max_tokens: int = 700
    temperature: float = 0.3
    created_at: datetime = Field(default_factory=now)


class Provider(SQLModel, table=True):
    """External helper model: an endpoint to hand auxiliary generations to."""
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    kind: str = "openai"  # openai (any compatible server) | anthropic
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    enabled: bool = True
    note: str = ""
    created_at: datetime = Field(default_factory=now)


class McpServer(SQLModel, table=True):
    """External MCP server, registered once and reusable by every agent and flow.

    Two transports: “http” (a server already running at a URL) and “stdio” (a program the app
    launches and talks to over standard input/output, which is how almost all well-known
    MCP servers ship: npx @modelcontextprotocol/server-…, uvx mcp-server-…)."""
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    transport: str = "stdio"          # stdio | http
    url: str = ""
    command: str = ""                 # p. ej. npx
    args: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    env: dict[str, str] = Field(default_factory=dict, sa_column=Column(JSON))  # tokens, in plain text
    enabled: bool = True
    note: str = ""
    created_at: datetime = Field(default_factory=now)


class Trigger(SQLModel, table=True):
    """Something that runs a flow or an agent without you clicking a button in the app.

    kind: webhook (a secret URL), telegram (a bot listening for messages) or schedule (a time).
    A flow nobody can trigger is a prototype: this is what connects it to the world."""
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    kind: str = "webhook"                 # webhook | telegram | schedule
    target_kind: str = "flow"             # flow | agent
    target_id: int = 0
    # webhook: {token} · telegram: {bot_token, allowed_chat_ids} · schedule: {cada_min | diario, input, avisar_a}
    config: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    enabled: bool = True
    last_runs: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))  # last 20
    created_at: datetime = Field(default_factory=now)


class AgentEvalRun(SQLModel, table=True):
    """Agent evaluation: does it pick the right tool and answer well?

    Evaluation measures models; this measures agents, where the question is a different one. With small models
    what fails isn't so much the text as the decision about which tool to use, and without measuring that
    you can only judge it by eye."""
    id: int | None = Field(default=None, primary_key=True)
    agent_id: int
    name: str = ""
    # [{question, tool, keywords}] · tool = the expected name or "none"
    cases: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    results: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    summary: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    status: str = "running"
    created_at: datetime = Field(default_factory=now)


class Flow(SQLModel, table=True):
    """Visual flow: nodes and connections compiled into a runnable graph."""
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
    # running | waiting (approval) | completed | failed | cancelled
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
    title: str = "New conversation"
    # [{role, content, sources?, stats?, ragMode?}]
    messages: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    # the playground settings used in this conversation (model, rag, params, system)
    settings: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)


def _migrate() -> None:
    """Adds new columns to existing tables (SQLite doesn't do it by itself with create_all)."""
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

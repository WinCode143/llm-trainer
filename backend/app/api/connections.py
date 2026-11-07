"""Connections: everything that talks to the outside world, in one place.

MCP plays two opposite roles that used to be mixed up: as a SERVER, this app lends its
collections, datasets and flows to others (Claude Code, Cursor…); as a CLIENT, the agents use
tools from external MCP servers. Both are shown here, next to the OpenAI-compatible API,
the helper models and the triggers.
"""
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from ..config import settings
from ..db import McpServer, get_session

router = APIRouter(prefix="/api/connections", tags=["connections"])


def _mask(env: dict) -> dict:
    # The values are usually tokens: they never go back to the interface in full
    return {k: (f"…{v[-4:]}" if v and len(v) > 4 else "…") for k, v in (env or {}).items()}


@router.get("/overview")
def overview(s: Session = Depends(get_session)):
    """Summary of what's connected."""
    from ..api.mcp_server import server as mcp
    from ..db import Provider

    tools_ = []
    try:
        import anyio
        tools_ = [t.name for t in anyio.run(mcp.list_tools)]
    except Exception as e:  # noqa: BLE001
        print("[connections] couldn't list the tools of our own MCP server:", e)
    return {
        "mcp_servidor": {"ruta": "/mcp/", "herramientas": tools_},
        "mcp_clientes": len(s.exec(select(McpServer)).all()),
        "proveedores": len(s.exec(select(Provider)).all()),
        "api_openai": "/v1",
    }


class McpBody(BaseModel):
    name: str
    transport: str = "stdio"
    url: str = ""
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True
    note: str = ""


@router.get("/mcp-servers")
def list_servers(s: Session = Depends(get_session)):
    out = []
    for m in s.exec(select(McpServer).order_by(McpServer.id)).all():
        d = m.model_dump(mode="json")
        d["env"] = _mask(m.env)
        out.append(d)
    return out


@router.post("/mcp-servers")
def create(body: McpBody, s: Session = Depends(get_session)):
    if s.exec(select(McpServer).where(McpServer.name == body.name)).first():
        raise HTTPException(409, "There's already a server with that name")
    if body.transport == "http" and not body.url:
        raise HTTPException(400, "An HTTP server needs a URL")
    if body.transport == "stdio" and not body.command:
        raise HTTPException(400, "A stdio server needs the command that launches it")
    m = McpServer(**body.model_dump())
    s.add(m)
    s.commit()
    s.refresh(m)
    return {"id": m.id}


@router.put("/mcp-servers/{mid}")
def edit(mid: int, body: McpBody, s: Session = Depends(get_session)):
    m = s.get(McpServer, mid)
    if not m:
        raise HTTPException(404)
    data = body.model_dump()
    # A masked value (“…abcd”) means it wasn't touched: the existing one is kept
    data["env"] = {k: (m.env.get(k, "") if str(v).startswith("…") else v) for k, v in data["env"].items()}
    for k, v in data.items():
        setattr(m, k, v)
    s.add(m)
    s.commit()
    return {"ok": True}


@router.delete("/mcp-servers/{mid}")
def delete(mid: int, s: Session = Depends(get_session)):
    m = s.get(McpServer, mid)
    if m:
        s.delete(m)
        s.commit()
    return {"ok": True}


@router.post("/mcp-servers/{mid}/test")
def test(mid: int):
    """Connects and lists its tools. The first time with npx can take a while: it downloads the package."""
    from ..services import tools as tools_svc
    t0 = time.time()
    try:
        listed = tools_svc._mcp_list(mid)
        return {"ok": True, "seconds": round(time.time() - t0, 1),
                "tools": [{"name": t["name"], "description": t["description"][:160],
                           "parameters": t["parameters"]} for t in listed]}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "seconds": round(time.time() - t0, 1), "error": tools_svc.error_real(e)[:400]}


# ------------------------------------------------------------------ catalog
# Checked against npm and PyPI on 2026-09-24: all published and none marked deprecated.
# Left out on purpose: server-github and server-brave-search (deprecated on npm) and mcp-server-sqlite
# (17 months without a release, and the app already has its own SQL tool).
CATALOG = [
    # The only remote one: maintained by Hugging Face and works without an account (with a token it also sees private repos).
    # It's the best fit for the app: that's where the models, datasets and papers come from.
    {"key": "huggingface", "nombre": "Hugging Face Hub", "transport": "http", "url": "https://huggingface.co/mcp",
     "command": "", "args": [],
     "da": "Search models and datasets, see their cards and read papers (hf://papers/…) as Markdown: "
           "the source of everything trained here.",
     "riesgo": "Read-only. If an HF token is configured, it's sent to huggingface.co to also see private repos."},
    {"key": "arxiv", "nombre": "arXiv", "command": "npx", "args": ["-y", "@cyanheads/arxiv-mcp-server"],
     "da": "Search arXiv papers by topic, see their metadata and read the full text. Complements Hugging Face, "
           "which lists featured papers but doesn't search by topic.",
     "riesgo": "Read-only. The first time, npx downloads the package (~30 s)."},
    {"key": "exa", "nombre": "Exa web search", "transport": "http", "url": "https://mcp.exa.ai/mcp",
     "command": "", "args": [],
     "da": "A web search engine built for agents: it returns each page's relevant text, not just the link. "
           "No account needed; better results than the built-in web search for technical topics.",
     "riesgo": "What you search for is sent to exa.ai. Without an account there's a query limit."},
    {"key": "whisper", "nombre": "Transcribe audio (local Whisper)", "command": "uvx",
     "args": ["--from", "whisper-transcribe-mcp[local]", "whisper-transcribe-mcp"],
     "env": {"WHISPER_MODEL": "small"},
     "da": "Turns recordings, lectures or meetings (mp3, wav, m4a…) into text on your machine, to index them "
           "in a collection or build a dataset from them. Runs on the CPU: it takes no VRAM from the model.",
     "riesgo": "Reads the audio file you point it to. The first time it downloads the small model (~500 MB)."},
    {"key": "filesystem", "nombre": "File system", "command": "npx",
     "args": ["-y", "@modelcontextprotocol/server-filesystem", "{folder}"],
     "param": {"key": "folder", "label": "Folder it can access", "placeholder": "/home/me/documents",
               "default": str(settings.data_dir / "inbox")},
     "da": "Read, write, search and list files INSIDE that folder. Defaults to data/inbox: "
           "drop PDFs, notes or CSVs there and an agent moves them into a collection or a dataset.",
     "riesgo": "It can modify and delete files in that folder. Give it a specific folder, never your whole /home."},
    {"key": "memory", "nombre": "Persistent memory", "command": "npx",
     "args": ["-y", "@modelcontextprotocol/server-memory"],
     # Without this it stores the graph inside npx's cache, which is wiped when the package updates
     "env": {"MEMORY_FILE_PATH": str(settings.data_dir / "mcp" / "memory.jsonl")},
     "da": "A knowledge graph where the agent stores and retrieves facts across conversations: the memory agents don't have.",
     "riesgo": "Stores whatever the agent decides to remember in a local file."},
    {"key": "fetch", "nombre": "Read web pages", "command": "uvx", "args": ["mcp-server-fetch"],
     "da": "Downloads a URL and turns it into clean text.",
     "riesgo": "Goes out to the internet, to whatever addresses the agent picks."},
    {"key": "playwright", "nombre": "Browser (Playwright)", "command": "npx", "args": ["-y", "@playwright/mcp@latest"],
     "da": "Drives a real browser: open pages, click, fill in forms, read what's on screen.",
     "riesgo": "Browses and interacts with websites on your behalf. The first time it downloads the browser (~150 MB)."},
    {"key": "git", "nombre": "Git repository", "command": "uvx",
     "args": ["mcp-server-git", "--repository", "{repository}"],
     "param": {"key": "repository", "label": "Repository path", "placeholder": "/home/me/project"},
     "da": "See status, diffs, history and branches; it can also commit.",
     "riesgo": "It can create commits in that repository."},
    {"key": "time", "nombre": "Time and time zones", "command": "uvx", "args": ["mcp-server-time"],
     "da": "The current time in any zone and conversions between zones.", "riesgo": "None: it doesn't touch anything."},
    {"key": "thinking", "nombre": "Sequential thinking", "command": "npx",
     "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
     "da": "A scratchpad where the agent breaks a hard problem into steps and reviews them.",
     "riesgo": "None. It helps big models more; it can confuse a small one."},
]


@router.get("/catalog")
def catalog(s: Session = Depends(get_session)):
    return [{**c, "instalado": _installed(s, c) is not None} for c in CATALOG]


def _installed(s: Session, c: dict) -> McpServer | None:
    for m in s.exec(select(McpServer)).all():
        if c.get("transport") == "http" and m.transport == "http" and m.url.rstrip("/") == c["url"].rstrip("/"):
            return m
        if c["command"] and (m.command, tuple(m.args or [])[:2]) == (c["command"], tuple(c["args"])[:2]):
            return m
    return None


class FromCatalog(BaseModel):
    key: str
    valor: str = ""
    name: str | None = None


@router.post("/catalog")
def from_catalog(body: FromCatalog, s: Session = Depends(get_session)):
    c = next((x for x in CATALOG if x["key"] == body.key), None)
    if not c:
        raise HTTPException(404, "It isn't in the catalog")
    p = c.get("param")
    if p and not body.valor.strip() and p.get("default"):
        body.valor = p["default"]
        Path(body.valor).mkdir(parents=True, exist_ok=True)   # the default folder is created automatically
    if p and not body.valor.strip():
        raise HTTPException(400, f"Missing: {p['label']}")
    if p:
        path = Path(body.valor).expanduser()
        if not path.exists():
            raise HTTPException(400, f"{path} doesn't exist")
        body.valor = str(path.resolve())
    args = [a.replace("{" + p["key"] + "}", body.valor) if p else a for a in c["args"]]
    name = body.name or c["key"]
    if s.exec(select(McpServer).where(McpServer.name == name)).first():
        raise HTTPException(409, f"There's already a server called “{name}”")
    if c.get("transport") == "http":
        cab = {"Authorization": f"Bearer {settings.hf_token}"} if c["key"] == "huggingface" and settings.hf_token else {}
        m = McpServer(name=name, transport="http", url=c["url"], env=cab, note=c["nombre"])
    else:
        for v in (c.get("env") or {}).values():
            if v.startswith(str(settings.data_dir)):
                Path(v).parent.mkdir(parents=True, exist_ok=True)
        m = McpServer(name=name, transport="stdio", command=c["command"], args=args, env=dict(c.get("env") or {}),
                      note=c["nombre"])
    s.add(m)
    s.commit()
    s.refresh(m)
    return {"id": m.id, "name": m.name}


# ------------------------------------------------------------------ recipes
# A lone MCP server does nothing: what's useful is an agent that combines it with this app. Each recipe
# registers whatever is needed (including our own server, so the agent can WRITE here)
# and creates the agent with only the tools it needs: with fewer tools, a local model
# chooses better (agent evaluation showed it: 67% → 100% when trimming).

OWN = "llm-trainer"

RECIPES = [
    {"key": "dataset-finder", "nombre": "Dataset finder",
     "da": "Searches Hugging Face for datasets for what you want to train, shows you the options and brings the chosen one into Datasets.",
     "ejemplo": "Find instruction datasets for an assistant and import the best one with 300 rows",
     "usa": {"huggingface": ["hub_repo_search", "hub_repo_details"], OWN: ["import_hf_dataset", "list_datasets"]},
     "prompt": ("You help find training data. 1) Search with hub_repo_search: repo_types=['dataset'], "
                "sort='downloads', limit=8, a short query (e.g. 'instruct', 'chat', 'qa') and the language "
                "as a filter (filters=['language:en'] for English). If nothing comes up, drop words from the query. "
                "2) Look at the most promising one with hub_repo_details. 3) Summarize 3 options (exact name, size, what "
                "it's for). 4) If the user asked to import one, call import_hf_dataset with the exact name, the name they "
                "asked for and a reasonable limit. If the tool returns a warning, pass it on as is. Don't say you did "
                "anything no tool confirmed.")},
    {"key": "paper-reader", "nombre": "Paper reader",
     "da": "Searches arXiv and Hugging Face Papers, reads the papers and saves the full PDF to a collection so you can query it with RAG.",
     "ejemplo": "Find the LoRA paper, tell me what it's about and save it to the papers collection",
     "usa": {"arxiv": ["arxiv_search"], "huggingface": ["hf_fs"], OWN: ["ingest_url", "list_collections"]},
     "prompt": ("You read AI research. To see papers use hf_fs with operations=[{\"cmd\": \"ls\", "
                "\"args\": [\"hf://papers/trending\"]}] (the featured ones) or with cmd \"cat\" and "
                "args [\"hf://papers/<arXiv id>/paper.md\"] (to read one; e.g. LoRA is 2106.09685). To search by "
                "topic use arxiv_search with a specific query (e.g. 'GRPO reinforcement learning reasoning') "
                "and max_results=8, WITHOUT sort_by (it sorts by relevance); of what comes back, keep what's really "
                "about the topic. "
                "To save one in full to a collection, call ingest_url with url='https://arxiv.org/pdf/<arXiv id>' "
                "(use list_collections to see the names). Summarize and cite the arXiv id.")},
    {"key": "archivist", "nombre": "Archivist",
     "da": "Checks what you drop in data/inbox and indexes it into the collection you name: drop a PDF in a folder and ask it questions.",
     "ejemplo": "Index all the new PDFs in the inbox folder into the manuals collection",
     "usa": {OWN: ["ingest_file", "list_collections"]}, "locales": ["list_files"],
     "prompt": ("You organize documents. With list_files you see what's in the inbox folder; with list_collections, "
                "the collections. To index a file call ingest_file with the collection and the path exactly as "
                "list_files shows it. When you finish, say what you indexed and how many chunks it produced.")},
    {"key": "memory-assistant", "nombre": "Assistant with memory",
     "da": "Remembers what you tell it across conversations (preferences, people, projects) in a local graph.",
     "ejemplo": "Remember that I prefer short answers and that I work with an RTX 3090",
     "usa": {"memory": ["search_nodes", "open_nodes", "create_entities", "add_observations", "create_relations"]},
     "prompt": ("You are an assistant with long-term memory. Your FIRST action on every message, without exception, is "
                "to call open_nodes with names=['User'] to remember who they are; answer with what you find. "
                "For other topics (a project, a person) use search_nodes. When the user shares a lasting fact "
                "about themselves, store it in the 'User' entity: add_observations if it already exists, "
                "create_entities (entityType 'Person') if open_nodes didn't find it. Don't store small talk.")},
]


def _ensure(s: Session, key: str) -> McpServer:
    """The server the recipe needs: the registered one if it exists; otherwise it's registered."""
    if key == OWN:
        url = f"http://127.0.0.1:{settings.port}/mcp/"
        m = next((x for x in s.exec(select(McpServer)).all()
                  if x.transport == "http" and x.url.rstrip("/").endswith(f":{settings.port}/mcp")), None)
        if not m:
            m = McpServer(name=OWN, transport="http", url=url, note="This same app")
            s.add(m)
            s.commit()
            s.refresh(m)
        return m
    c = next(x for x in CATALOG if x["key"] == key)
    m = _installed(s, c)
    if not m:
        r = from_catalog(FromCatalog(key=key, name=key if not s.exec(
            select(McpServer).where(McpServer.name == key)).first() else f"{key}-mcp"), s)
        m = s.get(McpServer, r["id"])
    if not m.enabled:
        m.enabled = True
        s.add(m)
        s.commit()
    return m


@router.get("/recipes")
def recipes(s: Session = Depends(get_session)):
    from ..db import Agent
    made = {a.name for a in s.exec(select(Agent)).all()}
    return [{**{k: r[k] for k in ("key", "nombre", "da", "ejemplo")},
             "servidores": list(r["usa"]), "creado": r["nombre"] in made} for r in RECIPES]


class Recipe(BaseModel):
    model_id: int | None = None


@router.post("/recipes/{key}")
def create_from_recipe(key: str, body: Recipe, s: Session = Depends(get_session)):
    from ..db import Agent
    r = next((x for x in RECIPES if x["key"] == key), None)
    if not r:
        raise HTTPException(404, "Unknown recipe")
    if s.exec(select(Agent).where(Agent.name == r["nombre"])).first():
        raise HTTPException(409, f"The agent “{r['nombre']}” already exists")
    tools_ = []
    for srv, only in r["usa"].items():
        m = _ensure(s, srv)
        tools_.append({"id": "mcp", "config": {"server_id": m.id, "only": ",".join(only)}})
    for tid in r.get("locales", []):
        cfg = {"root_dir": str(settings.data_dir / "inbox")} if tid in ("list_files", "read_file") else {}
        tools_.append({"id": tid, "config": cfg})
    (settings.data_dir / "inbox").mkdir(parents=True, exist_ok=True)
    a = Agent(name=r["nombre"], description=r["da"], system_prompt=r["prompt"], tools=tools_,
              model_id=body.model_id, max_steps=6, max_tokens=900, temperature=0.2)
    s.add(a)
    s.commit()
    s.refresh(a)
    return {"id": a.id, "name": a.name, "servidores": list(r["usa"])}

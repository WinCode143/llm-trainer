"""Tool catalog for the agents.

Each tool declares:
    label/desc/icon : how it shows up in the interface
    config          : settings fields (the UI draws them by itself from `type`)
    schema          : OpenAI/JSON-Schema definition passed to the model
    run(cfg, args)  : execution (returns text for the model)

Adding a tool = adding an entry to TOOLS. An external MCP server contributes all its tools
at once (discovered on the fly) without touching code.
"""
import ast
import json
import operator
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from ..config import settings

# ---------------------------------------------------------------- helpers


def _fmt_hits(hits: list[dict]) -> str:
    if not hits:
        return "No results."
    out = []
    for i, h in enumerate(hits, 1):
        src = h.get("source") or ""
        loc = f" · p. {h['page']}" if h.get("page") else ""
        out.append(f"[{i}] {src}{loc}{' · ' + h['section'] if h.get('section') else ''}\n{h['text'][:1200]}")
    return "\n\n".join(out)


# ---------------------------------------------------------------- execution


def _rag_search(cfg: dict, args: dict) -> str:
    from sqlmodel import Session

    from ..db import RagCollection, engine
    from . import rag
    cid = int(cfg.get("collection_id") or 0)
    with Session(engine, expire_on_commit=False) as s:
        coll = s.get(RagCollection, cid)
    if not coll:
        return "Error: the tool has no valid collection configured."
    hits = rag.retrieve(coll, str(args.get("query", "")), k=int(cfg.get("k") or coll.top_k or 4),
                        tags=cfg.get("tags") or None)
    return _fmt_hits(hits)


def _web_search(cfg: dict, args: dict) -> str:
    from ddgs import DDGS
    n = int(cfg.get("max_results") or 5)
    try:
        res = DDGS().text(str(args.get("query", "")), max_results=n, region=cfg.get("region") or "es-es")
    except Exception as e:  # noqa: BLE001
        return f"Web search error: {e}"
    if not res:
        return "No results."
    return "\n\n".join(f"[{i}] {r['title']}\n{r['href']}\n{r.get('body', '')[:400]}" for i, r in enumerate(res, 1))


def _fetch_url(cfg: dict, args: dict) -> str:
    from . import rag
    url = str(args.get("url", ""))
    if not url.startswith(("http://", "https://")):
        return "Error: invalid URL."
    try:
        raw = rag.fetch_url(url)
    except Exception as e:  # noqa: BLE001
        return f"Download error: {e}"
    text = "\n\n".join(t for t, _ in rag.extract(url, raw, rag.url_ext(url, raw)))
    limit = int(cfg.get("max_chars") or 6000)
    return text[:limit] + ("\n…(truncated)" if len(text) > limit else "")


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv, ast.USub: operator.neg}


DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]


def _calc(cfg: dict, args: dict) -> str:
    """Safe arithmetic evaluation (no eval): numbers and operators only."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("expression not allowed")
    expr = str(args.get("expression", "")).strip()
    # Make room for what models tend to write: thousands separators and “X% of Y”
    expr = re.sub(r"(\d)[.,](\d{3})\b", r"\1\2", expr)
    m = re.match(r"^\s*([\d.]+)\s*%\s*(?:de|of)\s*(.+)$", expr, re.I)
    if m:
        expr = f"({m.group(1)}/100)*({m.group(2)})"
    expr = expr.replace("×", "*").replace("÷", "/").replace(",", ".")
    try:
        return str(ev(ast.parse(expr, mode="eval").body))
    except Exception:  # noqa: BLE001
        return ("Error: only numbers and the operators + - * / ** % ( ) are allowed. "
                "Write the whole operation, for example: 0.12*3382662")


def _python(cfg: dict, args: dict) -> str:
    """Runs Python sandboxed with bubblewrap: no access to your /home or the project, no network unless
    enabled, and with memory, CPU and time limits."""
    from . import sandbox
    r = sandbox.run_python(str(args.get("code", "")), timeout_s=int(cfg.get("timeout_s") or 20),
                           memory_mb=int(cfg.get("memory_mb") or 2048),
                           allow_network=bool(cfg.get("allow_network")),
                           ro_dir=cfg.get("ro_dir") or None)
    parts = []
    if r["output"]:
        parts.append(r["output"])
    if r["error"]:
        parts.append(("[error]\n" if r["output"] else "") + r["error"])
    if not parts:
        parts.append("(no output: remember to use print)")
    if not r["sandboxed"]:
        parts.append("[warning] ran WITHOUT a sandbox (bubblewrap not available)")
    return "\n".join(parts)[:4000]


def _files_root(cfg: dict) -> Path:
    return Path(cfg.get("root_dir") or settings.data_dir).expanduser().resolve()


def _safe_path(cfg: dict, rel: str) -> Path | None:
    root = _files_root(cfg)
    p = (root / rel.lstrip("/")).resolve()
    return p if root in p.parents or p == root else None


def _list_files(cfg: dict, args: dict) -> str:
    p = _safe_path(cfg, str(args.get("path", "")))
    if not p or not p.exists():
        return "Error: invalid path or outside the allowed folder."
    if p.is_file():
        return f"{p.name} ({p.stat().st_size} bytes)"
    items = sorted(p.iterdir())[:200]
    return "\n".join(f"{'[dir] ' if i.is_dir() else ''}{i.name}" + ("" if i.is_dir() else f" ({i.stat().st_size} B)") for i in items) or "(empty)"


def _read_file(cfg: dict, args: dict) -> str:
    p = _safe_path(cfg, str(args.get("path", "")))
    if not p or not p.is_file():
        return "Error: file not found or outside the allowed folder."
    limit = int(cfg.get("max_chars") or 6000)
    try:
        return p.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError as e:
        return f"Error: {e}"


def _now(cfg: dict, args: dict) -> str:
    d = datetime.now().astimezone()
    return f"{DAYS[d.weekday()]}, {MONTHS[d.month - 1]} {d.day}, {d.year}, {d:%H:%M} ({d:%Z})"




# ---------------------------------------------------------------- more tools

def _sql(cfg: dict, args: dict) -> str:
    """Read-only SQL query on SQLite/Postgres/MySQL (a single SELECT statement)."""
    from sqlalchemy import create_engine, text
    url = (cfg.get("url") or "").strip()
    if not url:
        return "Error: the connection URL is missing from the tool's settings."
    q = str(args.get("query", "")).strip().rstrip(";")
    low = q.lower()
    if not low.startswith(("select", "with")) or ";" in q:
        return "Error: only a single SELECT (or WITH) statement is allowed, with no semicolon."
    if re.search(r"\b(insert|update|delete|drop|alter|create|truncate|grant|copy)\b", low):
        return "Error: the query contains a write operation."
    limit = int(cfg.get("max_rows") or 50)
    try:
        eng = create_engine(url)
        with eng.connect() as conn:
            rows = conn.execute(text(q)).mappings().fetchmany(limit)
    except Exception as e:  # noqa: BLE001
        return f"Database error: {str(e)[:400]}"
    if not rows:
        return "The query returned no rows."
    cols = list(rows[0].keys())
    out = [" | ".join(cols)] + [" | ".join(str(r[c])[:80] for c in cols) for r in rows]
    return "\n".join(out) + (f"\n({len(rows)} rows, maximum {limit})" if len(rows) == limit else "")


def _schema(cfg: dict, args: dict) -> str:
    """Tables and columns of the configured database (so the model knows what to query)."""
    from sqlalchemy import create_engine, inspect
    url = (cfg.get("url") or "").strip()
    if not url:
        return "Error: the connection URL is missing."
    try:
        insp = inspect(create_engine(url))
        out = []
        for t in insp.get_table_names()[:60]:
            cols = ", ".join(f"{c['name']}:{c['type']}" for c in insp.get_columns(t)[:40])
            out.append(f"{t}({cols})")
        return "\n".join(out) or "(no tables)"
    except Exception as e:  # noqa: BLE001
        return f"Error: {str(e)[:300]}"


def _write_file(cfg: dict, args: dict) -> str:
    """Saves text to a file inside the allowed folder (the agent can't leave it)."""
    rel = str(args.get("path", "")).strip()
    if not rel or rel.endswith("/"):
        return "Error: give a file name."
    p = _safe_path(cfg, rel)
    if not p:
        return "Error: path outside the allowed folder."
    if p.exists() and not cfg.get("overwrite"):
        return f"Error: '{rel}' already exists and the tool isn't allowed to overwrite (enable it in its settings)."
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        content = str(args.get("content", ""))
        p.write_text(content, encoding="utf-8")
        return f"Saved {p.name} ({len(content)} characters) in {p.parent}"
    except OSError as e:
        return f"Write error: {e}"


def _http(cfg: dict, args: dict) -> str:
    """HTTP request to an API, restricted to the domains you allow."""
    import urllib.error
    import urllib.request
    url = str(args.get("url", ""))
    allowed = [h.strip().lower() for h in (cfg.get("allowed_hosts") or "").split(",") if h.strip()]
    host = re.sub(r"^https?://", "", url).split("/")[0].lower()
    if not url.startswith(("http://", "https://")):
        return "Error: invalid URL."
    if not allowed:
        return "Error: the tool has no allowed domains in its settings."
    if not any(host == a or host.endswith("." + a) for a in allowed):
        return f"Error: the domain '{host}' isn't allowed. Allowed: {', '.join(allowed)}"
    from .rag import UA
    body = args.get("body")
    headers = {"User-Agent": UA, "Accept": "application/json, text/plain, */*"}
    if cfg.get("auth_header"):
        k, _, v = str(cfg["auth_header"]).partition(":")
        headers[k.strip()] = v.strip()
    data = None
    if body:
        data = (json.dumps(body) if isinstance(body, (dict, list)) else str(body)).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=str(args.get("method", "GET")).upper())
    try:
        with urllib.request.urlopen(req, timeout=int(cfg.get("timeout_s") or 20)) as r:
            text = r.read(400_000).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}: {e.read(500).decode('utf-8', 'replace')}"
    except Exception as e:  # noqa: BLE001
        return f"Network error: {e}"
    return text[: int(cfg.get("max_chars") or 4000)]


def _wikipedia(cfg: dict, args: dict) -> str:
    """Summary of a Wikipedia article (no API key)."""
    import urllib.parse
    from . import rag
    lang = cfg.get("lang") or "es"
    term = str(args.get("term", "")).strip()
    if not term:
        return "Error: give a term."
    try:
        search = json.loads(rag.fetch_url(
            f"https://{lang}.wikipedia.org/w/api.php?action=query&list=search&srsearch="
            f"{urllib.parse.quote(term)}&format=json&srlimit=3").decode())
        hits = search.get("query", {}).get("search", [])
        if not hits:
            return f"No results for “{term}”."
        title = hits[0]["title"]
        data = json.loads(rag.fetch_url(
            f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}").decode())
        extra = ", ".join(h["title"] for h in hits[1:])
        return (f"{data.get('title')}\n{data.get('extract', '')}\n{data.get('content_urls', {}).get('desktop', {}).get('page', '')}"
                + (f"\nOther articles: {extra}" if extra else ""))
    except Exception as e:  # noqa: BLE001
        return f"Error querying Wikipedia: {e}"


def _weather(cfg: dict, args: dict) -> str:
    """Current weather and forecast for a city (Open-Meteo, no key)."""
    import urllib.parse
    from . import rag
    place = str(args.get("place", "")).strip()
    try:
        geo = json.loads(rag.fetch_url(
            f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(place)}&count=1&language=es").decode())
        res = (geo.get("results") or [None])[0]
        if not res:
            return f"I can't find the place “{place}”."
        w = json.loads(rag.fetch_url(
            f"https://api.open-meteo.com/v1/forecast?latitude={res['latitude']}&longitude={res['longitude']}"
            "&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"
            "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&forecast_days=3&timezone=auto").decode())
        c, d = w["current"], w["daily"]
        out = [f"{res['name']}, {res.get('country', '')}: {c['temperature_2m']} °C, humidity {c['relative_humidity_2m']} %, "
               f"wind {c['wind_speed_10m']} km/h, precipitation {c['precipitation']} mm"]
        for i, day in enumerate(d["time"]):
            out.append(f"{day}: min {d['temperature_2m_min'][i]} °C, max {d['temperature_2m_max'][i]} °C, "
                       f"rain {d['precipitation_sum'][i]} mm")
        return "\n".join(out)
    except Exception as e:  # noqa: BLE001
        return f"Error querying the weather: {e}"


def _list_docs(cfg: dict, args: dict) -> str:
    """Which collections and documents are available (so the agent knows where to search)."""
    from sqlmodel import Session, select

    from ..db import RagCollection, engine
    from . import rag
    with Session(engine) as s:
        cols = s.exec(select(RagCollection)).all()
        cid = cfg.get("collection_id")
        cols = [c for c in cols if not cid or c.id == int(cid)]
        if not cols:
            return "There are no document collections."
        out = []
        for c in cols:
            tags = {t for m in (c.doc_meta or {}).values() for t in (m or {}).get("tags", [])}
            out.append(f"Collection “{c.name}” ({rag.count(c.id)} chunks)"
                       + (f" · tags: {', '.join(sorted(tags))}" if tags else ""))
            out += [f"  - {d}" for d in c.documents[:40]]
        return "\n".join(out)


def _graph_query(cfg: dict, args: dict) -> str:
    """Queries the knowledge graph: entities, relations and topics."""
    from sqlmodel import Session

    from ..db import RagCollection, engine
    from . import graph
    cid = int(cfg.get("collection_id") or 0)
    with Session(engine, expire_on_commit=False) as s:
        coll = s.get(RagCollection, cid)
    if not coll:
        return "Error: the tool has no valid collection configured."
    g = graph.query(coll, str(args.get("query", "")))
    if not g:
        return "There's no graph for that collection (build it in RAG → Graph) or no entity matches."
    return g["context"][:4000]


def _describe_image(cfg: dict, args: dict) -> str:
    """Describes an image on disk with the loaded vision model."""
    from .inference import GenParams, manager
    cur = manager.current
    if not cur or getattr(cur, "processor", None) is None:
        return "Error: no vision model is loaded (e.g. Qwen2.5-VL)."
    p = _safe_path(cfg, str(args.get("path", "")))
    if not p or not p.is_file():
        return "Error: image not found or outside the allowed folder."
    question = str(args.get("question") or "Describe the image in detail.")
    msgs = [{"role": "user", "content": [{"type": "image", "url": str(p)}, {"type": "text", "text": question}]}]
    try:
        return "".join(manager.stream_chat(msgs, GenParams(max_new_tokens=400, temperature=0.2, enable_thinking=False))).strip()
    except Exception as e:  # noqa: BLE001
        return f"Error describing the image: {e}"


# ---------------------------------------------------------------- catalog

def _p(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


TOOLS: dict[str, dict] = {
    "rag_search": {
        "label": "Search my documents (RAG)", "icon": "library",
        "desc": "Searches a collection with that collection's settings (hybrid, rerank, graph…).",
        "config": [
            {"key": "collection_id", "label": "Collection", "type": "collection", "required": True},
            {"key": "k", "label": "Chunks per search", "type": "number", "default": 4},
            {"key": "tags", "label": "Only documents with these tags", "type": "tags", "default": []},
        ],
        "schema": {"name": "search_documents", "description": "Searches the user's documents. Use it whenever the question could be answered from their documents.",
                   "parameters": _p({"query": {"type": "string", "description": "What to search for, in natural language"}}, ["query"])},
        "run": _rag_search,
    },
    "web_search": {
        "label": "Search the web", "icon": "globe",
        "desc": "DuckDuckGo search (no API key). Returns the title, URL and snippet.",
        "config": [
            {"key": "max_results", "label": "Results", "type": "number", "default": 5},
            {"key": "region", "label": "Region", "type": "select", "options": ["us-en", "uk-en", "es-es", "wt-wt"], "default": "wt-wt"},
        ],
        "schema": {"name": "search_web", "description": "Searches the internet for current information or anything not in the user's documents.",
                   "parameters": _p({"query": {"type": "string", "description": "Search query"}}, ["query"])},
        "run": _web_search,
    },
    "fetch_url": {
        "label": "Read a web page", "icon": "link",
        "desc": "Downloads a URL (HTML or PDF) and returns its text.",
        "config": [{"key": "max_chars", "label": "Max characters", "type": "number", "default": 6000}],
        "schema": {"name": "read_url", "description": "Downloads a web page or PDF and returns its content as text.",
                   "parameters": _p({"url": {"type": "string", "description": "Full URL (https://…)"}}, ["url"])},
        "run": _fetch_url,
    },
    "python": {
        "label": "Run Python", "icon": "terminal",
        "desc": "Runs Python sandboxed with bubblewrap: no access to your files or the network (unless you allow it), with memory and time limits. Useful for calculations and data analysis.",
        "config": [
            {"key": "timeout_s", "label": "Time limit (s)", "type": "number", "default": 20},
            {"key": "memory_mb", "label": "Max memory (MB)", "type": "number", "default": 2048},
            {"key": "allow_network", "label": "Allow internet access", "type": "bool", "default": False},
            {"key": "ro_dir", "label": "Read-only data folder (optional)", "type": "path",
             "placeholder": "mounted inside as /data"},
        ],
        "schema": {"name": "run_python", "description": "Runs Python code in a sandbox and returns what it prints (use print). There's no access to the user's files except /data if configured.",
                   "parameters": _p({"code": {"type": "string", "description": "Complete Python code"}}, ["code"])},
        "run": _python,
    },
    "calculator": {
        "label": "Calculator", "icon": "calculator",
        "desc": "Exact, safe arithmetic (without running code).",
        "config": [],
        "schema": {"name": "calculate", "description": "Evaluates an arithmetic expression (+ - * / ** %).",
                   "parameters": _p({"expression": {"type": "string", "description": "Ej.: (13*7)+2**5"}}, ["expression"])},
        "run": _calc,
    },
    "datetime": {
        "label": "Date and time", "icon": "clock",
        "desc": "The system's current date and time.",
        "config": [],
        "schema": {"name": "date_time", "description": "Returns the current date and time.", "parameters": _p({}, [])},
        "run": _now,
    },
    "list_files": {
        "label": "List files", "icon": "folder",
        "desc": "Lists files inside an allowed folder (it can't leave it).",
        "config": [{"key": "root_dir", "label": "Allowed folder", "type": "path", "default": str(settings.data_dir)}],
        "schema": {"name": "list_files", "description": "Lists the files in a folder.",
                   "parameters": _p({"path": {"type": "string", "description": "Relative path inside the allowed folder ('' = root)"}}, [])},
        "run": _list_files,
    },
    "read_file": {
        "label": "Read file", "icon": "file",
        "desc": "Reads a text file inside the allowed folder.",
        "config": [
            {"key": "root_dir", "label": "Allowed folder", "type": "path", "default": str(settings.data_dir)},
            {"key": "max_chars", "label": "Max characters", "type": "number", "default": 6000},
        ],
        "schema": {"name": "read_file", "description": "Reads the contents of a text file.",
                   "parameters": _p({"path": {"type": "string", "description": "Relative path inside the allowed folder"}}, ["path"])},
        "run": _read_file,
    },
    "sql_query": {
        "label": "Query a database (SQL)", "icon": "database",
        "desc": "Runs read-only SELECTs on SQLite, PostgreSQL or MySQL. Blocks any write.",
        "config": [
            {"key": "url", "label": "Connection URL", "type": "text", "required": True,
             "placeholder": "sqlite:////path/db.sqlite  ·  postgresql://user:password@host/db"},
            {"key": "max_rows", "label": "Max rows", "type": "number", "default": 50},
        ],
        "schema": {"name": "query_sql", "description": "Runs a SELECT query on the configured database and returns the rows.",
                   "parameters": _p({"query": {"type": "string", "description": "SELECT statement (no semicolon)"}}, ["query"])},
        "run": _sql,
    },
    "sql_schema": {
        "label": "See the database schema", "icon": "table",
        "desc": "Lists tables and columns. Use it together with the previous one so the model knows what to query.",
        "config": [{"key": "url", "label": "Connection URL", "type": "text", "required": True,
                    "placeholder": "sqlite:////path/db.sqlite"}],
        "schema": {"name": "sql_schema", "description": "Returns the database's tables and columns.",
                   "parameters": _p({}, [])},
        "run": _schema,
    },
    "write_file": {
        "label": "Save file", "icon": "save",
        "desc": "Writes text to a file inside the allowed folder (reports, notes, results).",
        "config": [
            {"key": "root_dir", "label": "Allowed folder", "type": "path", "default": str(settings.data_dir / "agents")},
            {"key": "overwrite", "label": "Allow overwriting", "type": "bool", "default": False},
        ],
        "schema": {"name": "save_file", "description": "Saves text to a file (path relative to the allowed folder).",
                   "parameters": _p({"path": {"type": "string", "description": "Name or relative path, e.g. report.md"},
                                     "content": {"type": "string", "description": "Full content"}}, ["path", "content"])},
        "run": _write_file,
    },
    "http_request": {
        "label": "Call an API (HTTP)", "icon": "plug",
        "desc": "GET/POST to the APIs you allow, by domain. You can add an authentication header.",
        "config": [
            {"key": "allowed_hosts", "label": "Allowed domains", "type": "text", "required": True,
             "placeholder": "api.example.com, my-server.local"},
            {"key": "auth_header", "label": "Auth header (optional)", "type": "text", "placeholder": "Authorization: Bearer xxx"},
            {"key": "max_chars", "label": "Max characters", "type": "number", "default": 4000},
            {"key": "timeout_s", "label": "Max time (s)", "type": "number", "default": 20},
        ],
        "schema": {"name": "call_api", "description": "Makes an HTTP request to an allowed API and returns the response.",
                   "parameters": _p({"url": {"type": "string", "description": "Full URL"},
                                     "method": {"type": "string", "description": "GET or POST (GET by default)"},
                                     "body": {"type": "object", "description": "JSON body for POST"}}, ["url"])},
        "run": _http,
    },
    "wikipedia": {
        "label": "Wikipedia", "icon": "book",
        "desc": "Looks up an article and returns its summary. More reliable than web search for encyclopedic facts.",
        "config": [{"key": "lang", "label": "Language", "type": "select", "options": ["en", "es", "pt", "fr"], "default": "en"}],
        "schema": {"name": "wikipedia", "description": "Summary of a Wikipedia article.",
                   "parameters": _p({"term": {"type": "string", "description": "Topic to look up"}}, ["term"])},
        "run": _wikipedia,
    },
    "weather": {
        "label": "Weather", "icon": "cloud",
        "desc": "Current weather and a 3-day forecast by place (Open-Meteo, no API key).",
        "config": [],
        "schema": {"name": "weather", "description": "Current weather and forecast for a place.",
                   "parameters": _p({"place": {"type": "string", "description": "City, e.g. “Montevideo”"}}, ["place"])},
        "run": _weather,
    },
    "list_documents": {
        "label": "List my documents", "icon": "folder",
        "desc": "Tells which collections and documents exist. Helps the agent decide where to search.",
        "config": [{"key": "collection_id", "label": "Only one collection (optional)", "type": "collection"}],
        "schema": {"name": "list_documents", "description": "Lists the available collections and documents.",
                   "parameters": _p({}, [])},
        "run": _list_docs,
    },
    "graph_query": {
        "label": "Query the knowledge graph", "icon": "share",
        "desc": "Entities, relations and topics from the graph (RAG → Graph). Useful for multi-hop questions.",
        "config": [{"key": "collection_id", "label": "Collection", "type": "collection", "required": True}],
        "schema": {"name": "query_graph", "description": "Returns graph entities and relations related to the question.",
                   "parameters": _p({"query": {"type": "string", "description": "Question or entities of interest"}}, ["query"])},
        "run": _graph_query,
    },
    "describe_image": {
        "label": "Describe an image", "icon": "image",
        "desc": "Describes an image on disk with the loaded vision model (needs a VLM).",
        "config": [{"key": "root_dir", "label": "Allowed folder", "type": "path", "default": str(settings.data_dir)}],
        "schema": {"name": "describe_image", "description": "Describes an image; optionally answers a question about it.",
                   "parameters": _p({"path": {"type": "string", "description": "Path relative to the allowed folder"},
                                     "question": {"type": "string", "description": "A specific question (optional)"}}, ["path"])},
        "run": _describe_image,
    },
    "mcp": {
        "label": "External MCP server", "icon": "plug",
        "desc": "Adds all the tools of an MCP (Model Context Protocol) server. They're discovered at run time.",
        "config": [
            {"key": "server_id", "label": "Registered server", "type": "mcp_server"},
            {"key": "url", "label": "…or a standalone URL", "type": "text", "placeholder": "http://localhost:8000/mcp/"},
            {"key": "only", "label": "Only these tools (comma-separated)", "type": "text", "default": ""},
        ],
        "schema": None,  # dynamic
        # In an agent the tools are discovered (resolve); in a flow node a specific one is called
        "run": lambda cfg, args: _mcp_run(cfg, args),
    },
}


def catalog() -> list[dict]:
    return [{"id": k, **{f: v[f] for f in ("label", "desc", "icon", "config")},
             "function": (v["schema"] or {}).get("name")} for k, v in TOOLS.items()]


# ---------------------------------------------------------------- external MCP

def error_real(e: BaseException) -> str:
    """The error that really happened, dug out of anyio's ExceptionGroups.

    The MCP client wraps any failure in “unhandled errors in a TaskGroup (1 sub-exception)”,
    which says nothing: not whether the command doesn't exist, nor whether the URL doesn't answer."""
    while isinstance(e, BaseExceptionGroup) and e.exceptions:
        e = e.exceptions[0]
    return f"{type(e).__name__}: {e}" if str(e) else type(e).__name__


def _mcp_server(ref) -> dict:
    """Normalizes which server to talk to: a registered id, a standalone URL or an already resolved dict."""
    if isinstance(ref, dict):
        return ref
    if isinstance(ref, int) or (isinstance(ref, str) and ref.isdigit()):
        from sqlmodel import Session

        from ..db import McpServer, engine
        with Session(engine, expire_on_commit=False) as s:
            m = s.get(McpServer, int(ref))
        if not m:
            raise RuntimeError(f"MCP server #{ref} no longer exists")
        if not m.enabled:
            raise RuntimeError(f"The MCP server “{m.name}” is disabled")
        return {"transport": m.transport, "url": m.url, "command": m.command, "args": m.args or [],
                "env": m.env or {}, "name": m.name}
    return {"transport": "http", "url": str(ref)}


def _open(server: dict):
    """Async context with the MCP session open, HTTP or stdio."""
    import contextlib
    import os

    @contextlib.asynccontextmanager
    async def ctx():
        from mcp import ClientSession
        if server.get("transport") == "stdio":
            from mcp.client.stdio import StdioServerParameters, stdio_client
            params = StdioServerParameters(command=server["command"], args=list(server.get("args") or []),
                                           env={**os.environ, **(server.get("env") or {})})
            async with stdio_client(params) as (r, w):
                async with ClientSession(r, w) as sess:
                    await sess.initialize()
                    yield sess
        else:
            from mcp.client.streamable_http import streamable_http_client
            from mcp.shared._httpx_utils import create_mcp_http_client
            # For HTTP servers, “env” holds headers: that's where “Authorization: Bearer …” goes
            client = create_mcp_http_client(headers=dict(server.get("env") or {})) if server.get("env") else None
            async with streamable_http_client(server["url"], http_client=client) as (r, w, *_):
                async with ClientSession(r, w) as sess:
                    await sess.initialize()
                    yield sess

    return ctx()


def _mcp_call(ref, name: str, args: dict, timeout: float = 120, max_chars: int = 6000) -> str:
    import asyncio
    server = _mcp_server(ref)

    async def go():
        async with _open(server) as s:
            res = await s.call_tool(name, args)
            if res.structured_content is not None:
                return json.dumps(res.structured_content, ensure_ascii=False)[:max_chars]
            return "\n".join(getattr(c, "text", "") for c in (res.content or []))[:max_chars]

    return asyncio.run(asyncio.wait_for(go(), timeout))


def _mcp_list(ref) -> list[dict]:
    import asyncio
    server = _mcp_server(ref)

    async def go():
        async with _open(server) as s:
            out = await s.list_tools()
            # mcp 2.x calls it input_schema; 1.x, inputSchema. With the old name, listing
            # tools always failed and the error was swallowed: agents ended up without them.
            return [{"name": t.name, "description": t.description or "",
                     "parameters": getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or _p({}, [])}
                    for t in out.tools]

    # stdio starts a process (npx may have to download the package the first time): more headroom
    return asyncio.run(asyncio.wait_for(go(), 120 if server.get("transport") == "stdio" else 30))


def _mcp_run(cfg: dict, args: dict) -> str:
    """Flow node “Tool → MCP”: one specific tool from a server, with fixed arguments."""
    ref = cfg.get("server_id") or (cfg.get("url") or "").strip()
    name = (cfg.get("tool") or "").strip()
    if not ref or not name:
        raise RuntimeError("Choose the MCP server and the tool to call")
    try:
        # In a flow the result goes into a prompt, not into an agent's history: there's room for more
        return _mcp_call(ref, name, _typed(args), max_chars=int(cfg.get("max_chars") or 12000))
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(error_real(e)) from e


def _typed(args: dict) -> dict:
    """A node's arguments arrive as text (after replacing {variables}); if they look like a number,
    boolean or JSON they're converted, because MCP servers validate types and reject “5” for 5."""
    out = {}
    for k, v in (args or {}).items():
        if isinstance(v, str):
            t = v.strip()
            if t and (t[0] in "[{" or t in ("true", "false", "null") or re.fullmatch(r"-?\d+(\.\d+)?", t)):
                try:
                    v = json.loads(t)
                except ValueError:
                    pass
        out[k] = v
    return out


def resolve(selected: list[dict]) -> list[dict]:
    """Turns the agent's selection [{id, config}] into runnable tools
    [{name, schema, run, tool_id, label}]. MCP ones are discovered by asking the server."""
    out: list[dict] = []
    for item in selected or []:
        tid, cfg = item.get("id"), item.get("config") or {}
        spec = TOOLS.get(tid)
        if not spec:
            continue
        if tid == "mcp":
            url = cfg.get("server_id") or (cfg.get("url") or "").strip()
            if not url:
                continue
            only = {x.strip() for x in (cfg.get("only") or "").split(",") if x.strip()}
            try:
                remote = _mcp_list(url)
            except Exception as e:  # noqa: BLE001
                print(f"[tools] couldn't list MCP server {url}: {error_real(e)}")
                continue
            for t in remote:
                if only and t["name"] not in only:
                    continue
                out.append({"name": f"mcp_{t['name']}", "label": f"MCP · {t['name']}", "tool_id": "mcp",
                            "schema": {"name": f"mcp_{t['name']}", "description": t["description"], "parameters": t["parameters"]},
                            "run": (lambda u, n: lambda c, a: _mcp_call(u, n, a))(url, t["name"])})
        else:
            out.append({"name": spec["schema"]["name"], "label": spec["label"], "tool_id": tid,
                        "schema": spec["schema"], "run": spec["run"], "config": with_defaults(tid, cfg)})
    return out


def with_defaults(tool_id: str, cfg: dict | None) -> dict:
    """Fills the config with the catalog's defaults (the UI fills them in when you tick the
    tool, but flows and templates may carry partial configs)."""
    spec = TOOLS.get(tool_id) or {}
    out = {c["key"]: c["default"] for c in spec.get("config", []) if "default" in c}
    out.update({k: v for k, v in (cfg or {}).items() if v not in (None, "")})
    return out


def execute(tool: dict, args: dict) -> tuple[str, float]:
    t0 = time.time()
    try:
        res = tool["run"](tool.get("config") or {}, args or {})
    except Exception as e:  # noqa: BLE001
        res = f"Error running the tool: {e}"
    return str(res), round(time.time() - t0, 2)


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", name)[:60]

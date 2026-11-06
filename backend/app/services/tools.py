"""Catálogo de herramientas para los agentes.

Cada herramienta declara:
  label/desc/icon : cómo se presenta en la interfaz
  config          : campos de ajuste (la UI los dibuja sola a partir de `type`)
  schema          : definición en formato OpenAI/JSON-Schema que se pasa al modelo
  run(cfg, args)  : ejecución (devuelve texto para el modelo)

Añadir una herramienta = añadir una entrada a TOOLS. Un servidor MCP externo aporta todas sus herramientas
de golpe (se descubren al vuelo) sin tocar código.
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

# ---------------------------------------------------------------- utilidades


def _fmt_hits(hits: list[dict]) -> str:
    if not hits:
        return "Sin resultados."
    out = []
    for i, h in enumerate(hits, 1):
        src = h.get("source") or ""
        loc = f" · pág. {h['page']}" if h.get("page") else ""
        out.append(f"[{i}] {src}{loc}{' · ' + h['section'] if h.get('section') else ''}\n{h['text'][:1200]}")
    return "\n\n".join(out)


# ---------------------------------------------------------------- ejecuciones


def _rag_search(cfg: dict, args: dict) -> str:
    from sqlmodel import Session

    from ..db import RagCollection, engine
    from . import rag
    cid = int(cfg.get("collection_id") or 0)
    with Session(engine, expire_on_commit=False) as s:
        coll = s.get(RagCollection, cid)
    if not coll:
        return "Error: la herramienta no tiene una colección válida configurada."
    hits = rag.retrieve(coll, str(args.get("query", "")), k=int(cfg.get("k") or coll.top_k or 4),
                        tags=cfg.get("tags") or None)
    return _fmt_hits(hits)


def _web_search(cfg: dict, args: dict) -> str:
    from ddgs import DDGS
    n = int(cfg.get("max_results") or 5)
    try:
        res = DDGS().text(str(args.get("query", "")), max_results=n, region=cfg.get("region") or "es-es")
    except Exception as e:  # noqa: BLE001
        return f"Error en la búsqueda web: {e}"
    if not res:
        return "Sin resultados."
    return "\n\n".join(f"[{i}] {r['title']}\n{r['href']}\n{r.get('body', '')[:400]}" for i, r in enumerate(res, 1))


def _fetch_url(cfg: dict, args: dict) -> str:
    from . import rag
    url = str(args.get("url", ""))
    if not url.startswith(("http://", "https://")):
        return "Error: URL no válida."
    try:
        raw = rag.fetch_url(url)
    except Exception as e:  # noqa: BLE001
        return f"Error al descargar: {e}"
    text = "\n\n".join(t for t, _ in rag.extract(url, raw, rag.url_ext(url, raw)))
    limit = int(cfg.get("max_chars") or 6000)
    return text[:limit] + ("\n…(truncado)" if len(text) > limit else "")


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv, ast.USub: operator.neg}


DAYS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MONTHS = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
          "noviembre", "diciembre"]


def _calc(cfg: dict, args: dict) -> str:
    """Evaluación aritmética segura (sin eval): sólo números y operadores."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("expresión no permitida")
    expr = str(args.get("expression", "")).strip()
    # Facilitamos lo que suelen escribir los modelos: separadores de miles y «X% de Y»
    expr = re.sub(r"(\d)[.,](\d{3})\b", r"\1\2", expr)
    m = re.match(r"^\s*([\d.]+)\s*%\s*(?:de|of)\s*(.+)$", expr, re.I)
    if m:
        expr = f"({m.group(1)}/100)*({m.group(2)})"
    expr = expr.replace("×", "*").replace("÷", "/").replace(",", ".")
    try:
        return str(ev(ast.parse(expr, mode="eval").body))
    except Exception:  # noqa: BLE001
        return ("Error: sólo se admiten números y los operadores + - * / ** % ( ). "
                "Escribe la operación completa, por ejemplo: 0.12*3382662")


def _python(cfg: dict, args: dict) -> str:
    """Ejecuta Python aislado con bubblewrap: sin acceso a tu /home ni al proyecto, sin red salvo que
    se active, y con límites de memoria, CPU y tiempo."""
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
        parts.append("(sin salida: recuerda usar print)")
    if not r["sandboxed"]:
        parts.append("[aviso] ejecutado SIN aislamiento (bubblewrap no disponible)")
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
        return "Error: ruta no válida o fuera de la carpeta permitida."
    if p.is_file():
        return f"{p.name} ({p.stat().st_size} bytes)"
    items = sorted(p.iterdir())[:200]
    return "\n".join(f"{'[dir] ' if i.is_dir() else ''}{i.name}" + ("" if i.is_dir() else f" ({i.stat().st_size} B)") for i in items) or "(vacío)"


def _read_file(cfg: dict, args: dict) -> str:
    p = _safe_path(cfg, str(args.get("path", "")))
    if not p or not p.is_file():
        return "Error: archivo no encontrado o fuera de la carpeta permitida."
    limit = int(cfg.get("max_chars") or 6000)
    try:
        return p.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError as e:
        return f"Error: {e}"


def _now(cfg: dict, args: dict) -> str:
    d = datetime.now().astimezone()
    return f"{DAYS[d.weekday()]} {d.day} de {MONTHS[d.month - 1]} de {d.year}, {d:%H:%M} ({d:%Z})"




# ---------------------------------------------------------------- herramientas adicionales

def _sql(cfg: dict, args: dict) -> str:
    """Consulta SQL de sólo lectura sobre SQLite/Postgres/MySQL (una sola sentencia SELECT)."""
    from sqlalchemy import create_engine, text
    url = (cfg.get("url") or "").strip()
    if not url:
        return "Error: falta la URL de conexión en la configuración de la herramienta."
    q = str(args.get("query", "")).strip().rstrip(";")
    low = q.lower()
    if not low.startswith(("select", "with")) or ";" in q:
        return "Error: sólo se permite una sentencia SELECT (o WITH), sin punto y coma."
    if re.search(r"\b(insert|update|delete|drop|alter|create|truncate|grant|copy)\b", low):
        return "Error: la consulta contiene una operación de escritura."
    limit = int(cfg.get("max_rows") or 50)
    try:
        eng = create_engine(url)
        with eng.connect() as conn:
            rows = conn.execute(text(q)).mappings().fetchmany(limit)
    except Exception as e:  # noqa: BLE001
        return f"Error de base de datos: {str(e)[:400]}"
    if not rows:
        return "La consulta no devolvió filas."
    cols = list(rows[0].keys())
    out = [" | ".join(cols)] + [" | ".join(str(r[c])[:80] for c in cols) for r in rows]
    return "\n".join(out) + (f"\n({len(rows)} filas, máximo {limit})" if len(rows) == limit else "")


def _schema(cfg: dict, args: dict) -> str:
    """Tablas y columnas de la base de datos configurada (para que el modelo sepa qué consultar)."""
    from sqlalchemy import create_engine, inspect
    url = (cfg.get("url") or "").strip()
    if not url:
        return "Error: falta la URL de conexión."
    try:
        insp = inspect(create_engine(url))
        out = []
        for t in insp.get_table_names()[:60]:
            cols = ", ".join(f"{c['name']}:{c['type']}" for c in insp.get_columns(t)[:40])
            out.append(f"{t}({cols})")
        return "\n".join(out) or "(sin tablas)"
    except Exception as e:  # noqa: BLE001
        return f"Error: {str(e)[:300]}"


def _write_file(cfg: dict, args: dict) -> str:
    """Guarda texto en un archivo dentro de la carpeta permitida (el agente no puede salir de ella)."""
    rel = str(args.get("path", "")).strip()
    if not rel or rel.endswith("/"):
        return "Error: indica un nombre de archivo."
    p = _safe_path(cfg, rel)
    if not p:
        return "Error: ruta fuera de la carpeta permitida."
    if p.exists() and not cfg.get("overwrite"):
        return f"Error: '{rel}' ya existe y la herramienta no puede sobrescribir (actívalo en su configuración)."
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        content = str(args.get("content", ""))
        p.write_text(content, encoding="utf-8")
        return f"Guardado {p.name} ({len(content)} caracteres) en {p.parent}"
    except OSError as e:
        return f"Error al escribir: {e}"


def _http(cfg: dict, args: dict) -> str:
    """Petición HTTP a una API, restringida a los dominios que autorices."""
    import urllib.error
    import urllib.request
    url = str(args.get("url", ""))
    allowed = [h.strip().lower() for h in (cfg.get("allowed_hosts") or "").split(",") if h.strip()]
    host = re.sub(r"^https?://", "", url).split("/")[0].lower()
    if not url.startswith(("http://", "https://")):
        return "Error: URL no válida."
    if not allowed:
        return "Error: la herramienta no tiene dominios autorizados en su configuración."
    if not any(host == a or host.endswith("." + a) for a in allowed):
        return f"Error: el dominio '{host}' no está autorizado. Permitidos: {', '.join(allowed)}"
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
        return f"Error de red: {e}"
    return text[: int(cfg.get("max_chars") or 4000)]


def _wikipedia(cfg: dict, args: dict) -> str:
    """Resumen de un artículo de Wikipedia (sin clave de API)."""
    import urllib.parse
    from . import rag
    lang = cfg.get("lang") or "es"
    term = str(args.get("term", "")).strip()
    if not term:
        return "Error: indica un término."
    try:
        search = json.loads(rag.fetch_url(
            f"https://{lang}.wikipedia.org/w/api.php?action=query&list=search&srsearch="
            f"{urllib.parse.quote(term)}&format=json&srlimit=3").decode())
        hits = search.get("query", {}).get("search", [])
        if not hits:
            return f"Sin resultados para «{term}»."
        title = hits[0]["title"]
        data = json.loads(rag.fetch_url(
            f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}").decode())
        extra = ", ".join(h["title"] for h in hits[1:])
        return (f"{data.get('title')}\n{data.get('extract', '')}\n{data.get('content_urls', {}).get('desktop', {}).get('page', '')}"
                + (f"\nOtros artículos: {extra}" if extra else ""))
    except Exception as e:  # noqa: BLE001
        return f"Error consultando Wikipedia: {e}"


def _weather(cfg: dict, args: dict) -> str:
    """Tiempo actual y previsión por ciudad (Open-Meteo, sin clave)."""
    import urllib.parse
    from . import rag
    place = str(args.get("place", "")).strip()
    try:
        geo = json.loads(rag.fetch_url(
            f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(place)}&count=1&language=es").decode())
        res = (geo.get("results") or [None])[0]
        if not res:
            return f"No encuentro la localidad «{place}»."
        w = json.loads(rag.fetch_url(
            f"https://api.open-meteo.com/v1/forecast?latitude={res['latitude']}&longitude={res['longitude']}"
            "&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"
            "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&forecast_days=3&timezone=auto").decode())
        c, d = w["current"], w["daily"]
        out = [f"{res['name']}, {res.get('country', '')}: {c['temperature_2m']} °C, humedad {c['relative_humidity_2m']} %, "
               f"viento {c['wind_speed_10m']} km/h, precipitación {c['precipitation']} mm"]
        for i, day in enumerate(d["time"]):
            out.append(f"{day}: min {d['temperature_2m_min'][i]} °C, máx {d['temperature_2m_max'][i]} °C, "
                       f"lluvia {d['precipitation_sum'][i]} mm")
        return "\n".join(out)
    except Exception as e:  # noqa: BLE001
        return f"Error consultando el tiempo: {e}"


def _list_docs(cfg: dict, args: dict) -> str:
    """Qué colecciones y documentos hay disponibles (para que el agente sepa dónde buscar)."""
    from sqlmodel import Session, select

    from ..db import RagCollection, engine
    from . import rag
    with Session(engine) as s:
        cols = s.exec(select(RagCollection)).all()
        cid = cfg.get("collection_id")
        cols = [c for c in cols if not cid or c.id == int(cid)]
        if not cols:
            return "No hay colecciones de documentos."
        out = []
        for c in cols:
            tags = {t for m in (c.doc_meta or {}).values() for t in (m or {}).get("tags", [])}
            out.append(f"Colección «{c.name}» ({rag.count(c.id)} fragmentos)"
                       + (f" · etiquetas: {', '.join(sorted(tags))}" if tags else ""))
            out += [f"  - {d}" for d in c.documents[:40]]
        return "\n".join(out)


def _graph_query(cfg: dict, args: dict) -> str:
    """Consulta el grafo de conocimiento: entidades, relaciones y temas."""
    from sqlmodel import Session

    from ..db import RagCollection, engine
    from . import graph
    cid = int(cfg.get("collection_id") or 0)
    with Session(engine, expire_on_commit=False) as s:
        coll = s.get(RagCollection, cid)
    if not coll:
        return "Error: la herramienta no tiene una colección válida configurada."
    g = graph.query(coll, str(args.get("query", "")))
    if not g:
        return "No hay grafo para esa colección (constrúyelo en RAG → Grafo) o ninguna entidad coincide."
    return g["context"][:4000]


def _describe_image(cfg: dict, args: dict) -> str:
    """Describe una imagen del disco usando el modelo de visión cargado."""
    from .inference import GenParams, manager
    cur = manager.current
    if not cur or getattr(cur, "processor", None) is None:
        return "Error: no hay ningún modelo de visión cargado (p. ej. Qwen2.5-VL)."
    p = _safe_path(cfg, str(args.get("path", "")))
    if not p or not p.is_file():
        return "Error: imagen no encontrada o fuera de la carpeta permitida."
    question = str(args.get("question") or "Describe la imagen con detalle.")
    msgs = [{"role": "user", "content": [{"type": "image", "url": str(p)}, {"type": "text", "text": question}]}]
    try:
        return "".join(manager.stream_chat(msgs, GenParams(max_new_tokens=400, temperature=0.2, enable_thinking=False))).strip()
    except Exception as e:  # noqa: BLE001
        return f"Error al describir la imagen: {e}"


# ---------------------------------------------------------------- catálogo

def _p(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


TOOLS: dict[str, dict] = {
    "rag_search": {
        "label": "Buscar en mis documentos (RAG)", "icon": "library",
        "desc": "Busca en una colección con la configuración de esa colección (híbrida, rerank, grafo…).",
        "config": [
            {"key": "collection_id", "label": "Colección", "type": "collection", "required": True},
            {"key": "k", "label": "Fragmentos por búsqueda", "type": "number", "default": 4},
            {"key": "tags", "label": "Sólo documentos con estas etiquetas", "type": "tags", "default": []},
        ],
        "schema": {"name": "buscar_documentos", "description": "Busca información en los documentos del usuario. Úsala siempre que la pregunta pueda responderse con sus documentos.",
                   "parameters": _p({"query": {"type": "string", "description": "Qué buscar, en lenguaje natural"}}, ["query"])},
        "run": _rag_search,
    },
    "web_search": {
        "label": "Buscar en la web", "icon": "globe",
        "desc": "Búsqueda en DuckDuckGo (sin clave de API). Devuelve título, URL y resumen.",
        "config": [
            {"key": "max_results", "label": "Resultados", "type": "number", "default": 5},
            {"key": "region", "label": "Región", "type": "select", "options": ["es-es", "es-ar", "mx-es", "wt-wt", "us-en"], "default": "es-es"},
        ],
        "schema": {"name": "buscar_web", "description": "Busca en internet información actual o que no esté en los documentos del usuario.",
                   "parameters": _p({"query": {"type": "string", "description": "Consulta de búsqueda"}}, ["query"])},
        "run": _web_search,
    },
    "fetch_url": {
        "label": "Leer una página web", "icon": "link",
        "desc": "Descarga una URL (HTML o PDF) y devuelve su texto.",
        "config": [{"key": "max_chars", "label": "Máx. caracteres", "type": "number", "default": 6000}],
        "schema": {"name": "leer_url", "description": "Descarga una página web o PDF y devuelve su contenido en texto.",
                   "parameters": _p({"url": {"type": "string", "description": "URL completa (https://…)"}}, ["url"])},
        "run": _fetch_url,
    },
    "python": {
        "label": "Ejecutar Python", "icon": "terminal",
        "desc": "Ejecuta código Python aislado con bubblewrap: sin acceso a tus archivos ni a la red (salvo que lo permitas), con límites de memoria y tiempo. Útil para cálculos y análisis de datos.",
        "config": [
            {"key": "timeout_s", "label": "Límite de tiempo (s)", "type": "number", "default": 20},
            {"key": "memory_mb", "label": "Memoria máx. (MB)", "type": "number", "default": 2048},
            {"key": "allow_network", "label": "Permitir acceso a internet", "type": "bool", "default": False},
            {"key": "ro_dir", "label": "Carpeta de datos en sólo lectura (opcional)", "type": "path",
             "placeholder": "se monta dentro como /datos"},
        ],
        "schema": {"name": "ejecutar_python", "description": "Ejecuta código Python en un entorno aislado y devuelve lo que imprima (usa print). No hay acceso a archivos del usuario salvo /datos si está configurado.",
                   "parameters": _p({"code": {"type": "string", "description": "Código Python completo"}}, ["code"])},
        "run": _python,
    },
    "calculator": {
        "label": "Calculadora", "icon": "calculator",
        "desc": "Aritmética exacta y segura (sin ejecutar código).",
        "config": [],
        "schema": {"name": "calcular", "description": "Evalúa una expresión aritmética (+ - * / ** %).",
                   "parameters": _p({"expression": {"type": "string", "description": "Ej.: (13*7)+2**5"}}, ["expression"])},
        "run": _calc,
    },
    "datetime": {
        "label": "Fecha y hora", "icon": "clock",
        "desc": "Fecha y hora actuales del sistema.",
        "config": [],
        "schema": {"name": "fecha_hora", "description": "Devuelve la fecha y hora actuales.", "parameters": _p({}, [])},
        "run": _now,
    },
    "list_files": {
        "label": "Listar archivos", "icon": "folder",
        "desc": "Lista archivos dentro de una carpeta permitida (no puede salir de ella).",
        "config": [{"key": "root_dir", "label": "Carpeta permitida", "type": "path", "default": str(settings.data_dir)}],
        "schema": {"name": "listar_archivos", "description": "Lista los archivos de una carpeta.",
                   "parameters": _p({"path": {"type": "string", "description": "Ruta relativa dentro de la carpeta permitida ('' = raíz)"}}, [])},
        "run": _list_files,
    },
    "read_file": {
        "label": "Leer archivo", "icon": "file",
        "desc": "Lee un archivo de texto dentro de la carpeta permitida.",
        "config": [
            {"key": "root_dir", "label": "Carpeta permitida", "type": "path", "default": str(settings.data_dir)},
            {"key": "max_chars", "label": "Máx. caracteres", "type": "number", "default": 6000},
        ],
        "schema": {"name": "leer_archivo", "description": "Lee el contenido de un archivo de texto.",
                   "parameters": _p({"path": {"type": "string", "description": "Ruta relativa dentro de la carpeta permitida"}}, ["path"])},
        "run": _read_file,
    },
    "sql_query": {
        "label": "Consultar base de datos (SQL)", "icon": "database",
        "desc": "Ejecuta SELECT de sólo lectura sobre SQLite, PostgreSQL o MySQL. Bloquea cualquier escritura.",
        "config": [
            {"key": "url", "label": "URL de conexión", "type": "text", "required": True,
             "placeholder": "sqlite:////ruta/base.db  ·  postgresql://usuario:clave@host/bd"},
            {"key": "max_rows", "label": "Máx. filas", "type": "number", "default": 50},
        ],
        "schema": {"name": "consultar_sql", "description": "Ejecuta una consulta SELECT sobre la base de datos configurada y devuelve las filas.",
                   "parameters": _p({"query": {"type": "string", "description": "Sentencia SELECT (sin punto y coma)"}}, ["query"])},
        "run": _sql,
    },
    "sql_schema": {
        "label": "Ver esquema de la base de datos", "icon": "table",
        "desc": "Lista tablas y columnas. Úsala junto a la anterior para que el modelo sepa qué consultar.",
        "config": [{"key": "url", "label": "URL de conexión", "type": "text", "required": True,
                    "placeholder": "sqlite:////ruta/base.db"}],
        "schema": {"name": "esquema_sql", "description": "Devuelve las tablas y columnas de la base de datos.",
                   "parameters": _p({}, [])},
        "run": _schema,
    },
    "write_file": {
        "label": "Guardar archivo", "icon": "save",
        "desc": "Escribe texto en un archivo dentro de la carpeta permitida (informes, notas, resultados).",
        "config": [
            {"key": "root_dir", "label": "Carpeta permitida", "type": "path", "default": str(settings.data_dir / "agentes")},
            {"key": "overwrite", "label": "Permitir sobrescribir", "type": "bool", "default": False},
        ],
        "schema": {"name": "guardar_archivo", "description": "Guarda texto en un archivo (ruta relativa a la carpeta permitida).",
                   "parameters": _p({"path": {"type": "string", "description": "Nombre o ruta relativa, p. ej. informe.md"},
                                     "content": {"type": "string", "description": "Contenido completo"}}, ["path", "content"])},
        "run": _write_file,
    },
    "http_request": {
        "label": "Llamar a una API (HTTP)", "icon": "plug",
        "desc": "GET/POST a las APIs que autorices por dominio. Puedes añadir una cabecera de autenticación.",
        "config": [
            {"key": "allowed_hosts", "label": "Dominios autorizados", "type": "text", "required": True,
             "placeholder": "api.ejemplo.com, mi-servidor.local"},
            {"key": "auth_header", "label": "Cabecera de auth (opcional)", "type": "text", "placeholder": "Authorization: Bearer xxx"},
            {"key": "max_chars", "label": "Máx. caracteres", "type": "number", "default": 4000},
            {"key": "timeout_s", "label": "Tiempo máx. (s)", "type": "number", "default": 20},
        ],
        "schema": {"name": "llamar_api", "description": "Hace una petición HTTP a una API autorizada y devuelve la respuesta.",
                   "parameters": _p({"url": {"type": "string", "description": "URL completa"},
                                     "method": {"type": "string", "description": "GET o POST (por defecto GET)"},
                                     "body": {"type": "object", "description": "Cuerpo JSON para POST"}}, ["url"])},
        "run": _http,
    },
    "wikipedia": {
        "label": "Wikipedia", "icon": "book",
        "desc": "Busca un artículo y devuelve su resumen. Más fiable que la búsqueda web para datos enciclopédicos.",
        "config": [{"key": "lang", "label": "Idioma", "type": "select", "options": ["es", "en", "pt", "fr"], "default": "es"}],
        "schema": {"name": "wikipedia", "description": "Resumen de un artículo de Wikipedia.",
                   "parameters": _p({"term": {"type": "string", "description": "Tema a buscar"}}, ["term"])},
        "run": _wikipedia,
    },
    "weather": {
        "label": "Tiempo meteorológico", "icon": "cloud",
        "desc": "Tiempo actual y previsión a 3 días por localidad (Open-Meteo, sin clave de API).",
        "config": [],
        "schema": {"name": "tiempo", "description": "Tiempo actual y previsión de una localidad.",
                   "parameters": _p({"place": {"type": "string", "description": "Ciudad, p. ej. «Montevideo»"}}, ["place"])},
        "run": _weather,
    },
    "list_documents": {
        "label": "Listar mis documentos", "icon": "folder",
        "desc": "Dice qué colecciones y documentos existen. Ayuda al agente a decidir dónde buscar.",
        "config": [{"key": "collection_id", "label": "Sólo una colección (opcional)", "type": "collection"}],
        "schema": {"name": "listar_documentos", "description": "Lista las colecciones y documentos disponibles.",
                   "parameters": _p({}, [])},
        "run": _list_docs,
    },
    "graph_query": {
        "label": "Consultar grafo de conocimiento", "icon": "share",
        "desc": "Entidades, relaciones y temas del grafo (RAG → Grafo). Útil para preguntas de varios saltos.",
        "config": [{"key": "collection_id", "label": "Colección", "type": "collection", "required": True}],
        "schema": {"name": "consultar_grafo", "description": "Devuelve entidades y relaciones del grafo relacionadas con la pregunta.",
                   "parameters": _p({"query": {"type": "string", "description": "Pregunta o entidades de interés"}}, ["query"])},
        "run": _graph_query,
    },
    "describe_image": {
        "label": "Describir una imagen", "icon": "image",
        "desc": "Describe una imagen del disco con el modelo de visión cargado (requiere un VLM).",
        "config": [{"key": "root_dir", "label": "Carpeta permitida", "type": "path", "default": str(settings.data_dir)}],
        "schema": {"name": "describir_imagen", "description": "Describe una imagen; opcionalmente responde una pregunta sobre ella.",
                   "parameters": _p({"path": {"type": "string", "description": "Ruta relativa a la carpeta permitida"},
                                     "question": {"type": "string", "description": "Pregunta concreta (opcional)"}}, ["path"])},
        "run": _describe_image,
    },
    "mcp": {
        "label": "Servidor MCP externo", "icon": "plug",
        "desc": "Añade todas las herramientas de un servidor MCP (Model Context Protocol). Se descubren al ejecutar.",
        "config": [
            {"key": "server_id", "label": "Servidor registrado", "type": "mcp_server"},
            {"key": "url", "label": "…o una URL suelta", "type": "text", "placeholder": "http://localhost:8000/mcp/"},
            {"key": "only", "label": "Sólo estas herramientas (coma)", "type": "text", "default": ""},
        ],
        "schema": None,  # dinámico
        # En un agente las herramientas se descubren (resolve); en un nodo de flujo se llama a una concreta
        "run": lambda cfg, args: _mcp_run(cfg, args),
    },
}


def catalog() -> list[dict]:
    return [{"id": k, **{f: v[f] for f in ("label", "desc", "icon", "config")},
             "function": (v["schema"] or {}).get("name")} for k, v in TOOLS.items()]


# ---------------------------------------------------------------- MCP externo

def error_real(e: BaseException) -> str:
    """El error que de verdad pasó, sacado de dentro de los ExceptionGroup de anyio.

    El cliente MCP envuelve cualquier fallo en «unhandled errors in a TaskGroup (1 sub-exception)»,
    que no dice nada: ni si el comando no existe, ni si la URL no responde."""
    while isinstance(e, BaseExceptionGroup) and e.exceptions:
        e = e.exceptions[0]
    return f"{type(e).__name__}: {e}" if str(e) else type(e).__name__


def _mcp_server(ref) -> dict:
    """Normaliza a qué servidor hablar: un id registrado, una URL suelta o un dict ya resuelto."""
    if isinstance(ref, dict):
        return ref
    if isinstance(ref, int) or (isinstance(ref, str) and ref.isdigit()):
        from sqlmodel import Session

        from ..db import McpServer, engine
        with Session(engine, expire_on_commit=False) as s:
            m = s.get(McpServer, int(ref))
        if not m:
            raise RuntimeError(f"El servidor MCP #{ref} ya no existe")
        if not m.enabled:
            raise RuntimeError(f"El servidor MCP «{m.name}» está desactivado")
        return {"transport": m.transport, "url": m.url, "command": m.command, "args": m.args or [],
                "env": m.env or {}, "name": m.name}
    return {"transport": "http", "url": str(ref)}


def _abrir(server: dict):
    """Contexto asíncrono con la sesión MCP abierta, sea HTTP o stdio."""
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
            # En los servidores HTTP, «env» son cabeceras: ahí va el «Authorization: Bearer …»
            cliente = create_mcp_http_client(headers=dict(server.get("env") or {})) if server.get("env") else None
            async with streamable_http_client(server["url"], http_client=cliente) as (r, w, *_):
                async with ClientSession(r, w) as sess:
                    await sess.initialize()
                    yield sess

    return ctx()


def _mcp_call(ref, name: str, args: dict, timeout: float = 120, max_chars: int = 6000) -> str:
    import asyncio
    server = _mcp_server(ref)

    async def go():
        async with _abrir(server) as s:
            res = await s.call_tool(name, args)
            if res.structured_content is not None:
                return json.dumps(res.structured_content, ensure_ascii=False)[:max_chars]
            return "\n".join(getattr(c, "text", "") for c in (res.content or []))[:max_chars]

    return asyncio.run(asyncio.wait_for(go(), timeout))


def _mcp_list(ref) -> list[dict]:
    import asyncio
    server = _mcp_server(ref)

    async def go():
        async with _abrir(server) as s:
            out = await s.list_tools()
            # mcp 2.x lo llama input_schema; la 1.x, inputSchema. Con el nombre viejo, listar
            # herramientas fallaba siempre y el error se tragaba: los agentes se quedaban sin ellas.
            return [{"name": t.name, "description": t.description or "",
                     "parameters": getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or _p({}, [])}
                    for t in out.tools]

    # stdio arranca un proceso (npx puede tener que descargar el paquete la primera vez): más margen
    return asyncio.run(asyncio.wait_for(go(), 120 if server.get("transport") == "stdio" else 30))


def _mcp_run(cfg: dict, args: dict) -> str:
    """Nodo de flujo «Herramienta → MCP»: una herramienta concreta de un servidor, con argumentos fijos."""
    ref = cfg.get("server_id") or (cfg.get("url") or "").strip()
    nombre = (cfg.get("tool") or "").strip()
    if not ref or not nombre:
        raise RuntimeError("Elige el servidor MCP y la herramienta a llamar")
    try:
        # En un flujo el resultado va a un prompt, no al historial de un agente: cabe más
        return _mcp_call(ref, nombre, _tipar(args), max_chars=int(cfg.get("max_chars") or 12000))
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(error_real(e)) from e


def _tipar(args: dict) -> dict:
    """Los argumentos de un nodo llegan como texto (tras sustituir {variables}); si parecen número,
    booleano o JSON se convierten, que los servidores MCP validan el tipo y rechazan «5» por 5."""
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
    """Convierte la selección del agente [{id, config}] en herramientas ejecutables
    [{name, schema, run, tool_id, label}]. Las de MCP se descubren consultando el servidor."""
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
                print(f"[tools] no se pudo listar el servidor MCP {url}: {error_real(e)}")
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
    """Completa la configuración con los valores por defecto del catálogo (la UI los rellena al marcar la
    herramienta, pero los flujos y las plantillas pueden traer configuraciones parciales)."""
    spec = TOOLS.get(tool_id) or {}
    out = {c["key"]: c["default"] for c in spec.get("config", []) if "default" in c}
    out.update({k: v for k, v in (cfg or {}).items() if v not in (None, "")})
    return out


def execute(tool: dict, args: dict) -> tuple[str, float]:
    t0 = time.time()
    try:
        res = tool["run"](tool.get("config") or {}, args or {})
    except Exception as e:  # noqa: BLE001
        res = f"Error ejecutando la herramienta: {e}"
    return str(res), round(time.time() - t0, 2)


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", name)[:60]

"""Conexiones: todo lo que habla con el exterior, en un solo sitio.

MCP cumple dos papeles opuestos que antes estaban mezclados: como SERVIDOR, esta app presta sus
colecciones, datasets y flujos a otros (Claude Code, Cursor…); como CLIENTE, los agentes usan
herramientas de servidores MCP externos. Aquí se ven los dos, junto a la API compatible con OpenAI,
los modelos ayudantes y los disparadores.
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
    # Los valores suelen ser tokens: nunca vuelven enteros a la interfaz
    return {k: (f"…{v[-4:]}" if v and len(v) > 4 else "…") for k, v in (env or {}).items()}


@router.get("/overview")
def overview(s: Session = Depends(get_session)):
    """Resumen de lo que está conectado."""
    from ..api.mcp_server import server as mcp
    from ..db import Provider

    herramientas = []
    try:
        import anyio
        herramientas = [t.name for t in anyio.run(mcp.list_tools)]
    except Exception as e:  # noqa: BLE001
        print("[connections] no se pudieron listar las herramientas del servidor MCP propio:", e)
    return {
        "mcp_servidor": {"ruta": "/mcp/", "herramientas": herramientas},
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
def listar(s: Session = Depends(get_session)):
    out = []
    for m in s.exec(select(McpServer).order_by(McpServer.id)).all():
        d = m.model_dump(mode="json")
        d["env"] = _mask(m.env)
        out.append(d)
    return out


@router.post("/mcp-servers")
def crear(body: McpBody, s: Session = Depends(get_session)):
    if s.exec(select(McpServer).where(McpServer.name == body.name)).first():
        raise HTTPException(409, "Ya hay un servidor con ese nombre")
    if body.transport == "http" and not body.url:
        raise HTTPException(400, "Un servidor HTTP necesita URL")
    if body.transport == "stdio" and not body.command:
        raise HTTPException(400, "Un servidor stdio necesita el comando que lo lanza")
    m = McpServer(**body.model_dump())
    s.add(m)
    s.commit()
    s.refresh(m)
    return {"id": m.id}


@router.put("/mcp-servers/{mid}")
def editar(mid: int, body: McpBody, s: Session = Depends(get_session)):
    m = s.get(McpServer, mid)
    if not m:
        raise HTTPException(404)
    data = body.model_dump()
    # Un valor enmascarado («…abcd») es que no lo tocaron: se conserva el que había
    data["env"] = {k: (m.env.get(k, "") if str(v).startswith("…") else v) for k, v in data["env"].items()}
    for k, v in data.items():
        setattr(m, k, v)
    s.add(m)
    s.commit()
    return {"ok": True}


@router.delete("/mcp-servers/{mid}")
def borrar(mid: int, s: Session = Depends(get_session)):
    m = s.get(McpServer, mid)
    if m:
        s.delete(m)
        s.commit()
    return {"ok": True}


@router.post("/mcp-servers/{mid}/test")
def probar(mid: int):
    """Conecta y lista sus herramientas. La primera vez con npx puede tardar: descarga el paquete."""
    from ..services import tools as tools_svc
    t0 = time.time()
    try:
        lista = tools_svc._mcp_list(mid)
        return {"ok": True, "seconds": round(time.time() - t0, 1),
                "tools": [{"name": t["name"], "description": t["description"][:160],
                           "parameters": t["parameters"]} for t in lista]}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "seconds": round(time.time() - t0, 1), "error": tools_svc.error_real(e)[:400]}


# ------------------------------------------------------------------ catálogo
# Verificados contra npm y PyPI el 24/09/2026: todos publicados y sin marca de deprecación.
# Fuera a propósito: server-github y server-brave-search (deprecados en npm) y mcp-server-sqlite
# (17 meses sin versión, y la app ya trae su propia herramienta SQL).
CATALOGO = [
    # El único remoto: lo mantiene Hugging Face y funciona sin cuenta (con token, además, ve lo privado).
    # Es el que más encaja con la app: de ahí salen los modelos, los datasets y los papers.
    {"key": "huggingface", "nombre": "Hugging Face Hub", "transport": "http", "url": "https://huggingface.co/mcp",
     "command": "", "args": [],
     "da": "Buscar modelos y datasets, ver sus fichas y leer papers (hf://papers/…) en Markdown: "
           "el origen de todo lo que se entrena aquí.",
     "riesgo": "Sólo lectura. Si hay token de HF configurado, se envía a huggingface.co para ver también lo privado."},
    {"key": "arxiv", "nombre": "arXiv", "command": "npx", "args": ["-y", "@cyanheads/arxiv-mcp-server"],
     "da": "Buscar papers por tema en arXiv, ver sus datos y leer el texto completo. Complementa a Hugging Face, "
           "que lista los destacados pero no busca por tema.",
     "riesgo": "Sólo lectura. La primera vez npx descarga el paquete (~30 s)."},
    {"key": "exa", "nombre": "Búsqueda web Exa", "transport": "http", "url": "https://mcp.exa.ai/mcp",
     "command": "", "args": [],
     "da": "Buscador web pensado para agentes: devuelve el texto relevante de cada página, no sólo el enlace. "
           "Sin cuenta; mejores resultados que la búsqueda web integrada para temas técnicos.",
     "riesgo": "Lo que buscas se envía a exa.ai. Sin cuenta tiene un límite de consultas."},
    {"key": "whisper", "nombre": "Transcribir audio (Whisper local)", "command": "uvx",
     "args": ["--from", "whisper-transcribe-mcp[local]", "whisper-transcribe-mcp"],
     "env": {"WHISPER_MODEL": "small"},
     "da": "Pasa a texto grabaciones, clases o reuniones (mp3, wav, m4a…) en tu máquina, para indexarlas "
           "en una colección o sacar de ahí un dataset. Corre en CPU: no le quita VRAM al modelo.",
     "riesgo": "Lee el archivo de audio que le indiques. La primera vez descarga el modelo small (~500 MB)."},
    {"key": "filesystem", "nombre": "Sistema de archivos", "command": "npx",
     "args": ["-y", "@modelcontextprotocol/server-filesystem", "{carpeta}"],
     "param": {"key": "carpeta", "label": "Carpeta a la que puede acceder", "placeholder": "/home/yo/documentos",
               "default": str(settings.data_dir / "entrada")},
     "da": "Leer, escribir, buscar y listar archivos DENTRO de esa carpeta. Por defecto data/entrada: "
           "deja ahí PDFs, notas o CSV y un agente los pasa a una colección o a un dataset.",
     "riesgo": "Puede modificar y borrar archivos en esa carpeta. Dale una carpeta concreta, nunca tu /home entero."},
    {"key": "memory", "nombre": "Memoria persistente", "command": "npx",
     "args": ["-y", "@modelcontextprotocol/server-memory"],
     # Sin esto guarda el grafo dentro de la caché de npx, que se borra al actualizar el paquete
     "env": {"MEMORY_FILE_PATH": str(settings.data_dir / "mcp" / "memoria.jsonl")},
     "da": "Un grafo de conocimiento donde el agente guarda y recupera hechos entre conversaciones: la memoria que los agentes no tienen.",
     "riesgo": "Guarda lo que el agente decida recordar en un archivo local."},
    {"key": "fetch", "nombre": "Leer páginas web", "command": "uvx", "args": ["mcp-server-fetch"],
     "da": "Descarga una URL y la convierte a texto limpio.",
     "riesgo": "Sale a internet a las direcciones que el agente elija."},
    {"key": "playwright", "nombre": "Navegador (Playwright)", "command": "npx", "args": ["-y", "@playwright/mcp@latest"],
     "da": "Controla un navegador de verdad: abrir páginas, hacer clic, rellenar formularios, leer lo que se ve.",
     "riesgo": "Navega e interactúa con webs en tu nombre. La primera vez descarga el navegador (~150 MB)."},
    {"key": "git", "nombre": "Repositorio git", "command": "uvx",
     "args": ["mcp-server-git", "--repository", "{repositorio}"],
     "param": {"key": "repositorio", "label": "Ruta del repositorio", "placeholder": "/home/yo/proyecto"},
     "da": "Ver estado, diferencias, historial y ramas; también hacer commits.",
     "riesgo": "Puede crear commits en ese repositorio."},
    {"key": "time", "nombre": "Hora y zonas horarias", "command": "uvx", "args": ["mcp-server-time"],
     "da": "La hora actual en cualquier zona y conversiones entre zonas.", "riesgo": "Ninguno: no toca nada."},
    {"key": "thinking", "nombre": "Pensamiento secuencial", "command": "npx",
     "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
     "da": "Un cuaderno donde el agente descompone un problema difícil en pasos y los revisa.",
     "riesgo": "Ninguno. Ayuda más a modelos grandes; a uno pequeño puede liarle."},
]


@router.get("/catalogo")
def catalogo(s: Session = Depends(get_session)):
    return [{**c, "instalado": _instalado(s, c) is not None} for c in CATALOGO]


def _instalado(s: Session, c: dict) -> McpServer | None:
    for m in s.exec(select(McpServer)).all():
        if c.get("transport") == "http" and m.transport == "http" and m.url.rstrip("/") == c["url"].rstrip("/"):
            return m
        if c["command"] and (m.command, tuple(m.args or [])[:2]) == (c["command"], tuple(c["args"])[:2]):
            return m
    return None


class DesdeCatalogo(BaseModel):
    key: str
    valor: str = ""
    name: str | None = None


@router.post("/catalogo")
def desde_catalogo(body: DesdeCatalogo, s: Session = Depends(get_session)):
    c = next((x for x in CATALOGO if x["key"] == body.key), None)
    if not c:
        raise HTTPException(404, "No está en el catálogo")
    p = c.get("param")
    if p and not body.valor.strip() and p.get("default"):
        body.valor = p["default"]
        Path(body.valor).mkdir(parents=True, exist_ok=True)   # la carpeta por defecto se crea sola
    if p and not body.valor.strip():
        raise HTTPException(400, f"Falta: {p['label']}")
    if p:
        ruta = Path(body.valor).expanduser()
        if not ruta.exists():
            raise HTTPException(400, f"No existe {ruta}")
        body.valor = str(ruta.resolve())
    args = [a.replace("{" + p["key"] + "}", body.valor) if p else a for a in c["args"]]
    nombre = body.name or c["key"]
    if s.exec(select(McpServer).where(McpServer.name == nombre)).first():
        raise HTTPException(409, f"Ya hay un servidor llamado «{nombre}»")
    if c.get("transport") == "http":
        cab = {"Authorization": f"Bearer {settings.hf_token}"} if c["key"] == "huggingface" and settings.hf_token else {}
        m = McpServer(name=nombre, transport="http", url=c["url"], env=cab, note=c["nombre"])
    else:
        for v in (c.get("env") or {}).values():
            if v.startswith(str(settings.data_dir)):
                Path(v).parent.mkdir(parents=True, exist_ok=True)
        m = McpServer(name=nombre, transport="stdio", command=c["command"], args=args, env=dict(c.get("env") or {}),
                      note=c["nombre"])
    s.add(m)
    s.commit()
    s.refresh(m)
    return {"id": m.id, "name": m.name}


# ------------------------------------------------------------------ recetas
# Un servidor MCP suelto no hace nada: lo útil es un agente que lo combine con esta app. Cada receta
# registra lo que haga falta (incluido nuestro propio servidor, para que el agente pueda ESCRIBIR aquí)
# y crea el agente con sólo las herramientas necesarias: con menos herramientas, un modelo local
# elige mejor (la evaluación de agentes lo mostró: 67 % → 100 % al recortar).

PROPIO = "llm-trainer"

RECETAS = [
    {"key": "buscador-datasets", "nombre": "Buscador de datasets",
     "da": "Busca en Hugging Face datasets para lo que quieres entrenar, te enseña las opciones y trae el elegido a Datasets.",
     "ejemplo": "Busca datasets de instrucciones en español para un asistente y trae el mejor con 300 filas",
     "usa": {"huggingface": ["hub_repo_search", "hub_repo_details"], PROPIO: ["import_hf_dataset", "list_datasets"]},
     "prompt": ("Ayudas a encontrar datos de entrenamiento. 1) Busca con hub_repo_search: repo_types=['dataset'], "
                "sort='downloads', limit=8, la query EN INGLÉS y corta (p. ej. 'instruct', 'chat', 'qa') y el idioma "
                "como filtro (filters=['language:es'] para español). Si no sale nada, quita palabras de la query. "
                "2) Mira el más prometedor con hub_repo_details. 3) Resume 3 opciones (nombre exacto, tamaño, para qué "
                "sirve). 4) Si el usuario pidió traer uno, llama a import_hf_dataset con el nombre exacto, el name que "
                "pidió y un limit razonable. Si la herramienta devuelve un aviso, díselo tal cual. No digas que hiciste "
                "algo que ninguna herramienta confirmó. Responde en español.")},
    {"key": "lector-papers", "nombre": "Lector de papers",
     "da": "Busca papers en arXiv y Hugging Face Papers, los lee y guarda el PDF completo en una colección para consultarlo con RAG.",
     "ejemplo": "Busca el paper de LoRA, dime de qué trata y guárdalo en la colección papers",
     "usa": {"arxiv": ["arxiv_search"], "huggingface": ["hf_fs"], PROPIO: ["ingest_url", "list_collections"]},
     "prompt": ("Lees investigación sobre IA. Para ver papers usa hf_fs con operations=[{\"cmd\": \"ls\", "
                "\"args\": [\"hf://papers/trending\"]}] (los destacados) o con cmd \"cat\" y "
                "args [\"hf://papers/<id arXiv>/paper.md\"] (leer uno; p. ej. LoRA es 2106.09685). Para buscar por "
                "tema usa arxiv_search con query en inglés, concreta (p. ej. 'GRPO reinforcement learning reasoning') "
                "y max_results=8, SIN sort_by (ordena por relevancia); de lo que vuelva, quédate con lo que trate "
                "de verdad el tema. "
                "Para guardarlo entero en una colección llama a ingest_url con url='https://arxiv.org/pdf/<id arXiv>' "
                "(usa list_collections para ver los nombres). Resume en español y cita el id arXiv.")},
    {"key": "archivista", "nombre": "Archivista",
     "da": "Revisa lo que dejas en data/entrada y lo indexa en la colección que digas: soltar un PDF en una carpeta y preguntarle.",
     "ejemplo": "Indexa en la colección manuales todos los PDF nuevos de la carpeta de entrada",
     "usa": {PROPIO: ["ingest_file", "list_collections"]}, "locales": ["list_files"],
     "prompt": ("Organizas documentos. Con list_files ves lo que hay en la carpeta de entrada; con list_collections, "
                "las colecciones. Para indexar un archivo llama a ingest_file con la colección y la ruta tal como "
                "sale en list_files. Al terminar di qué indexaste y cuántos fragmentos generó. Responde en español.")},
    {"key": "asistente-memoria", "nombre": "Asistente con memoria",
     "da": "Recuerda lo que le cuentas entre conversaciones (preferencias, personas, proyectos) en un grafo local.",
     "ejemplo": "Acuérdate de que prefiero respuestas cortas y que trabajo con una RTX 3090",
     "usa": {"memory": ["search_nodes", "open_nodes", "create_entities", "add_observations", "create_relations"]},
     "prompt": ("Eres un asistente con memoria a largo plazo. Tu PRIMERA acción en cada mensaje, sin excepción, es "
                "llamar a open_nodes con names=['Usuario'] para recordar quién es; responde con lo que encuentres. "
                "Para otros temas (un proyecto, una persona) usa search_nodes. Cuando el usuario cuente un hecho "
                "duradero sobre sí mismo, guárdalo en la entidad 'Usuario': add_observations si ya existe, "
                "create_entities (entityType 'Persona') si open_nodes no la encontró. No guardes charla trivial. "
                "Responde en español.")},
]


def _asegurar(s: Session, key: str) -> McpServer:
    """El servidor que la receta necesita: el ya registrado si existe; si no, se registra."""
    if key == PROPIO:
        url = f"http://127.0.0.1:{settings.port}/mcp/"
        m = next((x for x in s.exec(select(McpServer)).all()
                  if x.transport == "http" and x.url.rstrip("/").endswith(f":{settings.port}/mcp")), None)
        if not m:
            m = McpServer(name=PROPIO, transport="http", url=url, note="Esta misma app")
            s.add(m)
            s.commit()
            s.refresh(m)
        return m
    c = next(x for x in CATALOGO if x["key"] == key)
    m = _instalado(s, c)
    if not m:
        r = desde_catalogo(DesdeCatalogo(key=key, name=key if not s.exec(
            select(McpServer).where(McpServer.name == key)).first() else f"{key}-mcp"), s)
        m = s.get(McpServer, r["id"])
    if not m.enabled:
        m.enabled = True
        s.add(m)
        s.commit()
    return m


@router.get("/recetas")
def recetas(s: Session = Depends(get_session)):
    from ..db import Agent
    hechos = {a.name for a in s.exec(select(Agent)).all()}
    return [{**{k: r[k] for k in ("key", "nombre", "da", "ejemplo")},
             "servidores": list(r["usa"]), "creado": r["nombre"] in hechos} for r in RECETAS]


class Receta(BaseModel):
    model_id: int | None = None


@router.post("/recetas/{key}")
def crear_receta(key: str, body: Receta, s: Session = Depends(get_session)):
    from ..db import Agent
    r = next((x for x in RECETAS if x["key"] == key), None)
    if not r:
        raise HTTPException(404, "Receta desconocida")
    if s.exec(select(Agent).where(Agent.name == r["nombre"])).first():
        raise HTTPException(409, f"Ya existe el agente «{r['nombre']}»")
    herramientas = []
    for srv, solo in r["usa"].items():
        m = _asegurar(s, srv)
        herramientas.append({"id": "mcp", "config": {"server_id": m.id, "only": ",".join(solo)}})
    for tid in r.get("locales", []):
        cfg = {"root_dir": str(settings.data_dir / "entrada")} if tid in ("list_files", "read_file") else {}
        herramientas.append({"id": tid, "config": cfg})
    (settings.data_dir / "entrada").mkdir(parents=True, exist_ok=True)
    a = Agent(name=r["nombre"], description=r["da"], system_prompt=r["prompt"], tools=herramientas,
              model_id=body.model_id, max_steps=6, max_tokens=900, temperature=0.2)
    s.add(a)
    s.commit()
    s.refresh(a)
    return {"id": a.id, "name": a.name, "servidores": list(r["usa"])}

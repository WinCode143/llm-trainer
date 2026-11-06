from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from ..db import Agent, get_session
from ..services import agents as svc
from ..services import tools as tools_svc
from ..services.inference import GenParams

router = APIRouter(prefix="/api/agents", tags=["agents"])


@router.get("/tools")
def tool_catalog():
    """Catálogo de herramientas con sus campos de configuración (la UI los dibuja a partir de esto)."""
    return tools_svc.catalog()


class McpProbe(BaseModel):
    url: str


@router.post("/tools/mcp/probe")
def probe_mcp(body: McpProbe):
    """Lista las herramientas de un servidor MCP externo (para elegirlas en la interfaz)."""
    try:
        return {"tools": tools_svc._mcp_list(body.url)}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"No se pudo conectar: {e}")


class AgentBody(BaseModel):
    name: str
    description: str = ""
    model_id: int | None = None
    helper: int | None = None
    system_prompt: str = ""
    tools: list[dict] = Field(default_factory=list)
    max_steps: int = 5
    max_tokens: int = 700
    temperature: float = 0.3


@router.get("")
def list_agents(s: Session = Depends(get_session)):
    return s.exec(select(Agent).order_by(Agent.id)).all()


@router.post("")
def create(body: AgentBody, s: Session = Depends(get_session)):
    if s.exec(select(Agent).where(Agent.name == body.name)).first():
        raise HTTPException(409, "Ya existe un agente con ese nombre")
    a = Agent(**body.model_dump())
    s.add(a)
    s.commit()
    s.refresh(a)
    return a


@router.put("/{aid}")
def update(aid: int, body: AgentBody, s: Session = Depends(get_session)):
    a = s.get(Agent, aid)
    if not a:
        raise HTTPException(404)
    for k, v in body.model_dump().items():
        setattr(a, k, v)
    s.add(a)
    s.commit()
    s.refresh(a)
    return a


@router.delete("/{aid}")
def delete(aid: int, s: Session = Depends(get_session)):
    a = s.get(Agent, aid)
    if a:
        s.delete(a)
        s.commit()
    return {"ok": True}


class RunBody(BaseModel):
    messages: list[dict]
    max_steps: int | None = None


@router.post("/{aid}/run")
def run(aid: int, body: RunBody, s: Session = Depends(get_session)):
    """Ejecuta el agente en streaming (NDJSON): tokens, llamadas a herramientas y resultados."""
    from .chat import ndjson_stream
    a = s.get(Agent, aid)
    if not a:
        raise HTTPException(404)
    cfg = a.model_dump()
    if body.max_steps:
        cfg["max_steps"] = body.max_steps
    # La carga del modelo preferido la hace el propio servicio, para que sea igual desde aquí y desde un flujo
    params = GenParams(max_new_tokens=a.max_tokens, temperature=a.temperature, enable_thinking=False)
    return ndjson_stream(svc.run(cfg, body.messages, params))


# ------------------------------------------------------------------ evaluación de agentes

class Caso(BaseModel):
    question: str
    tool: str = ""
    keywords: list[str] = Field(default_factory=list)


class EvalBody(BaseModel):
    name: str = ""
    cases: list[Caso]


@router.post("/{aid}/eval")
def evaluar(aid: int, body: EvalBody, s: Session = Depends(get_session)):
    """Pasa los casos por el agente y mide si elige la herramienta que toca."""
    from ..db import AgentEvalRun
    from ..services import agent_eval
    from ..services import tasks
    if not s.get(Agent, aid):
        raise HTTPException(404)
    if not body.cases:
        raise HTTPException(400, "Escribe al menos un caso")
    r = AgentEvalRun(agent_id=aid, name=body.name, cases=[c.model_dump() for c in body.cases])
    s.add(r)
    s.commit()
    s.refresh(r)
    t = tasks.submit("agent-eval", f"Evaluar agente · {len(body.cases)} casos", lambda task: agent_eval.ejecutar(r.id, task))
    return {"run_id": r.id, "task": t.to_dict()}


@router.get("/{aid}/evals")
def evaluaciones(aid: int, s: Session = Depends(get_session)):
    from ..db import AgentEvalRun
    filas = s.exec(select(AgentEvalRun).where(AgentEvalRun.agent_id == aid).order_by(AgentEvalRun.id.desc())).all()
    return [f.model_dump(mode="json") for f in filas[:10]]


# ------------------------------------------------------------------ importar de CrewAI
# CrewAI describe cada agente con rol, objetivo y trasfondo en un agents.yaml. Eso es texto y se
# traslada tal cual. Sus herramientas NO: son clases de Python de su framework. Sólo se traducen las
# que tienen equivalente claro aquí; el resto se avisa, no se inventa.
EQUIVALENCIAS = {
    "serperdevtool": "web_search", "bravesearchtool": "web_search", "duckduckgosearchrun": "web_search",
    "websitesearchtool": "web_search", "exasearchtool": "web_search",
    "scrapewebsitetool": "fetch_url", "scrapeelementfromwebsitetool": "fetch_url",
    "filereadtool": "read_file", "filewritertool": "write_file", "directoryreadtool": "list_files",
    "codeinterpretertool": "python", "nl2sqltool": "sql_query",
    "pdfsearchtool": "rag_search", "txtsearchtool": "rag_search", "ragtool": "rag_search",
    "csvsearchtool": "rag_search", "docxsearchtool": "rag_search", "mdxsearchtool": "rag_search",
}


class CrewBody(BaseModel):
    yaml: str
    model_id: int | None = None


@router.post("/import/crewai")
def importar_crewai(body: CrewBody, s: Session = Depends(get_session)):
    import re

    import yaml
    try:
        datos = yaml.safe_load(body.yaml) or {}
    except yaml.YAMLError as e:
        raise HTTPException(400, f"El YAML no es válido: {e}") from e
    if not isinstance(datos, dict):
        raise HTTPException(400, "Se esperaba un agents.yaml: un diccionario nombre → {role, goal, backstory}")

    creados, avisos = [], []
    for clave, a in datos.items():
        if not isinstance(a, dict) or not (a.get("role") or a.get("goal")):
            avisos.append(f"«{clave}» no tiene role ni goal: se salta")
            continue
        rol, objetivo, fondo = (str(a.get(k) or "").strip() for k in ("role", "goal", "backstory"))
        prompt = "\n\n".join(x for x in (f"Eres {rol}." if rol else "", f"Tu objetivo: {objetivo}" if objetivo else "", fondo) if x)
        huecos = sorted(set(re.findall(r"\{(\w+)\}", prompt)))
        if huecos:
            avisos.append(f"«{clave}» tiene huecos que CrewAI rellena al lanzarlo ({', '.join('{' + h + '}' for h in huecos)}): "
                          "sustitúyelos en sus instrucciones")
        herramientas, sin_equivalente = [], []
        for t in a.get("tools") or []:
            nombre = re.sub(r"[^a-z0-9]", "", str(t).lower())
            if nombre in EQUIVALENCIAS:
                herramientas.append({"id": EQUIVALENCIAS[nombre], "config": {}})
            else:
                sin_equivalente.append(str(t))
        if sin_equivalente:
            avisos.append(f"«{clave}»: sin equivalente aquí para {', '.join(sin_equivalente)}. "
                          "Busca un servidor MCP que las cubra en Conexiones")
        if any(h["id"] == "rag_search" for h in herramientas):
            avisos.append(f"«{clave}» usa búsqueda en documentos: elige la colección en su herramienta")

        nombre, n = clave, 1
        while s.exec(select(Agent).where(Agent.name == nombre)).first():
            n += 1
            nombre = f"{clave}-{n}"
        ag = Agent(name=nombre, description=(rol or objetivo)[:200], model_id=body.model_id,
                   system_prompt=prompt, tools=herramientas)
        s.add(ag)
        s.commit()
        s.refresh(ag)
        creados.append({"id": ag.id, "name": ag.name, "tools": [h["id"] for h in herramientas]})
    if not creados:
        raise HTTPException(400, "No se encontró ningún agente que importar. " + " · ".join(avisos))
    return {"creados": creados, "avisos": avisos}

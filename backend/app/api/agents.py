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
    """Tool catalog with their settings fields (the UI draws them from this)."""
    return tools_svc.catalog()


class McpProbe(BaseModel):
    url: str


@router.post("/tools/mcp/probe")
def probe_mcp(body: McpProbe):
    """Lists the tools of an external MCP server (to pick them in the interface)."""
    try:
        return {"tools": tools_svc._mcp_list(body.url)}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Couldn't connect: {e}")


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
        raise HTTPException(409, "There's already an agent with that name")
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
    """Runs the agent with streaming (NDJSON): tokens, tool calls and results."""
    from .chat import ndjson_stream
    a = s.get(Agent, aid)
    if not a:
        raise HTTPException(404)
    cfg = a.model_dump()
    if body.max_steps:
        cfg["max_steps"] = body.max_steps
    # The service itself loads the preferred model, so it's the same from here and from a flow
    params = GenParams(max_new_tokens=a.max_tokens, temperature=a.temperature, enable_thinking=False)
    return ndjson_stream(svc.run(cfg, body.messages, params))


# ------------------------------------------------------------------ agent evaluation

class Case(BaseModel):
    question: str
    tool: str = ""
    keywords: list[str] = Field(default_factory=list)


class EvalBody(BaseModel):
    name: str = ""
    cases: list[Case]


@router.post("/{aid}/eval")
def evaluate(aid: int, body: EvalBody, s: Session = Depends(get_session)):
    """Runs the cases through the agent and measures whether it picks the right tool."""
    from ..db import AgentEvalRun
    from ..services import agent_eval
    from ..services import tasks
    if not s.get(Agent, aid):
        raise HTTPException(404)
    if not body.cases:
        raise HTTPException(400, "Write at least one case")
    r = AgentEvalRun(agent_id=aid, name=body.name, cases=[c.model_dump() for c in body.cases])
    s.add(r)
    s.commit()
    s.refresh(r)
    t = tasks.submit("agent-eval", f"Evaluate agent · {len(body.cases)} cases", lambda task: agent_eval.run_eval(r.id, task))
    return {"run_id": r.id, "task": t.to_dict()}


@router.get("/{aid}/evals")
def evaluations(aid: int, s: Session = Depends(get_session)):
    from ..db import AgentEvalRun
    rows = s.exec(select(AgentEvalRun).where(AgentEvalRun.agent_id == aid).order_by(AgentEvalRun.id.desc())).all()
    return [f.model_dump(mode="json") for f in rows[:10]]


# ------------------------------------------------------------------ import from CrewAI
# CrewAI describes each agent with a role, goal and backstory in an agents.yaml. That's text and
# carries over as is. Its tools DON'T: they're Python classes from its framework. Only the ones
# with a clear equivalent here are translated; the rest are flagged, not made up.
EQUIVALENTS = {
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
def import_crewai(body: CrewBody, s: Session = Depends(get_session)):
    import re

    import yaml
    try:
        data = yaml.safe_load(body.yaml) or {}
    except yaml.YAMLError as e:
        raise HTTPException(400, f"The YAML isn't valid: {e}") from e
    if not isinstance(data, dict):
        raise HTTPException(400, "An agents.yaml was expected: a dictionary name → {role, goal, backstory}")

    created, warnings = [], []
    for key, a in data.items():
        if not isinstance(a, dict) or not (a.get("role") or a.get("goal")):
            warnings.append(f"“{key}” has no role or goal: skipped")
            continue
        role, goal, backstory = (str(a.get(k) or "").strip() for k in ("role", "goal", "backstory"))
        prompt = "\n\n".join(x for x in (f"You are {role}." if role else "", f"Your goal: {goal}" if goal else "", backstory) if x)
        gaps = sorted(set(re.findall(r"\{(\w+)\}", prompt)))
        if gaps:
            warnings.append(f"“{key}” has placeholders that CrewAI fills in at launch ({', '.join('{' + h + '}' for h in gaps)}): "
                          "replace them in its instructions")
        tools_, no_equivalent = [], []
        for t in a.get("tools") or []:
            name = re.sub(r"[^a-z0-9]", "", str(t).lower())
            if name in EQUIVALENTS:
                tools_.append({"id": EQUIVALENTS[name], "config": {}})
            else:
                no_equivalent.append(str(t))
        if no_equivalent:
            warnings.append(f"“{key}”: no equivalent here for {', '.join(no_equivalent)}. "
                          "Look for an MCP server that covers them under Connections")
        if any(h["id"] == "rag_search" for h in tools_):
            warnings.append(f"“{key}” uses document search: choose the collection in its tool")

        name, n = key, 1
        while s.exec(select(Agent).where(Agent.name == name)).first():
            n += 1
            name = f"{key}-{n}"
        ag = Agent(name=name, description=(role or goal)[:200], model_id=body.model_id,
                   system_prompt=prompt, tools=tools_)
        s.add(ag)
        s.commit()
        s.refresh(ag)
        created.append({"id": ag.id, "name": ag.name, "tools": [h["id"] for h in tools_]})
    if not created:
        raise HTTPException(400, "No agent to import was found. " + " · ".join(warnings))
    return {"creados": created, "avisos": warnings}

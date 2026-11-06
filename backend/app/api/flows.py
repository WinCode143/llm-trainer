from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from ..db import Flow, FlowRun, engine, get_session, now
from ..services import flows as svc

router = APIRouter(prefix="/api/flows", tags=["flows"])


@router.get("/node-types")
def node_types():
    return {"types": svc.NODE_TYPES, "templates": {k: {"name": v["name"], "description": v["description"]}
                                                   for k, v in svc.TEMPLATES.items()}}


@router.get("/templates/{key}")
def template(key: str):
    import copy
    t = svc.TEMPLATES.get(key)
    if not t:
        raise HTTPException(404)
    t = copy.deepcopy(t)
    # Los nodos MCP de una plantilla nombran el servidor por su clave del catálogo; aquí se enlazan al que
    # tengas registrado. Si no lo tienes, el nodo queda sin servidor y el flujo avisa antes de ejecutarse.
    from sqlmodel import Session

    from ..db import engine
    from .connections import CATALOGO, PROPIO, _asegurar, _instalado
    with Session(engine) as s:
        for n in t["spec"]["nodes"]:
            clave = (n.get("config") or {}).get("mcp_catalogo")
            c = next((x for x in CATALOGO if x["key"] == clave), None)
            # El propio servidor siempre existe: se registra si hace falta, no toca nada fuera
            m = _asegurar(s, PROPIO) if clave == PROPIO else (_instalado(s, c) if c else None)
            if m:
                n["config"].setdefault("config", {})["server_id"] = m.id
    return t


class FlowBody(BaseModel):
    name: str
    description: str = ""
    spec: dict = Field(default_factory=dict)


@router.get("")
def list_flows(s: Session = Depends(get_session)):
    out = []
    for f in s.exec(select(Flow).order_by(Flow.updated_at.desc())).all():
        d = f.model_dump(mode="json")
        d["nodes"] = len(f.spec.get("nodes", []))
        out.append(d)
    return out


@router.post("")
def create(body: FlowBody, s: Session = Depends(get_session)):
    if s.exec(select(Flow).where(Flow.name == body.name)).first():
        raise HTTPException(409, "Ya existe un flujo con ese nombre")
    f = Flow(**body.model_dump())
    s.add(f)
    s.commit()
    s.refresh(f)
    return f


@router.get("/{fid}")
def get_flow(fid: int, s: Session = Depends(get_session)):
    f = s.get(Flow, fid)
    if not f:
        raise HTTPException(404)
    return {**f.model_dump(mode="json"), "warnings": svc.validate(f.spec)}


@router.put("/{fid}")
def update(fid: int, body: FlowBody, s: Session = Depends(get_session)):
    f = s.get(Flow, fid)
    if not f:
        raise HTTPException(404)
    f.name, f.description, f.spec, f.updated_at = body.name, body.description, body.spec, now()
    s.add(f)
    s.commit()
    s.refresh(f)
    return {**f.model_dump(mode="json"), "warnings": svc.validate(f.spec)}


@router.delete("/{fid}")
def delete(fid: int, s: Session = Depends(get_session)):
    f = s.get(Flow, fid)
    if f:
        s.delete(f)
        s.commit()
    return {"ok": True}


class RunBody(BaseModel):
    inputs: dict = Field(default_factory=dict)


def _stream_events(run_id: int, source):
    """Consume los eventos de una ejecución guardando el progreso en la base de datos."""
    from .chat import ndjson_stream

    def gen():
        trace, final_state = [], {}
        status, interrupt_data, error = "completed", None, None
        for ev in source:
            if ev["type"] == "node":
                trace.append({k: v for k, v in ev.items() if k not in ("type", "thread_id")})
            elif ev["type"] == "interrupt":
                status, interrupt_data = "waiting", ev["data"]
            elif ev["type"] == "done":
                final_state = ev.get("vars", {})
            elif ev["type"] == "error":
                status, error = "failed", ev["message"]
            elif ev["type"] == "token":
                yield {**ev, "run_id": run_id}   # los tokens no se guardan, sólo se ven pasar
                continue
            with Session(engine) as s:
                r = s.get(FlowRun, run_id)
                r.thread_id = ev.get("thread_id") or r.thread_id
                r.trace, r.state, r.status, r.interrupt, r.error = trace, final_state, status if status != "completed" else "running", interrupt_data, error
                s.add(r)
                s.commit()
            yield {**ev, "run_id": run_id}
        with Session(engine) as s:
            r = s.get(FlowRun, run_id)
            r.status, r.trace, r.state, r.interrupt, r.error = status, trace, final_state, interrupt_data, error
            if status != "waiting":
                r.finished_at = now()
            s.add(r)
            s.commit()

    return ndjson_stream(gen())


def _stream(fid: int, run_id: int, spec: dict, inputs: dict, thread_id: str | None, resume=None):
    """Ejecuta el flujo guardando el progreso en la base de datos (para reanudar tras una aprobación)."""
    return _stream_events(run_id, svc.run(spec, inputs, thread_id=thread_id, resume=resume))


@router.post("/{fid}/run")
def run_flow(fid: int, body: RunBody, s: Session = Depends(get_session)):
    f = s.get(Flow, fid)
    if not f:
        raise HTTPException(404)
    r = FlowRun(flow_id=fid, inputs=body.inputs, status="running")
    s.add(r)
    s.commit()
    s.refresh(r)
    return _stream(fid, r.id, f.spec, body.inputs, None)


class ResumeBody(BaseModel):
    approved: bool = True
    comment: str = ""
    edited: str | None = None


@router.post("/runs/{rid}/resume")
def resume_run(rid: int, body: ResumeBody, s: Session = Depends(get_session)):
    """Continúa un flujo detenido en un nodo de aprobación."""
    r = s.get(FlowRun, rid)
    if not r or r.status != "waiting":
        raise HTTPException(400, "Esa ejecución no está esperando aprobación")
    f = s.get(Flow, r.flow_id)
    payload = {"rejected": not body.approved, "comment": body.comment, "edited": body.edited}
    r.status = "running"
    s.add(r)
    s.commit()
    return _stream(r.flow_id, rid, f.spec, r.inputs, r.thread_id, resume=payload)


@router.get("/runs/{rid}/history")
def run_history(rid: int, s: Session = Depends(get_session)):
    """Puntos de guardado de una ejecución: desde cuáles se puede rebobinar."""
    r = s.get(FlowRun, rid)
    if not r or not r.thread_id:
        raise HTTPException(404, "Esa ejecución no tiene puntos de guardado")
    f = s.get(Flow, r.flow_id)
    return {"thread_id": r.thread_id, "checkpoints": svc.history(f.spec, r.thread_id)}


class ReplayBody(BaseModel):
    checkpoint_id: str
    edits: dict = Field(default_factory=dict)


@router.post("/runs/{rid}/replay")
def replay_run(rid: int, body: ReplayBody, s: Session = Depends(get_session)):
    """Reanuda desde un punto de guardado, opcionalmente cambiando variables.

    No se repite lo anterior: se aprovecha lo que ya se calculó hasta ese punto."""
    r = s.get(FlowRun, rid)
    if not r or not r.thread_id:
        raise HTTPException(404)
    f = s.get(Flow, r.flow_id)
    nuevo = FlowRun(flow_id=r.flow_id, inputs=r.inputs, status="running", thread_id=r.thread_id)
    s.add(nuevo)
    s.commit()
    s.refresh(nuevo)
    return _stream_events(nuevo.id, svc.replay(f.spec, r.thread_id, body.checkpoint_id, body.edits or None))


@router.get("/{fid}/export")
def export_flow(fid: int, s: Session = Depends(get_session)):
    """El flujo como código LangGraph, para no depender de esta aplicación."""
    f = s.get(Flow, fid)
    if not f:
        raise HTTPException(404)
    return {"filename": f"{f.name.replace(' ', '_')}.py", "code": svc.to_python(f.spec, f.name)}


@router.get("/{fid}/runs")
def list_runs(fid: int, s: Session = Depends(get_session)):
    rows = s.exec(select(FlowRun).where(FlowRun.flow_id == fid).order_by(FlowRun.id.desc())).all()
    return [r.model_dump(mode="json") for r in rows[:20]]


@router.get("/runs/{rid}")
def get_run(rid: int, s: Session = Depends(get_session)):
    r = s.get(FlowRun, rid)
    if not r:
        raise HTTPException(404)
    return r

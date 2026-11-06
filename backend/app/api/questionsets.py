"""Conjuntos de preguntas de evaluación reutilizables."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from ..db import EvalRun, QuestionSet, get_session, now

router = APIRouter(prefix="/api/question-sets", tags=["question-sets"])


class Pregunta(BaseModel):
    question: str
    expected: str | None = None
    keywords: list[str] = Field(default_factory=list)


class SetBody(BaseModel):
    name: str
    description: str = ""
    questions: list[Pregunta] = Field(default_factory=list)
    source: str = "manual"


@router.get("")
def listing(s: Session = Depends(get_session)):
    out = []
    for q in s.exec(select(QuestionSet).order_by(QuestionSet.updated_at.desc())).all():
        d = q.model_dump(mode="json")
        d["n"] = len(q.questions)
        # Sin respuesta esperada no hay similitud ni palabras clave: sólo latencia y juez
        d["con_referencia"] = sum(1 for x in q.questions if x.get("expected"))
        out.append(d)
    return out


@router.post("")
def create(body: SetBody, s: Session = Depends(get_session)):
    if s.exec(select(QuestionSet).where(QuestionSet.name == body.name)).first():
        raise HTTPException(409, "Ya existe un conjunto con ese nombre")
    q = QuestionSet(**body.model_dump())
    s.add(q)
    s.commit()
    s.refresh(q)
    return q


@router.get("/{qid}")
def get_one(qid: int, s: Session = Depends(get_session)):
    q = s.get(QuestionSet, qid)
    if not q:
        raise HTTPException(404)
    return q


@router.put("/{qid}")
def update(qid: int, body: SetBody, s: Session = Depends(get_session)):
    q = s.get(QuestionSet, qid)
    if not q:
        raise HTTPException(404)
    q.name, q.description, q.questions, q.updated_at = body.name, body.description, \
        [p.model_dump() for p in body.questions], now()
    s.add(q)
    s.commit()
    s.refresh(q)
    return q


@router.delete("/{qid}")
def delete(qid: int, s: Session = Depends(get_session)):
    q = s.get(QuestionSet, qid)
    if q:
        s.delete(q)
        s.commit()
    return {"ok": True}


class DesdeRun(BaseModel):
    name: str
    description: str = ""


@router.post("/from-run/{rid}")
def from_run(rid: int, body: DesdeRun, s: Session = Depends(get_session)):
    """Guarda las preguntas de una evaluación ya hecha como conjunto reutilizable."""
    r = s.get(EvalRun, rid)
    if not r:
        raise HTTPException(404)
    preguntas = r.config.get("questions") or []
    if not preguntas:
        raise HTTPException(400, "Esa evaluación no tiene preguntas guardadas")
    if s.exec(select(QuestionSet).where(QuestionSet.name == body.name)).first():
        raise HTTPException(409, "Ya existe un conjunto con ese nombre")
    q = QuestionSet(name=body.name, description=body.description, questions=preguntas, source="generado")
    s.add(q)
    s.commit()
    s.refresh(q)
    return q

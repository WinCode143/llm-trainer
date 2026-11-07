"""Reusable sets of evaluation questions."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from ..db import EvalRun, QuestionSet, get_session, now

router = APIRouter(prefix="/api/question-sets", tags=["question-sets"])


class Question(BaseModel):
    question: str
    expected: str | None = None
    keywords: list[str] = Field(default_factory=list)


class SetBody(BaseModel):
    name: str
    description: str = ""
    questions: list[Question] = Field(default_factory=list)
    source: str = "manual"


@router.get("")
def listing(s: Session = Depends(get_session)):
    out = []
    for q in s.exec(select(QuestionSet).order_by(QuestionSet.updated_at.desc())).all():
        d = q.model_dump(mode="json")
        d["n"] = len(q.questions)
        # Without an expected answer there's no similarity or keywords: only latency and the judge
        d["con_referencia"] = sum(1 for x in q.questions if x.get("expected"))
        out.append(d)
    return out


@router.post("")
def create(body: SetBody, s: Session = Depends(get_session)):
    if s.exec(select(QuestionSet).where(QuestionSet.name == body.name)).first():
        raise HTTPException(409, "There's already a set with that name")
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
    """Saves the questions of an evaluation that already ran as a reusable set."""
    r = s.get(EvalRun, rid)
    if not r:
        raise HTTPException(404)
    questions = r.config.get("questions") or []
    if not questions:
        raise HTTPException(400, "That evaluation has no saved questions")
    if s.exec(select(QuestionSet).where(QuestionSet.name == body.name)).first():
        raise HTTPException(409, "There's already a set with that name")
    q = QuestionSet(name=body.name, description=body.description, questions=questions, source="generated")
    s.add(q)
    s.commit()
    s.refresh(q)
    return q

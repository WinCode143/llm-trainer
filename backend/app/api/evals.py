from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from ..db import EvalRun, get_session
from ..services import evaluation, tasks

router = APIRouter(prefix="/api/evals", tags=["evals"])


class Question(BaseModel):
    question: str
    expected: str | None = None
    keywords: list[str] = Field(default_factory=list)


class EvalBody(BaseModel):
    name: str
    model_ids: list[int]
    questions: list[Question]
    system_prompt: str | None = None
    rag_collection_id: int | None = None
    rag_k: int | None = None  # None = top_k de la colección
    rag_modes: list[str] = Field(default_factory=lambda: ["off", "on"])  # evalúa cada modelo sin y/o con RAG
    rag_system_prompt: str | None = None
    judge_model_id: int | None = None
    judge_helper: int | None = None  # juez externo: no ocupa VRAM
    quantization: str = "none"
    params: dict = Field(default_factory=lambda: {"max_new_tokens": 512, "temperature": 0.2, "enable_thinking": False})


@router.get("")
def list_runs(limit: int = 200, offset: int = 0, s: Session = Depends(get_session)):
    return s.exec(select(EvalRun).order_by(EvalRun.id.desc()).limit(min(limit, 1000)).offset(offset)).all()


@router.post("")
def create(body: EvalBody, s: Session = Depends(get_session)):
    if not body.model_ids or not body.questions:
        raise HTTPException(400, "Selecciona al menos un modelo y una pregunta")
    if body.rag_collection_id and not set(body.rag_modes) & {"on", "off"}:
        raise HTTPException(400, "Elige al menos un modo: sin RAG y/o con RAG")
    run = EvalRun(name=body.name, config=body.model_dump(mode="json"))
    s.add(run)
    s.commit()
    s.refresh(run)
    t = tasks.submit("eval", f"Evaluación {body.name}", lambda task: evaluation.run_eval(run.id, task))
    return {"run": run, "task": t.to_dict()}


class GenQuestions(BaseModel):
    collection_id: int
    n_chunks: int = 10
    per_chunk: int = 2
    sources: list[str] | None = None


@router.post("/generate-questions")
def generate_questions(body: GenQuestions):
    """Genera preguntas+referencias desde los documentos con el modelo cargado (set de evaluación automático)."""
    from ..services.inference import manager
    if not manager.current:
        raise HTTPException(400, "Carga un modelo para generar las preguntas")
    return tasks.submit("eval-qgen", "Generar preguntas de evaluación", lambda task: evaluation.generate_questions(
        task, body.collection_id, body.n_chunks, body.per_chunk, body.sources)).to_dict()


@router.get("/{rid}")
def get_run(rid: int, s: Session = Depends(get_session)):
    r = s.get(EvalRun, rid)
    if not r:
        raise HTTPException(404)
    return {**r.model_dump(mode="json"), "judge_agreement": evaluation.judge_agreement(r.results)}


class Rating(BaseModel):
    index: int
    rating: int | None  # 1-5


@router.post("/{rid}/rate")
def rate(rid: int, body: Rating, s: Session = Depends(get_session)):
    r = s.get(EvalRun, rid)
    results = [dict(x) for x in r.results]
    results[body.index]["rating"] = body.rating
    r.results = results
    r.summary = evaluation.summarize(results)
    s.add(r)
    s.commit()
    return r


@router.delete("/{rid}")
def delete(rid: int, s: Session = Depends(get_session)):
    r = s.get(EvalRun, rid)
    if r:
        s.delete(r)
        s.commit()
    return {"ok": True}

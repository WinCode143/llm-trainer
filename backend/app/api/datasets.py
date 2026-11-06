import os

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlmodel import Session, select

from ..db import Dataset, get_session
from ..services import datasets as svc
from ..services import synth, tasks

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


def _save(s: Session, name: str, fmt: str, rows: list[dict], source: str) -> Dataset:
    if s.exec(select(Dataset).where(Dataset.name == name)).first():
        raise HTTPException(409, f"Ya existe un dataset llamado '{name}'")
    path = svc.write_jsonl(name, rows)
    d = Dataset(name=name, format=fmt, path=str(path), num_rows=len(rows), source=source)
    s.add(d)
    s.commit()
    s.refresh(d)
    return d


@router.get("/plantillas")
def plantillas():
    """Plantillas por caso de uso: qué forma tienen los datos y cuántos hacen falta."""
    from ..services.plantillas import listado
    return listado()


@router.post("/plantillas/{key}")
def crear_desde_plantilla(key: str, name: str, s: Session = Depends(get_session)):
    """Crea un dataset con los ejemplos de la plantilla, listo para editar y ampliar."""
    from ..services.plantillas import PLANTILLAS
    t = PLANTILLAS.get(key)
    if not t:
        raise HTTPException(404, "No existe esa plantilla")
    fmt, rows = svc.normalize(t["filas"], t["formato"])
    return _save(s, name, fmt, rows, f"plantilla:{key}")


@router.get("")
def list_datasets(limit: int = 500, offset: int = 0, s: Session = Depends(get_session)):
    return s.exec(select(Dataset).order_by(Dataset.created_at.desc()).limit(min(limit, 1000)).offset(offset)).all()


@router.post("/upload")
async def upload(file: UploadFile = File(...), name: str = Form(...), format: str | None = Form(None),
                 system_prompt: str | None = Form(None), s: Session = Depends(get_session)):
    try:
        fmt, rows = svc.import_file(file.filename, await file.read(), format or None, system_prompt or None)
    except Exception as e:
        raise HTTPException(400, str(e))
    return _save(s, name, fmt, rows, f"archivo:{file.filename}")


class HFImport(BaseModel):
    repo: str
    name: str
    split: str = "train"
    subset: str | None = None
    limit: int | None = None
    format: str | None = None
    system_prompt: str | None = None


@router.post("/import-hf")
def import_hf(body: HFImport):
    def work(task):
        task.update(message=f"Descargando {body.repo}")
        fmt, rows = svc.import_hf(body.repo, body.split, body.subset, body.limit, body.format, body.system_prompt)
        from ..db import engine
        with Session(engine) as s:
            return _save(s, body.name, fmt, rows, f"hf:{body.repo}").model_dump(mode="json")

    return tasks.submit("dataset-import", f"Importar {body.repo}", work).to_dict()


class ManualRows(BaseModel):
    name: str
    format: str = "chat"
    rows: list[dict]


@router.post("/manual")
def manual(body: ManualRows, s: Session = Depends(get_session)):
    try:
        fmt, rows = svc.normalize(body.rows, body.format)
    except Exception as e:
        raise HTTPException(400, str(e))
    return _save(s, body.name, fmt, rows, "manual")


def _get(s: Session, did: int) -> Dataset:
    d = s.get(Dataset, did)
    if not d:
        raise HTTPException(404, "Dataset no encontrado")
    return d


@router.get("/{did}/rows")
def rows(did: int, offset: int = 0, limit: int = 20, s: Session = Depends(get_session)):
    d = _get(s, did)
    return {"total": d.num_rows, "rows": svc.read_rows(d.path, offset, limit)}


@router.get("/{did}/stats")
def stats(did: int, s: Session = Depends(get_session)):
    d = _get(s, did)
    return svc.stats(d.path, d.format)


class RowsAppend(BaseModel):
    rows: list[dict]


@router.post("/{did}/append")
def append(did: int, body: RowsAppend, s: Session = Depends(get_session)):
    d = _get(s, did)
    _, new = svc.normalize(body.rows, d.format)
    all_rows = svc.read_all(d.path) + new
    svc.write_jsonl(d.name, all_rows)
    d.num_rows = len(all_rows)
    s.add(d)
    s.commit()
    return d


class SplitBody(BaseModel):
    eval_ratio: float = 0.1
    seed: int = 42


@router.post("/{did}/split")
def split(did: int, body: SplitBody, s: Session = Depends(get_session)):
    d = _get(s, did)
    train, ev = svc.split(svc.read_all(d.path), body.eval_ratio, body.seed)
    a = _save(s, f"{d.name}-train", d.format, train, f"split:{d.name}")
    b = _save(s, f"{d.name}-eval", d.format, ev, f"split:{d.name}")
    return [a, b]


@router.delete("/{did}")
def delete(did: int, s: Session = Depends(get_session)):
    d = _get(s, did)
    if os.path.exists(d.path):
        os.remove(d.path)
    s.delete(d)
    s.commit()
    return {"ok": True}


class SynthBody(BaseModel):
    name: str
    source: str = "docs"  # docs | topics
    collection_id: int | None = None
    domain: str | None = None
    num_chunks: int = 20
    num_topics: int = 10
    pairs_per_chunk: int = 3
    language: str = "español"
    styles: list[str] = ["hecho", "explicacion"]
    system_prompt: str | None = None
    temperature: float = 0.7
    dedupe_threshold: float = 0.92
    judge_min_score: float = 0
    helper: int | None = None  # modelo ayudante externo; None = el modelo cargado


@router.post("/synthesize")
def synthesize(body: SynthBody, s: Session = Depends(get_session)):
    if s.exec(select(Dataset).where(Dataset.name == body.name)).first():
        raise HTTPException(409, f"Ya existe un dataset llamado '{body.name}'")
    t = tasks.submit("synth", f"Dataset sintético {body.name}",
                     lambda task: synth.generate(task, **body.model_dump()))
    return t.to_dict()

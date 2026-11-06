import json
import threading
import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlmodel import Session

from ..db import Model, RagCollection, get_session
from ..services import rag
from ..services.inference import GenParams, manager
from ..services.jobs import is_gpu_busy

router = APIRouter(prefix="/api/chat", tags=["chat"])


class Params(BaseModel):
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.05
    seed: int | None = None
    enable_thinking: bool | None = None


class ChatBody(BaseModel):
    messages: list[dict]
    params: Params = Field(default_factory=Params)
    rag_collection_id: int | None = None
    rag_k: int | None = None  # None = valores de la colección
    rag_min_score: float | None = None
    rag_sources: list[str] | None = None
    rag_tags: list[str] | None = None
    system_prompt: str | None = None


def text_of(content) -> str:
    """Texto de un mensaje que puede ser multimodal (lista de partes)."""
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text").strip()
    return content or ""


def ndjson_stream(gen):
    def wrap():
        try:
            for item in gen:
                yield json.dumps(item, ensure_ascii=False) + "\n"
        except Exception as e:  # noqa: BLE001
            yield json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False) + "\n"

    return StreamingResponse(wrap(), media_type="application/x-ndjson")


@router.post("/stream")
def stream(body: ChatBody, s: Session = Depends(get_session)):
    coll = s.get(RagCollection, body.rag_collection_id) if body.rag_collection_id else None
    params = GenParams(**body.params.model_dump())

    def gen():
        messages = [m for m in body.messages if m.get("role") != "system"]
        sources = []
        if coll:
            question = text_of(messages[-1]["content"])
            msgs, sources, rag_mode = rag.prepare(coll, question, messages[:-1], body.system_prompt, body.system_prompt,
                                                  k=body.rag_k, min_score=body.rag_min_score,
                                                  sources=body.rag_sources, tags=body.rag_tags)
            # rag_mode: rag | no_context (nada superó el umbral → responde sin contexto) | refuse
            yield {"type": "sources", "sources": sources, "rag_mode": rag_mode}
        else:
            msgs = ([{"role": "system", "content": body.system_prompt}] if body.system_prompt else []) + messages
        t0, n = time.time(), 0
        first = None
        cancel = threading.Event()
        try:
            for piece in manager.stream_chat(msgs, params, cancel):
                if first is None:
                    first = time.time() - t0
                n += 1
                yield {"type": "token", "text": piece}
        finally:
            cancel.set()
        el = time.time() - t0
        yield {"type": "done", "elapsed_s": round(el, 2), "ttft_s": round(first or 0, 2), "chunks": n,
               "chunks_per_s": round(n / el, 1) if el else 0}

    return ndjson_stream(gen())


class CompareBody(BaseModel):
    model_ids: list[int]
    prompt: str
    system_prompt: str | None = None
    params: Params = Field(default_factory=Params)
    quantization: str = "none"


@router.post("/compare")
def compare(body: CompareBody, s: Session = Depends(get_session)):
    """Carga cada modelo en secuencia y genera la respuesta; emite resultados según terminan."""
    if is_gpu_busy():
        raise HTTPException(409, "La GPU está ocupada con un entrenamiento")
    models = [s.get(Model, i) for i in body.model_ids]
    params = GenParams(**body.params.model_dump())

    def gen():
        msgs = ([{"role": "system", "content": body.system_prompt}] if body.system_prompt else []) + \
               [{"role": "user", "content": body.prompt}]
        for m in models:
            yield {"type": "loading", "model_id": m.id, "name": m.name}
            manager.load(model_id=m.id, name=m.name, kind=m.kind, path=m.path, base_model=m.base_model,
                         quantization=body.quantization,
                         engine=manager.preferred_engine if m.kind != "gguf" else "auto")
            yield {"type": "result", "model_id": m.id, "name": m.name, **manager.chat(msgs, params)}
        yield {"type": "done"}

    return ndjson_stream(gen())

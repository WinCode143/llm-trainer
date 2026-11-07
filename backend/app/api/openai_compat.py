"""OpenAI-compatible API (/v1) to use LLM Trainer's models from any client
(Open WebUI, LibreChat, Continue, the openai SDK, LangChain…).

Our own extension: the `rag_collection` field (name or id) in /v1/chat/completions applies
that collection's RAG (with its configured hybrid search, rerank and threshold) before answering.
"""
import json
import threading
import time
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlmodel import Session, select

from ..config import settings
from ..db import Model, RagCollection, engine
from ..services import rag
from ..services.inference import GenParams, manager
from ..services.jobs import is_gpu_busy


def auth(authorization: str | None = Header(default=None)):
    if settings.api_key and authorization != f"Bearer {settings.api_key}":
        raise HTTPException(401, {"error": {"message": "Invalid API key", "type": "invalid_request_error"}})


router = APIRouter(prefix="/v1", tags=["openai"], dependencies=[Depends(auth)])


@router.get("/models")
def models():
    with Session(engine) as s:
        rows = s.exec(select(Model)).all()
    loaded = manager.current.name if manager.current else None
    return {"object": "list", "data": [
        {"id": m.name, "object": "model", "created": int(m.created_at.timestamp()), "owned_by": "llm-trainer",
         "kind": m.kind, "loaded": m.name == loaded} for m in rows]}


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stream: bool = False
    seed: int | None = None
    frequency_penalty: float | None = None
    # LLM Trainer extensions
    rag_collection: str | int | None = None
    top_k: int = 50
    repetition_penalty: float | None = None
    enable_thinking: bool | None = None


def _text(content) -> str:
    """content can be a str or a list of parts [{type: text, text}] (OpenAI's newer format)."""
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in ("text", None))
    return content or ""


def _ensure_model(name: str | None) -> str:
    cur = manager.current
    if not name or (cur and name == cur.name):
        if not cur:
            raise HTTPException(400, {"error": {"message": "No model is loaded; pass `model` or load one in the UI"}})
        return cur.name
    with Session(engine) as s:
        m = s.exec(select(Model).where(Model.name == name)).first()
    if not m:
        if cur:  # generic names (gpt-4o, default…) → use the loaded one
            return cur.name
        raise HTTPException(404, {"error": {"message": f"Model '{name}' isn't registered"}})
    if is_gpu_busy():
        raise HTTPException(409, {"error": {"message": "The GPU is busy with a training run"}})
    manager.load(model_id=m.id, name=m.name, kind=m.kind, path=m.path, base_model=m.base_model,
                 engine=manager.preferred_engine if m.kind != "gguf" else "auto")
    return m.name


def _rag_collection(ref) -> RagCollection | None:
    if ref in (None, ""):
        return None
    with Session(engine, expire_on_commit=False) as s:
        c = s.get(RagCollection, int(ref)) if str(ref).isdigit() else \
            s.exec(select(RagCollection).where(RagCollection.name == str(ref))).first()
    if not c:
        raise HTTPException(404, {"error": {"message": f"RAG collection '{ref}' not found"}})
    return c


@router.post("/chat/completions")
def chat_completions(req: ChatRequest):
    model_name = _ensure_model(req.model)
    msgs = [{"role": m.get("role", "user"), "content": _text(m.get("content"))} for m in req.messages]
    sources = []
    coll = _rag_collection(req.rag_collection)
    if coll:
        system = next((m["content"] for m in msgs if m["role"] == "system"), None)
        convo = [m for m in msgs if m["role"] != "system"]
        msgs, sources, _ = rag.prepare(coll, convo[-1]["content"], convo[:-1], system, None)
    # frequency_penalty (OpenAI, 0–2) is approximated with repetition_penalty (1 = no penalty)
    rp = req.repetition_penalty or 1.0 + max(0.0, req.frequency_penalty or 0) * 0.5
    params = GenParams(max_new_tokens=req.max_completion_tokens or req.max_tokens or 1024, temperature=req.temperature,
                       top_p=req.top_p, top_k=req.top_k, repetition_penalty=rp, seed=req.seed,
                       enable_thinking=req.enable_thinking)
    cid, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())
    extra = {"sources": [{"source": h.get("source"), "page": h.get("page"), "score": h["score"]} for h in sources]} if coll else {}

    if not req.stream:
        text = "".join(manager.stream_chat(msgs, params))
        return {"id": cid, "object": "chat.completion", "created": created, "model": model_name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": _usage(msgs, text), **extra}

    def sse():
        def chunk(delta, finish=None):
            return "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created,
                                          "model": model_name, "choices": [{"index": 0, "delta": delta,
                                                                            "finish_reason": finish}]},
                                         ensure_ascii=False) + "\n\n"
        cancel = threading.Event()
        try:
            yield chunk({"role": "assistant", "content": ""})
            for piece in manager.stream_chat(msgs, params, cancel):
                yield chunk({"content": piece})
            yield chunk({}, "stop")
        except Exception as e:  # noqa: BLE001
            yield "data: " + json.dumps({"error": {"message": str(e)}}) + "\n\n"
        finally:
            cancel.set()
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


def _usage(msgs: list[dict], text: str) -> dict:
    tok = manager.current.tokenizer if manager.current else None
    if tok is None:
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    p = len(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True))
    c = len(tok.encode(text, add_special_tokens=False))
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}


class EmbeddingsRequest(BaseModel):
    input: str | list[str]
    model: str | None = None


@router.post("/embeddings")
def embeddings(req: EmbeddingsRequest):
    texts = [req.input] if isinstance(req.input, str) else req.input
    name = req.model if req.model and "/" in req.model else settings.default_embed_model
    vecs = rag.embed(name, texts, "passage")
    return {"object": "list", "model": name, "data": [
        {"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)],
        "usage": {"prompt_tokens": sum(len(t) // 4 for t in texts), "total_tokens": sum(len(t) // 4 for t in texts)}}

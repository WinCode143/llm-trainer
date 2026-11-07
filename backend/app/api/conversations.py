from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from ..db import Conversation, get_session, now

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.get("")
def list_conversations(q: str | None = None, limit: int = 200, offset: int = 0, s: Session = Depends(get_session)):
    base = select(Conversation).order_by(Conversation.updated_at.desc())
    if q:
        # The filter searches inside the messages, so everything is walked and trimmed AFTERWARDS:
        # limiting first would make search only look at the most recent conversations.
        ql = q.lower()
        rows = [c for c in s.exec(base).all()
                if ql in c.title.lower() or any(ql in (m.get("content") or "").lower() for m in c.messages)]
        rows = rows[offset:offset + min(limit, 1000)]
    else:
        rows = s.exec(base.limit(min(limit, 1000)).offset(offset)).all()
    return [{"id": c.id, "title": c.title, "n": len(c.messages), "updated_at": c.updated_at,
             "model": (c.settings or {}).get("model")} for c in rows]


class ConvBody(BaseModel):
    title: str | None = None
    messages: list[dict] | None = None
    settings: dict | None = None


@router.post("")
def create(body: ConvBody, s: Session = Depends(get_session)):
    c = Conversation(title=body.title or "New conversation", messages=body.messages or [], settings=body.settings or {})
    s.add(c)
    s.commit()
    s.refresh(c)
    return c


@router.get("/{cid}")
def get(cid: int, s: Session = Depends(get_session)):
    c = s.get(Conversation, cid)
    if not c:
        raise HTTPException(404)
    return c


@router.put("/{cid}")
def update(cid: int, body: ConvBody, s: Session = Depends(get_session)):
    c = s.get(Conversation, cid)
    if not c:
        raise HTTPException(404)
    if body.messages is not None:
        c.messages = body.messages
        # Automatic title from the user's first message
        if c.title == "New conversation":
            first = next((m.get("content", "") for m in body.messages if m.get("role") == "user"), "")
            if first:
                c.title = (first[:60] + "…") if len(first) > 60 else first
    if body.title:
        c.title = body.title
    if body.settings is not None:
        c.settings = body.settings
    c.updated_at = now()
    s.add(c)
    s.commit()
    s.refresh(c)
    return c


@router.delete("/{cid}")
def delete(cid: int, s: Session = Depends(get_session)):
    c = s.get(Conversation, cid)
    if c:
        s.delete(c)
        s.commit()
    return {"ok": True}

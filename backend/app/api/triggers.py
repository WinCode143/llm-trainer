"""Triggers: creating them, testing them, and the public entry point for webhooks."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from ..db import Trigger, get_session
from ..services import triggers as svc

router = APIRouter(tags=["triggers"])


def _public(t: Trigger) -> dict:
    d = t.model_dump(mode="json")
    cfg = dict(t.config or {})
    if cfg.get("bot_token"):
        cfg["bot_token"] = f"…{cfg['bot_token'][-4:]}"   # never sent whole to the interface
    d["config"] = cfg
    return d


class TriggerBody(BaseModel):
    name: str
    kind: str = "webhook"
    target_kind: str = "flow"
    target_id: int
    config: dict = Field(default_factory=dict)
    enabled: bool = True


@router.get("/api/triggers")
def list_triggers(s: Session = Depends(get_session)):
    return [_public(t) for t in s.exec(select(Trigger).order_by(Trigger.id)).all()]


@router.post("/api/triggers")
def create(body: TriggerBody, s: Session = Depends(get_session)):
    if s.exec(select(Trigger).where(Trigger.name == body.name)).first():
        raise HTTPException(409, "There's already a trigger with that name")
    cfg = dict(body.config)
    if body.kind == "webhook":
        cfg["token"] = svc.new_token()
    if body.kind == "telegram" and not cfg.get("bot_token"):
        raise HTTPException(400, "The bot token is missing (get one from @BotFather on Telegram)")
    if body.kind == "schedule" and not (cfg.get("cada_min") or cfg.get("diario")):
        raise HTTPException(400, "Say every how many minutes, or at what time of day")
    t = Trigger(**{**body.model_dump(), "config": cfg})
    s.add(t)
    s.commit()
    s.refresh(t)
    if t.kind == "telegram" and t.enabled:
        svc.start_telegram(t.id)
    return _public(t)


@router.put("/api/triggers/{tid}")
def edit(tid: int, body: TriggerBody, s: Session = Depends(get_session)):
    t = s.get(Trigger, tid)
    if not t:
        raise HTTPException(404)
    cfg = dict(body.config)
    # The bot token comes back masked from the interface; if it wasn't changed, it's kept.
    # Empty or masked (“…abcd”) means “I didn't touch it”: the existing one is kept.
    # Checking only for the mask, leaving the field empty (as the interface suggests) used to delete the token.
    if not cfg.get("bot_token") or str(cfg["bot_token"]).startswith("…"):
        cfg["bot_token"] = t.config.get("bot_token", "")
    if t.kind == "webhook":
        cfg["token"] = t.config.get("token")          # the webhook token isn't edited, it's regenerated
    t.name, t.target_kind, t.target_id, t.config, t.enabled = body.name, body.target_kind, body.target_id, cfg, body.enabled
    s.add(t)
    s.commit()
    if t.kind == "telegram":
        svc.start_telegram(t.id) if t.enabled else svc.stop_telegram(t.id)
    return _public(t)


@router.post("/api/triggers/{tid}/regenerate-token")
def regenerate(tid: int, s: Session = Depends(get_session)):
    """New webhook token: the old URL stops working immediately."""
    t = s.get(Trigger, tid)
    if not t or t.kind != "webhook":
        raise HTTPException(404)
    t.config = {**t.config, "token": svc.new_token()}
    s.add(t)
    s.commit()
    return {"token": t.config["token"]}


@router.delete("/api/triggers/{tid}")
def delete(tid: int, s: Session = Depends(get_session)):
    t = s.get(Trigger, tid)
    if t:
        svc.stop_telegram(tid)
        s.delete(t)
        s.commit()
    return {"ok": True}


@router.post("/api/triggers/{tid}/test")
def test(tid: int, s: Session = Depends(get_session)):
    """Runs the trigger by hand with its input (or a sample one) to check that it answers."""
    t = s.get(Trigger, tid)
    if not t:
        raise HTTPException(404)
    if t.kind == "telegram":
        try:
            me = svc.tg(t.config.get("bot_token", ""), "getMe")
            return {"ok": True, "bot": f"@{me.get('username')}", "chats_permitidos": t.config.get("allowed_chat_ids") or []}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(e)}
    return svc.execute(t, t.config.get("input") or "hello", "test")


class WebhookInput(BaseModel):
    input: str = ""


@router.post("/api/hooks/{token}")
def webhook(token: str, body: WebhookInput):
    """Public entry point for webhooks. The token is the only key: treat it like a password."""
    t = svc.find_webhook(token)
    if not t:
        raise HTTPException(404)          # same error whether it exists or not, so nothing is revealed
    return svc.execute(t, body.input, "webhook")

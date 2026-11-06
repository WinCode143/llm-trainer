"""Disparadores: alta, prueba y el punto de entrada de los webhooks."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from ..db import Trigger, get_session
from ..services import triggers as svc

router = APIRouter(tags=["triggers"])


def _publico(t: Trigger) -> dict:
    d = t.model_dump(mode="json")
    cfg = dict(t.config or {})
    if cfg.get("bot_token"):
        cfg["bot_token"] = f"…{cfg['bot_token'][-4:]}"   # nunca entero hacia la interfaz
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
def listar(s: Session = Depends(get_session)):
    return [_publico(t) for t in s.exec(select(Trigger).order_by(Trigger.id)).all()]


@router.post("/api/triggers")
def crear(body: TriggerBody, s: Session = Depends(get_session)):
    if s.exec(select(Trigger).where(Trigger.name == body.name)).first():
        raise HTTPException(409, "Ya hay un disparador con ese nombre")
    cfg = dict(body.config)
    if body.kind == "webhook":
        cfg["token"] = svc.nuevo_token()
    if body.kind == "telegram" and not cfg.get("bot_token"):
        raise HTTPException(400, "Falta el token del bot (se lo pide a @BotFather en Telegram)")
    if body.kind == "schedule" and not (cfg.get("cada_min") or cfg.get("diario")):
        raise HTTPException(400, "Di cada cuántos minutos o a qué hora del día")
    t = Trigger(**{**body.model_dump(), "config": cfg})
    s.add(t)
    s.commit()
    s.refresh(t)
    if t.kind == "telegram" and t.enabled:
        svc.arrancar_telegram(t.id)
    return _publico(t)


@router.put("/api/triggers/{tid}")
def editar(tid: int, body: TriggerBody, s: Session = Depends(get_session)):
    t = s.get(Trigger, tid)
    if not t:
        raise HTTPException(404)
    cfg = dict(body.config)
    # El token del bot vuelve enmascarado de la interfaz; si no lo cambiaron, se conserva
    # Vacío o enmascarado («…abcd») significa «no lo he tocado»: se conserva el que había.
    # Con sólo mirar la máscara, dejar el campo vacío —como sugiere la interfaz— borraba el token.
    if not cfg.get("bot_token") or str(cfg["bot_token"]).startswith("…"):
        cfg["bot_token"] = t.config.get("bot_token", "")
    if t.kind == "webhook":
        cfg["token"] = t.config.get("token")          # el token del webhook no se edita, se regenera
    t.name, t.target_kind, t.target_id, t.config, t.enabled = body.name, body.target_kind, body.target_id, cfg, body.enabled
    s.add(t)
    s.commit()
    if t.kind == "telegram":
        svc.arrancar_telegram(t.id) if t.enabled else svc.parar_telegram(t.id)
    return _publico(t)


@router.post("/api/triggers/{tid}/regenerar")
def regenerar(tid: int, s: Session = Depends(get_session)):
    """Nuevo token de webhook: la URL anterior deja de funcionar al instante."""
    t = s.get(Trigger, tid)
    if not t or t.kind != "webhook":
        raise HTTPException(404)
    t.config = {**t.config, "token": svc.nuevo_token()}
    s.add(t)
    s.commit()
    return {"token": t.config["token"]}


@router.delete("/api/triggers/{tid}")
def borrar(tid: int, s: Session = Depends(get_session)):
    t = s.get(Trigger, tid)
    if t:
        svc.parar_telegram(tid)
        s.delete(t)
        s.commit()
    return {"ok": True}


@router.post("/api/triggers/{tid}/probar")
def probar(tid: int, s: Session = Depends(get_session)):
    """Lanza el disparador a mano con su entrada (o una de ejemplo) para comprobar que responde."""
    t = s.get(Trigger, tid)
    if not t:
        raise HTTPException(404)
    if t.kind == "telegram":
        try:
            yo = svc.tg(t.config.get("bot_token", ""), "getMe")
            return {"ok": True, "bot": f"@{yo.get('username')}", "chats_permitidos": t.config.get("allowed_chat_ids") or []}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(e)}
    return svc.ejecutar(t, t.config.get("input") or "hola", "prueba")


class Entrada(BaseModel):
    input: str = ""


@router.post("/api/hooks/{token}")
def webhook(token: str, body: Entrada):
    """Punto de entrada público de los webhooks. El token es la única llave: trátalo como una contraseña."""
    t = svc.buscar_webhook(token)
    if not t:
        raise HTTPException(404)          # mismo error exista o no, para no revelar nada
    return svc.ejecutar(t, body.input, "webhook")

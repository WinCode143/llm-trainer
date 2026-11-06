"""Disparadores: lanzar un flujo o un agente sin pulsar un botón en la app.

Tres tipos:
  - webhook:  una URL con un token secreto. Cualquier cosa que haga un POST lo lanza.
  - telegram: un bot que escucha mensajes con «long polling». Las conexiones son SALIENTES, así que
              funciona en un PC de casa sin abrir puertos ni exponer nada a internet.
  - schedule: a una hora cada día o cada N minutos, con una entrada fija.

La app no tiene autenticación (es local por diseño), así que cada disparador trae su propio
cerrojo: el webhook, un token imposible de adivinar; Telegram, una lista de chats permitidos. Un bot
sin lista no ejecuta nada: sólo le dice a quien le escribe cuál es su chat_id, para poder añadirlo.
"""
from __future__ import annotations

import hmac
import secrets
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Callable

import httpx
from sqlmodel import Session, select

from ..db import Agent, Flow, Trigger, engine, now

TELEGRAM = "https://api.telegram.org/bot{token}/{metodo}"
MAX_HISTORIAL = 20

# Aprobaciones pendientes que llegan por Telegram: clave corta -> lo necesario para reanudar
_pendientes: dict[str, dict] = {}
_hilos: dict[int, threading.Event] = {}      # id de disparador telegram -> señal de parada
_lock = threading.Lock()


def nuevo_token() -> str:
    return secrets.token_urlsafe(24)


# ------------------------------------------------------------------ ejecución común

def ejecutar(t: Trigger, texto: str, origen: str) -> dict:
    """Corre el flujo o el agente del disparador con ese texto. Devuelve salida o pausa."""
    t0 = time.time()
    try:
        if t.target_kind == "agent":
            res = _correr_agente(t.target_id, texto)
        else:
            res = _correr_flujo(t.target_id, texto)
    except Exception as e:  # noqa: BLE001
        res = {"estado": "error", "salida": f"Error: {e}"}
    res["segundos"] = round(time.time() - t0, 1)
    _anotar(t.id, {"cuando": now().isoformat(), "origen": origen, "entrada": texto[:200],
                   "estado": res["estado"], "salida": str(res.get("salida", ""))[:400], "segundos": res["segundos"]})
    return res


def _correr_flujo(fid: int, texto: str, thread_id: str | None = None, resume: Any = None) -> dict:
    from . import flows as flows_svc
    with Session(engine, expire_on_commit=False) as s:
        f = s.get(Flow, fid)
    if not f:
        raise RuntimeError(f"El flujo #{fid} ya no existe")
    tid = thread_id or f"trig-{uuid.uuid4().hex[:10]}"
    entrada = {"entrada": texto, "_entrada": texto}
    for ev in flows_svc.run(f.spec, entrada, thread_id=tid, resume=resume):
        if ev["type"] == "interrupt":
            return {"estado": "espera", "salida": ev["data"], "thread_id": tid, "flow_id": fid}
        if ev["type"] == "done":
            return {"estado": "ok", "salida": ev.get("output") or (ev.get("vars") or {}).get("_salida", "")}
        if ev["type"] == "error":
            return {"estado": "error", "salida": ev["message"]}
    return {"estado": "error", "salida": "El flujo terminó sin salida"}


def _correr_agente(aid: int, texto: str) -> dict:
    from .agents import run as run_agent
    with Session(engine) as s:
        a = s.get(Agent, aid)
    if not a:
        raise RuntimeError(f"El agente #{aid} ya no existe")
    final = ""
    for ev in run_agent(a.model_dump(), [{"role": "user", "content": texto}]):
        if ev["type"] == "done":
            final = ev["final"]
        elif ev["type"] == "error":
            return {"estado": "error", "salida": ev["message"]}
    return {"estado": "ok", "salida": final}


def _anotar(tid: int, fila: dict) -> None:
    with Session(engine) as s:
        t = s.get(Trigger, tid)
        if t:
            t.last_runs = ([fila] + list(t.last_runs or []))[:MAX_HISTORIAL]
            s.add(t)
            s.commit()


# ------------------------------------------------------------------ webhook

def buscar_webhook(token: str) -> Trigger | None:
    """Busca por token en tiempo constante, para no filtrar por el tiempo de respuesta."""
    with Session(engine, expire_on_commit=False) as s:
        for t in s.exec(select(Trigger).where(Trigger.kind == "webhook", Trigger.enabled == True)).all():  # noqa: E712
            if hmac.compare_digest(str(t.config.get("token", "")), token):
                return t
    return None


# ------------------------------------------------------------------ telegram

def tg(token: str, metodo: str, **datos) -> dict:
    r = httpx.post(TELEGRAM.format(token=token, metodo=metodo), json=datos, timeout=40)
    d = r.json()
    if not d.get("ok"):
        raise RuntimeError(d.get("description") or f"Telegram respondió {r.status_code}")
    return d.get("result")


def procesar_update(t: Trigger, upd: dict, enviar: Callable[..., Any]) -> str | None:
    """Atiende un mensaje o una pulsación de botón. Separado del bucle para poder probarlo sin red.

    `enviar(metodo, **datos)` es la llamada a la API de Telegram. Devuelve qué se hizo, para registro."""
    permitidos = {int(x) for x in (t.config.get("allowed_chat_ids") or []) if str(x).lstrip("-").isdigit()}

    # Un botón de Aprobar/Rechazar de un flujo detenido
    if "callback_query" in upd:
        cq = upd["callback_query"]
        chat = cq["message"]["chat"]["id"]
        if chat not in permitidos:
            return "botón de un chat no permitido: ignorado"
        accion, _, clave = (cq.get("data") or "").partition(":")
        with _lock:
            p = _pendientes.pop(clave, None)
        enviar("answerCallbackQuery", callback_query_id=cq["id"])
        if not p:
            enviar("sendMessage", chat_id=chat, text="Esa aprobación ya no está pendiente.")
            return "aprobación caducada"
        res = _correr_flujo(p["flow_id"], p["texto"], thread_id=p["thread_id"],
                            resume={"rejected": accion != "ok", "comment": "desde Telegram", "edited": None})
        _responder(t, chat, res, p["texto"], enviar)
        return f"aprobación {'aceptada' if accion == 'ok' else 'rechazada'}"

    msg = upd.get("message") or {}
    texto = (msg.get("text") or "").strip()
    chat = (msg.get("chat") or {}).get("id")
    if not chat or not texto:
        return None

    if not permitidos:
        # Sin lista no se ejecuta nada. Decirle su id es lo que permite configurarlo.
        enviar("sendMessage", chat_id=chat, text=(
            f"Este bot aún no tiene chats permitidos. Tu chat_id es {chat}: añádelo en LLM Trainer → "
            "Conexiones → Disparadores para poder usarlo."))
        return f"sin lista: se le dijo su chat_id a {chat}"
    if chat not in permitidos:
        return f"chat {chat} no permitido: ignorado en silencio"

    if texto in ("/start", "/help"):
        enviar("sendMessage", chat_id=chat, text=f"Hola. Escríbeme y se lo paso a «{_destino(t)}».")
        return "ayuda"

    enviar("sendChatAction", chat_id=chat, action="typing")
    res = ejecutar(t, texto, f"telegram:{chat}")
    _responder(t, chat, res, texto, enviar)
    return res["estado"]


def _responder(t: Trigger, chat: int, res: dict, texto: str, enviar) -> None:
    if res["estado"] == "espera":
        # El flujo llegó a un nodo de Aprobación: se pregunta por Telegram con dos botones
        d = res["salida"] or {}
        clave = uuid.uuid4().hex[:12]
        with _lock:
            _pendientes[clave] = {"flow_id": res["flow_id"], "thread_id": res["thread_id"], "texto": texto}
        cuerpo = f"{d.get('message') or '¿Continuamos?'}\n\n{str(d.get('preview') or '')[:3000]}"
        enviar("sendMessage", chat_id=chat, text=cuerpo, reply_markup={"inline_keyboard": [[
            {"text": "✅ Aprobar", "callback_data": f"ok:{clave}"},
            {"text": "❌ Rechazar", "callback_data": f"no:{clave}"}]]})
        return
    salida = str(res.get("salida") or "(sin respuesta)")
    # Telegram corta a 4096 caracteres por mensaje
    for i in range(0, len(salida), 4000):
        enviar("sendMessage", chat_id=chat, text=salida[i:i + 4000])


def _destino(t: Trigger) -> str:
    with Session(engine) as s:
        obj = s.get(Agent if t.target_kind == "agent" else Flow, t.target_id)
    return obj.name if obj else f"{t.target_kind} #{t.target_id}"


def _bucle_telegram(tid: int, parar: threading.Event) -> None:
    offset = 0
    while not parar.is_set():
        with Session(engine, expire_on_commit=False) as s:
            t = s.get(Trigger, tid)
        if not t or not t.enabled:
            return
        token = t.config.get("bot_token", "")
        try:
            updates = tg(token, "getUpdates", offset=offset, timeout=30, allowed_updates=["message", "callback_query"])
        except Exception as e:  # noqa: BLE001
            print(f"[telegram] «{t.name}»: {e}; reintento en 15 s")
            parar.wait(15)
            continue
        for u in updates or []:
            offset = u["update_id"] + 1
            try:
                hecho = procesar_update(t, u, lambda metodo, **d: tg(token, metodo, **d))
                if hecho:
                    print(f"[telegram] «{t.name}»: {hecho}")
            except Exception as e:  # noqa: BLE001
                print(f"[telegram] «{t.name}» al procesar un mensaje: {e}")


def arrancar_telegram(tid: int) -> None:
    parar_telegram(tid)
    ev = threading.Event()
    _hilos[tid] = ev
    threading.Thread(target=_bucle_telegram, args=(tid, ev), daemon=True, name=f"telegram-{tid}").start()


def parar_telegram(tid: int) -> None:
    ev = _hilos.pop(tid, None)
    if ev:
        ev.set()


# ------------------------------------------------------------------ programados

def toca(cfg: dict, ultima: datetime | None, ahora: datetime) -> bool:
    """¿Le toca ejecutarse? «cada_min» cada N minutos, o «diario» a una hora HH:MM."""
    if cfg.get("cada_min"):
        return ultima is None or (ahora - ultima).total_seconds() >= int(cfg["cada_min"]) * 60
    if cfg.get("diario"):
        hh, mm = (int(x) for x in str(cfg["diario"]).split(":"))
        objetivo = ahora.replace(hour=hh, minute=mm, second=0, microsecond=0)
        return ahora >= objetivo and (ultima is None or ultima < objetivo)
    return False


def _bucle_programados() -> None:
    while True:
        ahora = datetime.now()
        with Session(engine, expire_on_commit=False) as s:
            programados = s.exec(select(Trigger).where(Trigger.kind == "schedule", Trigger.enabled == True)).all()  # noqa: E712
        for t in programados:
            ultima = None
            if t.last_runs:
                try:
                    ultima = datetime.fromisoformat(t.last_runs[0]["cuando"]).replace(tzinfo=None)
                except (KeyError, ValueError):
                    ultima = None
            if not toca(t.config, ultima, ahora):
                continue
            res = ejecutar(t, t.config.get("input") or "", "programado")
            _avisar(t, res)
        time.sleep(30)


def _avisar(t: Trigger, res: dict) -> None:
    """Un resultado programado que nadie ve no sirve: se manda al chat de un bot de Telegram."""
    bot_id = t.config.get("avisar_a")
    if not bot_id:
        return
    with Session(engine, expire_on_commit=False) as s:
        bot = s.get(Trigger, int(bot_id))
    if not bot or bot.kind != "telegram":
        return
    for chat in bot.config.get("allowed_chat_ids") or []:
        try:
            _responder(bot, int(chat), res, t.config.get("input") or "", lambda m, **d: tg(bot.config["bot_token"], m, **d))
        except Exception as e:  # noqa: BLE001
            print(f"[programado] no se pudo avisar por Telegram: {e}")


def arrancar_todo() -> None:
    """Al arrancar la app: un hilo por bot de Telegram activo y uno para los programados."""
    with Session(engine) as s:
        for t in s.exec(select(Trigger).where(Trigger.kind == "telegram", Trigger.enabled == True)).all():  # noqa: E712
            arrancar_telegram(t.id)
    threading.Thread(target=_bucle_programados, daemon=True, name="programados").start()

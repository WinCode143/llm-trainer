"""Triggers: run a flow or an agent without clicking a button in the app.

Three kinds:
    - webhook:  a URL with a secret token. Anything that sends a POST runs it.
    - telegram: a bot that listens for messages with long polling. The connections are OUTBOUND, so
                            it works on a home PC without opening ports or exposing anything to the internet.
    - schedule: at a time every day or every N minutes, with a fixed input.

The app has no authentication (it's local by design), so each trigger brings its own
lock: the webhook, an unguessable token; Telegram, a list of allowed chats. A bot
without a list runs nothing: it only tells whoever messages it their chat_id, so it can be added.
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

TELEGRAM = "https://api.telegram.org/bot{token}/{method}"
MAX_HISTORY = 20

# Pending approvals that arrive through Telegram: short key -> what's needed to resume
_pending: dict[str, dict] = {}
_threads: dict[int, threading.Event] = {}      # telegram trigger id -> stop signal
_lock = threading.Lock()


def new_token() -> str:
    return secrets.token_urlsafe(24)


# ------------------------------------------------------------------ common execution

def execute(t: Trigger, text: str, source: str) -> dict:
    """Runs the trigger's flow or agent with that text. Returns the output or a pause."""
    t0 = time.time()
    try:
        if t.target_kind == "agent":
            res = _run_agent(t.target_id, text)
        else:
            res = _run_flow(t.target_id, text)
    except Exception as e:  # noqa: BLE001
        res = {"estado": "error", "salida": f"Error: {e}"}
    res["segundos"] = round(time.time() - t0, 1)
    _log_run(t.id, {"cuando": now().isoformat(), "origen": source, "entrada": text[:200],
                   "estado": res["estado"], "salida": str(res.get("salida", ""))[:400], "segundos": res["segundos"]})
    return res


def _run_flow(fid: int, text: str, thread_id: str | None = None, resume: Any = None) -> dict:
    from . import flows as flows_svc
    with Session(engine, expire_on_commit=False) as s:
        f = s.get(Flow, fid)
    if not f:
        raise RuntimeError(f"Flow #{fid} no longer exists")
    tid = thread_id or f"trig-{uuid.uuid4().hex[:10]}"
    inputs = {"input": text, "_input": text}
    for ev in flows_svc.run(f.spec, inputs, thread_id=tid, resume=resume):
        if ev["type"] == "interrupt":
            return {"estado": "waiting", "salida": ev["data"], "thread_id": tid, "flow_id": fid}
        if ev["type"] == "done":
            return {"estado": "ok", "salida": ev.get("output") or (ev.get("vars") or {}).get("_output", "")}
        if ev["type"] == "error":
            return {"estado": "error", "salida": ev["message"]}
    return {"estado": "error", "salida": "The flow ended without output"}


def _run_agent(aid: int, text: str) -> dict:
    from .agents import run as run_agent
    with Session(engine) as s:
        a = s.get(Agent, aid)
    if not a:
        raise RuntimeError(f"Agent #{aid} no longer exists")
    final = ""
    for ev in run_agent(a.model_dump(), [{"role": "user", "content": text}]):
        if ev["type"] == "done":
            final = ev["final"]
        elif ev["type"] == "error":
            return {"estado": "error", "salida": ev["message"]}
    return {"estado": "ok", "salida": final}


def _log_run(tid: int, row: dict) -> None:
    with Session(engine) as s:
        t = s.get(Trigger, tid)
        if t:
            t.last_runs = ([row] + list(t.last_runs or []))[:MAX_HISTORY]
            s.add(t)
            s.commit()


# ------------------------------------------------------------------ webhook

def find_webhook(token: str) -> Trigger | None:
    """Looks up the token in constant time, so response time leaks nothing."""
    with Session(engine, expire_on_commit=False) as s:
        for t in s.exec(select(Trigger).where(Trigger.kind == "webhook", Trigger.enabled == True)).all():  # noqa: E712
            if hmac.compare_digest(str(t.config.get("token", "")), token):
                return t
    return None


# ------------------------------------------------------------------ telegram

def tg(token: str, method: str, **data) -> dict:
    r = httpx.post(TELEGRAM.format(token=token, method=method), json=data, timeout=40)
    d = r.json()
    if not d.get("ok"):
        raise RuntimeError(d.get("description") or f"Telegram answered {r.status_code}")
    return d.get("result")


def handle_update(t: Trigger, upd: dict, send: Callable[..., Any]) -> str | None:
    """Handles a message or a button press. Separate from the loop so it can be tested without a network.

    `send(method, **data)` is the call to the Telegram API. Returns what was done, for the log."""
    allowed = {int(x) for x in (t.config.get("allowed_chat_ids") or []) if str(x).lstrip("-").isdigit()}

    # An Approve/Reject button from a paused flow
    if "callback_query" in upd:
        cq = upd["callback_query"]
        chat = cq["message"]["chat"]["id"]
        if chat not in allowed:
            return "button from a chat that isn't allowed: ignored"
        action, _, key = (cq.get("data") or "").partition(":")
        with _lock:
            p = _pending.pop(key, None)
        send("answerCallbackQuery", callback_query_id=cq["id"])
        if not p:
            send("sendMessage", chat_id=chat, text="That approval is no longer pending.")
            return "approval expired"
        res = _run_flow(p["flow_id"], p["texto"], thread_id=p["thread_id"],
                            resume={"rejected": action != "ok", "comment": "from Telegram", "edited": None})
        _reply(t, chat, res, p["texto"], send)
        return f"approval {'accepted' if action == 'ok' else 'rejected'}"

    msg = upd.get("message") or {}
    text = (msg.get("text") or "").strip()
    chat = (msg.get("chat") or {}).get("id")
    if not chat or not text:
        return None

    if not allowed:
        # Without a list nothing runs. Telling them their id is what lets them set it up.
        send("sendMessage", chat_id=chat, text=(
            f"This bot has no allowed chats yet. Your chat_id is {chat}: add it in LLM Trainer → "
            "Connections → Triggers to use it."))
        return f"no list: told {chat} their chat_id"
    if chat not in allowed:
        return f"chat {chat} not allowed: silently ignored"

    if text in ("/start", "/help"):
        send("sendMessage", chat_id=chat, text=f"Hi. Write to me and I'll pass it on to “{_target_name(t)}”.")
        return "help"

    send("sendChatAction", chat_id=chat, action="typing")
    res = execute(t, text, f"telegram:{chat}")
    _reply(t, chat, res, text, send)
    return res["estado"]


def _reply(t: Trigger, chat: int, res: dict, text: str, send) -> None:
    if res["estado"] == "waiting":
        # The flow reached an Approval node: ask through Telegram with two buttons
        d = res["salida"] or {}
        key = uuid.uuid4().hex[:12]
        with _lock:
            _pending[key] = {"flow_id": res["flow_id"], "thread_id": res["thread_id"], "texto": text}
        body = f"{d.get('message') or 'Shall we continue?'}\n\n{str(d.get('preview') or '')[:3000]}"
        send("sendMessage", chat_id=chat, text=body, reply_markup={"inline_keyboard": [[
            {"text": "✅ Approve", "callback_data": f"ok:{key}"},
            {"text": "❌ Reject", "callback_data": f"no:{key}"}]]})
        return
    output = str(res.get("salida") or "(no answer)")
    # Telegram cuts messages at 4096 characters
    for i in range(0, len(output), 4000):
        send("sendMessage", chat_id=chat, text=output[i:i + 4000])


def _target_name(t: Trigger) -> str:
    with Session(engine) as s:
        obj = s.get(Agent if t.target_kind == "agent" else Flow, t.target_id)
    return obj.name if obj else f"{t.target_kind} #{t.target_id}"


def _telegram_loop(tid: int, stop: threading.Event) -> None:
    offset = 0
    while not stop.is_set():
        with Session(engine, expire_on_commit=False) as s:
            t = s.get(Trigger, tid)
        if not t or not t.enabled:
            return
        token = t.config.get("bot_token", "")
        try:
            updates = tg(token, "getUpdates", offset=offset, timeout=30, allowed_updates=["message", "callback_query"])
        except Exception as e:  # noqa: BLE001
            print(f"[telegram] “{t.name}”: {e}; retrying in 15 s")
            stop.wait(15)
            continue
        for u in updates or []:
            offset = u["update_id"] + 1
            try:
                done = handle_update(t, u, lambda method, **d: tg(token, method, **d))
                if done:
                    print(f"[telegram] “{t.name}”: {done}")
            except Exception as e:  # noqa: BLE001
                print(f"[telegram] “{t.name}” while handling a message: {e}")


def start_telegram(tid: int) -> None:
    stop_telegram(tid)
    ev = threading.Event()
    _threads[tid] = ev
    threading.Thread(target=_telegram_loop, args=(tid, ev), daemon=True, name=f"telegram-{tid}").start()


def stop_telegram(tid: int) -> None:
    ev = _threads.pop(tid, None)
    if ev:
        ev.set()


# ------------------------------------------------------------------ scheduled

def is_due(cfg: dict, last: datetime | None, now_: datetime) -> bool:
    """Is it due? “cada_min” every N minutes, or “diario” at a time HH:MM."""
    if cfg.get("cada_min"):
        return last is None or (now_ - last).total_seconds() >= int(cfg["cada_min"]) * 60
    if cfg.get("diario"):
        hh, mm = (int(x) for x in str(cfg["diario"]).split(":"))
        target = now_.replace(hour=hh, minute=mm, second=0, microsecond=0)
        return now_ >= target and (last is None or last < target)
    return False


def _schedule_loop() -> None:
    while True:
        now_ = datetime.now()
        with Session(engine, expire_on_commit=False) as s:
            scheduled = s.exec(select(Trigger).where(Trigger.kind == "schedule", Trigger.enabled == True)).all()  # noqa: E712
        for t in scheduled:
            last = None
            if t.last_runs:
                try:
                    last = datetime.fromisoformat(t.last_runs[0]["cuando"]).replace(tzinfo=None)
                except (KeyError, ValueError):
                    last = None
            if not is_due(t.config, last, now_):
                continue
            res = execute(t, t.config.get("input") or "", "scheduled")
            _notify(t, res)
        time.sleep(30)


def _notify(t: Trigger, res: dict) -> None:
    """A scheduled result nobody sees is useless: it's sent to a Telegram bot's chat."""
    bot_id = t.config.get("avisar_a")
    if not bot_id:
        return
    with Session(engine, expire_on_commit=False) as s:
        bot = s.get(Trigger, int(bot_id))
    if not bot or bot.kind != "telegram":
        return
    for chat in bot.config.get("allowed_chat_ids") or []:
        try:
            _reply(bot, int(chat), res, t.config.get("input") or "", lambda m, **d: tg(bot.config["bot_token"], m, **d))
        except Exception as e:  # noqa: BLE001
            print(f"[scheduled] couldn't notify through Telegram: {e}")


def start_all() -> None:
    """On app start: one thread per active Telegram bot and one for the schedules."""
    with Session(engine) as s:
        for t in s.exec(select(Trigger).where(Trigger.kind == "telegram", Trigger.enabled == True)).all():  # noqa: E712
            start_telegram(t.id)
    threading.Thread(target=_schedule_loop, daemon=True, name="schedules").start()

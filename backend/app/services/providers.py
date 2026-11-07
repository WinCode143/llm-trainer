"""Helper models: besides the model loaded in VRAM, the app can lean on external models.

What for: generating synthetic datasets, extracting the knowledge graph, judging
evaluations or enriching RAG are tasks where a small model falls short. With this you can
hand them to a big model (local on another machine, or paid) without changing anything else.

Two families, both over HTTP:
    - “openai”:    any server compatible with the OpenAI API (Ollama, LM Studio, remote vLLM,
                                  OpenRouter, Groq, DeepSeek, OpenAI itself…). Only the base URL is needed.
    - “anthropic”: the Claude API (its own message format).

The local model stays the default and needs no setup: `complete(None, …)`.

Keys are stored in plain text in the local database. They never leave your machine except towards the
provider you configure, but anyone with access to the file can read them.
"""
from __future__ import annotations

import json
import re
from typing import Any

import httpx
from sqlmodel import Session, select

from ..db import Provider, engine

KINDS = {
    "openai": {"label": "OpenAI-compatible", "hint": "Ollama, LM Studio, remote vLLM, OpenRouter, Groq, DeepSeek, OpenAI…"},
    "anthropic": {"label": "Anthropic (Claude)", "hint": "The Claude API with your own key"},
}

# Presets so you don't have to remember the URLs
PRESETS = [
    {"key": "ollama", "label": "Ollama (local)", "kind": "openai", "base_url": "http://localhost:11434/v1",
     "api_key": "ollama", "model": "llama3.1:8b", "note": "Free; the model has to be pulled in Ollama"},
    {"key": "lmstudio", "label": "LM Studio (local)", "kind": "openai", "base_url": "http://localhost:1234/v1",
     "api_key": "lm-studio", "model": "", "note": "Free; start the local server in LM Studio"},
    {"key": "openrouter", "label": "OpenRouter", "kind": "openai", "base_url": "https://openrouter.ai/api/v1",
     "api_key": "", "model": "meta-llama/llama-3.3-70b-instruct", "note": "Hundreds of models with a single key"},
    {"key": "groq", "label": "Groq", "kind": "openai", "base_url": "https://api.groq.com/openai/v1",
     "api_key": "", "model": "llama-3.3-70b-versatile", "note": "Very fast and cheap"},
    {"key": "deepseek", "label": "DeepSeek", "kind": "openai", "base_url": "https://api.deepseek.com/v1",
     "api_key": "", "model": "deepseek-chat", "note": "Very cheap, good at generating data"},
    {"key": "openai", "label": "OpenAI", "kind": "openai", "base_url": "https://api.openai.com/v1",
     "api_key": "", "model": "gpt-4o-mini", "note": "Needs a paid key"},
    {"key": "anthropic", "label": "Anthropic (Claude)", "kind": "anthropic", "base_url": "https://api.anthropic.com/v1",
     "api_key": "", "model": "claude-sonnet-5", "note": "Needs a paid key; a Claude Code subscription won't work"},
    {"key": "vllm", "label": "vLLM on another machine", "kind": "openai", "base_url": "http://192.168.1.10:8000/v1",
     "api_key": "-", "model": "base", "note": "Another PC on your network serving a big model"},
]


def get(pid: int) -> Provider | None:
    with Session(engine, expire_on_commit=False) as s:
        return s.get(Provider, pid)


def listing(include_keys: bool = False) -> list[dict]:
    with Session(engine) as s:
        rows = s.exec(select(Provider).order_by(Provider.id)).all()
    out = []
    for p in rows:
        d = p.model_dump(mode="json")
        # The key never goes back whole to the interface: only the last characters, to recognize it
        d["api_key"] = p.api_key if include_keys else (f"…{p.api_key[-4:]}" if p.api_key else "")
        d["has_key"] = bool(p.api_key)
        out.append(d)
    return out


def _client(p: Provider) -> httpx.Client:
    headers = {"content-type": "application/json"}
    if p.kind == "anthropic":
        headers |= {"x-api-key": p.api_key, "anthropic-version": "2023-06-01"}
    elif p.api_key:
        headers["authorization"] = f"Bearer {p.api_key}"
    return httpx.Client(base_url=p.base_url.rstrip("/"), headers=headers, timeout=httpx.Timeout(180, connect=10))


def _err(r: httpx.Response) -> str:
    try:
        body = r.json()
        msg = body.get("error", {}).get("message") if isinstance(body.get("error"), dict) else body.get("error")
        return str(msg or body)[:300]
    except ValueError:
        return r.text[:300]


def remote_models(p: Provider) -> list[str]:
    """Models the provider offers (for the dropdown). Some don't expose the list."""
    if p.kind == "anthropic":
        with _client(p) as c:
            r = c.get("/models")
            if r.status_code != 200:
                raise RuntimeError(_err(r))
            return [m["id"] for m in r.json().get("data", [])]
    with _client(p) as c:
        r = c.get("/models")
        if r.status_code != 200:
            raise RuntimeError(_err(r))
        return sorted(m["id"] for m in r.json().get("data", []))


def _messages(prompt: str, system: str | None, history: list[dict] | None = None) -> list[dict]:
    return (([{"role": "system", "content": system}] if system else []) + list(history or [])
            + [{"role": "user", "content": prompt}])


def _merge_roles(msgs: list[dict]) -> list[dict]:
    """Joins consecutive messages with the same role (Anthropic requires user/assistant to alternate)."""
    out: list[dict] = []
    for m in msgs:
        if out and out[-1]["role"] == m["role"]:
            out[-1] = {"role": m["role"], "content": out[-1]["content"] + "\n\n" + m["content"]}
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def generate(p: Provider, prompt: str, *, system: str | None = None, max_tokens: int = 600,
             temperature: float = 0.3, json_mode: bool = False, history: list[dict] | None = None) -> str:
    """One generation against an external provider. Returns the text.

    `history` holds the earlier turns of a conversation (user/assistant), sent before `prompt`."""
    if p.kind == "anthropic":
        body: dict[str, Any] = {"model": p.model, "max_tokens": max_tokens, "temperature": temperature,
                                "messages": _merge_roles(list(history or []) + [{"role": "user", "content": prompt}])}
        if system:
            body["system"] = system
        if json_mode:  # Claude has no JSON mode: it's asked for in the prompt and cleaned up afterwards
            body["messages"][-1]["content"] += "\n\nReply ONLY with valid JSON, with no text around it."
        with _client(p) as c:
            r = c.post("/messages", json=body)
            if r.status_code != 200:
                raise RuntimeError(f"{p.name}: {_err(r)}")
            return "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text")

    body = {"model": p.model, "messages": _messages(prompt, system, history), "max_tokens": max_tokens,
            "temperature": temperature}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    with _client(p) as c:
        r = c.post("/chat/completions", json=body)
        if r.status_code == 400 and json_mode:  # not all of them support response_format
            body.pop("response_format")
            body["messages"][-1]["content"] += "\n\nReply ONLY with valid JSON, with no text around it."
            r = c.post("/chat/completions", json=body)
        if r.status_code != 200:
            raise RuntimeError(f"{p.name}: {_err(r)}")
        choices = r.json().get("choices") or [{}]
        return (choices[0].get("message") or {}).get("content") or ""


def test(p: Provider) -> dict:
    """A real ping: asks for one word and measures how long it takes."""
    import time

    from .evaluation import strip_think
    t0 = time.time()
    try:
        # 200 tokens: reasoning models spend the first ones in <think> and would return a cut-off answer
        out = strip_think(generate(p, "Reply with just the word: ready", max_tokens=200, temperature=0))
        return {"ok": True, "seconds": round(time.time() - t0, 2), "reply": out.strip()[:80]}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "seconds": round(time.time() - t0, 2), "error": str(e)[:300]}


# --------------------------------------------------------------- single generation entry point

def complete(helper: int | None, prompt: str, *, system: str | None = None, max_tokens: int = 600,
             temperature: float = 0.3, json_mode: bool = False, enable_thinking: bool | None = False,
             on_token=None, history: list[dict] | None = None) -> str:
    """Generates with the chosen helper, or with the model loaded in VRAM if `helper` is None.

    It's the only path auxiliary generations go through (synthetic data, graph, judge,
    enrichment, flow nodes), so switching models means changing one number.
    """
    from .evaluation import strip_think
    from .inference import GenParams, manager

    if helper:
        p = get(helper)
        if not p:
            raise RuntimeError(f"Helper model #{helper} no longer exists; check it in System → Helper models")
        if not p.enabled:
            raise RuntimeError(f"The helper model “{p.name}” is disabled")
        return strip_think(generate(p, prompt, system=system, max_tokens=max_tokens, temperature=temperature,
                                    json_mode=json_mode, history=history)).strip()

    if not manager.current:
        raise RuntimeError("No model is loaded. Load one in Models, or pick an external helper model.")
    msgs = _messages(prompt, system, history)
    params = GenParams(max_new_tokens=max_tokens, temperature=temperature, enable_thinking=enable_thinking)
    if on_token:
        parts = []
        for t in manager.stream_chat(msgs, params):
            parts.append(t)
            on_token(t)
        return strip_think("".join(parts)).strip()
    return strip_think("".join(manager.stream_chat(msgs, params))).strip()


JSON_RE = re.compile(r"\{.*\}|\[.*\]", re.S)


def complete_json(helper: int | None, prompt: str, **kw) -> Any:
    """Like `complete` but returning the already parsed JSON (or None if it couldn't)."""
    raw = complete(helper, prompt, json_mode=True, **kw)
    m = JSON_RE.search(raw)
    for candidate in (raw, m.group(0) if m else None):  # whole, and otherwise the first JSON block that shows up
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def describe(helper: int | None) -> str:
    """Short text for traces and logs."""
    if not helper:
        from .inference import manager
        return manager.current.name if manager.current else "local model (none loaded)"
    p = get(helper)
    return f"{p.name} ({p.model})" if p else f"helper #{helper} (deleted)"

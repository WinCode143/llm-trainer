"""Modelos ayudantes: además del modelo cargado en VRAM, la app puede apoyarse en modelos externos.

Para qué sirve: generar datasets sintéticos, extraer el grafo de conocimiento, hacer de juez en las
evaluaciones o enriquecer el RAG son tareas donde un modelo pequeño se queda corto. Con esto puedes
delegarlas en un modelo grande (local en otra máquina, o de pago) sin cambiar nada más.

Dos familias, las dos por HTTP:
  - «openai»:    cualquier servidor compatible con la API de OpenAI (Ollama, LM Studio, vLLM remoto,
                 OpenRouter, Groq, DeepSeek, la propia OpenAI…). Sólo hace falta la URL base.
  - «anthropic»: la API de Claude (formato de mensajes propio).

El modelo local sigue siendo el de por defecto y no requiere configurar nada: `complete(None, …)`.

Las claves se guardan en la base de datos local, en claro. No salen de tu máquina salvo hacia el
proveedor que configures, pero cualquiera con acceso al archivo puede leerlas.
"""
from __future__ import annotations

import json
import re
from typing import Any

import httpx
from sqlmodel import Session, select

from ..db import Provider, engine

KINDS = {
    "openai": {"label": "Compatible con OpenAI", "hint": "Ollama, LM Studio, vLLM remoto, OpenRouter, Groq, DeepSeek, OpenAI…"},
    "anthropic": {"label": "Anthropic (Claude)", "hint": "API de Claude con clave propia"},
}

# Atajos para no tener que recordar las URLs
PRESETS = [
    {"key": "ollama", "label": "Ollama (local)", "kind": "openai", "base_url": "http://localhost:11434/v1",
     "api_key": "ollama", "model": "llama3.1:8b", "note": "Sin coste; el modelo tiene que estar descargado en Ollama"},
    {"key": "lmstudio", "label": "LM Studio (local)", "kind": "openai", "base_url": "http://localhost:1234/v1",
     "api_key": "lm-studio", "model": "", "note": "Sin coste; arranca el servidor local en LM Studio"},
    {"key": "openrouter", "label": "OpenRouter", "kind": "openai", "base_url": "https://openrouter.ai/api/v1",
     "api_key": "", "model": "meta-llama/llama-3.3-70b-instruct", "note": "Cientos de modelos con una sola clave"},
    {"key": "groq", "label": "Groq", "kind": "openai", "base_url": "https://api.groq.com/openai/v1",
     "api_key": "", "model": "llama-3.3-70b-versatile", "note": "Muy rápido y barato"},
    {"key": "deepseek", "label": "DeepSeek", "kind": "openai", "base_url": "https://api.deepseek.com/v1",
     "api_key": "", "model": "deepseek-chat", "note": "Muy barato, bueno generando datos"},
    {"key": "openai", "label": "OpenAI", "kind": "openai", "base_url": "https://api.openai.com/v1",
     "api_key": "", "model": "gpt-4o-mini", "note": "Requiere clave de pago"},
    {"key": "anthropic", "label": "Anthropic (Claude)", "kind": "anthropic", "base_url": "https://api.anthropic.com/v1",
     "api_key": "", "model": "claude-sonnet-5", "note": "Requiere clave de pago; no sirve la suscripción de Claude Code"},
    {"key": "vllm", "label": "vLLM en otra máquina", "kind": "openai", "base_url": "http://192.168.1.10:8000/v1",
     "api_key": "-", "model": "base", "note": "Otro PC de tu red sirviendo un modelo grande"},
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
        # La clave nunca vuelve entera a la interfaz: sólo los últimos caracteres para reconocerla
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
    """Modelos que ofrece el proveedor (para el desplegable). Algunos no exponen el listado."""
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


def _messages(prompt: str, system: str | None) -> list[dict]:
    return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]


def generate(p: Provider, prompt: str, *, system: str | None = None, max_tokens: int = 600,
             temperature: float = 0.3, json_mode: bool = False) -> str:
    """Una generación contra un proveedor externo. Devuelve el texto."""
    if p.kind == "anthropic":
        body: dict[str, Any] = {"model": p.model, "max_tokens": max_tokens, "temperature": temperature,
                                "messages": [{"role": "user", "content": prompt}]}
        if system:
            body["system"] = system
        if json_mode:  # Claude no tiene modo JSON: se pide en el prompt y se limpia después
            body["messages"][0]["content"] += "\n\nResponde ÚNICAMENTE con JSON válido, sin texto alrededor."
        with _client(p) as c:
            r = c.post("/messages", json=body)
            if r.status_code != 200:
                raise RuntimeError(f"{p.name}: {_err(r)}")
            return "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text")

    body = {"model": p.model, "messages": _messages(prompt, system), "max_tokens": max_tokens,
            "temperature": temperature}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    with _client(p) as c:
        r = c.post("/chat/completions", json=body)
        if r.status_code == 400 and json_mode:  # no todos admiten response_format
            body.pop("response_format")
            body["messages"][-1]["content"] += "\n\nResponde ÚNICAMENTE con JSON válido, sin texto alrededor."
            r = c.post("/chat/completions", json=body)
        if r.status_code != 200:
            raise RuntimeError(f"{p.name}: {_err(r)}")
        choices = r.json().get("choices") or [{}]
        return (choices[0].get("message") or {}).get("content") or ""


def test(p: Provider) -> dict:
    """Ping real: pide una palabra y mide cuánto tarda."""
    import time

    from .evaluation import strip_think
    t0 = time.time()
    try:
        # 200 tokens: los modelos razonadores gastan los primeros en <think> y devolverían una respuesta cortada
        out = strip_think(generate(p, "Responde solo con la palabra: listo", max_tokens=200, temperature=0))
        return {"ok": True, "seconds": round(time.time() - t0, 2), "reply": out.strip()[:80]}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "seconds": round(time.time() - t0, 2), "error": str(e)[:300]}


# --------------------------------------------------------------- punto único de generación

def complete(helper: int | None, prompt: str, *, system: str | None = None, max_tokens: int = 600,
             temperature: float = 0.3, json_mode: bool = False, enable_thinking: bool | None = False,
             on_token=None) -> str:
    """Genera con el ayudante elegido, o con el modelo cargado en VRAM si `helper` es None.

    Es el único punto por el que pasan las generaciones auxiliares (datos sintéticos, grafo, juez,
    enriquecimiento, nodos de flujo), así que cambiar de modelo es cambiar un número.
    """
    from .evaluation import strip_think
    from .inference import GenParams, manager

    if helper:
        p = get(helper)
        if not p:
            raise RuntimeError(f"El modelo ayudante #{helper} ya no existe; revísalo en Sistema → Modelos ayudantes")
        if not p.enabled:
            raise RuntimeError(f"El modelo ayudante «{p.name}» está desactivado")
        return strip_think(generate(p, prompt, system=system, max_tokens=max_tokens, temperature=temperature,
                                    json_mode=json_mode)).strip()

    if not manager.current:
        raise RuntimeError("No hay ningún modelo cargado. Cárgalo en Modelos, o elige un modelo ayudante externo.")
    msgs = _messages(prompt, system)
    params = GenParams(max_new_tokens=max_tokens, temperature=temperature, enable_thinking=enable_thinking)
    if on_token:
        partes = []
        for t in manager.stream_chat(msgs, params):
            partes.append(t)
            on_token(t)
        return strip_think("".join(partes)).strip()
    return strip_think("".join(manager.stream_chat(msgs, params))).strip()


JSON_RE = re.compile(r"\{.*\}|\[.*\]", re.S)


def complete_json(helper: int | None, prompt: str, **kw) -> Any:
    """Como `complete` pero devolviendo el JSON ya interpretado (o None si no se pudo)."""
    raw = complete(helper, prompt, json_mode=True, **kw)
    m = JSON_RE.search(raw)
    for candidate in (raw, m.group(0) if m else None):  # entero, y si no el primer bloque JSON que aparezca
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def describe(helper: int | None) -> str:
    """Texto corto para las trazas y los registros."""
    if not helper:
        from .inference import manager
        return manager.current.name if manager.current else "modelo local (ninguno cargado)"
    p = get(helper)
    return f"{p.name} ({p.model})" if p else f"ayudante #{helper} (borrado)"

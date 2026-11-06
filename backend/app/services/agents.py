"""Bucle de agente: el modelo decide cuándo llamar a herramientas, se ejecutan y se le devuelve el resultado.

Funciona con modelos con tool-calling nativo (plantilla de chat con `tools`, p. ej. Qwen) y, si la plantilla no
lo soporta, describiendo las herramientas en el prompt del sistema con el mismo formato <tool_call>{…}</tool_call>.
"""
import json
import re
from typing import Iterator

from .evaluation import strip_think
from .inference import GenParams, manager
from .tools import execute, resolve

CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
# Formato nativo de Gemma 4: <|tool_call>call:nombre{…}<tool_call|>. Aunque se le pida el formato de
# arriba, en GGUF vuelve al suyo en torno a una de cada tres veces; sin reconocerlo, la llamada se
# devolvía al usuario como si fuera la respuesta.
GEMMA_CALL_RE = re.compile(r"<\|tool_call>\s*call:([\w.\-]+):?\s*(\{.*?\})\s*<tool_call\|>", re.S)
# Variante con atributos: <tool_call name="x" arguments='{…}'></tool_call> (o autocerrada). Gemma la usa
# a veces con muchas herramientas MCP; sin reconocerla, la llamada salía como respuesta final.
ATTR_CALL_RE = re.compile(r"""<tool_call\s+name=["']([\w.\-]+)["']\s+arguments=(?:'(\{.*?\})'|"(\{.*?\})")\s*/?>(?:\s*</tool_call>)?""", re.S)
# Canal de pensamiento de Gemma 4: <|channel>thought … <channel|>
GEMMA_CANAL_RE = re.compile(r"<\|channel>.*?<channel\|>", re.S)
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)

FALLBACK_SYSTEM = """Tienes estas herramientas disponibles:
{tools}

Para usar una, responde EXACTAMENTE con este formato y nada más:
<tool_call>{{"name": "<nombre>", "arguments": {{…}}}}</tool_call>
Recibirás el resultado y podrás usar otra herramienta o dar la respuesta final.
Cuando tengas la información suficiente, responde al usuario en lenguaje natural, sin <tool_call>."""


def _generar(agent: dict, convo: list[dict], params, tools_schema):
    """Un turno del agente. Va por el modelo en VRAM o por un ayudante externo.

    Con un ayudante no hay tool-calling nativo (cada proveedor lo expone distinto), así que se
    describen las herramientas en el prompt y se leen las llamadas del texto, que es el camino que
    ya existía para los modelos cuya plantilla no las soporta."""
    helper = agent.get("helper")
    if helper:
        from . import providers
        texto = providers.complete(helper, convo[-1]["content"],
                                   system="\n\n".join(m["content"] for m in convo[:-1] if m["role"] != "user") or None,
                                   max_tokens=params.max_new_tokens, temperature=params.temperature)
        yield texto
        return
    extra = {"tools": tools_schema} if tools_schema else {}
    yield from manager.stream_chat(convo, params, template_kwargs=extra)


def template_supports_tools() -> bool:
    tok = getattr(manager.current, "tokenizer", None)
    tpl = getattr(tok, "chat_template", None) or ""
    return "tools" in tpl


def _json_laxo(s: str):
    """JSON tal como lo escribe Gemma 4 en su formato nativo: claves sin comillas ({names:[…]}) y cadenas
    entre <|"|>. Primero se intenta como JSON normal; si no, se normaliza. None si no hay forma."""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    t = s.replace('<|"|>', '"')
    # Comillas a las claves sueltas, sin tocar lo que va dentro de una cadena
    partes = re.split(r'("(?:[^"\\]|\\.)*")', t)
    for i in range(0, len(partes), 2):
        partes[i] = re.sub(r"([{,]\s*)([A-Za-z_][\w\-]*)\s*:", r'\1"\2":', partes[i])
    try:
        return json.loads("".join(partes))
    except json.JSONDecodeError:
        return None


def parse_calls(text: str) -> list[dict]:
    """Extrae llamadas a herramientas del texto generado (varios formatos habituales)."""
    calls = []
    for m in GEMMA_CALL_RE.finditer(text):
        args = _json_laxo(m.group(2))
        if args is not None:
            calls.append({"name": m.group(1), "arguments": args})
    for m in ATTR_CALL_RE.finditer(text):
        try:
            calls.append({"name": m.group(1), "arguments": json.loads(m.group(2) or m.group(3))})
        except json.JSONDecodeError:
            continue
    for m in CALL_RE.finditer(text) or []:
        try:
            calls.append(json.loads(m.group(1)))
        except json.JSONDecodeError:
            continue
    if not calls:
        for m in JSON_BLOCK_RE.finditer(text):
            try:
                d = json.loads(m.group(1))
                if isinstance(d, dict) and d.get("name"):
                    calls.append(d)
            except json.JSONDecodeError:
                continue
    out = []
    for c in calls:
        args = c.get("arguments") if isinstance(c.get("arguments"), dict) else c.get("parameters") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"query": args}
        if c.get("name"):
            out.append({"name": str(c["name"]), "arguments": args or {}})
    return out


def visible_text(text: str) -> str:
    t = GEMMA_CANAL_RE.sub("", strip_think(text))
    return ATTR_CALL_RE.sub("", GEMMA_CALL_RE.sub("", CALL_RE.sub("", t))).strip()


def ensure_model(agent: dict) -> str | None:
    """Carga el modelo preferido del agente si no es el que está en VRAM.

    Vive aquí y no en la API porque el agente se ejecuta desde varios sitios (su panel de pruebas y
    los nodos de flujo): si la carga estuviera sólo en un endpoint, el mismo agente daría respuestas
    distintas según desde dónde se le llame, con el modelo que hubiera cargado por casualidad."""
    from sqlmodel import Session

    from ..db import Model, engine
    from .jobs import is_gpu_busy

    mid = agent.get("model_id")
    if not mid or (manager.current and manager.current.model_id == mid):
        return None
    if is_gpu_busy():
        raise RuntimeError("La GPU está ocupada con un entrenamiento; espera a que termine.")
    with Session(engine, expire_on_commit=False) as s:
        m = s.get(Model, mid)
    if not m:
        return None
    manager.load(model_id=m.id, name=m.name, kind=m.kind, path=m.path, base_model=m.base_model,
                 engine=manager.preferred_engine if m.kind != "gguf" else "auto")
    return m.name


def run(agent: dict, messages: list[dict], params: GenParams | None = None) -> Iterator[dict]:
    """Ejecuta el agente. Va emitiendo eventos: token, tool_call, tool_result, step, done, error."""
    usa_ayudante = bool(agent.get("helper"))
    if not usa_ayudante:
        try:
            cargado = ensure_model(agent)
            if cargado:
                yield {"type": "model", "name": cargado}
        except Exception as e:  # noqa: BLE001
            yield {"type": "error", "message": str(e)}
            return
    if not usa_ayudante and not manager.current:
        yield {"type": "error", "message": "No hay ningún modelo cargado (Playground o Modelos), "
                                           "y este agente no tiene modelo preferido asignado."}
        return
    tools = resolve(agent.get("tools") or [])
    by_name = {t["name"]: t for t in tools}
    params = params or GenParams(max_new_tokens=agent.get("max_tokens") or 700,
                                 temperature=agent.get("temperature", 0.3), enable_thinking=False)
    native = (not usa_ayudante) and template_supports_tools() and tools
    system = (agent.get("system_prompt") or "").strip()
    if tools and not native:
        desc = "\n".join(f"- {t['schema']['name']}: {t['schema']['description']} · argumentos: "
                         f"{json.dumps(t['schema']['parameters'].get('properties', {}), ensure_ascii=False)}" for t in tools)
        system = (system + "\n\n" if system else "") + FALLBACK_SYSTEM.format(tools=desc)
    convo: list[dict] = ([{"role": "system", "content": system}] if system else []) + list(messages)
    yield {"type": "tools", "tools": [{"name": t["name"], "label": t["label"]} for t in tools], "native": bool(native)}

    seen_calls: dict[str, int] = {}
    for step in range(int(agent.get("max_steps") or 5)):
        text = ""
        esquema = [t["schema"] for t in tools] if native else None
        try:
            for piece in _generar(agent, convo, params, esquema):
                text += piece
                yield {"type": "token", "text": piece, "step": step}
        except Exception as e:  # noqa: BLE001
            yield {"type": "error", "message": str(e)}
            return
        calls = parse_calls(text)
        if not calls:
            yield {"type": "done", "steps": step + 1, "final": visible_text(text)}
            return
        convo.append({"role": "assistant", "content": text})
        for call in calls[:4]:
            tool = by_name.get(call["name"]) or by_name.get(call["name"].replace("functions.", ""))
            yield {"type": "tool_call", "name": call["name"], "arguments": call["arguments"], "step": step,
                   "known": tool is not None}
            sig = call["name"] + json.dumps(call["arguments"], sort_keys=True, ensure_ascii=False)
            seen_calls[sig] = seen_calls.get(sig, 0) + 1
            if seen_calls[sig] > 2:  # el modelo se atascó repitiendo la misma llamada
                yield {"type": "tool_result", "name": call["name"], "result": "(llamada repetida: se detiene el bucle)",
                       "seconds": 0.0, "step": step, "repeated": True}
                yield {"type": "done", "steps": step + 1, "final": visible_text(text),
                       "stopped_repeating": True}
                return
            if not tool:
                result, secs = f"Error: no existe la herramienta '{call['name']}'. Disponibles: {', '.join(by_name)}", 0.0
            elif seen_calls[sig] == 2:
                result, secs = ("Ya llamaste a esta herramienta con los mismos argumentos y el resultado fue el mismo. "
                                "Cambia los argumentos o responde con lo que ya sabes."), 0.0
            else:
                result, secs = execute(tool, call["arguments"])
            yield {"type": "tool_result", "name": call["name"], "result": result[:4000], "seconds": secs, "step": step}
            if native:
                convo.append({"role": "tool", "name": call["name"], "content": result[:6000]})
            else:
                convo.append({"role": "user", "content": f"Resultado de {call['name']}:\n{result[:6000]}"})
    final = visible_text(text) or ("El agente alcanzó el límite de pasos sin dar una respuesta final. "
                                   "Sube «Pasos máximos» o simplifica la petición.")
    yield {"type": "done", "steps": agent.get("max_steps") or 5, "final": final, "limit_reached": True}

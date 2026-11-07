"""Agent loop: the model decides when to call tools, they're run and the result is handed back to it.

It works with models that have native tool calling (a chat template with `tools`, e.g. Qwen) and, if the template
doesn't support it, by describing the tools in the system prompt with the same <tool_call>{…}</tool_call> format.
"""
import json
import re
from typing import Iterator

from .evaluation import strip_think
from .inference import GenParams, manager
from .tools import execute, resolve

CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
# Gemma 4's native format: <|tool_call>call:name{…}<tool_call|>. Even when asked for the format
# above, in GGUF it falls back to its own about one time in three; without recognizing it, the call was
# returned to the user as if it were the answer.
GEMMA_CALL_RE = re.compile(r"<\|tool_call>\s*call:([\w.\-]+):?\s*(\{.*?\})\s*<tool_call\|>", re.S)
# Variant with attributes: <tool_call name="x" arguments='{…}'></tool_call> (or self-closing). Gemma uses it
# sometimes with many MCP tools; without recognizing it, the call came out as the final answer.
ATTR_CALL_RE = re.compile(r"""<tool_call\s+name=["']([\w.\-]+)["']\s+arguments=(?:'(\{.*?\})'|"(\{.*?\})")\s*/?>(?:\s*</tool_call>)?""", re.S)
# Gemma 4's thinking channel: <|channel>thought … <channel|>
GEMMA_CANAL_RE = re.compile(r"<\|channel>.*?<channel\|>", re.S)
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)

FALLBACK_SYSTEM = """You have these tools available:
{tools}

To use one, reply EXACTLY with this format and nothing else:
<tool_call>{{"name": "<name>", "arguments": {{…}}}}</tool_call>
You'll receive the result and can then use another tool or give the final answer.
When you have enough information, answer the user in natural language, without <tool_call>."""


def _generate(agent: dict, convo: list[dict], params, tools_schema):
    """One agent turn. It goes through the model in VRAM or through an external helper.

    With a helper there's no native tool calling (each provider exposes it differently), so the
    tools are described in the prompt and the calls are read from the text, which is the path that
    already existed for models whose template doesn't support them."""
    helper = agent.get("helper")
    if helper:
        from . import providers
        # The whole conversation goes along: after a tool call the last message is the tool's
        # result, and without the earlier turns the model would lose the user's question
        system = "\n\n".join(m["content"] for m in convo[:-1] if m["role"] == "system") or None
        history = [m for m in convo[:-1] if m["role"] in ("user", "assistant")]
        piece = providers.complete(helper, convo[-1]["content"], system=system, history=history,
                                   max_tokens=params.max_new_tokens, temperature=params.temperature)
        yield piece
        return
    extra = {"tools": tools_schema} if tools_schema else {}
    yield from manager.stream_chat(convo, params, template_kwargs=extra)


def template_supports_tools() -> bool:
    tok = getattr(manager.current, "tokenizer", None)
    tpl = getattr(tok, "chat_template", None) or ""
    return "tools" in tpl


def _lax_json(s: str):
    """JSON the way Gemma 4 writes it in its native format: unquoted keys ({names:[…]}) and strings
    between <|"|>. It's tried as normal JSON first; if that fails, it's normalized. None if there's no way."""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    t = s.replace('<|"|>', '"')
    # Quote the bare keys, without touching what's inside a string
    parts = re.split(r'("(?:[^"\\]|\\.)*")', t)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r"([{,]\s*)([A-Za-z_][\w\-]*)\s*:", r'\1"\2":', parts[i])
    try:
        return json.loads("".join(parts))
    except json.JSONDecodeError:
        return None


def parse_calls(text: str) -> list[dict]:
    """Extracts tool calls from the generated text (several common formats)."""
    calls = []
    for m in GEMMA_CALL_RE.finditer(text):
        args = _lax_json(m.group(2))
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
    """Loads the agent's preferred model if it isn't the one in VRAM.

    It lives here and not in the API because the agent runs from several places (its test panel and
    flow nodes): if loading only happened in one endpoint, the same agent would give different answers
    depending on where it was called from, with whatever model happened to be loaded."""
    from sqlmodel import Session

    from ..db import Model, engine
    from .jobs import is_gpu_busy

    mid = agent.get("model_id")
    if not mid or (manager.current and manager.current.model_id == mid):
        return None
    if is_gpu_busy():
        raise RuntimeError("The GPU is busy with a training run; wait for it to finish.")
    with Session(engine, expire_on_commit=False) as s:
        m = s.get(Model, mid)
    if not m:
        return None
    manager.load(model_id=m.id, name=m.name, kind=m.kind, path=m.path, base_model=m.base_model,
                 engine=manager.preferred_engine if m.kind != "gguf" else "auto")
    return m.name


def run(agent: dict, messages: list[dict], params: GenParams | None = None) -> Iterator[dict]:
    """Runs the agent. It emits events: token, tool_call, tool_result, step, done, error."""
    uses_helper = bool(agent.get("helper"))
    if not uses_helper:
        try:
            loaded = ensure_model(agent)
            if loaded:
                yield {"type": "model", "name": loaded}
        except Exception as e:  # noqa: BLE001
            yield {"type": "error", "message": str(e)}
            return
    if not uses_helper and not manager.current:
        yield {"type": "error", "message": "No model is loaded (Playground or Models), "
                                           "and this agent has no preferred model assigned."}
        return
    tools = resolve(agent.get("tools") or [])
    by_name = {t["name"]: t for t in tools}
    params = params or GenParams(max_new_tokens=agent.get("max_tokens") or 700,
                                 temperature=agent.get("temperature", 0.3), enable_thinking=False)
    native = (not uses_helper) and template_supports_tools() and tools
    system = (agent.get("system_prompt") or "").strip()
    if tools and not native:
        desc = "\n".join(f"- {t['schema']['name']}: {t['schema']['description']} · arguments: "
                         f"{json.dumps(t['schema']['parameters'].get('properties', {}), ensure_ascii=False)}" for t in tools)
        system = (system + "\n\n" if system else "") + FALLBACK_SYSTEM.format(tools=desc)
    convo: list[dict] = ([{"role": "system", "content": system}] if system else []) + list(messages)
    yield {"type": "tools", "tools": [{"name": t["name"], "label": t["label"]} for t in tools], "native": bool(native)}

    seen_calls: dict[str, int] = {}
    for step in range(int(agent.get("max_steps") or 5)):
        text = ""
        schema = [t["schema"] for t in tools] if native else None
        try:
            for piece in _generate(agent, convo, params, schema):
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
            if seen_calls[sig] > 2:  # the model got stuck repeating the same call
                yield {"type": "tool_result", "name": call["name"], "result": "(repeated call: stopping the loop)",
                       "seconds": 0.0, "step": step, "repeated": True}
                yield {"type": "done", "steps": step + 1, "final": visible_text(text),
                       "stopped_repeating": True}
                return
            if not tool:
                result, secs = f"Error: there's no tool called '{call['name']}'. Available: {', '.join(by_name)}", 0.0
            elif seen_calls[sig] == 2:
                result, secs = ("You already called this tool with the same arguments and the result was the same. "
                                "Change the arguments or answer with what you already know."), 0.0
            else:
                result, secs = execute(tool, call["arguments"])
            yield {"type": "tool_result", "name": call["name"], "result": result[:4000], "seconds": secs, "step": step}
            if native:
                convo.append({"role": "tool", "name": call["name"], "content": result[:6000]})
            else:
                convo.append({"role": "user", "content": f"Result of {call['name']}:\n{result[:6000]}"})
    final = visible_text(text) or ("The agent hit the step limit without giving a final answer. "
                                   "Raise “Max steps” or simplify the request.")
    yield {"type": "done", "steps": agent.get("max_steps") or 5, "final": final, "limit_reached": True}

"""Evaluate an agent with test cases: the expected tool and keywords in the answer.

What's measured, per case:
    - did it use the right tool? (or none, if the question didn't need one)
    - was the first tool it chose the right one? Getting there after two failed tries also
        counts as a hit, but it costs twice as much and with a small model it's a sign of hesitation.
    - does the answer contain what it should?
    - steps and seconds.
"""
import time

from sqlmodel import Session

from ..db import Agent, AgentEvalRun, engine
from .agents import run as run_agent


def _alias(agent: dict) -> dict[str, set[str]]:
    """Function name the model sees -> every name it can be called by.

    The model calls “calculate” and the catalog says “calculator”: comparing only one of the two,
    a correct call counted as a miss and the metric sank for no reason."""
    from .tools import resolve
    out: dict[str, set[str]] = {}
    for t in resolve(agent.get("tools") or []):
        out[t["name"].lower()] = {t["name"].lower(), str(t.get("tool_id", "")).lower(), str(t.get("label", "")).lower()}
    return out


def _matches(expected: str, calls: list[str], alias: dict[str, set[str]]) -> bool:
    e = expected.lower().strip()
    for n in calls:
        names = alias.get(n.lower(), {n.lower()})
        if e in names or any(e and e in x for x in names):
            return True
    return False


def evaluate_case(agent: dict, case: dict, alias: dict[str, set[str]] | None = None) -> dict:
    alias = alias if alias is not None else _alias(agent)
    t0 = time.time()
    calls: list[str] = []
    final, steps, error = "", 0, None
    for ev in run_agent(agent, [{"role": "user", "content": case["question"]}]):
        if ev["type"] == "tool_call":
            calls.append(ev["name"])
        elif ev["type"] == "done":
            final, steps = ev.get("final", ""), ev.get("steps", 0)
        elif ev["type"] == "error":
            error = ev["message"]
    expected = (case.get("tool") or "").strip()
    no_tool = expected.lower() in ("", "none", "ninguna", "-")
    if no_tool:
        hit, first_try = not calls, not calls
    else:
        hit = _matches(expected, calls, alias)
        first_try = bool(calls) and _matches(expected, calls[:1], alias)
    kws = [k for k in (case.get("keywords") or []) if k]
    lower_final = final.lower()
    return {
        "question": case["question"], "esperada": expected or "none", "llamadas": calls,
        "acierto": hit, "a_la_primera": first_try,
        "claves": round(sum(k.lower() in lower_final for k in kws) / len(kws), 2) if kws else None,
        "pasos": steps, "segundos": round(time.time() - t0, 1), "respuesta": final[:600], "error": error,
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows) or 1
    with_kws = [f["claves"] for f in rows if f["claves"] is not None]
    failures = {}
    for f in rows:
        if not f["acierto"]:
            key = f"expected {f['esperada']}, used {', '.join(f['llamadas']) or 'none'}"
            failures[key] = failures.get(key, 0) + 1
    return {
        "casos": len(rows),
        "herramienta_correcta": round(sum(f["acierto"] for f in rows) / n, 2),
        "a_la_primera": round(sum(f["a_la_primera"] for f in rows) / n, 2),
        "palabras_clave": round(sum(with_kws) / len(with_kws), 2) if with_kws else None,
        "pasos_media": round(sum(f["pasos"] for f in rows) / n, 1),
        "segundos_media": round(sum(f["segundos"] for f in rows) / n, 1),
        "errores": sum(1 for f in rows if f["error"]),
        # Grouped failures say more than the percentage: “expected rag_search, used web_search” ×3
        "fallos_frecuentes": sorted(failures.items(), key=lambda kv: -kv[1])[:5],
    }


def run_eval(run_id: int, task) -> dict:
    with Session(engine, expire_on_commit=False) as s:
        r = s.get(AgentEvalRun, run_id)
        a = s.get(Agent, r.agent_id)
    if not a:
        raise RuntimeError("The agent no longer exists")
    agent = a.model_dump()
    alias = _alias(agent)      # resolved once: with MCP servers, listing is expensive
    rows = []
    for i, case in enumerate(r.cases):
        task.update(i / max(1, len(r.cases)), f"Case {i + 1}/{len(r.cases)}: {case['question'][:50]}")
        if task.cancel_requested:
            break
        rows.append(evaluate_case(agent, case, alias))
        with Session(engine) as s:          # saved case by case: if it's cut off, the work done isn't lost
            x = s.get(AgentEvalRun, run_id)
            x.results, x.summary = rows, summarize(rows)
            s.add(x)
            s.commit()
    summary = summarize(rows)
    with Session(engine) as s:
        x = s.get(AgentEvalRun, run_id)
        x.results, x.summary, x.status = rows, summary, "completed"
        s.add(x)
        s.commit()
    return summary

"""Evaluar un agente con casos de prueba: herramienta esperada y palabras clave de la respuesta.

Lo que se mide, por caso:
  - ¿usó la herramienta que tocaba? (o ninguna, si la pregunta no la necesitaba)
  - ¿la primera herramienta que eligió fue la buena? Llegar tras dos intentos fallidos también
    cuenta como acierto, pero cuesta el doble y con un modelo pequeño es señal de que duda.
  - ¿la respuesta contiene lo que tenía que contener?
  - pasos y segundos.
"""
import time

from sqlmodel import Session

from ..db import Agent, AgentEvalRun, engine
from .agents import run as run_agent


def _alias(agent: dict) -> dict[str, set[str]]:
    """Nombre de función que ve el modelo -> todos los nombres por los que se le puede llamar.

    El modelo llama «calcular» y el catálogo dice «calculator»: comparando sólo uno de los dos,
    una llamada correcta se contaba como fallo y la métrica salía hundida sin motivo."""
    from .tools import resolve
    out: dict[str, set[str]] = {}
    for t in resolve(agent.get("tools") or []):
        out[t["name"].lower()] = {t["name"].lower(), str(t.get("tool_id", "")).lower(), str(t.get("label", "")).lower()}
    return out


def _coincide(esperada: str, llamadas: list[str], alias: dict[str, set[str]]) -> bool:
    e = esperada.lower().strip()
    for n in llamadas:
        nombres = alias.get(n.lower(), {n.lower()})
        if e in nombres or any(e and e in x for x in nombres):
            return True
    return False


def evaluar_caso(agent: dict, caso: dict, alias: dict[str, set[str]] | None = None) -> dict:
    alias = alias if alias is not None else _alias(agent)
    t0 = time.time()
    llamadas: list[str] = []
    final, pasos, error = "", 0, None
    for ev in run_agent(agent, [{"role": "user", "content": caso["question"]}]):
        if ev["type"] == "tool_call":
            llamadas.append(ev["name"])
        elif ev["type"] == "done":
            final, pasos = ev.get("final", ""), ev.get("steps", 0)
        elif ev["type"] == "error":
            error = ev["message"]
    esperada = (caso.get("tool") or "").strip()
    sin_herramienta = esperada.lower() in ("", "ninguna", "none", "-")
    if sin_herramienta:
        acierto, primera = not llamadas, not llamadas
    else:
        acierto = _coincide(esperada, llamadas, alias)
        primera = bool(llamadas) and _coincide(esperada, llamadas[:1], alias)
    claves = [k for k in (caso.get("keywords") or []) if k]
    bajo = final.lower()
    return {
        "question": caso["question"], "esperada": esperada or "ninguna", "llamadas": llamadas,
        "acierto": acierto, "a_la_primera": primera,
        "claves": round(sum(k.lower() in bajo for k in claves) / len(claves), 2) if claves else None,
        "pasos": pasos, "segundos": round(time.time() - t0, 1), "respuesta": final[:600], "error": error,
    }


def resumir(filas: list[dict]) -> dict:
    n = len(filas) or 1
    con_claves = [f["claves"] for f in filas if f["claves"] is not None]
    fallos = {}
    for f in filas:
        if not f["acierto"]:
            clave = f"esperaba {f['esperada']}, usó {', '.join(f['llamadas']) or 'ninguna'}"
            fallos[clave] = fallos.get(clave, 0) + 1
    return {
        "casos": len(filas),
        "herramienta_correcta": round(sum(f["acierto"] for f in filas) / n, 2),
        "a_la_primera": round(sum(f["a_la_primera"] for f in filas) / n, 2),
        "palabras_clave": round(sum(con_claves) / len(con_claves), 2) if con_claves else None,
        "pasos_media": round(sum(f["pasos"] for f in filas) / n, 1),
        "segundos_media": round(sum(f["segundos"] for f in filas) / n, 1),
        "errores": sum(1 for f in filas if f["error"]),
        # Los fallos agrupados dicen más que el porcentaje: «esperaba rag_search, usó web_search» ×3
        "fallos_frecuentes": sorted(fallos.items(), key=lambda kv: -kv[1])[:5],
    }


def ejecutar(run_id: int, task) -> dict:
    with Session(engine, expire_on_commit=False) as s:
        r = s.get(AgentEvalRun, run_id)
        a = s.get(Agent, r.agent_id)
    if not a:
        raise RuntimeError("El agente ya no existe")
    agente = a.model_dump()
    alias = _alias(agente)      # se resuelve una vez: con servidores MCP, listar cuesta
    filas = []
    for i, caso in enumerate(r.cases):
        task.update(i / max(1, len(r.cases)), f"Caso {i + 1}/{len(r.cases)}: {caso['question'][:50]}")
        if task.cancel_requested:
            break
        filas.append(evaluar_caso(agente, caso, alias))
        with Session(engine) as s:          # se guarda caso a caso: si se corta, no se pierde lo hecho
            x = s.get(AgentEvalRun, run_id)
            x.results, x.summary = filas, resumir(filas)
            s.add(x)
            s.commit()
    resumen = resumir(filas)
    with Session(engine) as s:
        x = s.get(AgentEvalRun, run_id)
        x.results, x.summary, x.status = filas, resumen, "completed"
        s.add(x)
        s.commit()
    return resumen

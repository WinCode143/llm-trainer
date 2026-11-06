"""Flujos: grafos visuales que se compilan a LangGraph y se ejecutan con nuestros modelos y herramientas.

Un flujo es {nodes, edges}. Cada nodo lee y escribe un **estado** compartido (`vars`), así que un nodo puede
usar lo que produjo otro escribiendo `{nombre}` en sus plantillas.

Tipos de nodo:
  input     campos que rellena el usuario al lanzar el flujo
  llm       llama al modelo cargado con un prompt (opcionalmente exige JSON)
  tool      ejecuta cualquier herramienta del catálogo de agentes
  rag       busca en una colección y deja el contexto y las fuentes
  agent     delega en un agente ya creado (con sus herramientas)
  router    bifurca: por decisión del modelo o por una condición sobre el estado
  approval  se detiene y espera tu aprobación (se puede reanudar días después)
  output    compone la respuesta final

Se usa LangGraph sólo para la orquestación (aristas condicionales, ciclos, puntos de control y pausas);
la generación pasa por nuestro gestor de inferencia, así funciona igual con transformers, llama.cpp o vLLM.
"""
import ast
import json
import operator
import re
import sqlite3
import string
import threading
import uuid
from typing import Annotated, Any, Iterator, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy, Send, interrupt

from ..config import settings

NODE_TYPES = {
    "input": {"label": "Entrada", "desc": "Campos que rellenas al lanzar el flujo", "icon": "log-in"},
    "llm": {"label": "Modelo", "desc": "Genera texto con el modelo cargado; puede exigir JSON", "icon": "brain"},
    "tool": {"label": "Herramienta", "desc": "Ejecuta una herramienta del catálogo (web, SQL, Python…)", "icon": "wrench"},
    "rag": {"label": "Buscar en documentos", "desc": "Recupera contexto de una colección RAG", "icon": "library"},
    "agent": {"label": "Agente", "desc": "Delega en un agente con sus herramientas", "icon": "bot"},
    "router": {"label": "Bifurcación", "desc": "Elige camino: por decisión del modelo o por condición", "icon": "git-branch"},
    "approval": {"label": "Aprobación", "desc": "Se detiene y espera tu visto bueno", "icon": "user-check"},
    "foreach": {"label": "Para cada", "desc": "Repite unos pasos por cada elemento de una lista (en paralelo o uno a uno)", "icon": "repeat"},
    "subflow": {"label": "Subflujo", "desc": "Ejecuta otro flujo guardado como si fuera un nodo", "icon": "box"},
    "code": {"label": "Código", "desc": "Python aislado para transformar variables (sin red por defecto)", "icon": "code"},
    "output": {"label": "Salida", "desc": "Respuesta final del flujo", "icon": "log-out"},
}


# ---------------------------------------------------------------- estado y plantillas
def _merge(a: dict, b: dict) -> dict:
    return {**(a or {}), **(b or {})}


class State(TypedDict, total=False):
    vars: Annotated[dict, _merge]
    trace: Annotated[list, operator.add]


def _legible(v: Any) -> str:
    """Cómo se ve una variable dentro de un texto.

    Una lista puesta tal cual sale como ['a', 'b'], con comillas y corchetes, y eso acaba dentro del
    prompt que lee el modelo. Se formatea como lista numerada, que es lo que se querría leer."""
    if isinstance(v, list):
        if all(not isinstance(x, (list, dict)) for x in v):
            return "\n".join(f"{i + 1}. {x}" for i, x in enumerate(v))
        return json.dumps(v, ensure_ascii=False, indent=2)
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False, indent=2)
    return str(v)


class _Fmt(string.Formatter):
    """Como str.format, pero una variable que aún no existe queda vacía y los valores se ven legibles.
    El valor crudo se entrega antes de formatear para que {x[campo]} y {x.atributo} lleguen dentro de un
    JSON; sólo el resultado final pasa por _legible."""
    def get_value(self, key, args, kwargs):
        return kwargs.get(key, "") if isinstance(key, str) else super().get_value(key, args, kwargs)

    def format_field(self, value, spec):
        return format(_legible(value), spec)


_FMT = _Fmt()


def render(template: str, vars: dict) -> str:
    """Sustituye {variables} del estado. Las llaves sueltas no rompen nada."""
    try:
        return _FMT.vformat(str(template or ""), (), dict(vars or {}))
    except (ValueError, IndexError, KeyError, TypeError, AttributeError):  # {x[campo]} sobre algo que no es dict
        return str(template or "")


# ---------------------------------------------------------------- condiciones seguras
_CMP = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
        ast.Gt: operator.gt, ast.GtE: operator.ge, ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b}


def eval_condition(expr: str, vars: dict) -> bool:
    """Evalúa una condición sencilla sobre el estado: comparaciones, and/or/not, in, len(). Sin ejecutar código."""
    def ev(n):
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Name):
            return vars.get(n.id, "")
        if isinstance(n, ast.BoolOp):
            vals = [ev(v) for v in n.values]
            return all(vals) if isinstance(n.op, ast.And) else any(vals)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            return not ev(n.operand)
        if isinstance(n, ast.Compare):
            left = ev(n.left)
            for op, comp in zip(n.ops, n.comparators):
                fn = _CMP.get(type(op))
                if not fn or not fn(left, ev(comp)):
                    return False
                left = ev(comp)
            return True
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("len", "int", "float", "str", "bool"):
            return {"len": len, "int": int, "float": float, "str": str, "bool": bool}[n.func.id](ev(n.args[0]))
        raise ValueError("expresión no permitida")
    try:
        return bool(ev(ast.parse(str(expr), mode="eval").body))
    except Exception as e:  # noqa: BLE001
        # Una condición con un error de escritura devolvía False y el flujo se iba por otra rama sin
        # decir nada: parecía que la lógica estaba mal cuando lo que estaba mal era la expresión.
        print(f"[flows] la condición «{expr}» no se pudo evaluar ({e}); se toma como falsa")
        return False


# ---------------------------------------------------------------- nodos
def _llm(prompt: str, *, system: str | None = None, max_tokens: int = 600, temperature: float = 0.3,
         json_mode: bool = False, helper: int | None = None, node: str | None = None) -> str:
    """Generación de un nodo. Cada nodo puede usar su propio modelo ayudante (uno barato para
    clasificar, uno bueno para redactar) o el modelo local si no se indica ninguno.

    Si hay lector de eventos, los tokens salen en vivo y se ven escribirse en el nodo."""
    from . import providers

    escritor = None
    if node:
        try:
            from langgraph.config import get_stream_writer
            w = get_stream_writer()
            escritor = lambda t: w({"node": node, "text": t})  # noqa: E731
        except Exception:  # noqa: BLE001  — fuera de un grafo no hay lector
            escritor = None
    return providers.complete(helper, prompt, system=system, max_tokens=max_tokens,
                              temperature=temperature, json_mode=json_mode, on_token=escritor)


def _parse_json(text: str) -> Any:
    m = re.search(r"[\[{].*[\]}]", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def make_node(node: dict, flow: dict):
    """Devuelve la función que ejecuta un nodo y actualiza el estado."""
    cfg = node.get("config") or {}
    nid, ntype, label = node["id"], node["type"], node.get("label") or node["type"]

    def step(state: State) -> dict:
        vars_ = dict(state.get("vars") or {})
        out_key = cfg.get("output_key") or f"{nid}_salida"
        info: dict[str, Any] = {}
        value: Any = ""

        if ntype == "input":
            value = vars_.get("_entrada", "")

        elif ntype == "llm":
            prompt = render(cfg.get("prompt", "{entrada}"), vars_)
            value = _llm(prompt, system=render(cfg.get("system", ""), vars_) or None,
                         max_tokens=int(cfg.get("max_tokens") or 600), temperature=float(cfg.get("temperature", 0.3)),
                         json_mode=bool(cfg.get("json")), helper=cfg.get("helper"), node=nid)
            if cfg.get("json"):
                parsed = _parse_json(value)
                if isinstance(parsed, dict):
                    vars_.update({str(k): v for k, v in parsed.items()})
                    info["campos"] = list(parsed)
                value = parsed if parsed is not None else value
            info["prompt"] = prompt[:400]

        elif ntype == "tool":
            from . import tools as tools_svc
            # «spec_t» y no «spec»: reasignar spec lo volvía local y rompía los nodos que usan el del flujo
            spec_t = tools_svc.TOOLS.get(cfg.get("tool_id"))
            if not spec_t:
                raise RuntimeError(f"Herramienta desconocida: {cfg.get('tool_id')}")
            args = {k: render(v, vars_) if isinstance(v, str) else v for k, v in (cfg.get("args") or {}).items()}
            value, secs = tools_svc.execute(
                {"run": spec_t["run"], "config": tools_svc.with_defaults(cfg["tool_id"], cfg.get("config"))}, args)
            info = {"herramienta": cfg["tool_id"], "argumentos": args, "segundos": secs}
            # Los servidores MCP suelen devolver JSON: se deja también ya leído en «<salida>_json» para
            # poder usar un campo en el siguiente nodo, p. ej. {transcripcion_json[text]}
            if cfg["tool_id"] == "mcp":
                parsed = _parse_json(value)
                if isinstance(parsed, (dict, list)):
                    vars_[f"{out_key}_json"] = parsed

        elif ntype == "rag":
            from sqlmodel import Session

            from ..db import RagCollection, engine
            from . import rag as rag_svc
            with Session(engine, expire_on_commit=False) as s:
                coll = s.get(RagCollection, int(cfg.get("collection_id") or 0))
            if not coll:
                raise RuntimeError("El nodo de búsqueda no tiene colección configurada")
            query = render(cfg.get("query", "{entrada}"), vars_)
            hits = rag_svc.retrieve(coll, query, k=int(cfg.get("k") or coll.top_k or 4))
            value = "\n\n".join(f"[{i + 1}] ({h.get('source')}) {h['text']}" for i, h in enumerate(hits))
            vars_[f"{out_key}_fuentes"] = [h.get("source") for h in hits]
            info = {"consulta": query, "fragmentos": len(hits)}

        elif ntype == "agent":
            from sqlmodel import Session

            from ..db import Agent, engine
            from .agents import run as run_agent
            with Session(engine) as s:
                agent = s.get(Agent, int(cfg.get("agent_id") or 0))
            if not agent:
                raise RuntimeError("El nodo de agente no tiene agente configurado")
            prompt = render(cfg.get("prompt", "{entrada}"), vars_)
            pasos = []
            for ev in run_agent(agent.model_dump(), [{"role": "user", "content": prompt}]):
                if ev["type"] == "tool_call":
                    pasos.append(ev["name"])
                elif ev["type"] == "done":
                    value = ev["final"]
                elif ev["type"] == "error":
                    raise RuntimeError(ev["message"])
            info = {"agente": agent.name, "herramientas usadas": pasos}

        elif ntype == "router":
            if cfg.get("mode") == "llm":
                opciones = [b.get("label", "") for b in (cfg.get("branches") or [])]
                prompt = (render(cfg.get("prompt", "{entrada}"), vars_) +
                          f"\n\nElige UNA opción de esta lista y responde sólo con ella: {', '.join(opciones)}")
                elegido = _llm(prompt, max_tokens=30, temperature=0, helper=cfg.get("helper"))
                value = next((o for o in opciones if o.lower() in elegido.lower()), opciones[0] if opciones else "")
                info = {"respuesta del modelo": elegido[:120]}
            else:
                value = ""
                for b in cfg.get("branches") or []:
                    if eval_condition(b.get("when", ""), vars_):
                        value = b.get("label", "")
                        break
                if not value:
                    value = (cfg.get("default") or ((cfg.get("branches") or [{}])[-1].get("label", "")))
                info = {"condiciones": [b.get("when") for b in (cfg.get("branches") or [])]}
            vars_[f"{nid}_rama"] = value

        elif ntype == "approval":
            mensaje = render(cfg.get("message", "¿Continuamos?"), vars_)
            previo = render(cfg.get("preview", ""), vars_)
            respuesta = interrupt({"node": nid, "label": label, "message": mensaje, "preview": previo,
                                   "editable": bool(cfg.get("editable"))})
            if isinstance(respuesta, dict):
                if respuesta.get("rejected"):
                    raise RuntimeError(f"Rechazado en «{label}»: {respuesta.get('comment') or 'sin comentario'}")
                if respuesta.get("edited"):
                    vars_[cfg.get("edit_key") or out_key] = respuesta["edited"]
                value = respuesta.get("comment") or "aprobado"
            else:
                value = str(respuesta)
            info = {"mensaje": mensaje[:200]}

        elif ntype == "foreach":
            value, extra = _foreach(cfg, vars_, flow)
            vars_.update(extra)
            info = {"elementos": len(value) if isinstance(value, list) else 0,
                    "en paralelo": int(cfg.get("parallel") or 1)}

        elif ntype == "subflow":
            from sqlmodel import Session

            from ..db import Flow, engine
            with Session(engine, expire_on_commit=False) as s:
                sub = s.get(Flow, int(cfg.get("flow_id") or 0))
            if not sub:
                raise RuntimeError("El nodo de subflujo no tiene flujo configurado")
            if sub.id in (state.get("vars") or {}).get("_pila", []):
                raise RuntimeError(f"El subflujo «{sub.name}» se llama a sí mismo: eso no puede ejecutarse")
            entrada_sub = render(cfg.get("input", "{entrada}"), vars_)
            value = _run_subflow(sub, entrada_sub, (vars_.get("_pila") or []) + [sub.id])
            info = {"subflujo": sub.name}

        elif ntype == "code":
            from . import sandbox
            # Las variables entran como un dict `vars` y el resultado sale por `resultado` o por lo impreso
            preludio = ("import json, sys\n"
                        f"vars = json.loads({json.dumps(json.dumps({k: v for k, v in vars_.items() if not k.startswith('_')}, ensure_ascii=False, default=str))})\n"
                        "resultado = None\n")
            final = "\nprint('<<<RESULTADO>>>'); print(json.dumps(resultado, ensure_ascii=False, default=str))\n"
            r = sandbox.run_python(preludio + (cfg.get("code") or "") + final,
                                   timeout_s=int(cfg.get("timeout_s") or 20),
                                   memory_mb=int(cfg.get("memory_mb") or 1024),
                                   allow_network=bool(cfg.get("allow_network")))
            if r.get("error") and "<<<RESULTADO>>>" not in (r.get("output") or ""):
                raise RuntimeError(f"El código falló: {r['error'][:400]}")
            salida, _, cola = (r.get("output") or "").partition("<<<RESULTADO>>>")
            try:
                devuelto = json.loads(cola.strip()) if cola.strip() else None
            except json.JSONDecodeError:
                devuelto = None
            value = devuelto if devuelto is not None else salida.strip()
            if isinstance(devuelto, dict):   # un dict se reparte como variables sueltas
                vars_.update({str(k): v for k, v in devuelto.items()})
                info["campos"] = list(devuelto)
            info |= {"aislado": r.get("sandboxed"), "impreso": salida.strip()[:300]}

        elif ntype == "output":
            value = render(cfg.get("template", "{entrada}"), vars_)
            vars_["_salida"] = value

        vars_[out_key] = value
        # «node_type» y no «type»: el evento de streaming ya usa «type» para su propia clase
        return {"vars": vars_, "trace": [{"node": nid, "node_type": ntype, "label": label,
                                          "output": value if isinstance(value, str) else json.dumps(value, ensure_ascii=False),
                                          "info": info}]}

    step.__name__ = f"node_{nid}"
    return step



# ------------------------------------------------------------------ «Para cada»
def _as_list(valor: Any) -> list:
    """Convierte lo que haya en la variable en una lista: ya lo es, o son líneas de texto."""
    if isinstance(valor, list):
        return valor
    if isinstance(valor, dict):
        return list(valor.values())
    return [l.strip() for l in str(valor or "").splitlines() if l.strip()]


def _foreach(cfg: dict, vars_: dict, flow: dict) -> tuple[list, dict]:
    """Repite una acción por cada elemento de una lista y junta los resultados.

    Se resuelve con hilos, no con el reparto en paralelo de LangGraph, a propósito: el modelo local
    vive detrás de un candado (hay uno solo en memoria), así que repartirlo por el grafo se
    serializaría igual. Con hilos, además, el paralelismo sí es real cuando el nodo usa un modelo
    ayudante externo, que es cuando de verdad compensa.
    """
    import concurrent.futures as cf

    elementos = _as_list(vars_.get(cfg.get("items_key") or "entrada"))
    limite = int(cfg.get("max_items") or 0)
    if limite:
        elementos = elementos[:limite]
    modo = cfg.get("mode") or "llm"
    politica = cfg.get("on_error") or "stop"   # stop | skip | null
    hilos = max(1, min(int(cfg.get("parallel") or 1), 10))

    def uno(par):
        i, item = par
        contexto = {**vars_, "item": item, "indice": i}
        if modo == "llm":
            return _llm(render(cfg.get("prompt", "{item}"), contexto),
                        max_tokens=int(cfg.get("max_tokens") or 400),
                        temperature=float(cfg.get("temperature", 0.3)),
                        helper=cfg.get("helper"))
        if modo == "tool":
            from . import tools as tools_svc
            spec_t = tools_svc.TOOLS.get(cfg.get("tool_id"))
            if not spec_t:
                raise RuntimeError(f"Herramienta desconocida: {cfg.get('tool_id')}")
            args = {k: render(v, contexto) if isinstance(v, str) else v for k, v in (cfg.get("args") or {}).items()}
            valor, _ = tools_svc.execute(
                {"run": spec_t["run"], "config": tools_svc.with_defaults(cfg["tool_id"], cfg.get("config"))}, args)
            return valor
        if modo == "subflow":
            from sqlmodel import Session

            from ..db import Flow, engine
            with Session(engine, expire_on_commit=False) as s:
                sub = s.get(Flow, int(cfg.get("flow_id") or 0))
            if not sub:
                raise RuntimeError("El «Para cada» en modo subflujo no tiene flujo configurado")
            return _run_subflow(sub, render(cfg.get("input", "{item}"), contexto), [sub.id])
        raise RuntimeError(f"Modo de «Para cada» desconocido: {modo}")

    resultados: list = [None] * len(elementos)
    fallos: list[str] = []
    with cf.ThreadPoolExecutor(max_workers=hilos) as ex:
        futuros = {ex.submit(uno, (i, it)): i for i, it in enumerate(elementos)}
        for fut in cf.as_completed(futuros):
            i = futuros[fut]
            try:
                resultados[i] = fut.result()
            except Exception as e:  # noqa: BLE001
                fallos.append(f"[{i}] {e}")
                if politica == "stop":
                    for f2 in futuros:
                        f2.cancel()
                    raise RuntimeError(f"El elemento {i} falló: {e}. Cambia la política de error si prefieres continuar.")
                resultados[i] = None

    if politica == "skip":
        resultados = [r for r in resultados if r is not None]
    clave = cfg.get("output_key") or "resultados"
    return resultados, {f"{clave}_texto": "\n\n".join(str(r) for r in resultados if r is not None),
                        f"{clave}_fallos": fallos, f"{clave}_n": len(resultados)}


def _run_subflow(sub, entrada: str, pila: list[int]) -> str:
    """Ejecuta otro flujo entero y devuelve su salida. La pila evita que un flujo se llame a sí mismo."""
    graph = compile_flow(sub.spec)
    config = {"configurable": {"thread_id": f"sub-{uuid.uuid4().hex[:10]}"},
              "recursion_limit": int(sub.spec.get("max_steps") or 50)}
    estado = graph.invoke({"vars": {"entrada": entrada, "_entrada": entrada, "_pila": pila}, "trace": []}, config)
    v = estado.get("vars") or {}
    if v.get("_salida"):
        return v["_salida"]
    traza = estado.get("trace") or []
    return str(traza[-1].get("output", "")) if traza else ""


# ---------------------------------------------------------------- compilación
_saver = None
_saver_lock = threading.Lock()


def _get_saver() -> SqliteSaver:
    """Puntos de control en SQLite: conexión propia y compartida entre hilos (la app sirve peticiones
    desde un pool, y las pausas de aprobación se reanudan más tarde, en otro hilo)."""
    global _saver
    with _saver_lock:
        if _saver is None:
            conn = sqlite3.connect(str(settings.data_dir / "flows.sqlite"), check_same_thread=False)
            _saver = SqliteSaver(conn)
        return _saver


def compile_flow(spec: dict):
    nodes = {n["id"]: n for n in spec.get("nodes", [])}
    edges = spec.get("edges", [])
    if not nodes:
        raise ValueError("El flujo no tiene nodos")
    g = StateGraph(State)
    for nid, node in nodes.items():
        intentos = int((node.get("config") or {}).get("retries") or 0)
        # Útil en nodos que salen a la red (herramientas web, modelos ayudantes): un fallo puntual no tumba el flujo
        politica = RetryPolicy(max_attempts=intentos + 1, initial_interval=1.0, backoff_factor=2.0) if intentos else None
        g.add_node(nid, make_node(node, spec), retry_policy=politica)

    starts = [n for n in nodes.values() if n["type"] == "input"] or [nodes[list(nodes)[0]]]
    g.add_edge(START, starts[0]["id"])

    salientes: dict[str, list[dict]] = {}
    for e in edges:
        salientes.setdefault(e["from"], []).append(e)
    for nid, node in nodes.items():
        outs = salientes.get(nid, [])
        if not outs:
            g.add_edge(nid, END)
        elif node["type"] == "router":
            destinos = {e.get("when") or "": e["to"] for e in outs}

            def elegir(state: State, _nid=nid, _dest=dict(destinos)):
                rama = (state.get("vars") or {}).get(f"{_nid}_rama", "")
                return _dest.get(rama) or next(iter(_dest.values()))

            g.add_conditional_edges(nid, elegir, list(destinos.values()))
        else:
            for e in outs:  # varias salidas = ramas en paralelo
                g.add_edge(nid, e["to"])
    return g.compile(checkpointer=_get_saver())


def validate(spec: dict) -> list[str]:
    """Avisos útiles antes de ejecutar (no bloquean)."""
    nodes = spec.get("nodes", [])
    edges = spec.get("edges", [])
    ids = {n["id"] for n in nodes}
    avisos = []
    if not nodes:
        return ["El flujo está vacío"]
    if not any(n["type"] == "input" for n in nodes):
        avisos.append("No hay nodo de entrada: se empezará por el primer nodo")
    if not any(n["type"] == "output" for n in nodes):
        avisos.append("No hay nodo de salida: la respuesta final será la del último nodo")
    for n in nodes:
        c = n.get("config") or {}
        etiqueta = n.get("label") or n["id"]
        if n["type"] == "rag" and not c.get("collection_id"):
            avisos.append(f"«{etiqueta}»: falta elegir la colección")
        if n["type"] == "agent" and not c.get("agent_id"):
            avisos.append(f"«{etiqueta}»: falta elegir el agente")
        if n["type"] == "tool" and not c.get("tool_id"):
            avisos.append(f"«{etiqueta}»: falta elegir la herramienta")
        mc = c.get("config") or {}
        if n["type"] == "tool" and c.get("tool_id") == "mcp" and not (mc.get("tool") and (mc.get("server_id") or mc.get("url"))):
            avisos.append(f"«{etiqueta}»: falta elegir el servidor MCP y cuál de sus herramientas llamar")
        if n["type"] == "subflow" and not c.get("flow_id"):
            avisos.append(f"«{etiqueta}»: falta elegir el flujo que se ejecuta dentro")
        if n["type"] == "code" and not (c.get("code") or "").strip():
            avisos.append(f"«{etiqueta}»: no tiene código")
        if n["type"] == "foreach":
            if not c.get("items_key"):
                avisos.append(f"«{etiqueta}»: falta decir qué variable contiene la lista")
            if (c.get("mode") or "llm") == "subflow" and not c.get("flow_id"):
                avisos.append(f"«{etiqueta}»: en modo subflujo hay que elegir el flujo")
        if n["type"] == "router" and not (c.get("branches") or []):
            avisos.append(f"«{etiqueta}»: una bifurcación sin ramas no decide nada")
    for e in edges:
        if e["from"] not in ids or e["to"] not in ids:
            avisos.append(f"Conexión suelta: {e['from']} → {e['to']}")
    destinos = {e["to"] for e in edges}
    for n in nodes:
        if n["type"] != "input" and n["id"] not in destinos:
            avisos.append(f"«{n.get('label') or n['id']}» no está conectado a nada que lo active")
        if n["type"] == "router" and not [e for e in edges if e["from"] == n["id"]]:
            avisos.append(f"La bifurcación «{n.get('label') or n['id']}» no tiene salidas")
    return avisos


# ---------------------------------------------------------------- ejecución
def run(spec: dict, inputs: dict, thread_id: str | None = None, resume: Any = None) -> Iterator[dict]:
    """Ejecuta (o reanuda) un flujo emitiendo eventos: node, interrupt, done, error."""
    graph = compile_flow(spec)
    thread_id = thread_id or uuid.uuid4().hex[:12]
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": int(spec.get("max_steps") or 50)}
    entrada = {"vars": {**inputs, "entrada": inputs.get("entrada") or next(iter(inputs.values()), "")}, "trace": []}
    payload = Command(resume=resume) if resume is not None else entrada
    yield {"type": "start", "thread_id": thread_id}
    try:
        # «custom» deja que un nodo emita tokens mientras genera; «updates», lo que devuelve al terminar
        for modo, chunk in graph.stream(payload, config, stream_mode=["updates", "custom"]):
            if modo == "custom":
                yield {"type": "token", "thread_id": thread_id, **chunk}
                continue
            for nid, update in chunk.items():
                if nid == "__interrupt__":
                    data = update[0].value if isinstance(update, tuple) else update
                    yield {"type": "interrupt", "thread_id": thread_id, "data": data}
                    return
                for t in (update or {}).get("trace", []):
                    yield {"type": "node", "thread_id": thread_id, **t}
        state = graph.get_state(config).values
        yield {"type": "done", "thread_id": thread_id, "vars": state.get("vars", {}),
               "output": (state.get("vars") or {}).get("_salida") or
                         ((state.get("trace") or [{}])[-1].get("output") if state.get("trace") else "")}
    except Exception as e:  # noqa: BLE001
        yield {"type": "error", "thread_id": thread_id, "message": str(e)}


# ----------------------------------------------------------- viaje en el tiempo
def history(spec: dict, thread_id: str) -> list[dict]:
    """Los puntos de guardado de una ejecución, del más reciente al más antiguo.

    Cada uno es un «antes de ejecutar X»: sirve para rebobinar ahí, cambiar algo y seguir
    sin repetir lo que ya costó (una búsqueda RAG, una llamada a un modelo grande)."""
    graph = compile_flow(spec)
    config = {"configurable": {"thread_id": thread_id}}
    etiquetas = {n["id"]: n.get("label") or n["id"] for n in spec.get("nodes", [])}
    out = []
    for snap in graph.get_state_history(config):
        siguiente = list(snap.next)
        vars_ = snap.values.get("vars") or {}
        out.append({
            "checkpoint_id": snap.config["configurable"].get("checkpoint_id"),
            # La configuración entera: reconstruirla a mano se deja fuera «checkpoint_ns» y falla al reanudar
            "config": snap.config["configurable"],
            "next": siguiente,
            "next_labels": [etiquetas.get(n, n) for n in siguiente],
            "done": [t.get("node") for t in (snap.values.get("trace") or [])],
            # Las variables internas (_entrada, _salida) no se enseñan: confunden más que ayudan
            "vars": {k: v for k, v in vars_.items() if not k.startswith("_")},
            "created_at": str(snap.created_at) if snap.created_at else None,
        })
    return out


def _config_de(graph, thread_id: str, checkpoint_id: str) -> dict:
    """Busca en el historial la configuración completa de ese punto (lleva «checkpoint_ns»)."""
    for snap in graph.get_state_history({"configurable": {"thread_id": thread_id}}):
        if snap.config["configurable"].get("checkpoint_id") == checkpoint_id:
            return snap.config
    raise RuntimeError("Ese punto de guardado ya no existe en esta ejecución")


def state_at(spec: dict, thread_id: str, checkpoint_id: str) -> dict:
    graph = compile_flow(spec)
    snap = graph.get_state(_config_de(graph, thread_id, checkpoint_id))
    return {"vars": snap.values.get("vars") or {}, "next": list(snap.next),
            "trace": snap.values.get("trace") or []}


def replay(spec: dict, thread_id: str, checkpoint_id: str, edits: dict | None = None) -> Iterator[dict]:
    """Reanuda desde un punto de guardado. Con `edits` se cambian variables antes de seguir.

    Editar bifurca: se crea una rama nueva y la ejecución original se conserva intacta."""
    graph = compile_flow(spec)
    base = _config_de(graph, thread_id, checkpoint_id)
    config = base
    if edits:
        actual = graph.get_state(base).values.get("vars") or {}
        config = graph.update_state(base, {"vars": {**actual, **edits}})
    config = {**config, "recursion_limit": int(spec.get("max_steps") or 50)}
    yield {"type": "start", "thread_id": thread_id, "replay_from": checkpoint_id, "edited": list(edits or {})}
    try:
        for modo, chunk in graph.stream(None, config, stream_mode=["updates", "custom"]):
            if modo == "custom":
                yield {"type": "token", "thread_id": thread_id, **chunk}
                continue
            for nid, update in chunk.items():
                if nid == "__interrupt__":
                    data = update[0].value if isinstance(update, tuple) else update
                    yield {"type": "interrupt", "thread_id": thread_id, "data": data}
                    return
                for t in (update or {}).get("trace", []):
                    yield {"type": "node", "thread_id": thread_id, **t}
        vals = graph.get_state({"configurable": {"thread_id": thread_id}}).values
        yield {"type": "done", "thread_id": thread_id, "vars": vals.get("vars", {}),
               "output": (vals.get("vars") or {}).get("_salida", "")}
    except Exception as e:  # noqa: BLE001
        yield {"type": "error", "thread_id": thread_id, "message": str(e)}


TEMPLATES = {
    "rag_verificado": {
        "name": "RAG con verificación",
        "description": "Busca en tus documentos, responde y comprueba que la respuesta se apoya en las fuentes.",
        "spec": {"nodes": [
            {"id": "in", "type": "input", "label": "Pregunta", "x": 40, "y": 160,
             "config": {"fields": [{"key": "entrada", "label": "Pregunta"}]}},
            {"id": "buscar", "type": "rag", "label": "Buscar contexto", "x": 260, "y": 160,
             "config": {"query": "{entrada}", "k": 4, "output_key": "contexto"}},
            {"id": "responder", "type": "llm", "label": "Responder", "x": 480, "y": 160,
             "config": {"prompt": "Contexto:\n{contexto}\n\nPregunta: {entrada}\n\nResponde usando sólo el contexto y cita [n].",
                        "output_key": "respuesta"}},
            {"id": "verificar", "type": "llm", "label": "Verificar", "x": 700, "y": 160,
             "config": {"json": True, "output_key": "verificacion",
                        "prompt": "Contexto:\n{contexto}\n\nRespuesta: {respuesta}\n\n¿Está todo respaldado por el contexto? Devuelve {{\"apoyada\": true/false, \"motivo\": \"...\"}}"}},
            {"id": "router", "type": "router", "label": "¿Apoyada?", "x": 920, "y": 160,
             "config": {"mode": "expr", "branches": [{"label": "sí", "when": "apoyada == True"}, {"label": "no", "when": "True"}]}},
            {"id": "ok", "type": "output", "label": "Respuesta", "x": 1140, "y": 80, "config": {"template": "{respuesta}"}},
            {"id": "aviso", "type": "output", "label": "Sin respaldo", "x": 1140, "y": 240,
             "config": {"template": "No puedo asegurarlo con tus documentos ({motivo}).\n\nBorrador: {respuesta}"}},
        ], "edges": [
            {"from": "in", "to": "buscar"}, {"from": "buscar", "to": "responder"},
            {"from": "responder", "to": "verificar"}, {"from": "verificar", "to": "router"},
            {"from": "router", "to": "ok", "when": "sí"}, {"from": "router", "to": "aviso", "when": "no"}]},
    },
    "investigar_redactar": {
        "name": "Investigar y redactar (con aprobación)",
        "description": "Busca en la web, redacta un informe, te lo enseña para aprobarlo y lo guarda.",
        "spec": {"nodes": [
            {"id": "in", "type": "input", "label": "Tema", "x": 40, "y": 160,
             "config": {"fields": [{"key": "entrada", "label": "Tema a investigar"}]}},
            {"id": "buscar", "type": "tool", "label": "Buscar en la web", "x": 260, "y": 160,
             "config": {"tool_id": "web_search", "config": {"max_results": 5}, "args": {"query": "{entrada}"},
                        "output_key": "resultados"}},
            {"id": "redactar", "type": "llm", "label": "Redactar informe", "x": 480, "y": 160,
             "config": {"max_tokens": 900, "output_key": "informe",
                        "prompt": "Fuentes:\n{resultados}\n\nEscribe un informe breve en español sobre «{entrada}» con 3 secciones y cita las fuentes."}},
            {"id": "aprobar", "type": "approval", "label": "Revisión", "x": 700, "y": 160,
             "config": {"message": "¿Guardo este informe?", "preview": "{informe}", "editable": True, "edit_key": "informe"}},
            {"id": "guardar", "type": "tool", "label": "Guardar archivo", "x": 920, "y": 160,
             "config": {"tool_id": "write_file", "config": {"overwrite": True}, "output_key": "guardado",
                        "args": {"path": "informe.md", "content": "{informe}"}}},
            {"id": "fin", "type": "output", "label": "Resultado", "x": 1140, "y": 160,
             "config": {"template": "{guardado}\n\n{informe}"}},
        ], "edges": [{"from": "in", "to": "buscar"}, {"from": "buscar", "to": "redactar"},
                     {"from": "redactar", "to": "aprobar"}, {"from": "aprobar", "to": "guardar"},
                     {"from": "guardar", "to": "fin"}]},
    },
    "papers_del_dia": {
        "name": "Papers del día (Hugging Face)",
        "description": "Trae los papers destacados de Hugging Face y los resume en español. Prográmalo en "
                       "Conexiones → Disparadores para recibirlo cada mañana por Telegram. Usa el servidor MCP "
                       "«Hugging Face Hub» (Conexiones → catálogo).",
        "spec": {"nodes": [
            {"id": "in", "type": "input", "label": "Interés", "x": 40, "y": 160,
             "config": {"fields": [{"key": "entrada", "label": "¿Qué te interesa? (vacío = todo)"}]}},
            {"id": "traer", "type": "tool", "label": "Papers destacados", "x": 260, "y": 160,
             "config": {"tool_id": "mcp", "mcp_catalogo": "huggingface", "output_key": "papers",
                        "config": {"tool": "hf_fs"},
                        "args": {"operations": '[{"cmd": "ls", "args": ["hf://papers/trending"]}]'}}},
            {"id": "resumir", "type": "llm", "label": "Resumir", "x": 500, "y": 160,
             "config": {"max_tokens": 900, "output_key": "resumen",
                        "prompt": "Estos son los papers destacados hoy en Hugging Face (JSON):\n{papers}\n\n"
                                  "Interés del lector: «{entrada}». Elige los 5 más relevantes para ese interés "
                                  "(si está vacío, los más votados) y para cada uno escribe en español: título, "
                                  "una frase de qué aporta y el enlace https://huggingface.co/papers/<id>."}},
            {"id": "fin", "type": "output", "label": "Resumen", "x": 740, "y": 160, "config": {"template": "{resumen}"}},
        ], "edges": [{"from": "in", "to": "traer"}, {"from": "traer", "to": "resumir"}, {"from": "resumir", "to": "fin"}]},
    },
    "audio_a_coleccion": {
        "name": "Audio a colección (Whisper)",
        "description": "Transcribe una grabación de data/entrada con Whisper local y la indexa en una colección "
                       "para poder preguntarle. Usa los servidores MCP «Transcribir audio» (Conexiones → catálogo) "
                       "y el propio de la app.",
        "spec": {"nodes": [
            {"id": "in", "type": "input", "label": "Audio", "x": 40, "y": 160,
             "config": {"fields": [{"key": "entrada", "label": "Archivo de audio en data/entrada (p. ej. clase.mp3)"},
                                   {"key": "coleccion", "label": "Colección donde guardarlo"}]}},
            {"id": "transcribir", "type": "tool", "label": "Transcribir", "x": 280, "y": 160,
             "config": {"tool_id": "mcp", "mcp_catalogo": "whisper", "output_key": "transcripcion",
                        "config": {"tool": "transcribe_file", "max_chars": 200000},
                        "args": {"file_path": str(settings.data_dir / "entrada") + "/{entrada}"}}},
            {"id": "guardar", "type": "tool", "label": "Indexar", "x": 520, "y": 160,
             "config": {"tool_id": "mcp", "mcp_catalogo": "llm-trainer", "output_key": "indexado",
                        "config": {"tool": "add_document"},
                        "args": {"collection": "{coleccion}", "filename": "{entrada}.md",
                                 "text": "{transcripcion_json[text]}"}}},
            {"id": "fin", "type": "output", "label": "Resultado", "x": 760, "y": 160,
             "config": {"template": "{indexado}\n\n{transcripcion_json[text]}"}},
        ], "edges": [{"from": "in", "to": "transcribir"}, {"from": "transcribir", "to": "guardar"},
                     {"from": "guardar", "to": "fin"}]},
    },
    "clasificar_derivar": {
        "name": "Clasificar y derivar",
        "description": "Clasifica la consulta y la envía al camino adecuado (documentos, cálculo o respuesta directa).",
        "spec": {"nodes": [
            {"id": "in", "type": "input", "label": "Consulta", "x": 40, "y": 200,
             "config": {"fields": [{"key": "entrada", "label": "Consulta"}]}},
            {"id": "clasificar", "type": "router", "label": "Tipo de consulta", "x": 260, "y": 200,
             "config": {"mode": "llm", "prompt": "Clasifica esta consulta: {entrada}",
                        "branches": [{"label": "documentos"}, {"label": "cálculo"}, {"label": "general"}]}},
            {"id": "docs", "type": "rag", "label": "Buscar en documentos", "x": 520, "y": 60,
             "config": {"query": "{entrada}", "output_key": "contexto"}},
            {"id": "calc", "type": "tool", "label": "Calcular", "x": 520, "y": 200,
             "config": {"tool_id": "python", "config": {"timeout_s": 20}, "output_key": "contexto",
                        "args": {"code": "# resuelve: {entrada}\nprint('escribe aquí el cálculo')"}}},
            {"id": "directo", "type": "llm", "label": "Responder directo", "x": 520, "y": 340,
             "config": {"prompt": "{entrada}", "output_key": "contexto"}},
            {"id": "redactar", "type": "llm", "label": "Redactar respuesta", "x": 800, "y": 200,
             "config": {"prompt": "Pregunta: {entrada}\n\nInformación disponible:\n{contexto}\n\nResponde en español, breve.",
                        "output_key": "respuesta"}},
            {"id": "fin", "type": "output", "label": "Respuesta", "x": 1060, "y": 200,
             "config": {"template": "{respuesta}"}},
        ], "edges": [{"from": "in", "to": "clasificar"},
                     {"from": "clasificar", "to": "docs", "when": "documentos"},
                     {"from": "clasificar", "to": "calc", "when": "cálculo"},
                     {"from": "clasificar", "to": "directo", "when": "general"},
                     {"from": "docs", "to": "redactar"}, {"from": "calc", "to": "redactar"},
                     {"from": "directo", "to": "redactar"}, {"from": "redactar", "to": "fin"}]},
    },
}


# ----------------------------------------------------------- exportar a código
def _py(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False)


class _AVars(ast.NodeTransformer):
    """Reescribe `apoyada == True` como `v.get("apoyada") == True`.

    En el editor las condiciones se escriben con los nombres de las variables sueltos, pero en el
    código exportado esos nombres no existen: viven en el diccionario del estado."""

    SEGUROS = {"len", "int", "float", "str", "bool", "True", "False", "None"}

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id not in self.SEGUROS:
            return ast.Call(func=ast.Attribute(value=ast.Name(id="v", ctx=ast.Load()), attr="get", ctx=ast.Load()),
                            args=[ast.Constant(value=node.id)], keywords=[])
        return node


def _cond_py(expr: str) -> str:
    """Condición del editor traducida a una expresión válida en el código exportado."""
    try:
        arbol = ast.parse(expr.strip(), mode="eval")
        return ast.unparse(ast.fix_missing_locations(_AVars().visit(arbol)).body)
    except SyntaxError:
        return "False  # no se pudo traducir: " + expr.replace("\n", " ")


def to_python(spec: dict, name: str = "mi_flujo") -> str:
    """Genera el equivalente en código LangGraph de un flujo dibujado.

    No es para ejecutarlo dentro de la app —para eso está el compilador— sino para que el trabajo
    invertido en un flujo no dependa de que esta aplicación siga existiendo: el archivo resultante
    se ejecuta con langgraph y las funciones que tú rellenes.
    """
    nodes = {n["id"]: n for n in spec.get("nodes", [])}
    edges = spec.get("edges", [])
    L = [
        '"""Flujo exportado desde LLM Trainer.',
        "",
        f"Flujo: {name}",
        "",
        "Cada nodo es una función que recibe el estado y devuelve las variables que añade.",
        "Los nodos que llamaban al modelo o al RAG quedan marcados con TODO: enchúfalos a tu",
        "propio cliente (OpenAI, Anthropic, Ollama…), que es la única parte atada a la app.",
        '"""',
        "from typing import Annotated, TypedDict",
        "",
        "from langgraph.graph import END, START, StateGraph",
        "",
        "",
        "class State(TypedDict, total=False):",
        "    vars: Annotated[dict, lambda a, b: {**(a or {}), **(b or {})}]",
        "",
        "",
        "def render(plantilla: str, v: dict) -> str:",
        '    """Sustituye {variables}; las que falten quedan vacías."""',
        "    import string",
        "",
        "    class Safe(dict):",
        "        def __missing__(self, k):",
        '            return ""',
        "",
        "    return string.Formatter().vformat(plantilla, (), Safe(v))",
        "",
        "",
    ]
    for nid, n in nodes.items():
        cfg = n.get("config") or {}
        clave = cfg.get("output_key") or f"{nid}_salida"
        L += [f"def nodo_{nid}(state: State) -> dict:", f'    """{n.get("label") or nid} ({n["type"]})."""',
              "    v = dict(state.get('vars') or {})"]
        t = n["type"]
        if t == "input":
            L.append("    valor = v.get('entrada', '')")
        elif t == "llm":
            L += [f"    prompt = render({_py(cfg.get('prompt', '{entrada}'))}, v)",
                  "    # TODO: llama aquí a tu modelo",
                  "    valor = tu_modelo(prompt)"]
        elif t == "rag":
            L += [f"    consulta = render({_py(cfg.get('query', '{entrada}'))}, v)",
                  f"    # TODO: recupera de tu índice (era la colección {cfg.get('collection_id')}, k={cfg.get('k', 4)})",
                  "    valor = tu_busqueda(consulta)"]
        elif t == "tool":
            L += [f"    # TODO: herramienta {cfg.get('tool_id')!r} con argumentos {cfg.get('args')!r}",
                  "    valor = ''"]
        elif t == "agent":
            L += [f"    # TODO: agente {cfg.get('agent_id')} con sus herramientas", "    valor = ''"]
        elif t == "code":
            L += ["    # Código que escribiste en el nodo:"] + \
                 [f"    {l}" for l in (cfg.get("code") or "").splitlines()] + \
                 ["    valor = locals().get('resultado')"]
        elif t == "foreach":
            L += [f"    elementos = v.get({_py(cfg.get('items_key'))}) or []",
                  "    valor = []",
                  "    for indice, item in enumerate(elementos):",
                  f"        prompt = render({_py(cfg.get('prompt', '{item}'))}, {{**v, 'item': item, 'indice': indice}})",
                  "        valor.append(tu_modelo(prompt))  # TODO"]
        elif t == "subflow":
            L += [f"    # TODO: aquí se ejecutaba el flujo {cfg.get('flow_id')}; expórtalo también y llámalo",
                  "    valor = ''"]
        elif t == "router":
            ramas = cfg.get("branches") or []
            L.append("    valor = ''")
            for b in ramas:
                if cfg.get("mode") != "llm" and b.get("when"):
                    L += [f"    if not valor and ({_cond_py(b['when'])}):", f"        valor = {_py(b.get('label',''))}"]
            L += [f"    valor = valor or {_py(ramas[-1].get('label', '') if ramas else '')}",
                  f"    v[{_py(nid + '_rama')}] = valor"]
        elif t == "approval":
            L += ["    # Aquí el flujo se detenía a esperar tu visto bueno (langgraph.types.interrupt)",
                  "    valor = 'aprobado'"]
        elif t == "output":
            L += [f"    valor = render({_py(cfg.get('template', '{entrada}'))}, v)", "    v['_salida'] = valor"]
        L += [f"    v[{_py(clave)}] = valor", "    return {'vars': v}", "", ""]

    L += ["def construir():", "    g = StateGraph(State)"]
    for nid in nodes:
        L.append(f"    g.add_node({_py(nid)}, nodo_{nid})")
    entradas = [n for n in nodes.values() if n["type"] == "input"] or list(nodes.values())
    L.append(f"    g.add_edge(START, {_py(entradas[0]['id'])})")
    salientes: dict[str, list] = {}
    for e in edges:
        salientes.setdefault(e["from"], []).append(e)
    for nid, n in nodes.items():
        outs = salientes.get(nid, [])
        if not outs:
            L.append(f"    g.add_edge({_py(nid)}, END)")
        elif n["type"] == "router":
            destinos = {e.get("when") or "": e["to"] for e in outs}
            L += [f"    def _elegir_{nid}(state):",
                  f"        return {_py(destinos)}.get((state.get('vars') or {{}}).get({_py(nid + '_rama')}, ''), {_py(list(destinos.values())[0])})",
                  f"    g.add_conditional_edges({_py(nid)}, _elegir_{nid}, {_py(list(destinos.values()))})"]
        else:
            for e in outs:
                L.append(f"    g.add_edge({_py(nid)}, {_py(e['to'])})")
    L += ["    return g.compile()", "", "",
          'if __name__ == "__main__":',
          "    app = construir()",
          "    salida = app.invoke({'vars': {'entrada': 'tu pregunta aquí'}})",
          "    print(salida['vars'].get('_salida'))",
          ""]
    return "\n".join(L)

"""Flows: visual graphs compiled to LangGraph and run with our own models and tools.

A flow is {nodes, edges}. Every node reads and writes a shared **state** (`vars`), so a node can
use what another one produced by writing `{name}` in its templates.

Node types:
    input     fields the user fills in when starting the flow
    llm       calls the loaded model with a prompt (optionally requiring JSON)
    tool      runs any tool from the agents' catalog
    rag       searches a collection and leaves the context and the sources
    agent     hands off to an existing agent (with its tools)
    router    branches: by the model's decision or by a condition on the state
    approval  stops and waits for your approval (it can be resumed days later)
    output    builds the final answer

LangGraph is used only for orchestration (conditional edges, cycles, checkpoints and pauses);
generation goes through our inference manager, so it works the same with transformers, llama.cpp or vLLM.
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
    "input": {"label": "Input", "desc": "Fields you fill in when you start the flow", "icon": "log-in"},
    "llm": {"label": "Model", "desc": "Generates text with the loaded model; can require JSON", "icon": "brain"},
    "tool": {"label": "Tool", "desc": "Runs a tool from the catalog (web, SQL, Python…)", "icon": "wrench"},
    "rag": {"label": "Search documents", "desc": "Retrieves context from a RAG collection", "icon": "library"},
    "agent": {"label": "Agent", "desc": "Hands off to an agent with its tools", "icon": "bot"},
    "router": {"label": "Branch", "desc": "Picks a path: by the model's decision or by a condition", "icon": "git-branch"},
    "approval": {"label": "Approval", "desc": "Stops and waits for your go-ahead", "icon": "user-check"},
    "foreach": {"label": "For each", "desc": "Repeats some steps for every item in a list (in parallel or one by one)", "icon": "repeat"},
    "subflow": {"label": "Sub-flow", "desc": "Runs another saved flow as if it were a node", "icon": "box"},
    "code": {"label": "Code", "desc": "Sandboxed Python to transform variables (no network by default)", "icon": "code"},
    "output": {"label": "Output", "desc": "The flow's final answer", "icon": "log-out"},
}


# ---------------------------------------------------------------- state and templates
def _merge(a: dict, b: dict) -> dict:
    return {**(a or {}), **(b or {})}


class State(TypedDict, total=False):
    vars: Annotated[dict, _merge]
    trace: Annotated[list, operator.add]


def _readable(v: Any) -> str:
    """How a variable looks inside a text.

    A list inserted as is comes out as ['a', 'b'], with quotes and brackets, and that ends up in the
    prompt the model reads. It's formatted as a numbered list, which is what you'd want to read."""
    if isinstance(v, list):
        if all(not isinstance(x, (list, dict)) for x in v):
            return "\n".join(f"{i + 1}. {x}" for i, x in enumerate(v))
        return json.dumps(v, ensure_ascii=False, indent=2)
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False, indent=2)
    return str(v)


class _Fmt(string.Formatter):
    """Like str.format, but a variable that doesn't exist yet is left empty and values are shown readably.
    The raw value is passed in before formatting so that {x[field]} and {x.attribute} reach inside a
    JSON; only the final result goes through _readable."""
    def get_value(self, key, args, kwargs):
        return kwargs.get(key, "") if isinstance(key, str) else super().get_value(key, args, kwargs)

    def format_field(self, value, spec):
        return format(_readable(value), spec)


_FMT = _Fmt()


def render(template: str, vars: dict) -> str:
    """Replaces state {variables}. Stray braces don't break anything."""
    try:
        return _FMT.vformat(str(template or ""), (), dict(vars or {}))
    except (ValueError, IndexError, KeyError, TypeError, AttributeError):  # {x[field]} on something that isn't a dict
        return str(template or "")


# ---------------------------------------------------------------- safe conditions
_CMP = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
        ast.Gt: operator.gt, ast.GtE: operator.ge, ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b}


def eval_condition(expr: str, vars: dict) -> bool:
    """Evaluates a simple condition on the state: comparisons, and/or/not, in, len(). Without running code."""
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
        raise ValueError("expression not allowed")
    try:
        return bool(ev(ast.parse(str(expr), mode="eval").body))
    except Exception as e:  # noqa: BLE001
        # A condition with a typo used to return False and the flow silently took another branch:
        # it looked like the logic was wrong when what was wrong was the expression.
        print(f"[flows] the condition “{expr}” couldn't be evaluated ({e}); treating it as false")
        return False


# ---------------------------------------------------------------- nodes
def _llm(prompt: str, *, system: str | None = None, max_tokens: int = 600, temperature: float = 0.3,
         json_mode: bool = False, helper: int | None = None, node: str | None = None) -> str:
    """Generation for one node. Each node can use its own helper model (a cheap one to
    classify, a good one to write) or the local model if none is given.

    If there's an event reader, tokens stream out live and you see the node writing."""
    from . import providers

    writer = None
    if node:
        try:
            from langgraph.config import get_stream_writer
            w = get_stream_writer()
            writer = lambda t: w({"node": node, "text": t})  # noqa: E731
        except Exception:  # noqa: BLE001  — outside a graph there's no reader
            writer = None
    return providers.complete(helper, prompt, system=system, max_tokens=max_tokens,
                              temperature=temperature, json_mode=json_mode, on_token=writer)


def _parse_json(text: str) -> Any:
    m = re.search(r"[\[{].*[\]}]", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def make_node(node: dict, flow: dict):
    """Returns the function that runs a node and updates the state."""
    cfg = node.get("config") or {}
    nid, ntype, label = node["id"], node["type"], node.get("label") or node["type"]

    def step(state: State) -> dict:
        vars_ = dict(state.get("vars") or {})
        out_key = cfg.get("output_key") or f"{nid}_output"
        info: dict[str, Any] = {}
        value: Any = ""

        if ntype == "input":
            value = vars_.get("_input", "")

        elif ntype == "llm":
            prompt = render(cfg.get("prompt", "{input}"), vars_)
            value = _llm(prompt, system=render(cfg.get("system", ""), vars_) or None,
                         max_tokens=int(cfg.get("max_tokens") or 600), temperature=float(cfg.get("temperature", 0.3)),
                         json_mode=bool(cfg.get("json")), helper=cfg.get("helper"), node=nid)
            if cfg.get("json"):
                parsed = _parse_json(value)
                if isinstance(parsed, dict):
                    vars_.update({str(k): v for k, v in parsed.items()})
                    info["fields"] = list(parsed)
                value = parsed if parsed is not None else value
            info["prompt"] = prompt[:400]

        elif ntype == "tool":
            from . import tools as tools_svc
            # “spec_t” and not “spec”: reassigning spec made it local and broke the nodes that use the flow's one
            spec_t = tools_svc.TOOLS.get(cfg.get("tool_id"))
            if not spec_t:
                raise RuntimeError(f"Unknown tool: {cfg.get('tool_id')}")
            args = {k: render(v, vars_) if isinstance(v, str) else v for k, v in (cfg.get("args") or {}).items()}
            value, secs = tools_svc.execute(
                {"run": spec_t["run"], "config": tools_svc.with_defaults(cfg["tool_id"], cfg.get("config"))}, args)
            info = {"tool": cfg["tool_id"], "arguments": args, "seconds": secs}
            # MCP servers usually return JSON: it's also left parsed in “<output>_json” so a field can
            # be used in the next node, e.g. {transcript_json[text]}
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
                raise RuntimeError("The search node has no collection configured")
            query = render(cfg.get("query", "{input}"), vars_)
            hits = rag_svc.retrieve(coll, query, k=int(cfg.get("k") or coll.top_k or 4))
            value = "\n\n".join(f"[{i + 1}] ({h.get('source')}) {h['text']}" for i, h in enumerate(hits))
            vars_[f"{out_key}_sources"] = [h.get("source") for h in hits]
            info = {"query": query, "chunks": len(hits)}

        elif ntype == "agent":
            from sqlmodel import Session

            from ..db import Agent, engine
            from .agents import run as run_agent
            with Session(engine) as s:
                agent = s.get(Agent, int(cfg.get("agent_id") or 0))
            if not agent:
                raise RuntimeError("The agent node has no agent configured")
            prompt = render(cfg.get("prompt", "{input}"), vars_)
            steps = []
            for ev in run_agent(agent.model_dump(), [{"role": "user", "content": prompt}]):
                if ev["type"] == "tool_call":
                    steps.append(ev["name"])
                elif ev["type"] == "done":
                    value = ev["final"]
                elif ev["type"] == "error":
                    raise RuntimeError(ev["message"])
            info = {"agent": agent.name, "tools used": steps}

        elif ntype == "router":
            if cfg.get("mode") == "llm":
                options = [b.get("label", "") for b in (cfg.get("branches") or [])]
                prompt = (render(cfg.get("prompt", "{input}"), vars_) +
                          f"\n\nPick ONE option from this list and answer with it only: {', '.join(options)}")
                chosen = _llm(prompt, max_tokens=30, temperature=0, helper=cfg.get("helper"))
                value = next((o for o in options if o.lower() in chosen.lower()), options[0] if options else "")
                info = {"model answer": chosen[:120]}
            else:
                value = ""
                for b in cfg.get("branches") or []:
                    if eval_condition(b.get("when", ""), vars_):
                        value = b.get("label", "")
                        break
                if not value:
                    value = (cfg.get("default") or ((cfg.get("branches") or [{}])[-1].get("label", "")))
                info = {"conditions": [b.get("when") for b in (cfg.get("branches") or [])]}
            vars_[f"{nid}_rama"] = value

        elif ntype == "approval":
            message = render(cfg.get("message", "Shall we continue?"), vars_)
            preview = render(cfg.get("preview", ""), vars_)
            answer = interrupt({"node": nid, "label": label, "message": message, "preview": preview,
                                   "editable": bool(cfg.get("editable"))})
            if isinstance(answer, dict):
                if answer.get("rejected"):
                    raise RuntimeError(f"Rejected at “{label}”: {answer.get('comment') or 'no comment'}")
                if answer.get("edited"):
                    vars_[cfg.get("edit_key") or out_key] = answer["edited"]
                value = answer.get("comment") or "approved"
            else:
                value = str(answer)
            info = {"message": message[:200]}

        elif ntype == "foreach":
            value, extra = _foreach(cfg, vars_, flow)
            vars_.update(extra)
            info = {"items": len(value) if isinstance(value, list) else 0,
                    "in parallel": int(cfg.get("parallel") or 1)}

        elif ntype == "subflow":
            from sqlmodel import Session

            from ..db import Flow, engine
            with Session(engine, expire_on_commit=False) as s:
                sub = s.get(Flow, int(cfg.get("flow_id") or 0))
            if not sub:
                raise RuntimeError("The sub-flow node has no flow configured")
            if sub.id in (state.get("vars") or {}).get("_stack", []):
                raise RuntimeError(f"The sub-flow “{sub.name}” calls itself: that can't run")
            sub_input = render(cfg.get("input", "{input}"), vars_)
            value = _run_subflow(sub, sub_input, (vars_.get("_stack") or []) + [sub.id])
            info = {"subflujo": sub.name}

        elif ntype == "code":
            from . import sandbox
            # Variables go in as a `vars` dict and the result comes out through `result` or whatever was printed
            prelude = ("import json, sys\n"
                        f"vars = json.loads({json.dumps(json.dumps({k: v for k, v in vars_.items() if not k.startswith('_')}, ensure_ascii=False, default=str))})\n"
                        "result = None\n")
            final = "\nprint('<<<RESULT>>>'); print(json.dumps(result, ensure_ascii=False, default=str))\n"
            r = sandbox.run_python(prelude + (cfg.get("code") or "") + final,
                                   timeout_s=int(cfg.get("timeout_s") or 20),
                                   memory_mb=int(cfg.get("memory_mb") or 1024),
                                   allow_network=bool(cfg.get("allow_network")))
            if r.get("error") and "<<<RESULT>>>" not in (r.get("output") or ""):
                raise RuntimeError(f"The code failed: {r['error'][:400]}")
            printed, _, tail = (r.get("output") or "").partition("<<<RESULT>>>")
            try:
                returned = json.loads(tail.strip()) if tail.strip() else None
            except json.JSONDecodeError:
                returned = None
            value = returned if returned is not None else printed.strip()
            if isinstance(returned, dict):   # a dict is spread out as separate variables
                vars_.update({str(k): v for k, v in returned.items()})
                info["fields"] = list(returned)
            info |= {"sandboxed": r.get("sandboxed"), "printed": printed.strip()[:300]}

        elif ntype == "output":
            value = render(cfg.get("template", "{input}"), vars_)
            vars_["_output"] = value

        vars_[out_key] = value
        # “node_type” and not “type”: the streaming event already uses “type” for its own kind
        return {"vars": vars_, "trace": [{"node": nid, "node_type": ntype, "label": label,
                                          "output": value if isinstance(value, str) else json.dumps(value, ensure_ascii=False),
                                          "info": info}]}

    step.__name__ = f"node_{nid}"
    return step



# ------------------------------------------------------------------ “For each”
def _as_list(value: Any) -> list:
    """Turns whatever is in the variable into a list: it already is one, or it's lines of text."""
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return list(value.values())
    return [l.strip() for l in str(value or "").splitlines() if l.strip()]


def _foreach(cfg: dict, vars_: dict, flow: dict) -> tuple[list, dict]:
    """Repeats an action for every item in a list and gathers the results.

    It uses threads, not LangGraph's parallel fan-out, on purpose: the local model
    lives behind a lock (there's only one in memory), so fanning out through the graph would
    serialize anyway. With threads, parallelism is real when the node uses an external helper
    model, which is when it really pays off.
    """
    import concurrent.futures as cf

    items = _as_list(vars_.get(cfg.get("items_key") or "input"))
    limit = int(cfg.get("max_items") or 0)
    if limit:
        items = items[:limit]
    mode = cfg.get("mode") or "llm"
    policy = cfg.get("on_error") or "stop"   # stop | skip | null
    threads = max(1, min(int(cfg.get("parallel") or 1), 10))

    def one(par):
        i, item = par
        ctx = {**vars_, "item": item, "index": i}
        if mode == "llm":
            return _llm(render(cfg.get("prompt", "{item}"), ctx),
                        max_tokens=int(cfg.get("max_tokens") or 400),
                        temperature=float(cfg.get("temperature", 0.3)),
                        helper=cfg.get("helper"))
        if mode == "tool":
            from . import tools as tools_svc
            spec_t = tools_svc.TOOLS.get(cfg.get("tool_id"))
            if not spec_t:
                raise RuntimeError(f"Unknown tool: {cfg.get('tool_id')}")
            args = {k: render(v, ctx) if isinstance(v, str) else v for k, v in (cfg.get("args") or {}).items()}
            value, _ = tools_svc.execute(
                {"run": spec_t["run"], "config": tools_svc.with_defaults(cfg["tool_id"], cfg.get("config"))}, args)
            return value
        if mode == "subflow":
            from sqlmodel import Session

            from ..db import Flow, engine
            with Session(engine, expire_on_commit=False) as s:
                sub = s.get(Flow, int(cfg.get("flow_id") or 0))
            if not sub:
                raise RuntimeError("The “For each” node in sub-flow mode has no flow configured")
            return _run_subflow(sub, render(cfg.get("input", "{item}"), ctx), [sub.id])
        raise RuntimeError(f"Unknown “For each” mode: {mode}")

    results: list = [None] * len(items)
    failures: list[str] = []
    with cf.ThreadPoolExecutor(max_workers=threads) as ex:
        futures = {ex.submit(one, (i, it)): i for i, it in enumerate(items)}
        for fut in cf.as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:  # noqa: BLE001
                failures.append(f"[{i}] {e}")
                if policy == "stop":
                    for f2 in futures:
                        f2.cancel()
                    raise RuntimeError(f"Item {i} failed: {e}. Change the error policy if you'd rather continue.")
                results[i] = None

    if policy == "skip":
        results = [r for r in results if r is not None]
    key = cfg.get("output_key") or "results"
    return results, {f"{key}_text": "\n\n".join(str(r) for r in results if r is not None),
                        f"{key}_failures": failures, f"{key}_n": len(results)}


def _run_subflow(sub, sub_input: str, stack: list[int]) -> str:
    """Runs another whole flow and returns its output. The stack keeps a flow from calling itself."""
    graph = compile_flow(sub.spec)
    config = {"configurable": {"thread_id": f"sub-{uuid.uuid4().hex[:10]}"},
              "recursion_limit": int(sub.spec.get("max_steps") or 50)}
    final_state = graph.invoke({"vars": {"input": sub_input, "_input": sub_input, "_stack": stack}, "trace": []}, config)
    v = final_state.get("vars") or {}
    if v.get("_output"):
        return v["_output"]
    trace = final_state.get("trace") or []
    return str(trace[-1].get("output", "")) if trace else ""


# ---------------------------------------------------------------- compilation
_saver = None
_saver_lock = threading.Lock()


def _get_saver() -> SqliteSaver:
    """SQLite checkpoints: a connection of their own, shared between threads (the app serves requests
    from a pool, and approval pauses are resumed later, on another thread)."""
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
        raise ValueError("The flow has no nodes")
    g = StateGraph(State)
    for nid, node in nodes.items():
        attempts = int((node.get("config") or {}).get("retries") or 0)
        # Useful for nodes that go out to the network (web tools, helper models): a one-off failure doesn't kill the flow
        policy = RetryPolicy(max_attempts=attempts + 1, initial_interval=1.0, backoff_factor=2.0) if attempts else None
        g.add_node(nid, make_node(node, spec), retry_policy=policy)

    starts = [n for n in nodes.values() if n["type"] == "input"] or [nodes[list(nodes)[0]]]
    g.add_edge(START, starts[0]["id"])

    outgoing: dict[str, list[dict]] = {}
    for e in edges:
        outgoing.setdefault(e["from"], []).append(e)
    for nid, node in nodes.items():
        outs = outgoing.get(nid, [])
        if not outs:
            g.add_edge(nid, END)
        elif node["type"] == "router":
            destinations = {e.get("when") or "": e["to"] for e in outs}

            def choose(state: State, _nid=nid, _dest=dict(destinations)):
                rama = (state.get("vars") or {}).get(f"{_nid}_rama", "")
                return _dest.get(rama) or next(iter(_dest.values()))

            g.add_conditional_edges(nid, choose, list(destinations.values()))
        else:
            for e in outs:  # several outgoing edges = parallel branches
                g.add_edge(nid, e["to"])
    return g.compile(checkpointer=_get_saver())


def validate(spec: dict) -> list[str]:
    """Useful warnings before running (they don't block)."""
    nodes = spec.get("nodes", [])
    edges = spec.get("edges", [])
    ids = {n["id"] for n in nodes}
    warnings = []
    if not nodes:
        return ["The flow is empty"]
    if not any(n["type"] == "input" for n in nodes):
        warnings.append("There's no input node: it will start from the first node")
    if not any(n["type"] == "output" for n in nodes):
        warnings.append("There's no output node: the final answer will be the last node's")
    for n in nodes:
        c = n.get("config") or {}
        label = n.get("label") or n["id"]
        if n["type"] == "rag" and not c.get("collection_id"):
            warnings.append(f"“{label}”: choose the collection")
        if n["type"] == "agent" and not c.get("agent_id"):
            warnings.append(f"“{label}”: choose the agent")
        if n["type"] == "tool" and not c.get("tool_id"):
            warnings.append(f"“{label}”: choose the tool")
        mc = c.get("config") or {}
        if n["type"] == "tool" and c.get("tool_id") == "mcp" and not (mc.get("tool") and (mc.get("server_id") or mc.get("url"))):
            warnings.append(f"“{label}”: choose the MCP server and which of its tools to call")
        if n["type"] == "subflow" and not c.get("flow_id"):
            warnings.append(f"“{label}”: choose the flow to run inside")
        if n["type"] == "code" and not (c.get("code") or "").strip():
            warnings.append(f"“{label}”: it has no code")
        if n["type"] == "foreach":
            if not c.get("items_key"):
                warnings.append(f"“{label}”: say which variable holds the list")
            if (c.get("mode") or "llm") == "subflow" and not c.get("flow_id"):
                warnings.append(f"“{label}”: in sub-flow mode you have to choose the flow")
        if n["type"] == "router" and not (c.get("branches") or []):
            warnings.append(f"“{label}”: a branch with no branches decides nothing")
    for e in edges:
        if e["from"] not in ids or e["to"] not in ids:
            warnings.append(f"Dangling connection: {e['from']} → {e['to']}")
    destinations = {e["to"] for e in edges}
    for n in nodes:
        if n["type"] != "input" and n["id"] not in destinations:
            warnings.append(f"“{n.get('label') or n['id']}” isn't connected to anything that triggers it")
        if n["type"] == "router" and not [e for e in edges if e["from"] == n["id"]]:
            warnings.append(f"The branch “{n.get('label') or n['id']}” has no outgoing connections")
    return warnings


# ---------------------------------------------------------------- running
def run(spec: dict, inputs: dict, thread_id: str | None = None, resume: Any = None) -> Iterator[dict]:
    """Runs (or resumes) a flow, emitting events: node, interrupt, done, error."""
    graph = compile_flow(spec)
    thread_id = thread_id or uuid.uuid4().hex[:12]
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": int(spec.get("max_steps") or 50)}
    start_state = {"vars": {**inputs, "input": inputs.get("input") or next(iter(inputs.values()), "")}, "trace": []}
    payload = Command(resume=resume) if resume is not None else start_state
    yield {"type": "start", "thread_id": thread_id}
    try:
        # “custom” lets a node emit tokens while it generates; “updates”, what it returns when it finishes
        for mode, chunk in graph.stream(payload, config, stream_mode=["updates", "custom"]):
            if mode == "custom":
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
               "output": (state.get("vars") or {}).get("_output") or
                         ((state.get("trace") or [{}])[-1].get("output") if state.get("trace") else "")}
    except Exception as e:  # noqa: BLE001
        yield {"type": "error", "thread_id": thread_id, "message": str(e)}


# ----------------------------------------------------------- time travel
def history(spec: dict, thread_id: str) -> list[dict]:
    """A run's checkpoints, from the most recent to the oldest.

    Each one is a “before running X”: you can rewind there, change something and continue
    without repeating what was expensive (a RAG search, a call to a big model)."""
    graph = compile_flow(spec)
    config = {"configurable": {"thread_id": thread_id}}
    labels = {n["id"]: n.get("label") or n["id"] for n in spec.get("nodes", [])}
    out = []
    for snap in graph.get_state_history(config):
        next_nodes = list(snap.next)
        vars_ = snap.values.get("vars") or {}
        out.append({
            "checkpoint_id": snap.config["configurable"].get("checkpoint_id"),
            # The whole config: rebuilding it by hand leaves out “checkpoint_ns” and resuming fails
            "config": snap.config["configurable"],
            "next": next_nodes,
            "next_labels": [labels.get(n, n) for n in next_nodes],
            "done": [t.get("node") for t in (snap.values.get("trace") or [])],
            # Internal variables (_input, _output) aren't shown: they confuse more than they help
            "vars": {k: v for k, v in vars_.items() if not k.startswith("_")},
            "created_at": str(snap.created_at) if snap.created_at else None,
        })
    return out


def _config_de(graph, thread_id: str, checkpoint_id: str) -> dict:
    """Finds the full config of that point in the history (it carries “checkpoint_ns”)."""
    for snap in graph.get_state_history({"configurable": {"thread_id": thread_id}}):
        if snap.config["configurable"].get("checkpoint_id") == checkpoint_id:
            return snap.config
    raise RuntimeError("That checkpoint no longer exists in this run")


def state_at(spec: dict, thread_id: str, checkpoint_id: str) -> dict:
    graph = compile_flow(spec)
    snap = graph.get_state(_config_de(graph, thread_id, checkpoint_id))
    return {"vars": snap.values.get("vars") or {}, "next": list(snap.next),
            "trace": snap.values.get("trace") or []}


def replay(spec: dict, thread_id: str, checkpoint_id: str, edits: dict | None = None) -> Iterator[dict]:
    """Resumes from a checkpoint. With `edits`, variables are changed before continuing.

    Editing forks: a new branch is created and the original run is kept intact."""
    graph = compile_flow(spec)
    base = _config_de(graph, thread_id, checkpoint_id)
    config = base
    if edits:
        actual = graph.get_state(base).values.get("vars") or {}
        config = graph.update_state(base, {"vars": {**actual, **edits}})
    config = {**config, "recursion_limit": int(spec.get("max_steps") or 50)}
    yield {"type": "start", "thread_id": thread_id, "replay_from": checkpoint_id, "edited": list(edits or {})}
    try:
        for mode, chunk in graph.stream(None, config, stream_mode=["updates", "custom"]):
            if mode == "custom":
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
               "output": (vals.get("vars") or {}).get("_output", "")}
    except Exception as e:  # noqa: BLE001
        yield {"type": "error", "thread_id": thread_id, "message": str(e)}


TEMPLATES = {
    "verified_rag": {
        "name": "RAG with verification",
        "description": "Searches your documents, answers, and checks that the answer is backed by the sources.",
        "spec": {"nodes": [
            {"id": "in", "type": "input", "label": "Question", "x": 40, "y": 160,
             "config": {"fields": [{"key": "input", "label": "Question"}]}},
            {"id": "search", "type": "rag", "label": "Find context", "x": 260, "y": 160,
             "config": {"query": "{input}", "k": 4, "output_key": "context"}},
            {"id": "answer", "type": "llm", "label": "Answer", "x": 480, "y": 160,
             "config": {"prompt": "Context:\n{context}\n\nQuestion: {input}\n\nAnswer using only the context and cite [n].",
                        "output_key": "answer"}},
            {"id": "verify", "type": "llm", "label": "Verify", "x": 700, "y": 160,
             "config": {"json": True, "output_key": "verification",
                        "prompt": "Context:\n{context}\n\nAnswer: {answer}\n\nIs everything backed by the context? Return {{\"supported\": true/false, \"reason\": \"...\"}}"}},
            {"id": "router", "type": "router", "label": "Supported?", "x": 920, "y": 160,
             "config": {"mode": "expr", "branches": [{"label": "yes", "when": "supported == True"}, {"label": "no", "when": "True"}]}},
            {"id": "ok", "type": "output", "label": "Answer", "x": 1140, "y": 80, "config": {"template": "{answer}"}},
            {"id": "unsupported", "type": "output", "label": "Not supported", "x": 1140, "y": 240,
             "config": {"template": "I can't confirm this from your documents ({reason}).\n\nDraft: {answer}"}},
        ], "edges": [
            {"from": "in", "to": "search"}, {"from": "search", "to": "answer"},
            {"from": "answer", "to": "verify"}, {"from": "verify", "to": "router"},
            {"from": "router", "to": "ok", "when": "yes"}, {"from": "router", "to": "unsupported", "when": "no"}]},
    },
    "research_and_write": {
        "name": "Research and write (with approval)",
        "description": "Searches the web, drafts a report, shows it to you for approval and saves it.",
        "spec": {"nodes": [
            {"id": "in", "type": "input", "label": "Topic", "x": 40, "y": 160,
             "config": {"fields": [{"key": "input", "label": "Topic to research"}]}},
            {"id": "search", "type": "tool", "label": "Search the web", "x": 260, "y": 160,
             "config": {"tool_id": "web_search", "config": {"max_results": 5}, "args": {"query": "{input}"},
                        "output_key": "results"}},
            {"id": "draft", "type": "llm", "label": "Draft the report", "x": 480, "y": 160,
             "config": {"max_tokens": 900, "output_key": "report",
                        "prompt": "Sources:\n{results}\n\nWrite a short report about “{input}” with 3 sections, citing the sources."}},
            {"id": "approve", "type": "approval", "label": "Review", "x": 700, "y": 160,
             "config": {"message": "Save this report?", "preview": "{report}", "editable": True, "edit_key": "report"}},
            {"id": "save", "type": "tool", "label": "Save file", "x": 920, "y": 160,
             "config": {"tool_id": "write_file", "config": {"overwrite": True}, "output_key": "saved",
                        "args": {"path": "report.md", "content": "{report}"}}},
            {"id": "fin", "type": "output", "label": "Result", "x": 1140, "y": 160,
             "config": {"template": "{saved}\n\n{report}"}},
        ], "edges": [{"from": "in", "to": "search"}, {"from": "search", "to": "draft"},
                     {"from": "draft", "to": "approve"}, {"from": "approve", "to": "save"},
                     {"from": "save", "to": "fin"}]},
    },
    "daily_papers": {
        "name": "Daily papers (Hugging Face)",
        "description": "Fetches Hugging Face's featured papers and summarizes them. Schedule it under "
                       "Connections → Triggers to get it every morning on Telegram. Uses the “Hugging Face Hub” "
                       "MCP server (Connections → catalog).",
        "spec": {"nodes": [
            {"id": "in", "type": "input", "label": "Interest", "x": 40, "y": 160,
             "config": {"fields": [{"key": "input", "label": "What are you interested in? (empty = everything)"}]}},
            {"id": "fetch", "type": "tool", "label": "Featured papers", "x": 260, "y": 160,
             "config": {"tool_id": "mcp", "mcp_catalog": "huggingface", "output_key": "papers",
                        "config": {"tool": "hf_fs"},
                        "args": {"operations": '[{"cmd": "ls", "args": ["hf://papers/trending"]}]'}}},
            {"id": "summarize", "type": "llm", "label": "Summarize", "x": 500, "y": 160,
             "config": {"max_tokens": 900, "output_key": "summary",
                        "prompt": "These are today's featured papers on Hugging Face (JSON):\n{papers}\n\n"
                                  "Reader's interest: “{input}”. Pick the 5 most relevant to that interest "
                                  "(if it's empty, the most upvoted) and for each one write: the title, "
                                  "one sentence on what it contributes and the link https://huggingface.co/papers/<id>."}},
            {"id": "fin", "type": "output", "label": "Summary", "x": 740, "y": 160, "config": {"template": "{summary}"}},
        ], "edges": [{"from": "in", "to": "fetch"}, {"from": "fetch", "to": "summarize"}, {"from": "summarize", "to": "fin"}]},
    },
    "audio_to_collection": {
        "name": "Audio to collection (Whisper)",
        "description": "Transcribes a recording from data/inbox with local Whisper and indexes it into a collection "
                       "so you can ask it questions. Uses the “Transcribe audio” MCP server (Connections → catalog) "
                       "and the app's own one.",
        "spec": {"nodes": [
            {"id": "in", "type": "input", "label": "Audio", "x": 40, "y": 160,
             "config": {"fields": [{"key": "input", "label": "Audio file in data/inbox (e.g. lecture.mp3)"},
                                   {"key": "collection", "label": "Collection to save it in"}]}},
            {"id": "transcribe", "type": "tool", "label": "Transcribe", "x": 280, "y": 160,
             "config": {"tool_id": "mcp", "mcp_catalog": "whisper", "output_key": "transcript",
                        "config": {"tool": "transcribe_file", "max_chars": 200000},
                        "args": {"file_path": str(settings.data_dir / "inbox") + "/{input}"}}},
            {"id": "save", "type": "tool", "label": "Index", "x": 520, "y": 160,
             "config": {"tool_id": "mcp", "mcp_catalog": "llm-trainer", "output_key": "indexed",
                        "config": {"tool": "add_document"},
                        "args": {"collection": "{collection}", "filename": "{input}.md",
                                 "text": "{transcript_json[text]}"}}},
            {"id": "fin", "type": "output", "label": "Result", "x": 760, "y": 160,
             "config": {"template": "{indexed}\n\n{transcript_json[text]}"}},
        ], "edges": [{"from": "in", "to": "transcribe"}, {"from": "transcribe", "to": "save"},
                     {"from": "save", "to": "fin"}]},
    },
    "classify_and_route": {
        "name": "Classify and route",
        "description": "Classifies the request and sends it down the right path (documents, calculation or a direct answer).",
        "spec": {"nodes": [
            {"id": "in", "type": "input", "label": "Request", "x": 40, "y": 200,
             "config": {"fields": [{"key": "input", "label": "Request"}]}},
            {"id": "classify", "type": "router", "label": "Request type", "x": 260, "y": 200,
             "config": {"mode": "llm", "prompt": "Classify this request: {input}",
                        "branches": [{"label": "documents"}, {"label": "calculation"}, {"label": "general"}]}},
            {"id": "docs", "type": "rag", "label": "Search documents", "x": 520, "y": 60,
             "config": {"query": "{input}", "output_key": "context"}},
            {"id": "calc", "type": "tool", "label": "Calculate", "x": 520, "y": 200,
             "config": {"tool_id": "python", "config": {"timeout_s": 20}, "output_key": "context",
                        "args": {"code": "# solve: {input}\nprint('write the calculation here')"}}},
            {"id": "direct", "type": "llm", "label": "Answer directly", "x": 520, "y": 340,
             "config": {"prompt": "{input}", "output_key": "context"}},
            {"id": "draft", "type": "llm", "label": "Write the answer", "x": 800, "y": 200,
             "config": {"prompt": "Question: {input}\n\nAvailable information:\n{context}\n\nAnswer briefly.",
                        "output_key": "answer"}},
            {"id": "fin", "type": "output", "label": "Answer", "x": 1060, "y": 200,
             "config": {"template": "{answer}"}},
        ], "edges": [{"from": "in", "to": "classify"},
                     {"from": "classify", "to": "docs", "when": "documents"},
                     {"from": "classify", "to": "calc", "when": "calculation"},
                     {"from": "classify", "to": "direct", "when": "general"},
                     {"from": "docs", "to": "draft"}, {"from": "calc", "to": "draft"},
                     {"from": "direct", "to": "draft"}, {"from": "draft", "to": "fin"}]},
    },
}


# ----------------------------------------------------------- export to code
def _py(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False)


class _AVars(ast.NodeTransformer):
    """Rewrites `supported == True` as `v.get("supported") == True`.

    In the editor conditions are written with bare variable names, but in the
    exported code those names don't exist: they live in the state dict."""

    SAFE = {"len", "int", "float", "str", "bool", "True", "False", "None"}

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id not in self.SAFE:
            return ast.Call(func=ast.Attribute(value=ast.Name(id="v", ctx=ast.Load()), attr="get", ctx=ast.Load()),
                            args=[ast.Constant(value=node.id)], keywords=[])
        return node


def _cond_py(expr: str) -> str:
    """An editor condition translated into a valid expression in the exported code."""
    try:
        tree = ast.parse(expr.strip(), mode="eval")
        return ast.unparse(ast.fix_missing_locations(_AVars().visit(tree)).body)
    except SyntaxError:
        return "False  # couldn't translate: " + expr.replace("\n", " ")


def to_python(spec: dict, name: str = "my_flow") -> str:
    """Generates the LangGraph code equivalent of a drawn flow.

    It isn't meant to run inside the app (that's what the compiler is for) but so that the work
    put into a flow doesn't depend on this application still existing: the resulting file
    runs with langgraph and the functions you fill in.
    """
    nodes = {n["id"]: n for n in spec.get("nodes", [])}
    edges = spec.get("edges", [])
    L = [
        '"""Flow exported from LLM Trainer.',
        "",
        f"Flow: {name}",
        "",
        "Each node is a function that takes the state and returns the variables it adds.",
        "Nodes that called the model or RAG are marked with TODO: plug them into your",
        "own client (OpenAI, Anthropic, Ollama…), which is the only part tied to the app.",
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
        "def render(template: str, v: dict) -> str:",
        '    """Replaces {variables}; missing ones are left empty."""',
        "    import string",
        "",
        "    class Safe(dict):",
        "        def __missing__(self, k):",
        '            return ""',
        "",
        "    return string.Formatter().vformat(template, (), Safe(v))",
        "",
        "",
    ]
    for nid, n in nodes.items():
        cfg = n.get("config") or {}
        key = cfg.get("output_key") or f"{nid}_output"
        L += [f"def node_{nid}(state: State) -> dict:", f'    """{n.get("label") or nid} ({n["type"]})."""',
              "    v = dict(state.get('vars') or {})"]
        t = n["type"]
        if t == "input":
            L.append("    value = v.get('input', '')")
        elif t == "llm":
            L += [f"    prompt = render({_py(cfg.get('prompt', '{input}'))}, v)",
                  "    # TODO: call your model here",
                  "    value = your_model(prompt)"]
        elif t == "rag":
            L += [f"    query = render({_py(cfg.get('query', '{input}'))}, v)",
                  f"    # TODO: retrieve from your index (it was collection {cfg.get('collection_id')}, k={cfg.get('k', 4)})",
                  "    value = your_search(query)"]
        elif t == "tool":
            L += [f"    # TODO: tool {cfg.get('tool_id')!r} with arguments {cfg.get('args')!r}",
                  "    value = ''"]
        elif t == "agent":
            L += [f"    # TODO: agent {cfg.get('agent_id')} with its tools", "    value = ''"]
        elif t == "code":
            L += ["    # The code you wrote in the node:"] + \
                 [f"    {l}" for l in (cfg.get("code") or "").splitlines()] + \
                 ["    value = locals().get('result')"]
        elif t == "foreach":
            L += [f"    items = v.get({_py(cfg.get('items_key'))}) or []",
                  "    value = []",
                  "    for index, item in enumerate(items):",
                  f"        prompt = render({_py(cfg.get('prompt', '{item}'))}, {{**v, 'item': item, 'index': index}})",
                  "        value.append(your_model(prompt))  # TODO"]
        elif t == "subflow":
            L += [f"    # TODO: flow {cfg.get('flow_id')} ran here; export it too and call it",
                  "    value = ''"]
        elif t == "router":
            branches = cfg.get("branches") or []
            L.append("    value = ''")
            for b in branches:
                if cfg.get("mode") != "llm" and b.get("when"):
                    L += [f"    if not value and ({_cond_py(b['when'])}):", f"        value = {_py(b.get('label',''))}"]
            L += [f"    value = value or {_py(branches[-1].get('label', '') if branches else '')}",
                  f"    v[{_py(nid + '_branch')}] = value"]
        elif t == "approval":
            L += ["    # Here the flow stopped to wait for your go-ahead (langgraph.types.interrupt)",
                  "    value = 'approved'"]
        elif t == "output":
            L += [f"    value = render({_py(cfg.get('template', '{input}'))}, v)", "    v['_output'] = value"]
        L += [f"    v[{_py(key)}] = value", "    return {'vars': v}", "", ""]

    L += ["def build():", "    g = StateGraph(State)"]
    for nid in nodes:
        L.append(f"    g.add_node({_py(nid)}, node_{nid})")
    inputs = [n for n in nodes.values() if n["type"] == "input"] or list(nodes.values())
    L.append(f"    g.add_edge(START, {_py(inputs[0]['id'])})")
    outgoing: dict[str, list] = {}
    for e in edges:
        outgoing.setdefault(e["from"], []).append(e)
    for nid, n in nodes.items():
        outs = outgoing.get(nid, [])
        if not outs:
            L.append(f"    g.add_edge({_py(nid)}, END)")
        elif n["type"] == "router":
            destinations = {e.get("when") or "": e["to"] for e in outs}
            L += [f"    def _choose_{nid}(state):",
                  f"        return {_py(destinations)}.get((state.get('vars') or {{}}).get({_py(nid + '_branch')}, ''), {_py(list(destinations.values())[0])})",
                  f"    g.add_conditional_edges({_py(nid)}, _choose_{nid}, {_py(list(destinations.values()))})"]
        else:
            for e in outs:
                L.append(f"    g.add_edge({_py(nid)}, {_py(e['to'])})")
    L += ["    return g.compile()", "", "",
          'if __name__ == "__main__":',
          "    app = build()",
          "    out = app.invoke({'vars': {'input': 'your question here'}})",
          "    print(out['vars'].get('_output'))",
          ""]
    return "\n".join(L)

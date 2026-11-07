"""Evaluation: a models × RAG modes matrix over a set of questions.

Per-answer metrics (null when they don't apply):
    similarity        semantic similarity answer ↔ reference (embeddings)
    keyword_score     share of keywords present
    answer_relevance  similarity question ↔ answer (does it answer what was asked?)
    context_relevance [RAG] mean score of the chunks retrieved for the question
    context_recall    [RAG] max similarity reference ↔ chunk (did the context contain the answer?)
    judge_score       [judge] correctness 1–10 against the reference
    faithfulness      [judge + RAG] 1–10: are the claims backed by the context? (hallucination)
"""
import json
import re
import time

import numpy as np
from sqlmodel import Session

from ..config import settings
from ..db import EvalRun, Model, RagCollection, engine
from . import rag
from .inference import GenParams, manager
from .jobs import is_gpu_busy
from .tasks import Task

METRICS = ["similarity", "keyword_score", "answer_relevance", "context_relevance", "context_recall",
           "judge_score", "faithfulness", "latency_s", "rating"]

JUDGE_PROMPT = """Evaluate the ANSWER to the QUESTION by comparing it with the REFERENCE ANSWER.
Score its correctness and completeness from 1 to 10 (10 = equivalent to or better than the reference).
Reply with JSON ONLY: {{"score": <1-10>, "reason": "<one sentence>"}}

QUESTION: {q}
REFERENCE ANSWER: {ref}
ANSWER: {ans}"""

FAITH_PROMPT = """I'm giving you a CONTEXT and an ANSWER generated from it.
Score from 1 to 10 how much of what the ANSWER claims is backed by the CONTEXT
(10 = everything is backed; 1 = almost everything is made up or contradicts the context).
Saying "I don't know" or "it's not in the context" when the context doesn't have the information counts as backed.
Reply with JSON ONLY: {{"score": <1-10>, "reason": "<one sentence>"}}

CONTEXT:
{ctx}

ANSWER: {ans}"""

THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.S)


def strip_think(text: str) -> str:
    """Strips the reasoning of models that think out loud.

    Three different shapes, all seen in real models:
        - the whole <think>…</think> block;
        - only the closing </think>, because the chat template already opened the block inside the prompt and
            the model never emits the opening tag (Qwen3.5 and derivatives in GGUF);
        - an unclosed opening, when the tokens ran out while thinking: there's no answer to
            rescue there, and returning the reasoning would be worse than returning nothing.
    """
    t = THINK_RE.sub("", text)
    # Gemma 4 thinks in its own channel: <|channel>thought … <channel|>
    t = re.sub(r"<\|channel>.*?<channel\|>", "", t, flags=re.S)
    if "</think>" in t:           # stray closing tag: what came before was thinking
        t = t.split("</think>")[-1]
    if "<think>" in t:            # opening without closing: it got stuck thinking and never answered
        t = t.split("<think>")[0]
    return t.strip()


def _emb(texts: list[str]) -> np.ndarray:
    return np.array(rag.embed(settings.default_embed_model, texts, "passage"))


def _sim(a: str, b: str) -> float:
    va, vb = _emb([a, b])
    return round(float(np.dot(va, vb)), 4)


def _contains(answer: str, keywords: list[str]) -> float | None:
    if not keywords:
        return None
    low = answer.lower()
    return round(sum(k.lower() in low for k in keywords) / len(keywords), 3)


def _judge(prompt: str, helper: int | None = None) -> tuple[float | None, str | None]:
    """Scores an answer. With `helper` an external model judges and nothing needs to be loaded into VRAM."""
    from . import providers
    out = providers.complete(helper, prompt, max_tokens=400, temperature=0)
    m = re.search(r"\{.*\}", out, re.S)
    try:
        j = json.loads(m.group(0))
        return max(1.0, min(10.0, float(j["score"]))), j.get("reason")
    except Exception:  # noqa: BLE001
        # Small models sometimes don't return valid JSON: look for the first number 1–10
        n = re.search(r"\b(10|[1-9])(?:\.\d+)?\b", out)
        return (float(n.group(0)), out[:200]) if n else (None, out[:200])


def _load_model(m: Model, quant: str):
    # Respect the engine the user chose (vLLM speeds batch evaluations up a lot)
    engine = manager.preferred_engine if m.kind != "gguf" else "auto"
    manager.load(model_id=m.id, name=m.name, kind=m.kind, path=m.path, base_model=m.base_model,
                 quantization=quant, engine=engine)


def variant_name(model_name: str, use_rag: bool) -> str:
    return f"{model_name} + RAG" if use_rag else model_name


def run_eval(run_id: int, task: Task) -> dict:
    if is_gpu_busy():
        raise RuntimeError("A GPU job is running; wait for it to finish.")
    # expire_on_commit=False: the objects are still used after the session closes
    with Session(engine, expire_on_commit=False) as s:
        run = s.get(EvalRun, run_id)
        cfg = run.config
        models = [s.get(Model, mid) for mid in cfg["model_ids"]]
        coll = s.get(RagCollection, cfg["rag_collection_id"]) if cfg.get("rag_collection_id") else None
        judge = s.get(Model, cfg["judge_model_id"]) if cfg.get("judge_model_id") else None
        judge_helper = cfg.get("judge_helper")  # external helper model acting as judge
        run.status = "running"
        s.add(run)
        s.commit()

    questions = cfg["questions"]  # [{question, expected?, keywords?[]}]
    # RAG modes to evaluate for each model: "off" (no context) and/or "on" (with retrieved context)
    modes = [x == "on" for x in dict.fromkeys(cfg.get("rag_modes") or ["on"])] if coll else [False]
    params = GenParams(**cfg.get("params", {}))
    system = cfg.get("system_prompt")
    k = cfg.get("rag_k") or (coll.top_k if coll else 4)

    # Retrieval doesn't depend on the model: it's computed once per question
    retrieved: dict[int, list[dict]] = {}
    if coll and any(modes):
        for qi, q in enumerate(questions):
            task.update(message=f"Retrieving context {qi + 1}/{len(questions)}")
            retrieved[qi] = rag.retrieve(coll, q["question"], k=k)

    n_answers = len(models) * len(modes) * len(questions)
    total = n_answers * (2 if judge else 1)
    done = 0
    results: list[dict] = []
    failed: list[dict] = []
    try:
        for m in models:
            task.update(message=f"Loading {m.name}")
            try:
                _load_model(m, cfg.get("quantization", "none"))
            except Exception as e:  # noqa: BLE001
                # A model that doesn't load (doesn't fit in VRAM, odd format…) can't throw away
                # the answers already generated by the others: it's noted down and the rest go on.
                print(f"[eval] {m.name} couldn't be loaded: {e}")
                failed.append({"model": m.name, "error": str(e)[:400]})
                done += len(questions) * len(modes) * (2 if judge else 1)
                continue
            for use_rag in modes:
                # Prepare all the conversations of this variant (so they can be generated in batches)
                prepared = []
                for qi, q in enumerate(questions):
                    hits = retrieved.get(qi, []) if use_rag else []
                    rag_mode = None
                    if use_rag:
                        msgs, rag_mode = rag.messages_for(coll, q["question"], hits, None, system,
                                                          cfg.get("rag_system_prompt") or None)
                    else:
                        msgs = ([{"role": "system", "content": system}] if system else []) + \
                               [{"role": "user", "content": q["question"]}]
                    prepared.append((q, hits, rag_mode, msgs))
                # vLLM generates them all at once (much faster); the other engines, one by one
                t_batch = time.time()
                batch = manager.batch_chat([p[3] for p in prepared], params)
                batch_latency = round((time.time() - t_batch) / max(1, len(prepared)), 2) if batch else None
                for qi, (q, hits, rag_mode, msgs) in enumerate(prepared):
                    task.update(done / total, f"{variant_name(m.name, use_rag)}: {q['question'][:50]}")
                    t0 = time.time()
                    raw = batch[qi] if batch else "".join(manager.stream_chat(msgs, params))
                    latency = batch_latency if batch else round(time.time() - t0, 2)
                    answer = strip_think(raw)
                    r = {"model_id": m.id, "model_name": m.name, "rag": use_rag, "rag_mode": rag_mode,
                         "variant": variant_name(m.name, use_rag),
                         "question": q["question"], "expected": q.get("expected"), "answer": answer, "raw_answer": raw,
                         "latency_s": latency, "rating": None,
                         "similarity": _sim(answer, q["expected"]) if q.get("expected") and answer else None,
                         "keyword_score": _contains(answer, q.get("keywords") or []),
                         "answer_relevance": _sim(q["question"], answer) if answer else None,
                         "context_relevance": None, "context_recall": None,
                         "sources": [{"source": h.get("source"), "page": h.get("page"), "score": h["score"],
                                      "text": h["text"]} for h in hits]}
                    if hits:
                        r["context_relevance"] = round(float(np.mean([h["score"] for h in hits])), 4)
                        if q.get("expected"):
                            vecs = _emb([q["expected"]] + [h["text"] for h in hits])
                            r["context_recall"] = round(float((vecs[1:] @ vecs[0]).max()), 4)
                    results.append(r)
                    done += 1

        if judge or judge_helper:
            if judge_helper:  # external judge: uses no VRAM and doesn't force the evaluated model out
                from . import providers
                task.update(message=f"External judge: {providers.describe(judge_helper)}")
            else:
                task.update(message=f"Loading judge {judge.name}")
                _load_model(judge, cfg.get("quantization", "none"))
            for r in results:
                task.update(done / total, f"Judge: {r['question'][:50]}")
                r["judge_score"], r["judge_reason"] = _judge(JUDGE_PROMPT.format(
                    q=r["question"], ref=r.get("expected") or "(no reference; judge the overall quality)", ans=r["answer"]),
                    judge_helper)
                if r["rag"] and r["sources"]:
                    ctx = "\n\n".join(f"[{i + 1}] {s['text']}" for i, s in enumerate(r["sources"]))
                    r["faithfulness"], r["faithfulness_reason"] = _judge(FAITH_PROMPT.format(ctx=ctx, ans=r["answer"]),
                                                                        judge_helper)
                done += 1
    except Exception as e:
        _save(run_id, results, "failed", str(e), summarize(results))
        raise
    summary = summarize(results)
    if failed:
        summary["_failed_to_load"] = failed
    status_ = "completed" if results else "failed"
    warning = None if not failed else "; ".join(f"{f['model']}: {f['error'][:120]}" for f in failed)
    _save(run_id, results, status_, warning, summary)
    return summary


def summarize(results: list[dict]) -> dict:
    """Mean of each metric per variant (model, or model + RAG)."""
    by: dict[str, dict] = {}
    for r in results:
        key = r.get("variant") or r["model_name"]
        d = by.setdefault(key, {"model_name": r["model_name"], "rag": bool(r.get("rag")), "n": 0,
                                **{k: [] for k in METRICS}})
        d["n"] += 1
        for k in METRICS:
            if r.get(k) is not None:
                d[k].append(r[k])
    return {name: {**{k: v for k, v in d.items() if k not in METRICS},
                   **{k: (round(sum(d[k]) / len(d[k]), 3) if d[k] else None) for k in METRICS}}
            for name, d in by.items()}


def _save(run_id, results, status, error, summary=None):
    with Session(engine) as s:
        run = s.get(EvalRun, run_id)
        run.results, run.status, run.error = results, status, error
        if summary is not None:
            run.summary = summary
        s.add(run)
        s.commit()


# ---------------- judge calibration ----------------
def judge_agreement(results: list[dict]) -> dict | None:
    """How much the judge agrees with your ratings: Spearman correlation and mean error (1–5 scale)."""
    pairs = [(r["judge_score"] / 2, r["rating"]) for r in results
             if r.get("judge_score") is not None and r.get("rating") is not None]
    if len(pairs) < 3:
        return {"n": len(pairs), "spearman": None, "mae": None}
    from scipy.stats import spearmanr
    j, h = zip(*pairs)
    rho = spearmanr(j, h).statistic
    return {"n": len(pairs), "spearman": None if rho != rho else round(float(rho), 3),
            "mae": round(float(np.mean(np.abs(np.array(j) - np.array(h)))), 2),
            "judge_bias": round(float(np.mean(np.array(j) - np.array(h))), 2)}


# ---------------- evaluation questions generated from documents ----------------
QGEN_PROMPT = """From the CHUNK, write {n} evaluation questions with their correct answer.
The questions must be specific and answerable ONLY from the chunk (facts, names, figures).
Return ONLY a JSON array: [{{"question": "...", "expected": "...", "keywords": ["1-3 keywords from the answer"]}}]

CHUNK:
{text}"""


def generate_questions(task: Task, collection_id: int, n_chunks: int, per_chunk: int,
                       sources: list[str] | None = None) -> list[dict]:
    """Creates an evaluation set (question, reference, keywords) by sampling chunks from a collection."""
    import random
    col = rag._col(collection_id)
    got = col.get(where={"source": {"$in": sources}} if sources else None, include=["documents", "metadatas"])
    pool = [(d, m) for d, m in zip(got["documents"], got["metadatas"]) if not m.get("level") and len(d) > 200]
    random.shuffle(pool)
    out: list[dict] = []
    failures = 0
    pieces = pool[:n_chunks]
    for i, (doc, meta) in enumerate(pieces):
        task.update(i / max(1, min(n_chunks, len(pool))), f"Chunk {i + 1} · {len(out)} questions")
        text = "".join(manager.stream_chat([{"role": "user", "content": QGEN_PROMPT.format(n=per_chunk, text=doc[:2500])}],
                                           GenParams(max_new_tokens=600, temperature=0.4, enable_thinking=False)))
        m = re.search(r"\[.*\]", strip_think(text), re.S)
        try:
            for q in json.loads(m.group(0)):
                if q.get("question") and q.get("expected"):
                    kws = [k for k in (q.get("keywords") or []) if isinstance(k, str) and k.lower() in doc.lower()]
                    out.append({"question": q["question"].strip(), "expected": q["expected"].strip(),
                                "keywords": kws[:3], "source": meta.get("source")})
        except Exception as e:  # noqa: BLE001
            # Without this, twenty chunks produced fourteen questions and there was no way to know how many
            # chunks had failed or why: it looked like the model generated few.
            failures += 1
            print(f"[eval] chunk discarded while generating questions: {e}")
            continue
    if failures:
        print(f"[eval] {failures} of {len(pieces)} chunks gave no usable questions")
    if not out:
        raise RuntimeError("The model didn't generate valid questions; try a more capable instruct model")
    return out

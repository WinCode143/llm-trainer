"""Synthetic dataset generation with the loaded model.

Sources:
    - docs:   question/answer pairs from chunks of a RAG collection (answers faithful to the text)
    - topics: a topic tree from the description of a domain (no documents needed)
Quality:
    - a variety of question types (fact, explanation, comparison, procedure, application)
    - semantic deduplication (embeddings) of near-identical questions
    - a judge filter: the model scores each pair 1–5 and the ones below the threshold are dropped
"""
import json
import random
import re

import numpy as np
from sqlmodel import Session

from ..config import settings
from ..db import Dataset, RagCollection, engine
from . import datasets as ds_service
from . import rag
from .evaluation import strip_think
from .inference import GenParams, manager
from .tasks import Task

STYLES = {
    "hecho": "questions about specific facts (what, who, when, how much)",
    "explicacion": "questions asking to explain causes or how something works (why, how does it work)",
    "comparacion": "questions that compare or contrast things",
    "procedimiento": "questions about steps or processes (how is it done)",
    "aplicacion": "practical or scenario questions (what would you do if…)",
}

DOC_PROMPT = """From the TEXT, generate {n} question and answer pairs in {lang}.
Question type: {style}.
The questions must be answerable from the text alone; the answers must be complete, precise and
self-contained (without saying “according to the text”).
Return ONLY a JSON array: [{{"question": "...", "answer": "..."}}]

TEXT:
{text}"""

TOPICS_PROMPT = """Domain: {domain}
List {n} different, specific subtopics of this domain that a user might ask about.
One per line, without numbering or explaining."""

TOPIC_QA_PROMPT = """Domain: {domain}. Subtopic: {topic}.
Generate {n} high-quality instruction and answer pairs in {lang}.
Type: {style}. Correct, useful, well-structured answers.
Return ONLY a JSON array: [{{"question": "...", "answer": "..."}}]"""

JUDGE_PROMPT = """Evaluate this question-answer pair for training an assistant.
{ctx}QUESTION: {q}
ANSWER: {a}

Score from 1 to 5 (5 = a clear question and a correct, complete answer{faithful}).
Reply with JSON ONLY: {{"score": <1-5>}}"""


def _gen(prompt: str, temperature: float, max_tokens: int = 1200, helper: int | None = None) -> str:
    """Generates with the chosen helper model, or with the loaded model if there's none."""
    from . import providers
    return providers.complete(helper, prompt, max_tokens=max_tokens, temperature=temperature)


def _pairs(out: str) -> list[dict]:
    """Extracts the pairs from the JSON the model returned.

    When it fails it returns an empty list, and that used to go unnoticed: forty attempts produced fourteen
    pairs and there was no way to tell whether the model wasn't cooperating or the bug was ours."""
    m = re.search(r"\[.*\]", out, re.S)
    if not m:
        print(f"[synth] the model didn't return a JSON list: {out[:120]!r}")
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        print(f"[synth] invalid JSON ({e}); the batch is dropped")
        return []
    good = [p for p in data if isinstance(p, dict) and p.get("question") and p.get("answer")]
    if len(good) < len(data):
        print(f"[synth] {len(data) - len(good)} of {len(data)} pairs were incomplete")
    return good


def _judge(q: str, a: str, ctx: str | None, helper: int | None = None) -> float | None:
    out = _gen(JUDGE_PROMPT.format(q=q, a=a, ctx=f"REFERENCE TEXT:\n{ctx[:2000]}\n\n" if ctx else "",
                                   faithful=" and faithful to the reference text" if ctx else ""), 0.0, 60, helper=helper)
    m = re.search(r"\b([1-5])\b", out)
    return float(m.group(1)) if m else None


def _dedupe(rows: list[dict], threshold: float) -> tuple[list[dict], int]:
    if len(rows) < 2:
        return rows, 0
    vecs = np.array(rag.embed(settings.default_embed_model, [r["question"] for r in rows], "query"))
    keep: list[int] = []
    for i, v in enumerate(vecs):
        if not keep or float((vecs[keep] @ v).max()) < threshold:
            keep.append(i)
    return [rows[i] for i in keep], len(rows) - len(keep)


def generate(task: Task, *, name: str, source: str = "docs", collection_id: int | None = None, domain: str | None = None,
             num_chunks: int = 20, num_topics: int = 10, pairs_per_chunk: int = 3, language: str = "English",
             styles: list[str] | None = None, system_prompt: str | None = None, temperature: float = 0.7,
             dedupe_threshold: float = 0.92, judge_min_score: float = 0, helper: int | None = None, **_) -> dict:
    from . import providers
    if not helper and not manager.current:
        raise RuntimeError("Load a model (Playground or Models) or pick a helper model to generate the dataset")
    styles = [s for s in (styles or ["hecho", "explicacion"]) if s in STYLES] or ["hecho"]
    raw: list[dict] = []
    failed = 0
    src_label = ""

    if source == "docs":
        with Session(engine) as s:
            coll = s.get(RagCollection, collection_id)
            if not coll:
                raise ValueError("Collection not found")
            src_label = f"synthetic:{coll.name}"
        chunks = [c for c in rag.sample_chunks(collection_id, num_chunks * 2) if len(c["text"]) > 150][:num_chunks]
        for i, c in enumerate(chunks):
            task.update(0.7 * i / max(1, len(chunks)), f"Chunk {i + 1}/{len(chunks)} · {len(raw)} pairs")
            style = styles[i % len(styles)]
            got = _pairs(_gen(DOC_PROMPT.format(n=pairs_per_chunk, lang=language, style=STYLES[style], text=c["text"][:3000]), temperature, helper=helper))
            failed += not got
            raw += [{**p, "_ctx": c["text"], "_style": style} for p in got]
    else:
        if not domain:
            raise ValueError("Describe the domain to generate the topic tree")
        src_label = f"synthetic:topics:{domain[:40]}"
        task.update(0.02, "Generating the topic tree…")
        topics = [t.strip(" -•\t0123456789.") for t in _gen(TOPICS_PROMPT.format(domain=domain, n=num_topics), 0.8, 400, helper=helper).splitlines()]
        topics = [t for t in dict.fromkeys(topics) if len(t) > 3][:num_topics]
        if not topics:
            raise RuntimeError("The model didn't generate subtopics; try a more capable instruct model")
        for i, t in enumerate(topics):
            task.update(0.05 + 0.65 * i / len(topics), f"Topic {i + 1}/{len(topics)}: {t[:40]} · {len(raw)} pairs")
            style = styles[i % len(styles)]
            got = _pairs(_gen(TOPIC_QA_PROMPT.format(domain=domain, topic=t, n=pairs_per_chunk, lang=language, style=STYLES[style]), temperature, helper=helper))
            failed += not got
            raw += [{**p, "_ctx": None, "_style": style, "_topic": t} for p in got]

    if not raw:
        raise RuntimeError("The model didn't produce any valid pair. Try a more capable instruct model.")
    task.update(0.72, f"Deduplicating {len(raw)} pairs…")
    rows, n_dupes = _dedupe(raw, dedupe_threshold)
    n_rejected = 0
    if judge_min_score > 0:
        kept = []
        for i, r in enumerate(rows):
            task.update(0.75 + 0.24 * i / len(rows), f"Judge: {i + 1}/{len(rows)} · {n_rejected} dropped")
            sc = _judge(r["question"], r["answer"], r["_ctx"], helper)
            if sc is None or sc >= judge_min_score:
                kept.append({**r, "_score": sc})
            else:
                n_rejected += 1
        rows = kept
    if not rows:
        raise RuntimeError("The judge dropped every pair; lower the threshold or use a better model")

    out = []
    for r in rows:
        msgs = [{"role": "user", "content": r["question"].strip()}, {"role": "assistant", "content": r["answer"].strip()}]
        if system_prompt:
            msgs.insert(0, {"role": "system", "content": system_prompt})
        out.append({"messages": msgs})
    random.shuffle(out)
    path = ds_service.write_jsonl(name, out)
    with Session(engine) as s:
        d = Dataset(name=name, format="chat", path=str(path), num_rows=len(out), source=src_label)
        s.add(d)
        s.commit()
        s.refresh(d)
        return {"dataset_id": d.id, "rows": len(out), "generated": len(raw), "duplicates_removed": n_dupes,
                "rejected_by_judge": n_rejected, "failed_generations": failed}

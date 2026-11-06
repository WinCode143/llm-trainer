"""Generación de datasets sintéticos con el modelo cargado.

Fuentes:
  - docs:   pares pregunta/respuesta a partir de fragmentos de una colección RAG (respuestas fieles al texto)
  - topics: árbol de temas a partir de la descripción de un dominio (no necesita documentos)
Calidad:
  - variedad de tipos de pregunta (hecho, explicación, comparación, procedimiento, aplicación)
  - deduplicación semántica (embeddings) de preguntas casi iguales
  - filtro con juez: el modelo puntúa 1–5 cada par y se descartan los que no llegan al umbral
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
    "hecho": "preguntas de datos concretos (qué, quién, cuándo, cuánto)",
    "explicacion": "preguntas que piden explicar causas o funcionamiento (por qué, cómo funciona)",
    "comparacion": "preguntas que comparan o contrastan elementos",
    "procedimiento": "preguntas sobre pasos o procesos (cómo se hace)",
    "aplicacion": "preguntas prácticas o de escenario (qué harías si…)",
}

DOC_PROMPT = """A partir del TEXTO, genera {n} pares de pregunta y respuesta en {lang}.
Tipo de preguntas: {style}.
Las preguntas deben poder responderse sólo con el texto; las respuestas deben ser completas, precisas y
autocontenidas (sin decir «según el texto»).
Devuelve SOLO un array JSON: [{{"question": "...", "answer": "..."}}]

TEXTO:
{text}"""

TOPICS_PROMPT = """Dominio: {domain}
Enumera {n} subtemas distintos y concretos de este dominio que un usuario podría preguntar.
Uno por línea, sin numerar ni explicar."""

TOPIC_QA_PROMPT = """Dominio: {domain}. Subtema: {topic}.
Genera {n} pares de instrucción y respuesta de alta calidad en {lang}.
Tipo: {style}. Respuestas correctas, útiles y bien estructuradas.
Devuelve SOLO un array JSON: [{{"question": "...", "answer": "..."}}]"""

JUDGE_PROMPT = """Evalúa este par pregunta-respuesta para entrenar un asistente.
{ctx}PREGUNTA: {q}
RESPUESTA: {a}

Puntúa de 1 a 5 (5 = pregunta clara y respuesta correcta, completa{faithful}).
Responde SOLO con JSON: {{"score": <1-5>}}"""


def _gen(prompt: str, temperature: float, max_tokens: int = 1200, helper: int | None = None) -> str:
    """Genera con el modelo ayudante elegido, o con el modelo cargado si no hay ninguno."""
    from . import providers
    return providers.complete(helper, prompt, max_tokens=max_tokens, temperature=temperature)


def _pairs(out: str) -> list[dict]:
    """Extrae los pares del JSON que devolvió el modelo.

    Cuando falla devuelve lista vacía, y eso antes no se veía: de cuarenta intentos salían catorce
    pares y no había forma de saber si el modelo no colaboraba o si el fallo era nuestro."""
    m = re.search(r"\[.*\]", out, re.S)
    if not m:
        print(f"[synth] el modelo no devolvió una lista JSON: {out[:120]!r}")
        return []
    try:
        datos = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        print(f"[synth] JSON inválido ({e}); se descarta la tanda")
        return []
    buenos = [p for p in datos if isinstance(p, dict) and p.get("question") and p.get("answer")]
    if len(buenos) < len(datos):
        print(f"[synth] {len(datos) - len(buenos)} de {len(datos)} pares venían incompletos")
    return buenos


def _judge(q: str, a: str, ctx: str | None, helper: int | None = None) -> float | None:
    out = _gen(JUDGE_PROMPT.format(q=q, a=a, ctx=f"TEXTO DE REFERENCIA:\n{ctx[:2000]}\n\n" if ctx else "",
                                   faithful=" y fiel al texto de referencia" if ctx else ""), 0.0, 60, helper=helper)
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
             num_chunks: int = 20, num_topics: int = 10, pairs_per_chunk: int = 3, language: str = "español",
             styles: list[str] | None = None, system_prompt: str | None = None, temperature: float = 0.7,
             dedupe_threshold: float = 0.92, judge_min_score: float = 0, helper: int | None = None, **_) -> dict:
    from . import providers
    if not helper and not manager.current:
        raise RuntimeError("Carga un modelo (Playground o Modelos) o elige un modelo ayudante para generar el dataset")
    styles = [s for s in (styles or ["hecho", "explicacion"]) if s in STYLES] or ["hecho"]
    raw: list[dict] = []
    failed = 0
    src_label = ""

    if source == "docs":
        with Session(engine) as s:
            coll = s.get(RagCollection, collection_id)
            if not coll:
                raise ValueError("Colección no encontrada")
            src_label = f"sintético:{coll.name}"
        chunks = [c for c in rag.sample_chunks(collection_id, num_chunks * 2) if len(c["text"]) > 150][:num_chunks]
        for i, c in enumerate(chunks):
            task.update(0.7 * i / max(1, len(chunks)), f"Fragmento {i + 1}/{len(chunks)} · {len(raw)} pares")
            style = styles[i % len(styles)]
            got = _pairs(_gen(DOC_PROMPT.format(n=pairs_per_chunk, lang=language, style=STYLES[style], text=c["text"][:3000]), temperature, helper=helper))
            failed += not got
            raw += [{**p, "_ctx": c["text"], "_style": style} for p in got]
    else:
        if not domain:
            raise ValueError("Describe el dominio para generar el árbol de temas")
        src_label = f"sintético:temas:{domain[:40]}"
        task.update(0.02, "Generando árbol de temas…")
        topics = [t.strip(" -•\t0123456789.") for t in _gen(TOPICS_PROMPT.format(domain=domain, n=num_topics), 0.8, 400, helper=helper).splitlines()]
        topics = [t for t in dict.fromkeys(topics) if len(t) > 3][:num_topics]
        if not topics:
            raise RuntimeError("El modelo no generó subtemas; prueba con un modelo instruct más capaz")
        for i, t in enumerate(topics):
            task.update(0.05 + 0.65 * i / len(topics), f"Tema {i + 1}/{len(topics)}: {t[:40]} · {len(raw)} pares")
            style = styles[i % len(styles)]
            got = _pairs(_gen(TOPIC_QA_PROMPT.format(domain=domain, topic=t, n=pairs_per_chunk, lang=language, style=STYLES[style]), temperature, helper=helper))
            failed += not got
            raw += [{**p, "_ctx": None, "_style": style, "_topic": t} for p in got]

    if not raw:
        raise RuntimeError("El modelo no produjo ningún par válido. Prueba con un modelo instruct más capaz.")
    task.update(0.72, f"Deduplicando {len(raw)} pares…")
    rows, n_dupes = _dedupe(raw, dedupe_threshold)
    n_rejected = 0
    if judge_min_score > 0:
        kept = []
        for i, r in enumerate(rows):
            task.update(0.75 + 0.24 * i / len(rows), f"Juez: {i + 1}/{len(rows)} · descartados {n_rejected}")
            sc = _judge(r["question"], r["answer"], r["_ctx"], helper)
            if sc is None or sc >= judge_min_score:
                kept.append({**r, "_score": sc})
            else:
                n_rejected += 1
        rows = kept
    if not rows:
        raise RuntimeError("El juez descartó todos los pares; baja el umbral o usa un modelo mejor")

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

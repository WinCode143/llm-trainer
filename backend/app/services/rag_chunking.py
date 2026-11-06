"""Troceado de documentos y búsqueda léxica BM25.

Partir bien es la mitad del trabajo de un RAG: un fragmento que corta una frase por la mitad o que
mete tres temas juntos hace que la búsqueda falle por mucho que el modelo de embeddings sea bueno.
Por eso hay plantillas por tipo de documento y troceo consciente de las secciones.

BM25 va aquí porque comparte el tokenizado con el troceo: ambos tienen que ver el texto igual.
"""
import csv
import io
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

# Índice BM25 por colección, que se rehace cuando cambian sus documentos
_bm25: dict[int, "BM25"] = {}


# ---------------- chunking ----------------
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
SEPS = ["\n\n", "\n", ". ", " "]


def split_sections(text: str, stack: list[tuple[int, str]] | None = None
                   ) -> tuple[list[tuple[str, str]], list[tuple[int, str]]]:
    """Parte Markdown por títulos → [(ruta de títulos, cuerpo)].

    `stack` es la pila de títulos abiertos [(nivel, título)] y permite continuar entre páginas.
    Un título cierra todos los de nivel igual o mayor (así dos h2 seguidos son hermanos aunque no haya h1).
    """
    stack = list(stack or [])
    sections, buf = [], []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append((" › ".join(t for _, t in stack), body))
        buf.clear()

    for line in text.splitlines():
        m = HEADING.match(line.strip())
        if m:
            flush()
            level = len(m.group(1))
            title = re.sub(r"[*_`]", "", m.group(2)).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            buf.append(line)
    flush()
    return sections, stack


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Troceo recursivo que respeta párrafos → líneas → frases → palabras, conservando los separadores."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    def split(t: str, level: int) -> list[str]:
        if len(t) <= size:
            return [t]
        if level >= len(SEPS):
            return [t[i:i + size] for i in range(0, len(t), size)]
        parts = re.split(f"(?<={re.escape(SEPS[level])})", t)
        out, cur = [], ""
        for p in parts:
            if len(cur) + len(p) <= size:
                cur += p
                continue
            if cur.strip():
                out.append(cur)
            if len(p) > size:
                out.extend(split(p, level + 1))
                cur = ""
            else:
                cur = p
        if cur.strip():
            out.append(cur)
        return out

    pieces = [p.strip() for p in split(text, 0) if p.strip()]
    if overlap <= 0 or len(pieces) < 2:
        return pieces
    # Solapamiento: cola del chunk anterior, cortada en límite de palabra
    result = [pieces[0]]
    for prev, cur in zip(pieces, pieces[1:]):
        tail = prev[-overlap:]
        sp = tail.find(" ")
        tail = tail[sp + 1:] if sp != -1 else tail
        result.append(f"{tail} {cur}".strip())
    return result


TEMPLATES = {
    "general": "Por secciones (títulos) y párrafos — documentos en general",
    "qa": "Preguntas frecuentes: cada pregunta con su respuesta",
    "table": "Tablas/CSV: una fila por fragmento con sus cabeceras",
    "laws": "Normativa: un fragmento por artículo («Artículo N»)",
    "page": "Una página/diapositiva por fragmento (presentaciones)",
    "one": "Documento entero en un fragmento (textos cortos)",
}
QA_Q = re.compile(r"^\s*(?:\*\*)?(?:P|Q|Pregunta|Question)\s*[:.\-]|^\s*(?:#+\s*)?[¿].+\?\s*(?:\*\*)?$", re.I)
ARTICLE = re.compile(r"^\s*(?:#+\s*)?(?:\*\*)?(Art[íi]culo|Art\.)\s+\d+", re.I | re.M)


def _qa_chunks(text: str) -> list[str]:
    out, cur = [], []
    for line in text.splitlines():
        if QA_Q.match(line) and cur:
            out.append("\n".join(cur).strip())
            cur = []
        cur.append(line)
    if cur:
        out.append("\n".join(cur).strip())
    return [c for c in out if c]


def _table_chunks(filename: str, raw_text: str) -> list[str]:
    import csv
    rows: list[list[str]] = []
    if filename.lower().endswith(".csv"):
        rows = [r for r in csv.reader(io.StringIO(raw_text)) if any(c.strip() for c in r)]
    else:  # tablas Markdown | a | b |
        for line in raw_text.splitlines():
            if line.strip().startswith("|") and not re.match(r"^\s*\|?\s*:?-{2,}", line):
                rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    if len(rows) < 2:
        return []
    head = rows[0]
    return ["; ".join(f"{h}: {v}" for h, v in zip(head, r) if v) for r in rows[1:]]


def make_chunks(filename: str, pages: list[tuple[str, dict]], size: int, overlap: int,
                template: str = "general") -> list[tuple[str, dict]]:
    chunks: list[tuple[str, dict]] = []

    def add(text: str, meta: dict, section: str = ""):
        chunks.append((text, {**meta, "source": filename, "section": section, "chunk": len(chunks)}))

    if template == "one":
        full = "\n\n".join(t for t, _ in pages).strip()
        if len(full) <= max(size * 6, 6000):
            add(full, pages[0][1] if pages else {})
            return chunks
        template = "general"
    if template == "page":
        for text, meta in pages:
            for c in (chunk_text(text, size * 3, overlap) if len(text) > size * 3 else [text.strip()]):
                if c:
                    add(c, meta)
        return chunks
    if template == "table":
        rows = _table_chunks(filename, "\n".join(t for t, _ in pages))
        if rows:
            for r in rows:
                add(r, {})
            return chunks
        template = "general"
    if template in ("qa", "laws"):
        full = "\n".join(t for t, _ in pages)
        if template == "qa":
            parts = _qa_chunks(full)
        else:
            idx = [m.start() for m in ARTICLE.finditer(full)]
            parts = [full[i:j].strip() for i, j in zip([0] + idx, idx + [len(full)])] if idx else []
        parts = [p_ for p_ in parts if p_.strip()]  # el texto puede empezar ya por un artículo/pregunta
        if len(parts) > 1:
            for p_ in parts:
                title = p_.strip().splitlines()[0].strip("#* ")[:120]
                for c in (chunk_text(p_, size * 2, overlap) if len(p_) > size * 2 else [p_]):
                    add(c, {}, title if template == "laws" else "")
            return chunks
        template = "general"
    stack: list = []
    for text, meta in pages:
        sections, stack = split_sections(text, stack)
        for section, body in sections:
            for c in chunk_text(body, size, overlap):
                add(c, meta, section)
    return chunks


def contextual(text: str, meta: dict, with_questions: bool = False) -> str:
    """Texto que se indexa: cabecera documento › sección + chunk.

    Las preguntas generadas NO se mezclan en el embedding del fragmento (diluían su vector y empeoraban la
    búsqueda): se indexan como vectores propios en la colección auxiliar de preguntas (multi-vector) y sólo
    se añaden al texto de BM25 (with_questions=True) para el emparejamiento léxico."""
    head = Path(meta.get("source", "")).stem if not str(meta.get("source", "")).startswith("http") else meta["source"]
    if meta.get("section"):
        head += f" › {meta['section']}"
    out = f"{head}\n{text}" if head else text
    if with_questions and meta.get("questions"):
        out += "\nPreguntas que responde:\n" + meta["questions"]
    return out




# ---------------- BM25 ----------------
def normalize_tokens(text: str) -> list[str]:
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return re.findall(r"[a-z0-9]+", t)


class BM25:
    def __init__(self, ids: list[str], texts: list[str], k1: float = 1.5, b: float = 0.75, sources: list[str] | None = None):
        self.ids = ids
        self.sources = sources or [""] * len(ids)
        self.docs = [Counter(normalize_tokens(t)) for t in texts]
        self.lens = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avg = float(self.lens.mean()) if len(self.lens) else 0.0
        df = Counter(tok for d in self.docs for tok in d)
        n = len(self.docs)
        self.idf = {tok: math.log(1 + (n - f + 0.5) / (f + 0.5)) for tok, f in df.items()}
        self.k1, self.b = k1, b

    def search(self, query: str, n: int, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        q = [t for t in normalize_tokens(query) if t in self.idf]
        if not q or not self.docs:
            return []
        scores = np.zeros(len(self.docs))
        norm = self.k1 * (1 - self.b + self.b * self.lens / (self.avg or 1))
        for tok in set(q):
            tf = np.array([d.get(tok, 0) for d in self.docs], dtype=float)
            scores += self.idf[tok] * tf * (self.k1 + 1) / (tf + norm)
        if allowed is not None:
            scores[[i for i, src in enumerate(self.sources) if src not in allowed]] = 0
        order = np.argsort(-scores)[:n]
        return [(self.ids[i], float(scores[i])) for i in order if scores[i] > 0]


def bm25_index(collection_id: int) -> BM25:
    from .rag import _col
    col = _col(collection_id)
    idx = _bm25.get(collection_id)
    if idx is None or len(idx.ids) != col.count():
        got = col.get(include=["documents", "metadatas"])
        idx = BM25(got["ids"], [contextual(d, m, with_questions=True) for d, m in zip(got["documents"], got["metadatas"])],
                   sources=[m.get("source", "") for m in got["metadatas"]])
        _bm25[collection_id] = idx
    return idx


def invalidate(collection_id: int) -> None:
    _bm25.pop(collection_id, None)



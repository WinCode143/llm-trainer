"""Document chunking and BM25 lexical search.

Splitting well is half the work in RAG: a chunk that cuts a sentence in half or
lumps three topics together makes search fail no matter how good the embedding model is.
That's why there are templates per document type and section-aware chunking.

BM25 lives here because it shares tokenization with chunking: both have to see the text the same way.
"""
import csv
import io
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

# BM25 index per collection, rebuilt when its documents change
_bm25: dict[int, "BM25"] = {}


# ---------------- chunking ----------------
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
SEPS = ["\n\n", "\n", ". ", " "]


def split_sections(text: str, stack: list[tuple[int, str]] | None = None
                   ) -> tuple[list[tuple[str, str]], list[tuple[int, str]]]:
    """Splits Markdown by headings → [(heading path, body)].

    `stack` is the stack of open headings [(level, title)] and lets it carry on across pages.
    A heading closes all those of the same or higher level (so two h2 in a row are siblings even without an h1).
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
    """Recursive chunking that respects paragraphs → lines → sentences → words, keeping the separators."""
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
    # Overlap: the tail of the previous chunk, cut at a word boundary
    result = [pieces[0]]
    for prev, cur in zip(pieces, pieces[1:]):
        tail = prev[-overlap:]
        sp = tail.find(" ")
        tail = tail[sp + 1:] if sp != -1 else tail
        result.append(f"{tail} {cur}".strip())
    return result


TEMPLATES = {
    "general": "By section (headings) and paragraph — documents in general",
    "qa": "FAQs: each question with its answer",
    "table": "Tables/CSV: one row per chunk with its headers",
    "laws": "Legal texts: one chunk per article (“Article N”)",
    "page": "One page/slide per chunk (presentations)",
    "one": "The whole document in one chunk (short texts)",
}
QA_Q = re.compile(r"^\s*(?:\*\*)?(?:P|Q|Pregunta|Question)\s*[:.\-]|^\s*(?:#+\s*)?[¿]?[A-Z¿][^?]*\?\s*(?:\*\*)?$", re.I)
ARTICLE = re.compile(r"^\s*(?:#+\s*)?(?:\*\*)?(Art[íi]culo|Article|Art\.|Section|Sec\.|§)\s*\d+", re.I | re.M)


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
    else:  # Markdown tables | a | b |
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
        parts = [p_ for p_ in parts if p_.strip()]  # the text may already start with an article/question
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
    """Text that gets indexed: a document › section header + the chunk.

    Generated questions are NOT mixed into the chunk's embedding (they diluted its vector and made
    search worse): they're indexed as their own vectors in the auxiliary questions collection (multi-vector) and only
    added to the BM25 text (with_questions=True) for lexical matching."""
    head = Path(meta.get("source", "")).stem if not str(meta.get("source", "")).startswith("http") else meta["source"]
    if meta.get("section"):
        head += f" › {meta['section']}"
    out = f"{head}\n{text}" if head else text
    if with_questions and meta.get("questions"):
        out += "\nQuestions it answers:\n" + meta["questions"]
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



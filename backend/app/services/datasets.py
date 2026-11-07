"""Importing and normalizing datasets.

Everything is stored as JSONL in one of three canonical formats:
  chat:       {"messages": [{"role": "system|user|assistant", "content": "..."}]}
  text:       {"text": "..."}
  preference: {"prompt": "...", "chosen": "...", "rejected": "..."}
"""
import csv
import io
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

from ..config import settings

ROLE_MAP = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "bot": "assistant",
            "model": "assistant", "system": "system"}


def slugify(name: str) -> str:
    s = re.sub(r"[^\w\-]+", "-", name.strip().lower()).strip("-")
    return s or "dataset"


def _read_records(filename: str, raw: bytes) -> list[dict[str, Any]]:
    ext = Path(filename).suffix.lower()
    text = raw.decode("utf-8-sig", errors="replace")
    if ext == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if ext == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            # {"data": [...]} or another key holding a list
            for v in data.values():
                if isinstance(v, list):
                    return v
            return [data]
        return data
    if ext == ".csv":
        return list(csv.DictReader(io.StringIO(text)))
    if ext == ".parquet":
        import pandas as pd
        return pd.read_parquet(io.BytesIO(raw)).to_dict(orient="records")
    if ext in (".txt", ".md"):
        # Plain text: one example per paragraph (separated by blank lines)
        return [{"text": p.strip()} for p in re.split(r"\n\s*\n", text) if p.strip()]
    raise ValueError(f"Unsupported extension: {ext}")


def detect_format(rec: dict[str, Any]) -> str:
    keys = set(rec.keys())
    if {"chosen", "rejected"} <= keys:
        return "preference"
    if "messages" in keys or "conversations" in keys:
        return "chat"
    if keys & {"instruction", "question", "prompt", "input"} and keys & {"output", "answer", "response", "completion"}:
        return "chat"
    if "text" in keys:
        return "text"
    raise ValueError(f"The record's format isn't recognized. Keys: {sorted(keys)}")


def _to_chat(rec: dict[str, Any], system_prompt: str | None) -> dict | None:
    msgs: list[dict[str, str]]
    if "messages" in rec or "conversations" in rec:
        raw = rec.get("messages") or rec.get("conversations") or []
        if isinstance(raw, str):
            raw = json.loads(raw)
        msgs = []
        for m in raw:
            role = ROLE_MAP.get(str(m.get("role", m.get("from", ""))).lower())
            content = m.get("content", m.get("value", ""))
            if role and content is not None:
                msgs.append({"role": role, "content": str(content)})
    else:
        instr = rec.get("instruction") or rec.get("question") or rec.get("prompt") or ""
        inp = rec.get("input") or ""
        out = rec.get("output") or rec.get("answer") or rec.get("response") or rec.get("completion") or ""
        user = f"{instr}\n\n{inp}".strip() if inp and instr else (instr or inp)
        if not user or not out:
            return None
        msgs = [{"role": "user", "content": str(user)}, {"role": "assistant", "content": str(out)}]
        if rec.get("system"):
            msgs.insert(0, {"role": "system", "content": str(rec["system"])})
    if system_prompt and not any(m["role"] == "system" for m in msgs):
        msgs.insert(0, {"role": "system", "content": system_prompt})
    if not any(m["role"] == "assistant" for m in msgs):
        return None
    return {"messages": msgs}


def _flatten(v: Any) -> str:
    # chosen/rejected may come as a list of messages
    if isinstance(v, list):
        return next((str(m.get("content", "")) for m in reversed(v) if m.get("role") == "assistant"), "")
    return str(v)


def normalize(records: Iterable[dict[str, Any]], fmt: str | None = None, system_prompt: str | None = None) -> tuple[str, list[dict]]:
    records = list(records)
    if not records:
        raise ValueError("The dataset is empty")
    fmt = fmt or detect_format(records[0])
    out: list[dict] = []
    for rec in records:
        if fmt == "chat":
            r = _to_chat(rec, system_prompt)
        elif fmt == "text":
            t = rec.get("text")
            r = {"text": str(t)} if t else None
        elif fmt == "preference":
            prompt = rec.get("prompt") or rec.get("question") or rec.get("instruction") or ""
            if isinstance(prompt, list):
                prompt = next((m.get("content", "") for m in prompt if m.get("role") == "user"), "")
            r = {"prompt": str(prompt), "chosen": _flatten(rec["chosen"]), "rejected": _flatten(rec["rejected"])}
        else:
            raise ValueError(f"Unknown format: {fmt}")
        if r:
            out.append(r)
    if not out:
        raise ValueError("No valid record left after normalization")
    return fmt, out


def write_jsonl(name: str, rows: list[dict]) -> Path:
    path = settings.datasets_dir / f"{slugify(name)}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def import_file(filename: str, raw: bytes, fmt: str | None, system_prompt: str | None) -> tuple[str, list[dict]]:
    return normalize(_read_records(filename, raw), fmt, system_prompt)


def import_hf(repo: str, split: str, subset: str | None, limit: int | None, fmt: str | None,
              system_prompt: str | None) -> tuple[str, list[dict]]:
    from datasets import load_dataset

    ds = load_dataset(repo, subset, split=split, token=settings.hf_token)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return normalize(ds, fmt, system_prompt)


def read_rows(path: str, offset: int = 0, limit: int = 50) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            if len(rows) >= limit:
                break
            rows.append(json.loads(line))
    return rows


def read_all(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def stats(path: str, fmt: str) -> dict:
    rows = read_all(path)
    lengths = []
    turns = []
    for r in rows:
        if fmt == "chat":
            lengths.append(sum(len(m["content"]) for m in r["messages"]))
            turns.append(len(r["messages"]))
        elif fmt == "text":
            lengths.append(len(r["text"]))
        else:
            lengths.append(len(r["prompt"]) + len(r["chosen"]))
    lengths.sort()
    n = len(lengths)
    return {
        "rows": n,
        "chars_avg": round(sum(lengths) / n) if n else 0,
        "chars_p50": lengths[n // 2] if n else 0,
        "chars_p95": lengths[int(n * 0.95)] if n else 0,
        "chars_max": lengths[-1] if n else 0,
        "approx_tokens_total": sum(lengths) // 4,
        "turns_avg": round(sum(turns) / len(turns), 2) if turns else None,
    }


def split(rows: list[dict], eval_ratio: float, seed: int = 42) -> tuple[list[dict], list[dict]]:
    rows = rows[:]
    random.Random(seed).shuffle(rows)
    k = max(1, int(len(rows) * eval_ratio)) if eval_ratio > 0 else 0
    return rows[k:], rows[:k]

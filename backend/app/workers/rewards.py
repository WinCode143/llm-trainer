"""Reward functions for GRPO (TRL's signature: f(prompts, completions, **columns) -> list[float])."""
import re
import unicodedata
from collections import Counter

THINK = re.compile(r"<think>(.*?)</think>\s*(.*)", re.S)


def _text(c) -> str:
    # conversational completions: [{"role": "assistant", "content": ...}]
    return c[0]["content"] if isinstance(c, list) else str(c)


def _answer_part(text: str) -> str:
    m = THINK.search(text)
    return m.group(2) if m else text


def _norm(s: str) -> list[str]:
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.findall(r"[a-z0-9]+", s)


def _f1(pred: str, ref: str) -> float:
    p, r = _norm(pred), _norm(ref)
    if not p or not r:
        return 0.0
    common = sum((Counter(p) & Counter(r)).values())
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def correctness(prompts, completions, answer, **_):
    """0–2: word F1 against the reference answer (+1 if it contains it literally)."""
    out = []
    for c, ref in zip(completions, answer):
        ans = _answer_part(_text(c))
        bonus = 1.0 if " ".join(_norm(ref)) and " ".join(_norm(ref)) in " ".join(_norm(ans)) else 0.0
        out.append(round(_f1(ans, ref) + bonus, 4))
    return out


def reasoning_format(prompts, completions, **_):
    """1 if it reasons inside <think>…</think> and then gives a non-empty answer."""
    out = []
    for c in completions:
        m = THINK.search(_text(c))
        out.append(1.0 if m and len(m.group(1).strip()) > 10 and m.group(2).strip() else 0.0)
    return out


def make_length(target_words: int):
    def length(prompts, completions, **_):
        """1 up to `target` words; decays linearly to 0 at twice that."""
        out = []
        for c in completions:
            n = len(_answer_part(_text(c)).split())
            out.append(1.0 if n <= target_words else max(0.0, 1 - (n - target_words) / target_words))
        return out
    length.__name__ = "length"
    return length


def make_regex(pattern: str):
    rx = re.compile(pattern, re.S)

    def regex(prompts, completions, **_):
        """1 if the answer matches the user's regular expression."""
        return [1.0 if rx.search(_text(c)) else 0.0 for c in completions]
    regex.__name__ = "regex"
    return regex


def build(cfg: dict) -> list:
    names = cfg.get("rewards") or ["correctness"]
    fns = []
    for n in names:
        if n == "correctness":
            fns.append(correctness)
        elif n == "format":
            fns.append(reasoning_format)
        elif n == "length":
            fns.append(make_length(int(cfg.get("reward_length_target", 80))))
        elif n == "regex" and cfg.get("reward_regex"):
            fns.append(make_regex(cfg["reward_regex"]))
    if not fns:
        raise ValueError("GRPO needs at least one reward function")
    return fns

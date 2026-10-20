"""Sentence-decoding metrics shared by every Phase 4 model (one implementation, same normalization).

WER follows BIT's eval_llm.py exactly: strip everything except letters, hyphens, apostrophes and
spaces, lowercase, then word-level edit distance summed over trials / total reference words.
"""

from __future__ import annotations

import re

import editdistance


def normalize(sentence: str) -> str:
    """BIT's remove_punctuation (eval_llm.py)."""
    s = re.sub(r"[^a-zA-Z\- ']", "", sentence or "")
    s = s.replace("- ", " ").lower().replace("--", "").replace(" '", "'").strip()
    return " ".join(w for w in s.split() if w)


def word_errors(ref: str, hyp: str) -> tuple[int, int]:
    r, h = normalize(ref).split(), normalize(hyp).split()
    return editdistance.eval(r, h), len(r)


def wer(refs: list[str], hyps: list[str]) -> float:
    """Aggregate WER (total edits / total reference words), as BIT reports it."""
    errs = [word_errors(r, h) for r, h in zip(refs, hyps, strict=True)]
    return sum(e for e, _ in errs) / max(sum(n for _, n in errs), 1)

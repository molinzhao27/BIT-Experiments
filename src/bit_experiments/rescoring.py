"""N-best rescoring SIDE TEST (docs/RESCORING.md): candidate files, decoder and OPT-6.7B scores, selection rules and
their metrics. Never a headline number: docs/DESIGN.md keeps LM rescoring out of every headline result.

A Stage 2 decoder writes N candidates per trial (scripts/nbest_decode.py: its greedy decode plus K nucleus samples,
duplicates kept). Per candidate: the decoder's own log-probability, the cosine between the trial's neural embedding
and the candidate's text embedding under the run's own contrastive head (BIT's ModalityAligner), optionally a diffusion
pseudo-log-likelihood; then OPT-6.7B's log-likelihood of the candidate text (scripts/rescore_nbest.py opt). A rule
picks one candidate per trial; rules with weights are tuned on BIT-val only (corpus WER) and applied unchanged to
BIT-test and to the shuffled-input nulls. Candidates are compared after BIT's normalization (textmetrics.normalize):
two samples that normalize alike are one candidate with a count of two.

numpy / pandas at import; torch is imported only by the functions that need it (decoder scores, OPT).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from bit_experiments.textmetrics import normalize, word_errors

FEATURES = ("dec_sum", "dec_mean", "opt_sum", "opt_mean", "n_words", "contrastive", "log_count", "mbr_risk", "pll_sum")


# ---------------------------------------------------------------- candidate files (resume-safe)

class CandidateFile:
    """candidates_<split>.jsonl: one JSON line per candidate, a trial's N lines written in one write. A trial is
    complete when it has as many lines as its rows' `n`; on opening, incomplete trials and a torn last line (a crash
    mid-write) are dropped, so a rerun redoes exactly the trials that were not finished."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.rows = self._load()
        self.done = {r["trial"] for r in self.rows}

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows, torn = [], False
        for line in self.path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                torn = True
        count: dict[int, int] = {}
        for r in rows:
            count[r["trial"]] = count.get(r["trial"], 0) + 1
        keep = [r for r in rows if count[r["trial"]] == r["n"]]
        if torn or len(keep) != len(rows):
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(r) + "\n" for r in keep))
            os.replace(tmp, self.path)
        return keep

    def append(self, rows: list[dict]) -> None:
        if not rows or any(r["trial"] != rows[0]["trial"] or r["n"] != len(rows) for r in rows):
            raise ValueError("append one whole trial at a time (every row with that trial and n = its row count)")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write("".join(json.dumps(r) + "\n" for r in rows))
            f.flush()
            os.fsync(f.fileno())
        self.rows += rows
        self.done.add(rows[0]["trial"])


def read_candidates(path: str | Path) -> pd.DataFrame:
    """A candidate file as a frame: one row per candidate, `scores` flattened into columns, `norm` = normalized text."""
    rows = CandidateFile(path).rows
    if not rows:
        raise FileNotFoundError(f"{path}: no candidates")
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "scores"} | r.get("scores", {}) for r in rows])
    df["norm"] = df.candidate.map(normalize)
    return df


# ---------------------------------------------------------------- decoder scores (AR)

def ar_sequence_scores(logits, ids, eos_id: int) -> list[dict]:
    """Per generated row: summed and mean log-probability of its tokens under the decoder (log-softmax of the raw,
    untempered logits HF generate returns with output_logits), up to and including the first EOS; the rest is padding.
    logits [B, T, V], ids [B, T] (the new tokens only)."""
    import torch

    lp = torch.log_softmax(logits.float(), -1).gather(-1, ids[..., None]).squeeze(-1)  # [B, T]
    is_eos = ids == eos_id
    after = (is_eos.cumsum(-1) - is_eos.long()) > 0  # strictly after the first EOS
    keep = ~after
    out = []
    for b in range(ids.shape[0]):
        n = int(keep[b].sum())
        s = float(lp[b][keep[b]].sum())
        out.append({"dec_sum": s, "dec_mean": s / max(n, 1), "dec_n_tokens": n, "ended_eos": bool(is_eos[b].any())})
    return out


# ---------------------------------------------------------------- OPT

def opt_text(candidate: str, cfg: dict) -> str:
    return normalize(candidate) if cfg.get("text", "normalized") == "normalized" else candidate.strip()


def opt_logprobs(texts: list[str], model, tokenizer, device, batch_size: int = 32) -> list[dict]:
    """OPT log-likelihood per text, as Card et al. 2024 rescore with OPT: the tokenizer prepends </s> (OPT's BOS) and
    every later token is scored given the ones before it (right padding, attention mask; positions are absolute, so
    padding changes nothing). Also log p(</s> | text) (`opt_eos`, the end of the document). An empty text has no
    scored token: sum 0."""
    import torch

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    out = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        if not bool((ids[:, 0] == bos).all()):
            raise ValueError("expected the tokenizer to prepend OPT's BOS (</s>) to every text")
        with torch.no_grad():
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits.float()
        lsm = torch.log_softmax(logits, -1).cpu()
        tok_lp = lsm[:, :-1].gather(-1, ids[:, 1:, None]).squeeze(-1) * mask[:, 1:]
        n = mask.sum(1) - 1
        last = lsm[torch.arange(len(chunk)), n]  # the position after the last real token
        for b in range(len(chunk)):
            out.append({"opt_sum": float(tok_lp[b].sum()), "opt_n_tokens": int(n[b]), "opt_eos": float(last[b, eos])})
    return out


class ScoreCache:
    """An append-only JSONL cache {key: ..., <scores>} (OPT scores of a text are the same for every run)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.data[r.pop("key")] = r

    def missing(self, keys) -> list[str]:
        return sorted({k for k in keys if k not in self.data})

    def add(self, items: dict[str, dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write("".join(json.dumps({"key": k} | v) + "\n" for k, v in items.items()))
        self.data.update(items)


# ---------------------------------------------------------------- per-trial candidate tables

def mbr_risk(norms: list[str], counts: np.ndarray) -> np.ndarray:
    """Per unique candidate: its mean word edit distance (BIT's normalization) to the other samples of the trial,
    weighted by how often each was sampled. Lower = closer to the consensus."""
    import editdistance

    words = [s.split() for s in norms]
    n = int(counts.sum())
    d = np.array([[editdistance.eval(a, b) for b in words] for a in words], float)
    return (d @ counts) / max(n - 1, 1)


def trial_table(cands: pd.DataFrame, opt: dict[str, dict] | None = None, opt_cfg: dict | None = None,
                bertscore: dict | None = None) -> pd.DataFrame:
    """Unique candidates per trial with every feature. Duplicates (same normalized text) merge: count = their number;
    decoder score = the best path's; is_greedy if any of them was the greedy decode. opt: ScoreCache.data keyed by
    opt_text; bertscore: {(normalized reference, normalized candidate): F1}."""
    opt_cfg = opt_cfg or {}
    cands = cands.assign(order=np.arange(len(cands)))
    agg = {"candidate": "first", "order": "min", "is_greedy": "any", "reference": "first", "k": "size"}
    for col in ("dec_sum", "dec_mean", "contrastive", "pll_sum"):
        if col in cands and cands[col].notna().any():
            agg[col] = "max"
    if "seen_in_train" in cands:
        agg["seen_in_train"] = "first"
    t = cands.groupby(["trial", "norm"], sort=False).agg(agg).rename(columns={"k": "count"}).reset_index()
    t = t.sort_values(["trial", "order"]).reset_index(drop=True)
    n_trial = t.groupby("trial")["count"].transform("sum")
    t["n_samples"] = n_trial
    t["log_count"] = np.log(t["count"])
    t["agreement"] = t["count"] / n_trial
    t["n_words"] = t.norm.str.split().str.len().fillna(0).astype(int)
    t["mbr_risk"] = 0.0
    for _, idx in t.groupby("trial").indices.items():
        t.loc[t.index[idx], "mbr_risk"] = mbr_risk(t.norm.iloc[idx].tolist(), t["count"].iloc[idx].to_numpy(float))
    errs = [word_errors(r, h) for r, h in zip(t.reference, t.norm)]
    t["errors"] = [e for e, _ in errs]
    t["ref_words"] = [n for _, n in errs]
    if opt is not None:
        keys = [opt_text(c, opt_cfg) for c in t.candidate]
        miss = [k for k in keys if k not in opt]
        if miss:
            raise KeyError(f"{len(miss)} candidates have no OPT score (run scripts/rescore_nbest.py opt), e.g. {miss[0]!r}")
        s = pd.DataFrame([opt[k] for k in keys], index=t.index)
        t["opt_sum"] = s.opt_sum + (s.opt_eos if opt_cfg.get("eos") else 0.0)
        t["opt_tokens"] = s.opt_n_tokens + (1 if opt_cfg.get("eos") else 0)
        t["opt_mean"] = t.opt_sum / t.opt_tokens.clip(lower=1)
    if bertscore is not None:
        t["bs_f1"] = [bertscore.get((normalize(r), c), np.nan) for r, c in zip(t.reference, t.norm)]
    return t


# ---------------------------------------------------------------- rules

def rule_weights(rule: str, params: dict | None = None) -> dict[str, float]:
    """A rule's linear score over the features; the selected candidate maximizes it (ties: the greedy decode, then
    the most sampled, then the first)."""
    p = params or {}
    if rule == "greedy":
        return {"is_greedy": 1.0}
    if rule == "decoder":
        return {"dec_sum": 1.0}
    if rule == "opt":
        return {"opt_sum": 1.0}
    if rule == "opt_norm":
        return {"opt_mean": 1.0}
    if rule == "dec_opt":
        return {"dec_sum": 1.0, "opt_sum": p.get("alpha", 0.0), "n_words": p.get("beta", 0.0)}
    if rule == "mbr":
        return {"mbr_risk": -1.0, "log_count": 1e-9}  # consensus; among equal risk, the most sampled
    if rule == "contrastive":
        return {"contrastive": 1.0}
    if rule == "dec_contrastive":
        return {"dec_sum": 1.0, "contrastive": p.get("gamma", 0.0)}
    if rule == "all":
        return dict(p)
    raise ValueError(f"unknown rule {rule!r}")


class Selector:
    """Vectorized argmax per trial over a trial_table (fast enough to tune weights by search)."""

    def __init__(self, table: pd.DataFrame):
        self.t = table.reset_index(drop=True)
        self.group = self.t.trial.to_numpy()
        self.trials, self.first = np.unique(self.group, return_index=True)
        if not (np.diff(self.group) >= 0).all():
            raise ValueError("table must be sorted by trial")
        self.tie = (self.t.is_greedy.to_numpy(float) * 1e6 + self.t["count"].to_numpy(float) * 1e3
                    - self.t.order.to_numpy(float) * 1e-9)
        self.errors = self.t.errors.to_numpy(float)
        self.words = self.t.groupby("trial").ref_words.first().to_numpy(float)

    def available(self, weights: dict[str, float]) -> bool:
        return all(k in self.t and self.t[k].notna().all() for k, w in weights.items() if w != 0)

    def select(self, weights: dict[str, float]) -> np.ndarray:
        """Row index of the selected candidate per trial (in trial order)."""
        score = np.zeros(len(self.t))
        for k, w in weights.items():
            if w != 0:
                score = score + w * self.t[k].to_numpy(float)
        order = np.lexsort((-self.tie, -score, self.group))
        return order[np.searchsorted(self.group[order], self.trials)]

    def wer(self, weights: dict[str, float]) -> float:
        return float(self.errors[self.select(weights)].sum() / max(self.words.sum(), 1))


def tune(rule: str, val: Selector, cfg: dict) -> tuple[dict, float]:
    """The rule's parameters with the lowest BIT-val corpus WER (grid order breaks ties: the first, i.e. the smallest
    weights). Only the val table is ever seen here. Returns (params, val WER)."""
    grid = cfg["grid"]
    if rule == "dec_opt":
        cands = [{"alpha": a, "beta": b} for a in grid["alpha"] for b in grid["beta"]]
    elif rule == "dec_contrastive":
        cands = [{"gamma": g} for g in grid["gamma"]]
    elif rule == "all":
        return tune_all(val, cfg["all"])
    else:
        return {}, val.wer(rule_weights(rule))
    scored = [(val.wer(rule_weights(rule, c)), i, c) for i, c in enumerate(cands)]
    best = min(scored, key=lambda x: (x[0], x[1]))
    return best[2], best[0]


def tune_all(val: Selector, cfg: dict) -> tuple[dict, float]:
    """Every feature at once: directions on the unit sphere of the standardized features (seeded), the axes and the
    greedy-equivalent start, then coordinate refinement. The argmax is scale-invariant, so only directions count."""
    feats = [f for f in cfg["features"] if f in val.t and val.t[f].notna().all()]
    sd = np.array([max(float(val.t[f].std()), 1e-9) for f in feats])
    rng = np.random.default_rng(cfg.get("seed", 0))
    dirs = [np.eye(len(feats))[i] for i in range(len(feats))]
    dirs += list(rng.standard_normal((int(cfg.get("n_random", 3000)), len(feats))))

    def weights(v):
        return {f: float(x) for f, x in zip(feats, v / sd)}

    best_v, best = None, math.inf
    for v in dirs:
        w = val.wer(weights(v))
        if w < best - 1e-12:
            best_v, best = np.array(v, float), w
    for _ in range(3):  # coordinate refinement
        improved = False
        for j in range(len(feats)):
            for step in (-1.0, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1.0):
                v = best_v.copy()
                v[j] += step * max(abs(v[j]), 0.1)
                w = val.wer(weights(v))
                if w < best - 1e-12:
                    best_v, best, improved = v, w, True
        if not improved:
            break
    return weights(best_v), best


# ---------------------------------------------------------------- metrics

def bootstrap_indices(n: int, n_resamples: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))


def paired_bootstrap(err: np.ndarray, base: np.ndarray, words: np.ndarray, idx: np.ndarray) -> dict:
    """Corpus WER(rule) - WER(base) over trial resamples `idx` [R, n]: point, 95% percentile CI, and the one-sided
    share of resamples where the rule is not better (delta >= 0)."""
    d = (err[idx].sum(1) - base[idx].sum(1)) / np.maximum(words[idx].sum(1), 1)
    point = float((err.sum() - base.sum()) / max(words.sum(), 1))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": point, "ci_low": float(lo), "ci_high": float(hi), "p_not_better": float((d >= 0).mean())}


def coverage_wer(errors: np.ndarray, words: np.ndarray, confidence: np.ndarray, coverages) -> dict:
    """Selective prediction: corpus WER over the most confident fraction of trials (abstain on the rest; ties keep
    trial order)."""
    order = np.argsort(-confidence, kind="stable")
    out = {}
    for c in coverages:
        k = max(1, int(round(c * len(order))))
        keep = order[:k]
        out[f"{c:g}"] = float(errors[keep].sum() / max(words[keep].sum(), 1))
    return out

#!/usr/bin/env python
"""Novel test words: does a decoder get reference words that never occur in its training sentences? (CPU only.)

    .venv/bin/python scripts/novel_word_analysis.py [--train-sentences FILE] [--train-source ZIP_OR_DIR]
        [--runs RUN ...] [--n-boot 10000] [--seed 0] [--no-phonetic] [--out results/stage2/novel_words]

A cheap stand-in for a held-out-phoneme test. The encoder and the LLM adapter never saw neural data paired with these
words, so a decoder can only produce one from sub-word structure in the signal (or from the LLM's prior via context).

Training sentences. Stage 2 trains on every `sentence_label` of every session's data_train.hdf5 (Brain-to-Text '25,
T15; label fraction 1 in every run here). They are read as BIT's loader reads them (punctuation except apostrophes
removed, lower-cased) and then normalized as the evaluator does (textmetrics.normalize). Source: --train-source, the
release zip or an extracted hdf5_data_final/ directory (attributes only, no neural data), or --train-sentences, one
raw sentence_label per line. --train-sentences is written as a cache after the first read from --train-source.
Checks: every run's seen_in_train column equals "the normalized reference is a training sentence", and all runs share
the same references.

1. Vocabulary. A reference token is novel if its word never occurs in the training sentences, otherwise in-vocabulary
   (in-vocab), with its training frequency = occurrences over all training trials.
2. Alignment. Unit-cost word Levenshtein, the WER's edit distance (checked against textmetrics.word_errors per trial).
   Among the alignments with that fewest edits, one with the most correct words (a plain backtrace can turn a hit into
   two substitutions: ref "a b", hyp "b c" is del a, ok b, ins c, not sub, sub). Each reference word is correct (ok),
   substituted (sub) or deleted (del); recall = ok / reference tokens of that kind. For a substituted novel word:
   is the decoded word in the training vocabulary (did the decoder fall back to a known word)? The same share for
   substituted in-vocab words is the comparison.
3. Paired cluster bootstrap over trials (--n-boot resamples, --seed; one resampling for every run and measure, so
   every contrast is paired). Recall is a ratio of sums within each resample. 95% percentile CIs; p is two-sided
   (2 x the smaller tail share of the resampled differences at 0). Contrasts: CONTRASTS and the difference-in-
   differences DID.
4. Frequency and length. Novel words are rarer and longer, so in-vocab recall is also given by training frequency
   (1-2, 3-10, > 10) and against matched in-vocab words. Seen sentences contain no novel word and are partly
   memorized, so the frequency bins and the matched pools use unseen-sentence trials only. Matched = direct
   standardization: in-vocab recall per cell, weighted by the novel tokens' share of each cell (cells: letters
   <= 3, 4, ..., 9, >= 10; English Zipf frequency (wordfreq) < 3, 3-4, 4-5, >= 5), within each resample; novel tokens
   in cells without in-vocab tokens drop out (coverage reported). gap = novel recall - matched in-vocab recall over the
   same cells.
5. Sound (optional, --no-phonetic skips it). For substituted words, the phoneme similarity of the decoded word to the
   reference (scripts/phonetic_error_analysis.py, read-only reuse: CMU dictionary, length-matched null from the run's
   own substitutes, excess = observed - null), novel vs in-vocab reference words; pairs with a word outside the
   dictionary drop out (novel words more often).

Outputs: <out>.json and <out>.md.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import re
import string
import sys
import tempfile
import zipfile
import zlib
from collections import Counter
from pathlib import Path

import editdistance
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bit_experiments.textmetrics import normalize, word_errors  # noqa: E402

RESULTS = ROOT / "results" / "stage2"
PHONETIC_SCRIPT = ROOT / "scripts" / "phonetic_error_analysis.py"
DEFAULT_SOURCE = ROOT / "data" / "t15_copyTask_neuralData.zip"
DEFAULT_CACHE = Path(tempfile.gettempdir()) / "bit-novel-words" / "train_sentences.txt"  # outside the repo

RUNS = {  # run -> label; runs without test decodes are skipped
    "B_ssl_s42": "B AR (Aero)",
    "B_ssl_aero-1-audio-diffusion_s42": "B diffusion (Aero)",
    "A_ssl_s42": "A AR (Aero)",
    "A_ssl_aero-1-audio-diffusion_s42": "A diffusion (Aero)",
    "C_ssl_l14_b1024_s42": "C AR (Aero)",
    "C_ssl_aero-1-audio-diffusion_s42": "C diffusion (Aero)",
    "B_ssl_qwen25-1.5b-instruct_s42": "B AR (Instruct)",
    "B_ssl_qwen25-1.5b-instruct-diffusion_s42": "B diffusion (Instruct)",
    "B_ssl_fast-dllm-v2-1.5b_s42": "B Fast-dLLM v2",
}
CONTRASTS = [  # label, a, b: a - b
    ("B diff - B AR", "B_ssl_aero-1-audio-diffusion_s42", "B_ssl_s42"),
    ("A diff - A AR", "A_ssl_aero-1-audio-diffusion_s42", "A_ssl_s42"),
    ("C diff - C AR", "C_ssl_aero-1-audio-diffusion_s42", "C_ssl_l14_b1024_s42"),
    ("C AR - B AR", "C_ssl_l14_b1024_s42", "B_ssl_s42"),
    ("C diff - B diff", "C_ssl_aero-1-audio-diffusion_s42", "B_ssl_aero-1-audio-diffusion_s42"),
    ("A AR - B AR", "A_ssl_s42", "B_ssl_s42"),
    ("A diff - B diff", "A_ssl_aero-1-audio-diffusion_s42", "B_ssl_aero-1-audio-diffusion_s42"),
    ("Instruct diff - Instruct AR", "B_ssl_qwen25-1.5b-instruct-diffusion_s42", "B_ssl_qwen25-1.5b-instruct_s42"),
]
DID = [  # label, (a1, b1), (a2, b2): (a1 - b1) - (a2 - b2)
    ("(A diff - A AR) - (B diff - B AR)", ("A_ssl_aero-1-audio-diffusion_s42", "A_ssl_s42"),
     ("B_ssl_aero-1-audio-diffusion_s42", "B_ssl_s42")),
    ("(C diff - C AR) - (B diff - B AR)", ("C_ssl_aero-1-audio-diffusion_s42", "C_ssl_l14_b1024_s42"),
     ("B_ssl_aero-1-audio-diffusion_s42", "B_ssl_s42")),
]
FREQ_BINS = [("1-2", 1, 2), ("3-10", 3, 10), (">10", 11, None)]
LEN_EDGES = [4, 5, 6, 7, 8, 9, 10]  # letters: <= 3 | 4 | ... | 9 | >= 10
ZIPF_EDGES = [3.0, 4.0, 5.0]
MEASURES = ["novel", "inv", "inv_unseen", "gap_len", "gap_len_zipf", "gap_rare_len"]  # bootstrapped per run


# ---------------------------------------------------------------- training sentences

def bit_sentence(label: str) -> str:
    """BIT's processing of a sentence_label (data/brandman_2024_text/prepare_data.py)."""
    return label.translate(str.maketrans("", "", string.punctuation.replace("'", ""))).lower().strip()


def _session_files(source: Path):
    """(session, opener) for every data_train.hdf5 under a release zip or an extracted directory."""
    import h5py

    if source.is_file():
        zf = zipfile.ZipFile(source)
        found = {}
        for n in zf.namelist():
            m = re.search(r"hdf5_data_final/([^/]+)/data_train\.hdf5$", n)
            if m:
                found[m.group(1)] = n
        return [(s, lambda n=found[s]: h5py.File(io.BytesIO(zf.read(n)), "r")) for s in sorted(found)]
    base = source / "hdf5_data_final" if (source / "hdf5_data_final").is_dir() else source
    return [(d.name, lambda p=d / "data_train.hdf5": h5py.File(p, "r"))
            for d in sorted(base.iterdir()) if (d / "data_train.hdf5").is_file()]


def read_train_labels(source: Path) -> tuple[list[str], int]:
    """Every training trial's raw sentence_label (sessions sorted, trials in file order), and the session count."""
    labels, files = [], _session_files(source)
    for _, opener in files:
        with opener() as f:
            for key in f.keys():
                v = f[key].attrs["sentence_label"]
                labels.append(v.decode() if isinstance(v, bytes) else str(v))
    if not labels:
        raise SystemExit(f"no data_train.hdf5 under {source}")
    return labels, len(files)


def load_train_labels(cache: Path | None, source: Path | None) -> tuple[list[str], dict]:
    side = cache.with_suffix(".source.json") if cache is not None else None
    if cache is not None and cache.is_file():
        labels = cache.read_text().split("\n")
        labels = labels[:-1] if labels and labels[-1] == "" else labels
        prov = {"read_from": "cached sentence list", "n_sessions": None}
        if side.is_file():
            prov.update(json.loads(side.read_text()), cached=True)
    else:
        if source is None or not source.exists():
            raise SystemExit("no training sentences: give --train-sentences FILE or --train-source ZIP_OR_DIR")
        labels, n_sessions = read_train_labels(source)
        prov = {"read_from": f"{source.name} (data_train.hdf5 sentence_label attributes)", "n_sessions": n_sessions}
        if cache is not None:
            if any("\n" in s for s in labels):
                raise SystemExit("a sentence_label contains a newline; cannot cache one per line")
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text("\n".join(labels) + "\n")
            side.write_text(json.dumps(prov))
    prov["sha256_labels"] = hashlib.sha256("\n".join(labels).encode()).hexdigest()
    prov["n_train_trials"] = len(labels)
    return labels, prov


def train_counts(labels: list[str]) -> tuple[Counter, set[str]]:
    """Word counts over all training trials and the set of normalized training sentences."""
    norm = [normalize(bit_sentence(s)) for s in labels]
    return Counter(w for s in norm for w in s.split()), set(norm)


# ---------------------------------------------------------------- alignment and per-word records

def align(ref: list[str], hyp: list[str]) -> list[tuple[str, int | None, int | None]]:
    """Fewest-edit word alignment with the most hits among those: [(op, ref index, hyp index)], op in
    ok / sub / del / ins. Cost = edits * big - hits, so edits come first."""
    n, m = len(ref), len(hyp)
    big = n + m + 1
    d = np.zeros((n + 1, m + 1), dtype=np.int64)
    d[:, 0] = np.arange(n + 1) * big
    d[0, :] = np.arange(m + 1) * big
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = d[i - 1, j - 1] + (-1 if ref[i - 1] == hyp[j - 1] else big)
            d[i, j] = min(diag, d[i - 1, j] + big, d[i, j - 1] + big)
    ops, i, j = [], n, m
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (-1 if ref[i - 1] == hyp[j - 1] else big):
            ops.append(("ok" if ref[i - 1] == hyp[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and d[i, j] == d[i - 1, j] + big:
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("ins", None, j - 1))
            j -= 1
    return ops[::-1]


def letters(w: str) -> int:
    return sum(c.isalpha() for c in w)


def cell(length: int, zipf: float) -> tuple[int, int]:
    """(letters bin, Zipf bin) indices."""
    return int(np.searchsorted(LEN_EDGES, length, side="right")), int(np.searchsorted(ZIPF_EDGES, zipf, side="right"))


INFLECTIONS = ("s", "es", "ed", "d", "ing", "'s")  # inflections only: -er / -ly would pair corn / corner


def inflection_of_train(w: str, vocab) -> bool:
    """w is an inflection of a training word (foxes / fox, raiding / raid, running / run, tried / try) or its base
    (deliver / delivered). Heuristic; for the novel-word split only."""
    for suf in INFLECTIONS:
        if (w + suf) in vocab or (w.endswith("e") and (w[:-1] + suf) in vocab):
            return True
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            stem = w[: -len(suf)]
            cands = {stem, stem + "e"}
            if stem[-1] == stem[-2]:
                cands.add(stem[:-1])
            if stem.endswith("i"):
                cands.add(stem[:-1] + "y")
            if any(c in vocab for c in cands):
                return True
    return False


def word_records(df: pd.DataFrame, counts: Counter, zipf=None) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Per reference word (trial, pos, word, op, hyp, novel, freq, seen, letters, zipf, cells, inflection,
    in_decode), per hypothesis word (trial, word, op, novel) and substitution pairs (phonetic_error_analysis's pair
    fields + novel)."""
    zipf = zipf or _zipf
    vocab = {w for w, c in counts.items() if c > 0}
    refs, hyps, pairs = [], [], []
    for row in df.itertuples(index=False):
        r, h = normalize(row.reference).split(), normalize(row.decoded).split()
        hset = set(h)
        ops = align(r, h)
        errs, n = word_errors(row.reference, row.decoded)
        assert sum(op != "ok" for op, _, _ in ops) == errs, f"trial {row.trial}: alignment != edit distance"
        for op, i, j in ops:
            if j is not None:
                hyps.append({"trial": int(row.trial), "word": h[j], "op": op, "novel": counts[h[j]] == 0})
            if i is None:
                continue
            w = r[i]
            z = zipf(w)
            hw = h[j] if op == "sub" else None
            refs.append({"trial": int(row.trial), "pos": i, "word": w, "op": op, "hyp": hw, "novel": counts[w] == 0,
                         "freq": counts[w], "seen": bool(row.seen_in_train), "letters": letters(w), "zipf": z,
                         "len_cell": cell(letters(w), z)[0], "lz_cell": 10 * cell(letters(w), z)[0] + cell(letters(w), z)[1],
                         "hyp_known": (counts[hw] > 0) if hw is not None else None, "in_decode": w in hset,
                         "inflection": counts[w] == 0 and inflection_of_train(w, vocab)})
            if op == "sub":
                pairs.append({"trial": int(row.trial), "seen": bool(row.seen_in_train), "trial_wer": errs / max(n, 1),
                              "pos": i, "ref": w, "hyp": hw, "novel": counts[w] == 0})
    return pd.DataFrame(refs), pd.DataFrame(hyps), pairs


_ZIPF_CACHE: dict[str, float] = {}


def _zipf(w: str) -> float:
    if w not in _ZIPF_CACHE:
        import wordfreq

        _ZIPF_CACHE[w] = float(wordfreq.zipf_frequency(w, "en"))
    return _ZIPF_CACHE[w]


def freq_mask(rec: pd.DataFrame, lo: int, hi: int | None) -> np.ndarray:
    f = rec["freq"].to_numpy()
    return (f >= lo) & (f <= hi if hi is not None else True)


# ---------------------------------------------------------------- bootstrap

class Boot:
    """Cluster bootstrap over trials; every statistic of every run uses the same resamples."""

    def __init__(self, n_trials: int, n_boot: int, seed: int):
        self.n = n_trials
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, n_trials, size=(n_boot, n_trials))
        self.w = np.stack([np.bincount(d, minlength=n_trials) for d in draws]).astype(float)

    def per_trial(self, trials: np.ndarray, values=None, cells: np.ndarray | None = None, n_cells: int = 1) -> np.ndarray:
        """(n_trials,) or (n_trials, n_cells) sums of values (1 if None) per trial (and cell)."""
        values = np.ones(len(trials)) if values is None else np.asarray(values, dtype=float)
        if cells is None:
            return np.bincount(trials, weights=values, minlength=self.n)
        out = np.zeros((self.n, n_cells))
        np.add.at(out, (trials, cells), values)
        return out

    def ratio(self, rec: pd.DataFrame, mask: np.ndarray, hit: np.ndarray) -> tuple[float, np.ndarray]:
        """Point estimate and resampled values of sum(hit & mask) / sum(mask) over trials."""
        t = rec["trial"].to_numpy()[mask]
        s, c = self.per_trial(t, hit[mask]), self.per_trial(t)
        with np.errstate(invalid="ignore", divide="ignore"):
            return float(s.sum() / c.sum()) if c.sum() else np.nan, (self.w @ s) / (self.w @ c)

    def matched(self, rec: pd.DataFrame, target: np.ndarray, pool: np.ndarray, hit: np.ndarray, cell_col: str):
        """Recall of `target` tokens, and of `pool` tokens standardized to target's cell distribution, over the cells
        both occupy; per resample. Returns (point dict, resampled target recall, resampled matched recall)."""
        keys = sorted(set(rec.loc[target, cell_col]) | set(rec.loc[pool, cell_col]))
        idx = {k: u for u, k in enumerate(keys)}
        cells = rec[cell_col].map(idx).fillna(-1).astype(int).to_numpy()  # -1: neither target nor pool
        t = rec["trial"].to_numpy()
        k = len(keys)
        tn = self.per_trial(t[target], None, cells[target], k)
        th = self.per_trial(t[target], hit[target], cells[target], k)
        pn = self.per_trial(t[pool], None, cells[pool], k)
        ph = self.per_trial(t[pool], hit[pool], cells[pool], k)

        def est(TN, TH, PN, PH):
            ok = PN > 0
            TNk = np.where(ok, TN, 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                tgt = np.where(ok, TH, 0.0).sum(-1) / TNk.sum(-1)
                pr = np.where(ok, PH / np.where(ok, PN, 1.0), 0.0)
                return tgt, (TNk * pr).sum(-1) / TNk.sum(-1), TNk.sum(-1) / TN.sum(-1)

        tgt, mat, cov = est(tn.sum(0), th.sum(0), pn.sum(0), ph.sum(0))
        btgt, bmat, _ = est(self.w @ tn, self.w @ th, self.w @ pn, self.w @ ph)
        return {"target": float(tgt), "matched": float(mat), "coverage": float(cov),
                "n_target": int(tn.sum()), "n_pool": int(pn.sum())}, btgt, bmat


def ci(boot: np.ndarray) -> list[float]:
    b = boot[~np.isnan(boot)]
    return [float(x) for x in np.percentile(b, [2.5, 97.5])] if len(b) else [np.nan, np.nan]


def p_two_sided(boot: np.ndarray) -> float:
    b = boot[~np.isnan(boot)]
    return float(min(1.0, 2 * min(np.mean(b <= 0), np.mean(b >= 0)))) if len(b) else np.nan


def summary(point: float, boot: np.ndarray) -> dict:
    return {"est": point, "ci": ci(boot), "p": p_two_sided(boot)}


# ---------------------------------------------------------------- per run

def analyse_run(rec: pd.DataFrame, hyp: pd.DataFrame, boot: Boot) -> tuple[dict, dict]:
    """Per-run numbers and the resampled values of MEASURES (for the contrasts)."""
    hit = (rec["op"] == "ok").to_numpy()
    nov = rec["novel"].to_numpy()
    unseen = ~rec["seen"].to_numpy()
    inv = ~nov
    out, draws = {}, {}

    def rate(mask, h):
        return float(h[mask].sum() / mask.sum()) if mask.sum() else np.nan

    out["n_ref"] = int(len(rec))
    out["wer_edits_over_ref"] = None  # filled by the caller (needs insertions)
    rec_tab = {}
    for name, mask in (("novel", nov), ("inv", inv), ("inv_unseen", inv & unseen)):
        pt, b = boot.ratio(rec, mask, hit)
        rec_tab[name] = {"n": int(mask.sum()), "ok": int(hit[mask].sum()), **summary(pt, b)}
        draws[name] = b
    for label, lo, hi in FREQ_BINS:
        m = inv & unseen & freq_mask(rec, lo, hi)
        pt, b = boot.ratio(rec, m, hit)
        rec_tab[f"inv_unseen_freq_{label}"] = {"n": int(m.sum()), "ok": int(hit[m].sum()), **summary(pt, b)}
    infl = rec["inflection"].to_numpy(dtype=bool)
    for name, mask in (("novel_inflection", nov & infl), ("novel_new_stem", nov & ~infl)):
        pt, b = boot.ratio(rec, mask, hit)
        rec_tab[name] = {"n": int(mask.sum()), "ok": int(hit[mask].sum()), **summary(pt, b)}
    out["recall"] = rec_tab
    out["novel_in_decode_anywhere"] = int(rec.loc[nov, "in_decode"].sum())  # alignment-free count of novel hits

    sub = (rec["op"] == "sub").to_numpy()
    dele = (rec["op"] == "del").to_numpy()
    out["novel_ops"] = {"sub_rate": rate(nov, sub), "del_rate": rate(nov, dele), "n_sub": int((nov & sub).sum()),
                        "n_del": int((nov & dele).sum())}
    out["inv_unseen_ops"] = {"sub_rate": rate(inv & unseen, sub), "del_rate": rate(inv & unseen, dele)}
    known = rec["hyp_known"].fillna(False).to_numpy(dtype=bool)
    fb = {}
    for name, mask in (("novel", nov & sub), ("inv", inv & sub), ("inv_unseen", inv & unseen & sub)):
        pt, b = boot.ratio(rec, mask, known)
        fb[name] = {"n_sub": int(mask.sum()), "known": int(known[mask].sum()), **summary(pt, b)}
    out["fallback_known"] = fb

    hn = hyp["novel"].to_numpy()
    hok = (hyp["op"] == "ok").to_numpy()
    out["novel_outputs"] = {"n_hyp_tokens": int(len(hyp)), "n_novel": int(hn.sum()),
                            "share": float(hn.mean()) if len(hyp) else np.nan,
                            "correct": int((hn & hok).sum()),
                            "precision": float((hn & hok).sum() / hn.sum()) if hn.sum() else np.nan}

    matched = {}
    rare = freq_mask(rec, 1, 2)
    for name, pool, col in (("gap_len", inv & unseen, "len_cell"), ("gap_len_zipf", inv & unseen, "lz_cell"),
                            ("gap_rare_len", inv & unseen & rare, "len_cell")):
        pt, bt, bm = boot.matched(rec, nov, pool, hit, col)
        g = bt - bm
        matched[name] = {**pt, "gap": summary(pt["target"] - pt["matched"], g), "matched_ci": ci(bm)}
        draws[name] = g
    out["matched"] = matched
    return out, draws


def contrast(draws: dict, points: dict, measure: str, a: str, b: str) -> dict:
    return summary(points[a][measure] - points[b][measure], draws[a][measure] - draws[b][measure])


def point_of(res: dict, measure: str) -> float:
    if measure in ("novel", "inv", "inv_unseen"):
        return res["recall"][measure]["est"]
    return res["matched"][measure]["gap"]["est"]


# ---------------------------------------------------------------- sound (phonetic_error_analysis, read-only)

def load_phonetic_module():
    """scripts/phonetic_error_analysis.py as a module, without leaving its HF_HUB_OFFLINE default behind."""
    if "phonetic_error_analysis" in sys.modules:
        return sys.modules["phonetic_error_analysis"]
    before = os.environ.get("HF_HUB_OFFLINE")
    spec = importlib.util.spec_from_file_location("phonetic_error_analysis", PHONETIC_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    finally:
        if before is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = before
    sys.modules["phonetic_error_analysis"] = mod
    return mod


def pair_analyses(all_pairs: dict[str, list[dict]], ref_words: dict[int, list[str]], embed_model: str | None,
                  min_null: int = 10) -> dict:
    """phonetic_error_analysis.RunAnalysis per run, built on these pairs: every measure of that script (phon, alike;
    with embed_model also sem, related and ctx) with its length-matched null. One lexicon for all runs, so one
    embedding centring (sem values differ a little from phonetic_errors.md, which centred on its own words)."""
    pea = load_phonetic_module()
    words = {w for pairs in all_pairs.values() for p in pairs for w in (p["ref"], p["hyp"])}
    lex = pea.Lexicon(words, embed_model)
    ctx_lp = None
    if embed_model:
        slots = {(p["trial"], p["pos"]) for pairs in all_pairs.values() for p in pairs}
        ctx_lp = lex.context_logprobs({k: ref_words[k[0]] for k in slots})
    costs = pea.substitution_costs()
    ras = {}
    for run, pairs in all_pairs.items():
        ra = pea.RunAnalysis(run, "real", None, pairs, Counter(), {}, lex, costs, min_null)
        ra.build(ctx_lp)
        ras[run] = ra
    return ras


def pair_summary(ras: dict, boot: Boot, keys: tuple[str, ...], diff_keys: tuple[str, ...]) -> tuple[dict, dict]:
    """Per run, novel vs in-vocab substitutions: pooled mean of each per-pair key (pairs where it is defined) with
    its cluster-bootstrap CI, and novel - in-vocab for diff_keys. Pairs outside the CMU dictionary are not in
    RunAnalysis.pairs (its null is phoneme-length matched); coverage is per key."""
    out, draws = {}, {}
    for run, ra in ras.items():
        res, dr = {}, {}
        for name, keep in (("novel", lambda p: p["novel"]), ("inv", lambda p: not p["novel"])):
            allp = [p for p in ra.all_pairs if keep(p)]
            sp = [p for p in ra.pairs if keep(p)]
            r = {"n_sub": len(allp), "n_in_dict": len(sp)}
            for key in keys:
                vals = np.array([p.get(key, np.nan) for p in sp], dtype=float)
                t = np.array([p["trial"] for p in sp], dtype=int)
                ok = ~np.isnan(vals)
                s, c = boot.per_trial(t[ok], vals[ok]), boot.per_trial(t[ok])
                with np.errstate(invalid="ignore", divide="ignore"):
                    b = (boot.w @ s) / (boot.w @ c)
                r[key] = {**summary(float(s.sum() / c.sum()) if c.sum() else np.nan, b), "n": int(ok.sum()),
                          "coverage": float(ok.sum() / max(len(allp), 1))}
                dr[f"{name}_{key}"] = b
            res[name] = r
        for key in diff_keys:
            res[f"novel_minus_inv_{key}"] = summary(res["novel"][key]["est"] - res["inv"][key]["est"],
                                                    dr[f"novel_{key}"] - dr[f"inv_{key}"])
            dr[f"novel_minus_inv_{key}"] = dr[f"novel_{key}"] - dr[f"inv_{key}"]
        out[run], draws[run] = res, dr
    return out, draws


def phonetic(ras: dict, boot: Boot) -> tuple[dict, dict]:
    """Per run: phoneme-similarity excess and sound-alike share of substitutions, novel vs in-vocab reference words."""
    out, draws = pair_summary(ras, boot, ("exc_phon", "obs_alike", "null_alike"), ("exc_phon",))
    for res in out.values():
        for name in ("novel", "inv"):
            res[name]["coverage"] = res[name]["n_in_dict"] / max(res[name]["n_sub"], 1)
    return out, draws


# ---------------------------------------------------------------- meaning: contextual similarity, sentences, examples

CTX_MODEL, CTX_LAYER = "roberta-large", 17  # BERTScore's model and layer (configs/eval.yaml)
MEANING_KEYS = ("exc_sem", "obs_sem", "null_sem", "exc_related", "obs_related", "null_related", "exc_ctx",
                "exc_csim", "obs_csim", "null_csim")
MEANING_DIFF = ("exc_sem", "exc_related", "exc_ctx", "exc_csim")
ERR_EDGES = [1, 2, 3, 4, 5, 6, 8, 11]  # word errors: 0 | 1 | 2 | 3 | 4 | 5 | 6-7 | 8-10 | >= 11
REFLEN_EDGES = [7, 10, 13]  # reference words: <= 6 | 7-9 | 10-12 | >= 13
EXAMPLE_RUNS = ["B_ssl_s42", "B_ssl_aero-1-audio-diffusion_s42", "A_ssl_s42", "A_ssl_aero-1-audio-diffusion_s42",
                "C_ssl_l14_b1024_s42", "C_ssl_aero-1-audio-diffusion_s42"]


class ContextEmbedder:
    """Contextual vector of one word in a sentence: roberta-large, the output of layer 17 (BERTScore's), mean over
    the word's BPE pieces; CPU. A cache file (outside the repo) keeps every vector computed."""

    def __init__(self, model: str = CTX_MODEL, layer: int = CTX_LAYER, batch: int = 64, cache: Path | None = None):
        self.model_name, self.layer, self.batch, self.cache = model, layer, batch, cache
        self.vecs: dict[str, np.ndarray] = {}
        if cache is not None and cache.is_file():
            z = np.load(cache, allow_pickle=False)
            self.vecs = dict(zip(z["keys"].tolist(), z["vecs"].astype(np.float32)))
        self.tok = self.model = None

    @staticmethod
    def key(words: tuple[str, ...], pos: int) -> str:
        return f"{pos}\t{' '.join(words)}"

    def _load(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(self.model_name, add_prefix_space=True)
        self.model = AutoModel.from_pretrained(self.model_name).eval().to("cpu")
        self.model.encoder.layer = self.model.encoder.layer[: self.layer]
        self.torch = torch

    def __call__(self, items: list[tuple[tuple[str, ...], int]]) -> dict[str, np.ndarray]:
        todo = sorted({(w, p) for w, p in items if self.key(w, p) not in self.vecs}, key=lambda x: (len(x[0]), x))
        if todo:
            if self.model is None:
                self._load()
            with self.torch.no_grad():
                for s in range(0, len(todo), self.batch):
                    chunk = todo[s:s + self.batch]
                    enc = self.tok([list(w) for w, _ in chunk], is_split_into_words=True, return_tensors="pt",
                                   padding=True)
                    h = self.model(**enc).last_hidden_state.float().numpy()
                    for b, (w, p) in enumerate(chunk):
                        idx = [k for k, wid in enumerate(enc.word_ids(b)) if wid == p]
                        self.vecs[self.key(w, p)] = h[b, idx].mean(0)
                    if (s // self.batch) % 50 == 0:
                        print(f"  contextual vectors {s + len(chunk)}/{len(todo)}", flush=True)
            self.save()
        return {self.key(w, p): self.vecs[self.key(w, p)] for w, p in items}

    def save(self):
        if self.cache is None:
            return
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self.vecs)
        tmp = self.cache.with_name(self.cache.stem + ".tmp.npz")
        np.savez(tmp, keys=np.array(keys), vecs=np.stack([self.vecs[k] for k in keys]).astype(np.float16))
        tmp.replace(self.cache)


def contextual_similarity(ras: dict, ref_words: dict[int, list[str]], embed, k: int, seed: int) -> dict:
    """obs_csim / null_csim / exc_csim on every in-dictionary pair (in place): cosine between the reference word's
    contextual vector in the reference sentence and the vector of the decoded word put in that slot; the null is
    k draws from the pair's length-matched null words (phonetic_error_analysis's, weighted the same way), each put
    in the same slot. Vectors are centred on the mean of every vector used. Returns counts."""
    plan, items = [], []
    for run, ra in ras.items():
        rng = np.random.default_rng([seed, zlib.crc32(run.encode())])
        for p in ra.pairs:
            words = list(ref_words[p["trial"]])
            pos = p["pos"]
            assert words[pos] == p["ref"], (run, p["trial"], pos)
            ref = (tuple(words), pos)
            obs = (tuple(words[:pos] + [p["hyp"]] + words[pos + 1:]), pos)
            idx = p["null_idx"]["len"]
            if len(idx) == 0:
                p["obs_csim"] = p["null_csim"] = p["exc_csim"] = np.nan
                continue
            w = ra.weight[idx] / ra.weight[idx].sum()
            nulls = [(tuple(words[:pos] + [ra.pool[d]] + words[pos + 1:]), pos) for d in rng.choice(idx, size=k, p=w)]
            plan.append((p, ref, obs, nulls))
            items += [ref, obs, *nulls]
    if not plan:
        return {"n_pairs": 0, "n_vectors": 0}
    vecs = embed(items)
    mean = np.mean(np.stack(list(vecs.values())), axis=0)

    def unit(item):
        v = vecs[ContextEmbedder.key(*item)] - mean
        return v / np.linalg.norm(v)

    for p, ref, obs, nulls in plan:
        r = unit(ref)
        p["obs_csim"] = float(unit(obs) @ r)
        p["null_csim"] = float(np.mean([unit(n) @ r for n in nulls]))
        p["exc_csim"] = p["obs_csim"] - p["null_csim"]
    return {"n_pairs": len(plan), "n_vectors": len(vecs), "k_null": k}


def per_trial_path(run: str) -> Path | None:
    config = run[:-4] if re.search(r"_s\d+$", run) else run
    seed = run.rsplit("_s", 1)[-1]
    hits = sorted((ROOT / "results" / "per_trial").glob(
        f"{config}__*__real__brandman_2024_text_test__greedy__seed{seed}.csv"))
    return hits[0] if len(hits) == 1 else None


def load_per_trial(run: str, df: pd.DataFrame) -> pd.DataFrame | None:
    """The run's per-trial BERTScore (scripts/evaluate_decodes.py score), checked against its decodes."""
    path = per_trial_path(run)
    if path is None:
        return None
    pt = pd.read_csv(path, keep_default_na=False).sort_values("trial").reset_index(drop=True)
    same = (len(pt) == len(df) and pt["trial"].tolist() == df["trial"].tolist()
            and pt["reference"].map(normalize).equals(df["reference"].map(normalize))
            and pt["decoded"].map(normalize).equals(df["decoded"].map(normalize)))
    if not same:
        raise SystemExit(f"{path.name}: per-trial file does not match {run}'s decodes")
    return pt


def sentence_meaning(rec: pd.DataFrame, pt: pd.DataFrame, boot: Boot) -> tuple[dict, dict]:
    """BERTScore F1 of unseen-sentence trials that miss >= 1 novel word vs unseen trials without a novel word,
    standardized to the same word-error count (and, second, word errors x reference length)."""
    g = rec.groupby("trial")
    novel = g["novel"].any()
    missed = rec.assign(m=rec["novel"] & (rec["op"] != "ok")).groupby("trial")["m"].any()
    seen = g["seen"].first()
    n_ref = g.size()
    trials = pd.DataFrame({"trial": pt["trial"].to_numpy(), "f1": pt["bs_f1"].astype(float).to_numpy(),
                           "errors": pt["word_errors"].astype(int).to_numpy()})
    trials["novel"] = trials["trial"].map(novel).fillna(False).astype(bool).to_numpy()
    trials["missed"] = trials["trial"].map(missed).fillna(False).astype(bool).to_numpy()
    trials["seen"] = trials["trial"].map(seen).fillna(False).astype(bool).to_numpy()
    trials["ref_len"] = trials["trial"].map(n_ref).fillna(0).astype(int).to_numpy()
    e = np.searchsorted(ERR_EDGES, trials["errors"].to_numpy(), side="right")
    trials["err_cell"] = e
    trials["err_len_cell"] = 10 * e + np.searchsorted(REFLEN_EDGES, trials["ref_len"].to_numpy(), side="right")
    unseen = ~trials["seen"].to_numpy()
    target = (trials["missed"].to_numpy()) & unseen
    pool = (~trials["novel"].to_numpy()) & unseen
    f1 = trials["f1"].to_numpy()
    out, draws = {}, {}
    for name, mask in (("all_unseen_novel", trials["novel"].to_numpy() & unseen), ("missed", target),
                       ("no_novel", pool)):
        pt_, b = boot.ratio(trials, mask, f1)
        out[f"f1_{name}"] = {"n": int(mask.sum()), **summary(pt_, b)}
    for name, col in (("errors", "err_cell"), ("errors_len", "err_len_cell")):
        m, bt, bm = boot.matched(trials, target, pool, f1, col)
        out[f"matched_{name}"] = {**m, "gap": summary(m["target"] - m["matched"], bt - bm)}
        draws[f"gap_{name}"] = bt - bm
    out["mean_errors_missed"] = float(trials.loc[target, "errors"].mean()) if target.any() else np.nan
    return out, draws


def pick_examples(recs: dict[str, pd.DataFrame], ref_words: dict[int, list[str]], ras: dict | None,
                  n: int = 20, seed: int = 0) -> list[dict]:
    """n novel reference tokens (seeded), the same for every run, with each run's aligned output there."""
    runs = [r for r in EXAMPLE_RUNS if r in recs]
    if not runs:
        return []
    base = recs[runs[0]]
    tokens = base.loc[base["novel"], ["trial", "pos", "word"]]
    wrong_somewhere = np.zeros(len(tokens), dtype=bool)
    for r in runs:
        ok = recs[r].set_index(["trial", "pos"])["op"].reindex(list(zip(tokens.trial, tokens.pos))) == "ok"
        wrong_somewhere |= ~ok.to_numpy()
    tokens = tokens[wrong_somewhere]
    pick = np.sort(np.random.default_rng(seed).choice(len(tokens), size=min(n, len(tokens)), replace=False))
    related = {}
    if ras:
        for r in runs:
            if r in ras:
                for p in ras[r].pairs:
                    related[(r, p["trial"], p["pos"])] = p.get("obs_related")
    out = []
    for t, pos, word in tokens.iloc[pick].itertuples(index=False):
        words = list(ref_words[t])
        row = {"trial": int(t), "pos": int(pos), "word": word,
               "sentence": " ".join(w.upper() if k == pos else w for k, w in enumerate(words)), "outputs": {}}
        for r in runs:
            x = recs[r].set_index(["trial", "pos"]).loc[(t, pos)]
            if x["op"] == "ok":
                row["outputs"][r] = "= (correct)"
            elif x["op"] == "del":
                row["outputs"][r] = "(deleted)"
            else:
                rel = related.get((r, int(t), int(pos)))
                row["outputs"][r] = x["hyp"] + (" *" if rel == 1 else "")
        out.append(row)
    return out


# ---------------------------------------------------------------- report

def pct(x, nd=1):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{nd}f}"


def fp(p: float) -> str:
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def pp_ci(s: dict, nd=1) -> str:
    return f"{100 * s['est']:+.{nd}f} [{100 * s['ci'][0]:+.{nd}f}, {100 * s['ci'][1]:+.{nd}f}]"


def r_ci(s: dict, nd=1) -> str:
    return f"{100 * s['est']:.{nd}f} [{100 * s['ci'][0]:.{nd}f}, {100 * s['ci'][1]:.{nd}f}]"


def markdown(res: dict) -> str:
    v, runs, prov = res["vocab"], res["runs"], res["train"]
    L = ["# Novel test words: recall of words absent from the training sentences", "",
         "`scripts/novel_word_analysis.py` (CPU; saved BIT-test greedy decodes, real input). "
         f"Paired cluster bootstrap over the {v['n_trials']} test trials, {res['settings']['n_boot']} resamples, "
         f"seed {res['settings']['seed']}; 95% percentile CIs. Recall = reference words aligned as correct "
         "(fewest-edit word alignment, most hits among ties). One seed per run.", "",
         "## Training vocabulary", "",
         f"- Training sentences: {prov['n_train_trials']} train trials"
         + (f" from {prov['n_sessions']} sessions" if prov.get("n_sessions") else "")
         + f", {prov['n_unique_sentences']} distinct; {v['n_train_types']} word types. Every run's `seen_in_train` "
         f"flags match these sentences: {'yes' if res['checks']['seen_matches'] else 'NO'}.",
         f"- Test references: {v['n_test_tokens']} tokens, {v['n_test_types']} types; "
         f"**novel: {v['n_novel_tokens']} tokens ({pct(v['novel_token_share'])}%), {v['n_novel_types']} types "
         f"({pct(v['novel_type_share'])}%)**, in {v['n_trials_with_novel']} of {v['n_trials']} trials "
         f"({v['n_unseen_trials']} trials are unseen sentences; seen sentences have no novel word by construction).",
         f"- Novel vs in-vocab (unseen trials) tokens: median letters {v['median_letters_novel']:.0f} vs "
         f"{v['median_letters_inv_unseen']:.0f}; median English Zipf {v['median_zipf_novel']:.2f} vs "
         f"{v['median_zipf_inv_unseen']:.2f}; {v['n_novel_once']} of the novel types occur once in the test set.",
         f"- Most frequent novel words: {', '.join(f'{w} ({c})' for w, c in v['top_novel'])}.", "",
         "## Per run", "",
         "| Run | WER | In-vocab recall (all) | In-vocab recall (unseen trials) | **Novel recall** [CI] | Novel sub | "
         "Novel del | Wrong novel -> known word | Wrong in-vocab -> known word | Novel outputs (correct) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for run, r in runs.items():
        rc, fb, no = r["recall"], r["fallback_known"], r["novel_outputs"]
        L.append(f"| {r['label']} | {pct(r['wer'])} | {pct(rc['inv']['est'])} | {pct(rc['inv_unseen']['est'])} | "
                 f"**{r_ci(rc['novel'])}** | {pct(r['novel_ops']['sub_rate'])} | {pct(r['novel_ops']['del_rate'])} | "
                 f"{pct(fb['novel']['est'])} ({fb['novel']['known']}/{fb['novel']['n_sub']}) | "
                 f"{pct(fb['inv']['est'])} | {no['n_novel']} ({no['correct']}) |")
    L += ["", "Percentages. Novel sub / del: share of novel reference tokens substituted / deleted (recall + sub + del "
          "= 100). Wrong -> known word: share of substituted reference words whose decoded word is in the training "
          "vocabulary. Novel outputs: decoded tokens not in the training vocabulary (how many are aligned as correct).",
          "", "## By training frequency (in-vocab words, unseen-sentence trials)", "",
          "| Run | Novel (0) | 1-2 | 3-10 | >10 |", "|---|---|---|---|---|"]
    for run, r in runs.items():
        rc = r["recall"]
        cells = [f"{pct(rc['novel']['est'])} (n={rc['novel']['n']})"]
        for label, _, _ in FREQ_BINS:
            x = rc[f"inv_unseen_freq_{label}"]
            cells.append(f"{pct(x['est'])} (n={x['n']})")
        L.append(f"| {r['label']} | " + " | ".join(cells) + " |")
    L += ["", "## Novel words: inflections of training words vs new stems", "",
          f"Inflection (heuristic: -s, -es, -ed, -d, -ing, -'s either way, with e-drop, doubling, y/i): "
          f"{v['n_novel_inflection_tokens']} of {v['n_novel_tokens']} novel tokens, e.g. "
          f"{', '.join(v['examples_inflection'][:8])}. New stems, e.g. {', '.join(v['examples_new_stem'][:8])}.", "",
          "| Run | Inflection recall [CI] | New-stem recall [CI] | Novel words found anywhere in the decode |",
          "|---|---|---|---|"]
    for run, r in runs.items():
        rc = r["recall"]
        L.append(f"| {r['label']} | {r_ci(rc['novel_inflection'])} (n={rc['novel_inflection']['n']}) | "
                 f"{r_ci(rc['novel_new_stem'])} (n={rc['novel_new_stem']['n']}) | "
                 f"{r['novel_in_decode_anywhere']} (aligned correct: {rc['novel']['ok']}) |")
    L += ["", "## Matched in-vocab words (unseen-sentence trials)", "",
          "Novel recall minus the recall of in-vocab words standardized to the novel words' cells. len: letters; "
          "len x Zipf: letters and English word frequency; rare len: in-vocab words seen 1-2 times in training, by "
          "letters. Coverage = share of novel tokens in cells that have in-vocab tokens.", "",
          "| Run | Novel | len: matched / gap | len x Zipf: matched / gap | rare len: matched / gap |",
          "|---|---|---|---|---|"]
    for run, r in runs.items():
        m = r["matched"]
        L.append(f"| {r['label']} | {pct(r['recall']['novel']['est'])} | "
                 + " | ".join(f"{pct(m[k]['matched'])} / {pp_ci(m[k]['gap'])}"
                              for k in ("gap_len", "gap_len_zipf", "gap_rare_len")) + " |")
    any_run = next(iter(runs.values()))["matched"]
    L += ["", "Coverage: " + ", ".join(f"{k} {pct(any_run[k]['coverage'])}% (n novel {any_run[k]['n_target']}, "
                                       f"pool {any_run[k]['n_pool']})" for k in ("gap_len", "gap_len_zipf", "gap_rare_len")) + ".",
          "", "## Paired contrasts (pp; a - b)", "",
          "| Contrast | Novel recall | In-vocab recall (unseen) | Gap vs len x Zipf-matched |", "|---|---|---|---|"]
    for c in res["contrasts"]:
        L.append(f"| {c['label']} | {pp_ci(c['novel'])} ({fp(c['novel']['p'])}) | {pp_ci(c['inv_unseen'])} | "
                 f"{pp_ci(c['gap_len_zipf'])} ({fp(c['gap_len_zipf']['p'])}) |")
    if res.get("phonetic"):
        L += ["", "## Sound of the wrong word (substitutions)", "",
              "Phoneme-similarity excess over the run's own length-matched substitutes and the sound-alike share "
              "(phon >= 0.5; null in brackets), from `scripts/phonetic_error_analysis.py` (read-only reuse). Only "
              "pairs with both words in the CMU dictionary.", "",
              "| Run | Novel: n (in dict) | Novel: excess | Novel: alike (null) | In-vocab: excess | "
              "In-vocab: alike (null) | Novel - in-vocab excess |", "|---|---|---|---|---|---|---|"]
        for run, p in res["phonetic"].items():
            n, i = p["novel"], p["inv"]
            L.append(f"| {runs[run]['label']} | {n['n_sub']} ({n['n_in_dict']}) | {n['exc_phon']['est']:+.3f} "
                     f"[{n['exc_phon']['ci'][0]:+.3f}, {n['exc_phon']['ci'][1]:+.3f}] | {pct(n['obs_alike']['est'])} "
                     f"({pct(n['null_alike']['est'])}) | {i['exc_phon']['est']:+.3f} | {pct(i['obs_alike']['est'])} "
                     f"({pct(i['null_alike']['est'])}) | {p['novel_minus_inv_exc_phon']['est']:+.3f} "
                     f"[{p['novel_minus_inv_exc_phon']['ci'][0]:+.3f}, {p['novel_minus_inv_exc_phon']['ci'][1]:+.3f}] |")
        if res.get("phonetic_contrasts"):
            L += ["", "Novel-word excess, paired: " + "; ".join(
                f"{c['label']} {c['est']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]"
                for c in res["phonetic_contrasts"]) + "."]
    if res.get("meaning"):
        L += meaning_markdown(res)
    L += ["", "## Caveats", ""] + [f"- {c}" for c in res["caveats"]] + [""]
    return "\n".join(L)


def x_ci(s: dict, nd=3) -> str:
    if s is None or np.isnan(s["est"]):
        return "n/a"
    return f"{s['est']:+.{nd}f} [{s['ci'][0]:+.{nd}f}, {s['ci'][1]:+.{nd}f}]"


def meaning_markdown(res: dict) -> list[str]:
    runs, me, st = res["runs"], res["meaning"], res["meaning_settings"]
    info = res.get("contextual_info") or {}
    L = ["", "## Meaning of the wrong word (substitutions)", "",
         "When a decoder replaces a word, is the replacement related in meaning? Excess = observed - null, the null "
         "being the run's own substitutes of the same phoneme length (phonetic_error_analysis.py's, read-only reuse). "
         f"sem: cosine of decontextualized {st['embed_model']} word vectors (centred); related: sem >= 0.4 (share, "
         "null in brackets); ctx: masked-LM log-probability (nats) of the decoded word in the reference word's slot "
         "(one-piece decoded words only; coverage in brackets); contextual: cosine of roberta-large layer-17 vectors "
         "of the reference word in the reference sentence and of the decoded word put in its slot (centred; "
         f"{info.get('k_null', 'n/a')} null draws per pair, {info.get('n_vectors', 0)} vectors). "
         "Pairs with both words in the CMU dictionary.", "",
         "| Run | Novel subs (in dict) | sem excess | related (null) | ctx excess (coverage) | contextual excess |",
         "|---|---|---|---|---|---|"]
    for run, m in me.items():
        n = m["novel"]
        L.append(f"| {runs[run]['label']} | {n['n_sub']} ({n['n_in_dict']}) | {x_ci(n['exc_sem'])} | "
                 f"{pct(n['obs_related']['est'])} ({pct(n['null_related']['est'])}) | {x_ci(n['exc_ctx'], 2)} "
                 f"({pct(n['exc_ctx']['coverage'], 0)}%) | {x_ci(n['exc_csim'])} |")
    L += ["", "In-vocab substitutions, and novel - in-vocab:", "",
          "| Run | In-vocab subs (in dict) | sem excess | related (null) | ctx excess | contextual excess | "
          "Novel - in-vocab: sem | related | contextual |", "|---|---|---|---|---|---|---|---|---|"]
    for run, m in me.items():
        i = m["inv"]
        L.append(f"| {runs[run]['label']} | {i['n_sub']} ({i['n_in_dict']}) | {x_ci(i['exc_sem'])} | "
                 f"{pct(i['obs_related']['est'])} ({pct(i['null_related']['est'])}) | {x_ci(i['exc_ctx'], 2)} | "
                 f"{x_ci(i['exc_csim'])} | {x_ci(m['novel_minus_inv_exc_sem'])} | "
                 f"{pp_ci(m['novel_minus_inv_exc_related']) if not np.isnan(m['novel_minus_inv_exc_related']['est']) else 'n/a'} | "
                 f"{x_ci(m['novel_minus_inv_exc_csim'])} |")
    L += ["", "Paired contrasts of the novel-word excess (a - b; related in pp):", "",
          "| Contrast | sem | related | ctx | contextual |", "|---|---|---|---|---|"]
    for c in res["meaning_contrasts"]:
        L.append(f"| {c['label']} | {x_ci(c['exc_sem'])} | "
                 f"{pp_ci(c['exc_related']) if not np.isnan(c['exc_related']['est']) else 'n/a'} | "
                 f"{x_ci(c['exc_ctx'], 2)} | {x_ci(c['exc_csim'])} |")
    sm = res.get("sentence_meaning")
    if sm:
        L += ["", "## Sentence meaning when a novel word is missed", "",
              "BERTScore F1 (roberta-large L17, rescaled; results/per_trial/) of unseen-sentence trials where the run "
              "missed >= 1 novel word, against unseen trials without a novel word, standardized to the same "
              "word-error count (second: word errors x reference length). Negative gap = missing a novel word costs "
              "more sentence meaning than other errors of the same count.", "",
              "| Run | Trials missing a novel word | Mean word errors | F1 there | Matched F1 (errors) | Gap [CI] | "
              "Gap, errors x length [CI] | F1, unseen trials without novel words |",
              "|---|---|---|---|---|---|---|---|"]
        for run, x in sm.items():
            me_, ml = x["matched_errors"], x["matched_errors_len"]
            L.append(f"| {runs[run]['label']} | {x['f1_missed']['n']} | {x['mean_errors_missed']:.2f} | "
                     f"{x['f1_missed']['est']:.3f} | {me_['matched']:.3f} | {x_ci(me_['gap'])} | {x_ci(ml['gap'])} "
                     f"({pct(ml['coverage'], 0)}%) | {x['f1_no_novel']['est']:.3f} |")
        if res.get("sentence_contrasts"):
            L += ["", "Paired contrasts of the gap (errors-matched; errors x length in brackets): " + "; ".join(
                f"{c['label']} {x_ci(c['errors'])} ({x_ci(c['errors_len'])})" for c in res["sentence_contrasts"])
                + "."]
    ex = res.get("examples")
    if ex:
        cols = [r for r in EXAMPLE_RUNS if r in ex[0]["outputs"]]
        L += ["", "## Examples: novel reference words and each decoder's output there", "",
              f"{len(ex)} novel tokens drawn at random (seed {res['settings']['seed']}) among those some Aero run gets "
              "wrong; the same trials for every run. Novel word in CAPS; * = related in meaning (sem >= 0.4).", "",
              "| Reference | " + " | ".join(runs[r]["label"] for r in cols) + " |",
              "|---|" + "---|" * len(cols)]
        for row in ex:
            L.append(f"| {row['sentence']} | " + " | ".join(row["outputs"][r] for r in cols) + " |")
    return L


# ---------------------------------------------------------------- main

def load_decodes(run: str) -> pd.DataFrame | None:
    path = RESULTS / run / "decodes" / "test_greedy_real.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path, keep_default_na=False)
    df["seen_in_train"] = df["seen_in_train"].astype(str).str.lower().eq("true")
    return df


def analyse(decodes: dict[str, pd.DataFrame], labels: list[str], n_boot: int, seed: int, do_phonetic: bool,
            meaning: dict | None = None) -> dict:
    """meaning (None skips): embed_model (sem / related / ctx; None leaves them out), contextual (a ContextEmbedder
    or a stand-in with its call signature; None skips), k (null draws per pair), per_trial (True: load
    results/per_trial/; or {run: DataFrame}; None skips), examples (count)."""
    counts, train_set = train_counts(labels)
    first = next(iter(decodes.values()))
    refs = first["reference"].map(normalize)
    seen_ok = {}
    for run, df in decodes.items():
        if not df["reference"].map(normalize).equals(refs) or not df["trial"].equals(first["trial"]):
            raise SystemExit(f"{run}: references / trials differ from {next(iter(decodes))}")
        seen_ok[run] = bool((df["reference"].map(normalize).isin(train_set).to_numpy()
                             == df["seen_in_train"].to_numpy()).all())
    if not all(seen_ok.values()):
        raise SystemExit(f"seen_in_train does not match the training sentences: {seen_ok}")
    trials = first["trial"].to_numpy()
    if not np.array_equal(trials, np.arange(len(trials))):
        raise SystemExit("expected trials 0..n-1")
    boot = Boot(len(trials), n_boot, seed)

    recs, runs, draws, pairs = {}, {}, {}, {}
    for run, df in decodes.items():
        rec, hyp, pr = word_records(df, counts)
        recs[run], pairs[run] = rec, pr
        res, dr = analyse_run(rec, hyp, boot)
        n_ins = int((hyp["op"] == "ins").sum())
        errs = int((rec["op"] != "ok").sum()) + n_ins
        res["wer"] = errs / len(rec)
        res.pop("wer_edits_over_ref")
        res["label"] = RUNS.get(run, run)
        runs[run], draws[run] = res, dr

    rec = next(iter(recs.values()))
    nov = rec["novel"].to_numpy()
    unseen = ~rec["seen"].to_numpy()
    tok_counts = Counter(rec.loc[nov, "word"])
    types = set(rec["word"])
    nov_types = {w for w in types if counts[w] == 0}
    vocab = {"n_trials": len(trials), "n_unseen_trials": int((~first["seen_in_train"]).sum()),
             "n_train_types": len(counts), "n_test_tokens": len(rec), "n_test_types": len(types),
             "n_novel_tokens": int(nov.sum()), "novel_token_share": float(nov.mean()),
             "n_novel_types": len(nov_types), "novel_type_share": len(nov_types) / len(types),
             "n_trials_with_novel": int(rec.loc[nov, "trial"].nunique()),
             "n_novel_once": sum(c == 1 for c in tok_counts.values()),
             "median_letters_novel": float(rec.loc[nov, "letters"].median()),
             "median_letters_inv_unseen": float(rec.loc[~nov & unseen, "letters"].median()),
             "median_zipf_novel": float(rec.loc[nov, "zipf"].median()),
             "median_zipf_inv_unseen": float(rec.loc[~nov & unseen, "zipf"].median()),
             "top_novel": tok_counts.most_common(15),
             "n_novel_inflection_tokens": int(rec.loc[nov, "inflection"].sum()),
             "examples_inflection": sorted(set(rec.loc[nov & rec["inflection"].to_numpy(dtype=bool), "word"]))[:12],
             "examples_new_stem": sorted(set(rec.loc[nov & ~rec["inflection"].to_numpy(dtype=bool), "word"]))[:12]}

    points = {run: {m: point_of(runs[run], m) for m in MEASURES} for run in runs}
    contrasts = []
    for label, a, b in CONTRASTS:
        if a in runs and b in runs:
            contrasts.append({"label": label, "a": a, "b": b,
                              **{m: contrast(draws, points, m, a, b) for m in MEASURES}})
    for label, (a1, b1), (a2, b2) in DID:
        if all(x in runs for x in (a1, b1, a2, b2)):
            contrasts.append({"label": label, "a": [a1, b1], "b": [a2, b2], **{
                m: summary(points[a1][m] - points[b1][m] - points[a2][m] + points[b2][m],
                           draws[a1][m] - draws[b1][m] - draws[a2][m] + draws[b2][m]) for m in MEASURES}})

    out = {"settings": {"n_boot": n_boot, "seed": seed, "split": "test", "decoding": "greedy", "condition": "real",
                        "len_edges_letters": LEN_EDGES, "zipf_edges": ZIPF_EDGES,
                        "freq_bins": [b[0] for b in FREQ_BINS]},
           "train": {"n_unique_sentences": len(train_set)}, "checks": {"seen_matches": all(seen_ok.values()),
                                                                       "per_run": seen_ok},
           "vocab": vocab, "runs": runs, "contrasts": contrasts}
    ref_words = {int(t): normalize(r).split() for t, r in zip(first["trial"], first["reference"])}
    meaning = meaning or {}
    ras = pair_analyses(pairs, ref_words, meaning.get("embed_model")) if (do_phonetic or meaning) else None
    if do_phonetic:
        ph, phd = phonetic(ras, boot)
        out["phonetic"] = ph
        pc = []
        for label, a, b in CONTRASTS:
            if a in ph and b in ph:
                pc.append({"label": label, **summary(ph[a]["novel"]["exc_phon"]["est"] - ph[b]["novel"]["exc_phon"]["est"],
                                                     phd[a]["novel_exc_phon"] - phd[b]["novel_exc_phon"])})
        out["phonetic_contrasts"] = pc
    if meaning:
        out.update(meaning_section(decodes, recs, ras, ref_words, boot, meaning, seed))
    return out


def meaning_section(decodes, recs, ras, ref_words, boot: Boot, meaning: dict, seed: int) -> dict:
    out = {"meaning_settings": {"embed_model": meaning.get("embed_model"),
                                "contextual": (f"{CTX_MODEL} layer {CTX_LAYER}, word in the reference slot"
                                               if meaning.get("contextual") is not None else None),
                                "k_null": meaning.get("k")}}
    if meaning.get("contextual") is not None:
        out["contextual_info"] = contextual_similarity(ras, ref_words, meaning["contextual"], meaning["k"], seed)
    mres, mdr = pair_summary(ras, boot, MEANING_KEYS, MEANING_DIFF)
    out["meaning"] = mres
    out["meaning_contrasts"] = [
        {"label": label, **{key: summary(mres[a]["novel"][key]["est"] - mres[b]["novel"][key]["est"],
                                         mdr[a][f"novel_{key}"] - mdr[b][f"novel_{key}"]) for key in MEANING_DIFF}}
        for label, a, b in CONTRASTS if a in mres and b in mres]
    per_trial = meaning.get("per_trial")
    if per_trial is not None:
        sm, smd = {}, {}
        for run, df in decodes.items():
            pt = per_trial.get(run) if isinstance(per_trial, dict) else load_per_trial(run, df)
            if pt is None:
                continue
            sm[run], smd[run] = sentence_meaning(recs[run], pt, boot)
        out["sentence_meaning"] = sm
        out["sentence_contrasts"] = [
            {"label": label, **{k: summary(sm[a][f"matched_{k}"]["gap"]["est"] - sm[b][f"matched_{k}"]["gap"]["est"],
                                           smd[a][f"gap_{k}"] - smd[b][f"gap_{k}"]) for k in ("errors", "errors_len")}}
            for label, a, b in CONTRASTS if a in sm and b in sm]
    out["examples"] = pick_examples(recs, ref_words, ras, meaning.get("examples", 20), seed)
    return out


def caveats(res: dict) -> list[str]:
    v = res["vocab"]
    n_nov = v["n_novel_tokens"]
    c = [f"Small sample: {n_nov} novel tokens ({v['n_novel_types']} types) in {v['n_trials_with_novel']} trials, "
         "most types once, so novel recall has CIs of several pp and run contrasts on novel words alone are weakly "
         "powered; one seed per run, so seed variance is not in the CIs.",
         "Novel means absent from the T15 training transcripts only. The LLMs know every word from pretraining, and "
         "SSL pretraining saw unlabeled neural data; novel recall mixes sub-word signal with the LLM's guess from "
         "context. It is a stand-in for held-out phonemes, not a test of them (every phoneme occurs in training).",
         "Novel words are longer and rarer in English than in-vocab words; the matched gaps adjust for letters and "
         "English Zipf frequency (direct standardization), not for position in the sentence, part of speech or "
         "proper names, and drop novel tokens in cells without in-vocab words.",
         "In-vocab words in seen sentences are partly memorized; the frequency bins, matched pools and the in-vocab "
         "contrast column use unseen-sentence trials only.",
         "Recall depends on the alignment (among fewest-edit alignments, the one with the most hits; WER unchanged). "
         + ("For novel words it does not matter here: in every run the aligned-correct count equals the "
            "alignment-free count (the novel word appears anywhere in the decode)."
            if all(r["novel_in_decode_anywhere"] == r["recall"]["novel"]["ok"] for r in res["runs"].values())
            else "Aligned-correct and alignment-free novel counts differ in some run (see the inflection table)."),
         "The inflection / new-stem split is a suffix heuristic (-s, -es, -ed, -d, -ing, -'s) with a few false hits "
         "(a name like tim / timed); its 65-token side is small."]
    if res.get("meaning"):
        c += ["Meaning of a word pair: sem is distilbert's decontextualized word vector (centred on every word here, "
              "so values differ a little from phonetic_errors.md), related = sem >= 0.4 (calibrated by "
              "phonetic_error_analysis.py on common words); a rare novel word is several word pieces averaged, so "
              "its vector is noisier. The contextual measure (roberta-large layer 17, BERTScore's) puts the decoded "
              "word in the reference slot: it scores fit to the true sentence as much as the word's own meaning, "
              "and its null is a few draws per pair (Monte Carlo noise is small next to the bootstrap CI). Only "
              "pairs with both words in the CMU dictionary (the null is phoneme-length matched); ctx needs a "
              "one-piece decoded word (coverage in the table).",
              "Sentence meaning: BERTScore F1 (rescaled) from results/per_trial/; trials that miss a novel word are "
              "compared with unseen trials without one at the same word-error count, not the same error types or "
              "positions; which trials miss a novel word differs by run, so run contrasts of the gap compare "
              "different trial sets."]
    return c


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-sentences", type=Path, default=DEFAULT_CACHE,
                    help="one raw sentence_label per line; read if present, else written from --train-source")
    ap.add_argument("--train-source", type=Path, default=DEFAULT_SOURCE,
                    help="the T15 release zip or an extracted hdf5_data_final/ directory")
    ap.add_argument("--runs", nargs="*", default=list(RUNS))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-phonetic", action="store_true")
    ap.add_argument("--no-meaning", action="store_true", help="skip the meaning measures (needs local HF models)")
    ap.add_argument("--embed-model", default="distilbert-base-uncased",
                    help="phonetic_error_analysis's sem / ctx model (local HF cache only)")
    ap.add_argument("--contextual-k", type=int, default=3, help="null draws per pair for the contextual measure; 0 skips")
    ap.add_argument("--contextual-cache", type=Path, default=DEFAULT_CACHE.with_name("contextual_vectors.npz"),
                    help="contextual vectors cache (outside the repo)")
    ap.add_argument("--examples", type=int, default=20)
    ap.add_argument("--out", type=Path, default=RESULTS / "novel_words")
    args = ap.parse_args()

    decodes = {}
    for run in args.runs:
        df = load_decodes(run)
        if df is None:
            print(f"skip {run}: no test greedy decodes")
            continue
        decodes[run] = df
    if not decodes:
        raise SystemExit("no decodes")
    labels, prov = load_train_labels(args.train_sentences, args.train_source)
    meaning = None
    if not args.no_meaning:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # models come from the local cache only
        meaning = {"embed_model": args.embed_model or None, "k": args.contextual_k, "per_trial": True,
                   "examples": args.examples,
                   "contextual": ContextEmbedder(cache=args.contextual_cache) if args.contextual_k > 0 else None}
    res = analyse(decodes, labels, args.n_boot, args.seed, not args.no_phonetic, meaning)
    res["train"] = {**prov, **res["train"]}
    res["caveats"] = caveats(res)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(res, indent=1, default=float))
    md = markdown(res)
    args.out.with_suffix(".md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

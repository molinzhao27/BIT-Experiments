#!/usr/bin/env python
"""Sound-alike or language-driven? The word substitutions of Stage 2 decodes (CPU only; reads saved decodes).

    .venv/bin/python scripts/phonetic_error_analysis.py [--runs RUN ...] [--split test] [--controls null_shuffled]
        [--seed 0] [--n-boot 5000] [--n-perm 5000] [--out results/stage2/phonetic_errors]

Question: when a phoneme-free decoder gets a word wrong, does the wrong word *sound* like the right one (the confusion
a phonetic code in the encoder would make), or does it fit the sentence or mean something similar (the LLM's prior
filling in)? This tests only the error pattern of the outputs, not internal representations; the frozen-encoder
phoneme probe and the foil contrasts (DESIGN.md §5) are the stronger tests.

1. Align. Reference and decode are normalized exactly as the project's WER (textmetrics.normalize) and aligned by
   word-level Levenshtein with unit costs (the edit count is checked against textmetrics.word_errors per trial). On a
   tie the backtrace prefers a substitution. The costs ignore what the words are, so the alignment cannot favour
   sound-alike pairs. Each substitution is a pair (r = reference word, h = decoded word).
2. Measures per pair; "similarity" = 1 - edit distance / longer length:
   phon       phoneme similarity from the CMU Pronouncing Dictionary (cmudict 1.1.3 via `pronouncing`, stress
              removed), the best over both words' listed pronunciations; a hyphenated word not in the dictionary
              is its parts joined. Pairs with a word not in the dictionary are dropped (rate reported).
   feat       the same with an articulatory-feature substitution cost (FEATURES below; consonant<->vowel = 1).
   first/last first / last phoneme identical (for some pair of pronunciations).
   alike      phon >= 0.5: at most half the phonemes changed ("sound-alike").
   homophone  phon == 1: some listed pronunciations identical (four / for, their / there).
   orth       spelling similarity (characters): the obvious confound, since the LLMs read and write BPE pieces.
   sem        cosine of decontextualized word embeddings: distilbert-base-uncased (local HF cache, offline), the word
              alone, last layer averaged over its word pieces, centred on the mean vector of every word analysed.
   related    sem >= 0.4 ("related in meaning"; for scale: cat/dog 0.56, happy/glad 0.46, car/truck 0.39,
              cat/hat 0.16).
   zipf       zipf(h) - zipf(r), English word frequency on the Zipf scale (`wordfreq` 3.1.1); commoner = share > 0.
   ctx        log-probability (nats) of the word in r's slot of the *reference* sentence under distilbert's masked
              LM: does the substitute fit the true context? ctx>r = it fits better than r itself. Only words that
              are one word piece (coverage reported).
3. Null. For each pair, h is replaced by a random decoded word from the same run's own substitutions (weighted by
   how often it was a substitute, never r itself) with as many phonemes as h (widened to +-1, +-2 until at least
   --min-null distinct words). The exact null expectation per pair gives the excess (observed - null). A cluster
   bootstrap over trials (the same resampled trials for every run, so run contrasts are paired) gives 95% CIs; a
   Monte Carlo permutation of h gives a one-sided p (observed >= null).
   phon|orth, feat|orth: a second null for the spelling confound. The random substitutes are spelled as similarly to
   r as h is (|d orth| <= 0.05, widened by 0.05), so the excess is sound similarity beyond spelling.
4. Negative control: the same on each run's --controls decodes (null_shuffled = another trial's neural data), where
   any excess comes from alignment or sentence position, not from the neural signal.
5. Slices: seen / unseen sentences; trials decoded mostly right (sentence WER <= 0.5) vs mostly wrong; content words
   only (neither word in FUNCTION); without pairs sharing a stem (use/used); pairs that do not sound alike
   (phon < 0.5: is there meaning or context beyond sound?).

Outputs: <out>/summary.json, <out>/tables.md, <out>/pairs_<run>_<condition>.csv (one row per in-dictionary pair).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # distilbert comes from the local cache only

import editdistance  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bit_experiments.textmetrics import normalize, word_errors  # noqa: E402

RESULTS = ROOT / "results" / ("phase4" if (ROOT / "results" / "phase4").is_dir() else "") / "stage2"

RUNS = {  # run -> label; runs without decodes for --split are skipped
    "B_ssl_fast-dllm-v2-1.5b_s42": "Fast-dLLM v2 (diffusion)",
    "B_ssl_s42": "Aero",
    "B_ssl_qwen25-1.5b-noprompt_s42": "Qwen base, no prompt",
    "B_ssl_qwen25-1.5b-instruct_s42": "Qwen Instruct",
    "B_rand_bit4a": "Aero, random-init encoder (4a)",
    "B_ssl_qwen25-1.5b-instruct-diffusion_s42": "Qwen Instruct + diffusion",
    "B_ssl_aero-1-audio-diffusion_s42": "Aero + diffusion",
    "A_ssl_s42": "Arm A: phoneme CTC encoder + Aero",
    "A_ssl_aero-1-audio-diffusion_s42": "Arm A + diffusion",
    "C_ssl_l14_b1024_s42": "Arm C: contrastive (Aero L14) + Aero",
    "C_ssl_aero-1-audio-diffusion_s42": "Arm C + diffusion",
}

# Articulatory features of the 39 ARPAbet phonemes. Vowels: height (0 high, 1 mid, 2 low), backness (0 front,
# 1 central, 2 back), rounded, offglide (0 none, 1 front, 2 back), tense, r-coloured; height and backness count
# |difference| / 2, the rest 0/1. Consonants: voiced, place (bilabial, labiodental, dental, alveolar, postalveolar,
# palatal, velar, glottal), manner (stop, fricative, affricate, nasal, liquid, glide), each 0/1. The cost is the mean
# over a class's features.
VOWELS = {
    "IY": (0, 0, 0, 0, 1, 0), "IH": (0, 0, 0, 0, 0, 0), "EY": (1, 0, 0, 1, 1, 0), "EH": (1, 0, 0, 0, 0, 0),
    "AE": (2, 0, 0, 0, 0, 0), "AA": (2, 2, 0, 0, 1, 0), "AO": (1, 2, 1, 0, 1, 0), "AH": (1, 1, 0, 0, 0, 0),
    "ER": (1, 1, 0, 0, 1, 1), "OW": (1, 2, 1, 2, 1, 0), "OY": (1, 2, 1, 1, 1, 0), "UH": (0, 2, 1, 0, 0, 0),
    "UW": (0, 2, 1, 0, 1, 0), "AW": (2, 1, 0, 2, 1, 0), "AY": (2, 1, 0, 1, 1, 0),
}
CONSONANTS = {
    "P": (0, 0, 0), "B": (1, 0, 0), "T": (0, 3, 0), "D": (1, 3, 0), "K": (0, 6, 0), "G": (1, 6, 0),
    "F": (0, 1, 1), "V": (1, 1, 1), "TH": (0, 2, 1), "DH": (1, 2, 1), "S": (0, 3, 1), "Z": (1, 3, 1),
    "SH": (0, 4, 1), "ZH": (1, 4, 1), "HH": (0, 7, 1), "CH": (0, 4, 2), "JH": (1, 4, 2),
    "M": (1, 0, 3), "N": (1, 3, 3), "NG": (1, 6, 3), "L": (1, 3, 4), "R": (1, 4, 4), "W": (1, 0, 5), "Y": (1, 5, 5),
}
PHONES = list(VOWELS) + list(CONSONANTS)
PIDX = {p: k for k, p in enumerate(PHONES)}

# Closed-class words for the content-word slice: articles, pronouns, prepositions, conjunctions, auxiliaries, modals
# and their contractions.
FUNCTION = set("""
a an the this that these those some any no every each all both either neither another such what which who whom whose
i me my mine myself you your yours yourself yourselves he him his himself she her hers herself it its itself we us
our ours ourselves they them their theirs themselves one ones
in on at by for from with without to of off up down out over under into onto upon about above below between through
during before after since until till against among around near across along behind beyond toward towards via per
and or but nor so yet if then than because as while whether though although unless
be am is are was were been being have has had having do does did doing done
will would shall should can could may might must ought
i'm you're he's she's it's we're they're i've you've we've they've i'd you'd he'd she'd we'd they'd i'll you'll
he'll she'll we'll they'll isn't aren't wasn't weren't hasn't haven't hadn't don't doesn't didn't won't wouldn't
can't couldn't shouldn't mustn't that's there's here's what's who's let's
not there here very just too also only
""".split())
SUFFIXES = ("s", "es", "ed", "d", "ing", "er", "est", "ly", "n", "en", "'s", "'d", "'ll", "'re", "'ve")

METRICS = [  # name, per-pair vector, null (len = length-matched, orth = spelling-matched)
    ("phon", "phon", "len"), ("feat", "feat", "len"), ("first", "first", "len"), ("last", "last", "len"),
    ("alike", "alike", "len"), ("homophone", "homophone", "len"), ("orth", "orth", "len"), ("phon|orth", "phon", "orth"), ("feat|orth", "feat", "orth"),
    ("sem", "sem", "len"), ("related", "related", "len"), ("zipf", "zipf", "len"), ("commoner", "commoner", "len"), ("ctx", "ctx", "len"),
    ("ctx>r", "ctx_gt", "len"),
]
SLICES = ["all", "seen", "unseen", "mostly_right", "mostly_wrong", "content", "no_stem", "not_alike"]


def align(ref: list[str], hyp: list[str]) -> list[tuple[str, int | None, int | None]]:
    """Unit-cost word Levenshtein with backtrace: [(op, ref index, hyp index)], op in ok / sub / del / ins."""
    n, m = len(ref), len(hyp)
    d = [[i + j if i == 0 or j == 0 else 0 for j in range(m + 1)] for i in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]), d[i - 1][j] + 1, d[i][j - 1] + 1)
    ops, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("ok" if ref[i - 1] == hyp[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("ins", None, j - 1))
            j -= 1
    return ops[::-1]


def pronunciations(word: str) -> list[tuple[str, ...]]:
    """CMU pronunciations without stress; a hyphenated word not in the dictionary is its parts joined (<= 4)."""
    import pronouncing

    def lookup(w):
        return [tuple(re.sub(r"\d", "", p.split("#")[0]).split()) for p in pronouncing.phones_for_word(w)]  # "# name"

    prons = lookup(word)
    if not prons and "-" in word:
        parts = [lookup(p) for p in word.split("-") if p]
        if parts and all(parts):
            prons = [()]
            for variants in parts:
                prons = [a + b for a in prons for b in variants[:2]][:4]
    return list(dict.fromkeys(prons))


def substitution_costs() -> tuple[np.ndarray, np.ndarray]:
    """(unit, feature) phoneme substitution cost matrices over PHONES."""
    k = len(PHONES)
    feat = np.ones((k, k))
    for a in PHONES:
        for b in PHONES:
            if a in VOWELS and b in VOWELS:
                x, y = VOWELS[a], VOWELS[b]
                feat[PIDX[a], PIDX[b]] = (abs(x[0] - y[0]) / 2 + abs(x[1] - y[1]) / 2
                                          + sum(u != v for u, v in zip(x[2:], y[2:]))) / 6
            elif a in CONSONANTS and b in CONSONANTS:
                feat[PIDX[a], PIDX[b]] = sum(u != v for u, v in zip(CONSONANTS[a], CONSONANTS[b])) / 3
    return 1.0 - np.eye(k), feat


def batched_lev(a: np.ndarray, rows: np.ndarray, sub: np.ndarray) -> np.ndarray:
    """Levenshtein (insert / delete 1, substitute sub[x, y]) between a (m,) and every row of rows (K, n)."""
    k, n = rows.shape
    prev = np.tile(np.arange(n + 1, dtype=float), (k, 1))
    for i in range(1, len(a) + 1):
        cur = np.empty_like(prev)
        cur[:, 0] = i
        cost = sub[a[i - 1], rows]
        for j in range(1, n + 1):
            cur[:, j] = np.minimum(np.minimum(prev[:, j], cur[:, j - 1]) + 1, prev[:, j - 1] + cost[:, j - 1])
        prev = cur
    return prev[:, n]


def similarity(a: str, b: str) -> float:
    return 1.0 - editdistance.eval(a, b) / max(len(a), len(b), 1)


def same_stem(a: str, b: str) -> bool:
    """Inflectional / derivational neighbours (use / used, go / going): a shared 4-letter prefix, or the shorter
    word (>= 3 letters) plus a common suffix."""
    short, long_ = sorted((a, b), key=len)
    if len(os.path.commonprefix([a, b])) >= 4:
        return True
    return len(short) >= 3 and long_.startswith(short) and long_[len(short):] in SUFFIXES


class Lexicon:
    """Every word analysed: pronunciations, Zipf frequency, centred embedding, masked-LM token id."""

    def __init__(self, words: set[str], embed_model: str | None):
        import wordfreq

        self.words = sorted(words)
        self.index = {w: k for k, w in enumerate(self.words)}
        self.prons = {w: [np.array([PIDX[p] for p in pr]) for pr in pronunciations(w)] for w in self.words}
        self.zipf = np.array([wordfreq.zipf_frequency(w, "en") or np.nan for w in self.words])
        self.embed_model = embed_model
        self.emb = self.piece = None
        if embed_model:
            self._embed()

    def _embed(self, batch: int = 256):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(self.embed_model)
        self.mlm = AutoModelForMaskedLM.from_pretrained(self.embed_model).eval()
        encoder = self.mlm.base_model
        vecs = []
        with torch.no_grad():
            for s in range(0, len(self.words), batch):
                enc = self.tok(self.words[s:s + batch], return_tensors="pt", padding=True,
                               return_special_tokens_mask=True)
                keep = (enc.pop("special_tokens_mask") == 0) & (enc["attention_mask"] == 1)
                h = encoder(**enc).last_hidden_state
                vecs.append(((h * keep[..., None]).sum(1) / keep.sum(1, keepdim=True)).numpy())
        e = np.concatenate(vecs)
        e -= e.mean(0)
        self.emb = e / np.linalg.norm(e, axis=1, keepdims=True)
        unk = self.tok.unk_token_id
        ids = [self.tok.convert_tokens_to_ids(w) if self.tok.tokenize(w) == [w] else unk for w in self.words]
        self.piece = np.array([-1 if i == unk else i for i in ids])

    def context_logprobs(self, sentences: dict[tuple[int, int], list[str]], batch: int = 64) -> dict:
        """(trial, ref position) -> masked-LM log-probability of every lexicon word in that slot (NaN unless the
        word is one word piece)."""
        import torch

        keys = sorted(sentences)
        single = np.flatnonzero(self.piece >= 0)
        out = {}
        with torch.no_grad():
            for s in range(0, len(keys), batch):
                chunk = keys[s:s + batch]
                texts = [" ".join(self.tok.mask_token if k == key[1] else w for k, w in enumerate(sentences[key]))
                         for key in chunk]
                enc = self.tok(texts, return_tensors="pt", padding=True)
                logits = self.mlm(**enc).logits
                rows, cols = (enc["input_ids"] == self.tok.mask_token_id).nonzero(as_tuple=True)
                assert rows.tolist() == list(range(len(chunk))), "exactly one mask per sentence"
                lp = torch.log_softmax(logits[rows, cols], -1)[:, self.piece[single]].numpy()
                for key, row in zip(chunk, lp):
                    full = np.full(len(self.words), np.nan, dtype=np.float32)
                    full[single] = row
                    out[key] = full
        return out


def load_decodes(run: str, split: str, condition: str) -> pd.DataFrame | None:
    path = RESULTS / run / "decodes" / f"{split}_greedy_{condition}.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path, keep_default_na=False)
    df["seen_in_train"] = df["seen_in_train"].astype(str).str.lower().eq("true")
    return df


def extract(df: pd.DataFrame) -> tuple[list[dict], Counter, dict]:
    """Substitution pairs (with trial context), op counts, and reference-word frequency inputs."""
    pairs, ops_count, ref_words = [], Counter(), {"ok": [], "sub": [], "del": []}
    for row in df.itertuples(index=False):
        ref, hyp = normalize(row.reference).split(), normalize(row.decoded).split()
        ops = align(ref, hyp)
        errs, n = word_errors(row.reference, row.decoded)
        assert sum(op != "ok" for op, _, _ in ops) == errs, f"trial {row.trial}: alignment != edit distance"
        ops_count.update(op for op, _, _ in ops)
        for op, i, j in ops:
            if i is not None:
                ref_words[op].append(ref[i])
            if op == "sub":
                pairs.append({"trial": int(row.trial), "seen": bool(row.seen_in_train), "trial_wer": errs / max(n, 1),
                              "pos": i, "ref": ref[i], "hyp": hyp[j]})
    return pairs, ops_count, ref_words


class RunAnalysis:
    """One run and condition: the in-dictionary pairs, their observed values and their null candidate sets."""

    def __init__(self, name, condition, df, pairs, ops_count, ref_words, lex: Lexicon, costs, min_null: int):
        self.name, self.condition, self.lex, self.costs, self.min_null = name, condition, lex, costs, min_null
        self.df, self.ops_count, self.ref_words, self.all_pairs = df, ops_count, ref_words, pairs
        self.pairs = [p for p in pairs if lex.prons[p["ref"]] and lex.prons[p["hyp"]]]
        targets = Counter(p["hyp"] for p in self.pairs)
        self.pool = sorted(targets)
        self.pool_index = {w: k for k, w in enumerate(self.pool)}
        self.weight = np.array([targets[w] for w in self.pool], dtype=float)
        self.pool_lex = np.array([lex.index[w] for w in self.pool])
        self.plen = np.array([min(len(p) for p in lex.prons[w]) for w in self.pool])
        groups: dict[int, list] = {}
        for u, w in enumerate(self.pool):
            for pr in lex.prons[w]:
                groups.setdefault(len(pr), []).append((u, pr))
        self.groups = {n: (np.array([u for u, _ in g]), np.stack([pr for _, pr in g])) for n, g in groups.items()}
        self._ref_cache: dict[str, dict] = {}
        self.ctx_lp = None

    def ref_vectors(self, r: str) -> dict[str, np.ndarray]:
        """Every word-pair measure between r and each pool word (NaN where undefined)."""
        if r in self._ref_cache:
            return self._ref_cache[r]
        lex, n_pool = self.lex, len(self.pool)
        v = {key: np.full(n_pool, np.nan) for key in ("phon", "feat", "first", "last")}
        for a in lex.prons[r]:
            for n, (u, rows) in self.groups.items():
                denom = max(len(a), n)
                for key, sub in (("phon", self.costs[0]), ("feat", self.costs[1])):
                    np.fmax.at(v[key], u, 1.0 - batched_lev(a, rows, sub) / denom)
                np.fmax.at(v["first"], u, (rows[:, 0] == a[0]).astype(float))
                np.fmax.at(v["last"], u, (rows[:, -1] == a[-1]).astype(float))
        v["alike"] = np.where(np.isnan(v["phon"]), np.nan, (v["phon"] >= 0.5).astype(float))
        v["homophone"] = np.where(np.isnan(v["phon"]), np.nan, (v["phon"] >= 1 - 1e-9).astype(float))
        v["orth"] = np.array([similarity(r, w) for w in self.pool])
        ri = lex.index[r]
        v["zipf"] = lex.zipf[self.pool_lex] - lex.zipf[ri]
        v["commoner"] = np.where(np.isnan(v["zipf"]), np.nan, (v["zipf"] > 0).astype(float))
        v["sem"] = lex.emb[self.pool_lex] @ lex.emb[ri] if lex.emb is not None else np.full(n_pool, np.nan)
        v["related"] = np.where(np.isnan(v["sem"]), np.nan, (v["sem"] >= 0.4).astype(float))
        self._ref_cache[r] = v
        return v

    def vector(self, p: dict, key: str) -> np.ndarray | None:
        """Measure `key` between p's reference word and every pool word (ctx: in p's reference slot)."""
        if key in ("ctx", "ctx_gt"):
            if self.ctx_lp is None:
                return None
            lp = self.ctx_lp[(p["trial"], p["pos"])]
            ctx = lp[self.pool_lex].astype(float)
            if key == "ctx":
                return ctx
            ref = lp[self.lex.index[p["ref"]]]
            return np.where(np.isnan(ctx) | np.isnan(ref), np.nan, (ctx > ref).astype(float))
        return self.ref_vectors(p["ref"])[key]

    def build(self, ctx_lp: dict | None):
        """Null candidates, observed value, null expectation and excess for every pair and metric."""
        self.ctx_lp = ctx_lp
        for p in self.pairs:
            r, h = p["ref"], p["hyp"]
            orth = self.ref_vectors(r)["orth"]
            hi = self.pool_index[h]
            not_r = np.arange(len(self.pool)) != self.pool_index.get(r, -1)
            masks = {"len": self._widen(lambda w: (np.abs(self.plen - self.plen[hi]) <= w) & not_r, [0, 1, 2]),
                     "orth": self._widen(lambda w: (np.abs(orth - orth[hi]) <= w + 1e-9) & not_r,
                                         [0.05, 0.10, 0.15, 0.20, 0.25, 0.30])}
            p["null_idx"] = {k: np.flatnonzero(m) for k, m in masks.items()}
            p["hi"] = hi
            p["content"] = r not in FUNCTION and h not in FUNCTION
            p["no_stem"] = not same_stem(r, h)
            for name, key, null in METRICS:
                obs, vals, w = self.null_values(p, key, null)
                p[f"obs_{name}"] = obs
                p[f"null_{name}"] = float(np.dot(vals, w) / w.sum()) if vals is not None else np.nan
                p[f"exc_{name}"] = p[f"obs_{name}"] - p[f"null_{name}"]

    def _widen(self, mask_for, widths) -> np.ndarray:
        for w in widths:
            m = mask_for(w)
            if m.sum() >= self.min_null:
                return m
        return m

    def null_values(self, p, key, null):
        """(observed, candidate values, candidate weights) for one pair; (NaN, None, None) if undefined."""
        v = self.vector(p, key)
        if v is None:
            return np.nan, None, None
        obs = v[p["hi"]]
        idx = p["null_idx"][null]
        vals = v[idx]
        ok = ~np.isnan(vals)
        if np.isnan(obs) or not ok.any():
            return np.nan, None, None
        return float(obs), vals[ok], self.weight[idx][ok]

    def in_slice(self, p, s: str) -> bool:
        return {"all": True, "seen": p["seen"], "unseen": not p["seen"], "mostly_right": p["trial_wer"] <= 0.5,
                "mostly_wrong": p["trial_wer"] > 0.5, "content": p["content"], "no_stem": p["no_stem"],
                "not_alike": p["obs_phon"] < 0.5}[s]


class Stats:
    """Cluster bootstrap over trials (one resampling shared by every run) and permutation p-values."""

    def __init__(self, trials: list[int], n_boot: int, n_perm: int, seed: int):
        self.pos = {t: k for k, t in enumerate(trials)}
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(trials), size=(n_boot, len(trials)))
        self.counts = np.stack([np.bincount(d, minlength=len(trials)) for d in draws]).astype(float)
        self.perm_rng = np.random.default_rng(seed + 1)
        self.n_perm = n_perm

    def boot(self, pairs: list[dict], col: str) -> np.ndarray:
        """Bootstrap distribution of the pooled mean of p[col] over pairs (NaN pairs dropped)."""
        keep = [p for p in pairs if not np.isnan(p[col])]
        t = np.array([self.pos[p["trial"]] for p in keep], dtype=int)
        s = np.bincount(t, weights=[p[col] for p in keep], minlength=len(self.pos))
        c = np.bincount(t, minlength=len(self.pos)).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (self.counts @ s) / (self.counts @ c)

    def perm_p(self, ra: RunAnalysis, pairs: list[dict], name: str, key: str, null: str) -> float:
        """One-sided Monte Carlo p: the mean with every h replaced by a null draw >= the observed mean."""
        total, obs, n = np.zeros(self.n_perm), 0.0, 0
        for p in pairs:
            o, vals, w = ra.null_values(p, key, null)
            if vals is None:
                continue
            c = np.cumsum(w)
            k = np.minimum(np.searchsorted(c, self.perm_rng.random(self.n_perm) * c[-1], side="right"), len(vals) - 1)
            total += vals[k]
            obs += o
            n += 1
        if n == 0:
            return np.nan
        return float((1 + np.sum(total / n >= obs / n - 1e-12)) / (self.n_perm + 1))


def summarize(ra: RunAnalysis, stats: Stats, slices: list[str], perm: bool) -> dict:
    lex = ra.lex
    zipf = {op: np.nanmedian([lex_zipf(lex, w) for w in ws]) if ws else np.nan for op, ws in ra.ref_words.items()}
    n_ref = sum(len(normalize(r).split()) for r in ra.df["reference"])
    out = {"run": ra.name, "condition": ra.condition, "n_trials": len(ra.df), "n_ref_words": n_ref,
           "wer": (ra.ops_count["sub"] + ra.ops_count["del"] + ra.ops_count["ins"]) / n_ref,
           "ops": dict(ra.ops_count), "n_sub": len(ra.all_pairs), "n_pairs": len(ra.pairs),
           "oov_rate": 1 - len(ra.pairs) / max(len(ra.all_pairs), 1),
           "ref_zipf_median": {"correct": zipf["ok"], "substituted": zipf["sub"], "deleted": zipf["del"]},
           "slices": {}}
    both = [p for p in ra.pairs if not np.isnan(p["obs_alike"]) and not np.isnan(p["obs_related"])]
    out["crosstab"] = {f"{'alike' if a else 'not_alike'}+{'related' if r else 'unrelated'}":
                       float(np.mean([(p["obs_alike"] == a) and (p["obs_related"] == r) for p in both]))
                       for a in (1, 0) for r in (1, 0)}
    for s in slices:
        sp = [p for p in ra.pairs if ra.in_slice(p, s)]
        res = {"n_pairs": len(sp)}
        for name, key, null in METRICS:
            valid = [p for p in sp if not np.isnan(p[f"exc_{name}"])]
            if not valid:
                continue
            boot = stats.boot(valid, f"exc_{name}")
            res[name] = {"n": len(valid), "obs": float(np.mean([p[f"obs_{name}"] for p in valid])),
                         "null": float(np.mean([p[f"null_{name}"] for p in valid])),
                         "excess": float(np.mean([p[f"exc_{name}"] for p in valid])),
                         "ci": [float(x) for x in np.nanpercentile(boot, [2.5, 97.5])]}
            if perm and s == "all":
                res[name]["p_perm"] = stats.perm_p(ra, valid, name, key, null)
        out["slices"][s] = res
    return out


def lex_zipf(lex: Lexicon, w: str) -> float:
    return lex.zipf[lex.index[w]] if w in lex.index else np.nan


def contrast(stats: Stats, a: RunAnalysis, b: RunAnalysis, name: str, slice_: str = "all") -> dict:
    """Paired (same resampled trials) difference in mean excess, a - b."""
    pa = [p for p in a.pairs if a.in_slice(p, slice_) and not np.isnan(p[f"exc_{name}"])]
    pb = [p for p in b.pairs if b.in_slice(p, slice_) and not np.isnan(p[f"exc_{name}"])]
    d = stats.boot(pa, f"exc_{name}") - stats.boot(pb, f"exc_{name}")
    diff = np.mean([p[f"exc_{name}"] for p in pa]) - np.mean([p[f"exc_{name}"] for p in pb])
    return {"a": a.name, "b": b.name, "a_condition": a.condition, "b_condition": b.condition, "metric": name,
            "slice": slice_, "diff": float(diff),
            "ci": [float(x) for x in np.nanpercentile(d, [2.5, 97.5])]}


def slice_contrast(stats: Stats, ra: RunAnalysis, name: str, s1: str, s2: str) -> dict:
    p1 = [p for p in ra.pairs if ra.in_slice(p, s1) and not np.isnan(p[f"exc_{name}"])]
    p2 = [p for p in ra.pairs if ra.in_slice(p, s2) and not np.isnan(p[f"exc_{name}"])]
    d = stats.boot(p1, f"exc_{name}") - stats.boot(p2, f"exc_{name}")
    diff = np.mean([p[f"exc_{name}"] for p in p1]) - np.mean([p[f"exc_{name}"] for p in p2])
    return {"run": ra.name, "metric": name, "a": s1, "b": s2, "diff": float(diff),
            "ci": [float(x) for x in np.nanpercentile(d, [2.5, 97.5])]}


def fmt_exc(m: dict | None, digits: int = 3) -> str:
    if not m:
        return "–"
    return f"{m['excess']:+.{digits}f} [{m['ci'][0]:+.{digits}f}, {m['ci'][1]:+.{digits}f}]"


def fmt_p(m: dict | None) -> str:
    if not m or "p_perm" not in m:
        return "–"
    return "<0.001" if m["p_perm"] < 1e-3 else f"{m['p_perm']:.3f}"


def tables(summaries: list[dict], contrasts: list[dict], slice_diffs: list[dict], top: dict, labels: dict) -> str:
    lines = []

    def label(s):
        return labels.get(s["run"], s["run"]) + ("" if s["condition"] == "real" else f" · {s['condition']}")

    lines += ["## Phonetic side (all pairs; excess = observed − length-matched null, 95% trial-bootstrap CI)", "",
              "| Run | WER | subs | pairs (OOV) | phon obs / null | phon excess | p | feat excess | sound-alike obs / null "
              "| excess sound-alikes / 100 ref words | homophones obs / null | first / last phoneme excess "
              "| phon excess beyond spelling |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        a = s["slices"]["all"]
        lines.append(
            f"| {label(s)} | {100 * s['wer']:.1f}% | {s['n_sub']} | {s['n_pairs']} ({100 * s['oov_rate']:.1f}%) "
            f"| {a['phon']['obs']:.3f} / {a['phon']['null']:.3f} | {fmt_exc(a.get('phon'))} | {fmt_p(a.get('phon'))} "
            f"| {fmt_exc(a.get('feat'))} | {100 * a['alike']['obs']:.1f}% / {100 * a['alike']['null']:.1f}% "
            f"| {100 * a['alike']['excess'] * a['alike']['n'] / s['n_ref_words']:.1f} "
            f"| {100 * a['homophone']['obs']:.1f}% / {100 * a['homophone']['null']:.1f}% "
            f"| {a['first']['excess']:+.3f} / {a['last']['excess']:+.3f} | {fmt_exc(a.get('phon|orth'))} |")
    lines += ["", "## Language side (all pairs)", "",
              "| Run | spelling excess | meaning (sem) obs / null | sem excess | p | related obs / null "
              "| commoner obs / null "
              "| context fit (ctx, nats) excess | ctx p | fits context better than ref obs / null | ctx coverage "
              "| median zipf: correct / substituted ref |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        a = s["slices"]["all"]
        rel = a.get("related")
        sem, ctx, gt, com = a.get("sem"), a.get("ctx"), a.get("ctx>r"), a.get("commoner")
        z = s["ref_zipf_median"]
        if not (sem and ctx and gt and com):
            lines.append(f"| {label(s)} | {fmt_exc(a.get('orth'))} | – |")
            continue
        lines.append(
            f"| {label(s)} | {fmt_exc(a.get('orth'))} "
            f"| {sem['obs']:.3f} / {sem['null']:.3f} | {fmt_exc(sem)} | {fmt_p(sem)} "
            f"| {100 * rel['obs']:.1f}% / {100 * rel['null']:.1f}% "
            f"| {100 * com['obs']:.0f}% / {100 * com['null']:.0f}% | {fmt_exc(ctx, 2)} | {fmt_p(ctx)} "
            f"| {100 * gt['obs']:.0f}% / {100 * gt['null']:.0f}% | {100 * ctx['n'] / max(s['n_pairs'], 1):.0f}% "
            f"| {z['correct']:.2f} / {z['substituted']:.2f} |")
    lines += ["", "## Sound-alike (phon >= 0.5) x related in meaning (sem >= 0.4), share of pairs", "",
              "| Run | alike + related | alike only | related only | neither |", "|---|---|---|---|---|"]
    for s in summaries:
        x = s["crosstab"]
        lines.append(f"| {label(s)} | {100 * x['alike+related']:.1f}% | {100 * x['alike+unrelated']:.1f}% "
                     f"| {100 * x['not_alike+related']:.1f}% | {100 * x['not_alike+unrelated']:.1f}% |")
    lines += ["", "## Slices (excess with 95% CI; n pairs)", "",
              "| Run | slice | n | phon | phon beyond spelling | sem | ctx (nats) |", "|---|---|---|---|---|---|---|"]
    for s in summaries:
        if s["condition"] != "real":
            continue
        for sl, a in s["slices"].items():
            if sl == "all":
                continue
            lines.append(f"| {label(s)} | {sl} | {a['n_pairs']} | {fmt_exc(a.get('phon'))} "
                         f"| {fmt_exc(a.get('phon|orth'))} | {fmt_exc(a.get('sem'))} | {fmt_exc(a.get('ctx'), 2)} |")
    lines += ["", "## Slice differences within a run (a − b, 95% CI)", "", "| Run | metric | a − b | diff |",
              "|---|---|---|---|"]
    for c in slice_diffs:
        lines.append(f"| {labels.get(c['run'], c['run'])} | {c['metric']} | {c['a']} − {c['b']} "
                     f"| {c['diff']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] |")
    lines += ["", "## Run contrasts (difference in mean excess, a − b; paired trial bootstrap)", "",
              "| a | b | metric | slice | diff |", "|---|---|---|---|---|"]
    for c in contrasts:
        a = labels.get(c["a"], c["a"]) + ("" if c["a_condition"] == "real" else f" · {c['a_condition']}")
        b = labels.get(c["b"], c["b"]) + ("" if c["b_condition"] == "real" else f" · {c['b_condition']}")
        lines.append(f"| {a} | {b} | {c['metric']} | {c['slice']} "
                     f"| {c['diff']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] |")
    lines += ["", "## 20 most common substitutions (reference → decoded, count)", ""]
    runs = list(top)
    lines += ["| # | " + " | ".join(labels.get(r, r) for r in runs) + " |", "|---" * (len(runs) + 1) + "|"]
    for k in range(20):
        cells = [f"{top[r][k][0][0]} → {top[r][k][0][1]} ({top[r][k][1]})" if k < len(top[r]) else "" for r in runs]
        lines.append(f"| {k + 1} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", nargs="+", default=list(RUNS), help="run directories under results/stage2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--controls", nargs="*", default=["null_shuffled"], help="null decode conditions as controls")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--min-null", type=int, default=10, help="minimum distinct null words per pair")
    ap.add_argument("--embed-model", default="distilbert-base-uncased", help="'' skips sem and ctx")
    ap.add_argument("--out", type=Path, default=RESULTS / "phonetic_errors")
    args = ap.parse_args()

    loaded = []  # (run, condition, df, pairs, ops, ref_words)
    for run in args.runs:
        for cond in ["real", *args.controls]:
            df = load_decodes(run, args.split, cond)
            if df is None:
                print(f"skip {run} {cond}: no {args.split}_greedy_{cond}.csv")
                continue
            loaded.append((run, cond, df, *extract(df)))
    if not loaded:
        sys.exit("no decodes found")
    refs = {k: v for _, _, df, *_ in loaded for k, v in zip(df["trial"], df["reference"])}
    for run, cond, df, *_ in loaded:  # every run decodes the same trials
        assert all(refs[t] == r for t, r in zip(df["trial"], df["reference"])), f"{run} {cond}: references differ"
    words = {p[k] for *_, pairs, _, _ in loaded for p in pairs for k in ("ref", "hyp")}
    words |= {w for *_, rw in loaded for ws in rw.values() for w in ws}
    print(f"{len(loaded)} decode sets, {len(words)} distinct words; building the lexicon ...")
    lex = Lexicon(words, args.embed_model or None)
    costs = substitution_costs()

    analyses = [RunAnalysis(run, cond, df, pairs, ops, rw, lex, costs, args.min_null)
                for run, cond, df, pairs, ops, rw in loaded]
    ctx_lp = None
    if lex.emb is not None:
        slots = {(p["trial"], p["pos"]) for ra in analyses for p in ra.pairs}
        ref_words = {t: normalize(r).split() for t, r in refs.items()}
        print(f"masked-LM context scores for {len(slots)} reference slots ...")
        ctx_lp = lex.context_logprobs({(t, i): ref_words[t] for t, i in slots})
    for ra in analyses:
        ra.build(ctx_lp)

    stats = Stats(sorted(refs), args.n_boot, args.n_perm, args.seed)
    summaries = []
    for ra in analyses:
        print(f"{ra.name} {ra.condition}: {len(ra.pairs)} pairs ...")
        summaries.append(summarize(ra, stats, SLICES if ra.condition == "real" else ["all"], perm=True))

    real = {ra.name: ra for ra in analyses if ra.condition == "real"}
    contrasts = []
    order = [r for r in RUNS if r in real] + [r for r in real if r not in RUNS]
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            for name in ("phon", "phon|orth", "sem", "ctx"):
                for sl in ("all", "mostly_right"):
                    contrasts.append(contrast(stats, real[a], real[b], name, sl))
    for ra in analyses:
        if ra.condition != "real":
            base = real.get(ra.name)
            if base:
                for name in ("phon", "phon|orth", "sem", "ctx"):
                    contrasts.append(contrast(stats, base, ra, name))
    slice_diffs = [slice_contrast(stats, real[r], name, a, b) for r in order for name in ("phon", "sem", "ctx")
                   for a, b in (("seen", "unseen"), ("mostly_right", "mostly_wrong"))]
    top = {r: Counter((p["ref"], p["hyp"]) for p in real[r].all_pairs).most_common(20) for r in order}

    args.out.mkdir(parents=True, exist_ok=True)
    cols = ["trial", "seen", "trial_wer", "pos", "ref", "hyp", "content", "no_stem"]
    for ra in analyses:
        pd.DataFrame([{k: p[k] for k in cols} | {f"{w}_{name}": p[f"{w}_{name}"] for name, _, _ in METRICS
                                                  for w in ("obs", "null")} for p in ra.pairs]).to_csv(
            args.out / f"pairs_{ra.name}_{ra.condition}.csv", index=False, float_format="%.4f")
    meta = {"split": args.split, "seed": args.seed, "n_boot": args.n_boot, "n_perm": args.n_perm,
            "min_null": args.min_null, "embed_model": args.embed_model, "cmudict": _version("cmudict"),
            "pronouncing": _version("pronouncing"), "wordfreq": _version("wordfreq"), "n_lexicon": len(lex.words)}
    (args.out / "summary.json").write_text(json.dumps(
        {"meta": meta, "runs": summaries, "contrasts": contrasts, "slice_diffs": slice_diffs,
         "top20": {r: [[a, b, n] for (a, b), n in t] for r, t in top.items()}}, indent=1, default=float))
    labels = {r: RUNS.get(r, r) for r in order}
    (args.out / "tables.md").write_text(tables(summaries, contrasts, slice_diffs, top, labels))
    print(f"wrote {args.out}/summary.json, tables.md, pairs_*.csv")


def _version(pkg: str) -> str:
    from importlib.metadata import version

    return version(pkg)


if __name__ == "__main__":
    main()

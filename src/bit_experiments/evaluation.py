"""Phase 4 shared evaluator: every arm's decoded sentences are scored here, with one fixed configuration.

Per trial: word errors / reference words (textmetrics, BIT's WER) and BERTScore P/R/F1 (bert-score,
settings from configs/eval.yaml, both strings normalized with textmetrics.normalize first; an
empty decode or reference scores 0). Per decode file: aggregate WER, mean per-sentence WER, sentence
error rate, mean BERTScore. On top of that: meaning-preserving vs -changing error analysis over a τ grid,
mean ± SD over seeds, side-by-side examples, and paired-bootstrap null-control comparisons.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bit_experiments.textmetrics import normalize, word_errors

EVALUATOR = "evaluation"
META_KEYS = ("model", "arm", "seed", "split", "decoding", "condition")
CONDITIONS = ("real", "null_shuffled", "null_noise", "null_prompt_only")
NULLS = CONDITIONS[1:]
# config: the run's configuration across seeds (e.g. B_ssl vs B_ssl_qwen25-1.5b-instruct, both arm B), so runs of one
# arm and seed with different LLMs / inits / prompts neither replace each other nor pool as seeds
RUN_KEY = ("config", "arm", "seed", "split", "decoding", "condition", "evaluator")
# metrics.csv columns written by run_bit_baseline.py (which appends without a header): always first, in order
LEGACY_COLS = ("model", "arm", "seed", "split", "decoding", "n_trials", "wer", "per")
REPO = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG = {
    "bertscore": {"model_type": "roberta-large", "num_layers": 17, "idf": False, "lang": "en",
                  "rescale_with_baseline": True, "batch_size": 64, "device": "auto"},
    "error_analysis": {"tau": 0.75, "tau_grid": [0.6, 0.7, 0.75, 0.8, 0.9]},
    "bootstrap": {"n_resamples": 10000, "seed": 0},
    "examples": {"n": 20, "seed": 0},
}


# ---------------------------------------------------------------- config / metadata / io

def load_config(path: str | Path | None = None) -> dict:
    """Evaluator settings: DEFAULT_CONFIG updated section-wise by the YAML at `path`."""
    cfg = {k: dict(v) for k, v in DEFAULT_CONFIG.items()}
    if path is not None:
        for k, v in (yaml.safe_load(Path(path).read_text()) or {}).items():
            cfg[k] = cfg.get(k, {}) | v if isinstance(v, dict) else v
    return cfg


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def sidecar(path: str | Path) -> Path | None:
    """`<stem>.meta.{yaml,yml,json}` next to a decode CSV, if present."""
    p = Path(path)
    for ext in (".meta.yaml", ".meta.yml", ".meta.json"):
        if (q := p.with_name(p.stem + ext)).exists():
            return q
    return None


def run_meta(path: str | Path, meta_file: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Run metadata: columns embedded in the CSV < sidecar (or `meta_file`) < `overrides` (CLI)."""
    meta: dict = {}
    head = pd.read_csv(path, nrows=1, keep_default_na=False)
    meta |= {k: head[k].iloc[0] for k in META_KEYS if k in head and len(head)}
    if (f := Path(meta_file) if meta_file else sidecar(path)) is not None:
        meta |= (json.loads(f.read_text()) if f.suffix == ".json" else yaml.safe_load(f.read_text())) or {}
    meta |= {k: v for k, v in (overrides or {}).items() if v is not None}
    meta.setdefault("condition", "real")
    missing = [k for k in META_KEYS if k not in meta or meta[k] in ("", None)]
    if missing:
        raise ValueError(f"{path}: missing run metadata {missing} (give them on the CLI or in a sidecar)")
    if meta["condition"] not in CONDITIONS:
        raise ValueError(f"{path}: condition {meta['condition']!r} not in {CONDITIONS}")
    meta["seed"] = int(meta["seed"])
    meta["config"] = run_config(meta)
    return meta


def run_config(meta: dict) -> str:
    """meta's `config` if given, else its model name without a "stage2 " prefix and a trailing _s<seed> (Stage 2's
    "stage2 B_ssl_qwen25-1.5b-instruct_s42" -> "B_ssl_qwen25-1.5b-instruct"), else its arm."""
    if meta.get("config") not in (None, ""):
        return str(meta["config"])
    name = re.sub(r"^stage2 ", "", str(meta.get("model") or ""))
    return re.sub(rf"_s{meta.get('seed')}$", "", name) or str(meta["arm"])


def read_decodes(path: str | Path) -> pd.DataFrame:
    """Decode CSV (trial, reference, decoded, ...); empty cells stay empty strings."""
    df = pd.read_csv(path, keep_default_na=False)
    missing = {"trial", "reference", "decoded"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if df.trial.duplicated().any():
        raise ValueError(f"{path}: duplicate trial ids")
    df["reference"], df["decoded"] = df.reference.astype(str), df.decoded.astype(str)
    return df


def _norm_key(df: pd.DataFrame, cols) -> pd.Series:
    def one(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        if isinstance(v, (float, np.floating)) and float(v).is_integer():
            return str(int(v))
        return str(v)
    return df[list(cols)].apply(lambda r: "\x1f".join(one(v) for v in r), axis=1)


def upsert_csv(path: str | Path, rows: pd.DataFrame, key: tuple[str, ...]) -> pd.DataFrame:
    """Append `rows` to the CSV at `path` with the union of columns (existing order first); existing
    rows with the same `key` are replaced. Rows lacking any key column (e.g. older writers) are kept."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()
    if len(old) and all(k in old for k in key):
        old = old[~_norm_key(old, key).isin(set(_norm_key(rows, key)))]
    out = pd.concat([old, rows], ignore_index=True, sort=False) if len(old) else rows.reset_index(drop=True)
    out.to_csv(path, index=False)
    return out


def slug(meta: dict) -> str:
    s = f"{run_config(meta)}__{meta['arm']}__{meta['condition']}__{meta['split']}__{meta['decoding']}__seed{meta['seed']}"
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", s).strip("_")


# ---------------------------------------------------------------- BERTScore

def bertscore_settings(cfg: dict) -> dict:
    """The fixed BERTScore settings plus bert-score's own hash of them (recorded with every result)."""
    b = cfg["bertscore"]
    out = {"bertscore_model": b["model_type"], "bertscore_layer": int(b["num_layers"]),
           "bertscore_idf": bool(b["idf"]), "bertscore_lang": b["lang"],
           "bertscore_rescale": bool(b["rescale_with_baseline"])}
    try:
        import bert_score
        from bert_score.utils import get_hash
        out["bertscore_hash"] = get_hash(b["model_type"], int(b["num_layers"]), bool(b["idf"]),
                                         bool(b["rescale_with_baseline"]), False, False)
        out["bertscore_version"] = bert_score.__version__
    except ImportError:
        out["bertscore_hash"] = (f"{b['model_type']}_L{b['num_layers']}_idf{b['idf']}"
                                 f"_rescaled{b['rescale_with_baseline']}")
    return out


def bertscore(cands: list[str], refs: list[str], cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BERTScore P, R, F1 per pair after textmetrics.normalize(); pairs with an empty side score 0."""
    c, r = [normalize(x) for x in cands], [normalize(x) for x in refs]
    P, R, F = (np.zeros(len(c)) for _ in range(3))
    ok = np.array([bool(a) and bool(b) for a, b in zip(c, r, strict=True)], bool)
    if ok.any():
        import bert_score
        b = cfg["bertscore"]
        p, rr, f = bert_score.score([x for x, k in zip(c, ok) if k], [x for x, k in zip(r, ok) if k],
                                    model_type=b["model_type"], num_layers=int(b["num_layers"]),
                                    idf=bool(b["idf"]), lang=b["lang"],
                                    rescale_with_baseline=bool(b["rescale_with_baseline"]),
                                    batch_size=int(b["batch_size"]), device=resolve_device(b["device"]))
        P[ok], R[ok], F[ok] = (np.asarray(x, float) for x in (p, rr, f))
    return P, R, F


# ---------------------------------------------------------------- per trial / per file

def score_trials(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per-trial word errors, reference length, sentence WER and BERTScore."""
    out = df[["trial", "reference", "decoded"]].copy()
    errs = [word_errors(r, h) for r, h in zip(out.reference, out.decoded, strict=True)]
    out["word_errors"] = [e for e, _ in errs]
    out["ref_words"] = [n for _, n in errs]
    out["sent_wer"] = out.word_errors / out.ref_words.clip(lower=1)
    out["bs_p"], out["bs_r"], out["bs_f1"] = bertscore(out.decoded.tolist(), out.reference.tolist(), cfg)
    return out


def load_or_score(path: str | Path, cfg: dict) -> pd.DataFrame:
    """Per-trial table for `path`: reused if it already holds BERTScore columns, otherwise scored."""
    df = read_decodes(path)
    if {"bs_f1", "word_errors", "ref_words"} <= set(df.columns):
        if "bertscore_hash" in df and set(df.bertscore_hash) != {bertscore_settings(cfg)["bertscore_hash"]}:
            raise ValueError(f"{path}: scored with a different BERTScore configuration")
        df["sent_wer"] = df.word_errors / df.ref_words.clip(lower=1)
        return df
    return score_trials(df, cfg)


def aggregate(trials: pd.DataFrame) -> dict:
    """Per-file metrics; `wer` is BIT's aggregate (total edits / total reference words)."""
    e, n = trials.word_errors.to_numpy(float), trials.ref_words.to_numpy(float)
    return {"n_trials": len(trials), "wer": e.sum() / max(n.sum(), 1), "mean_sent_wer": float(np.mean(e / np.maximum(n, 1))),
            "ser": float(np.mean(e > 0)), "bertscore_p": float(trials.bs_p.mean()),
            "bertscore_r": float(trials.bs_r.mean()), "bertscore_f1": float(trials.bs_f1.mean())}


def error_analysis(trials: pd.DataFrame, taus, tau_default: float) -> pd.DataFrame:
    """Among sentences with >=1 word error: meaning-preserving if BERTScore F1 >= τ, else meaning-changing."""
    err = trials[trials.word_errors > 0]
    rows = []
    for tau in sorted(set(map(float, taus)) | {float(tau_default)}):
        keep = int((err.bs_f1 >= tau).sum())
        k = len(err)
        rows.append({"tau": tau, "is_default_tau": np.isclose(tau, tau_default), "n_trials": len(trials),
                     "n_error_sentences": k, "n_meaning_preserving": keep, "n_meaning_changing": k - keep,
                     "frac_meaning_preserving": keep / k if k else np.nan,
                     "frac_meaning_changing": (k - keep) / k if k else np.nan,
                     "frac_meaning_changing_of_all": (k - keep) / len(trials) if len(trials) else np.nan})
    return pd.DataFrame(rows)


def repo_path(p: str | Path) -> str:
    """p relative to the repository when it is inside it (metrics.csv is committed: no machine-specific paths)."""
    p = Path(p).resolve()
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


def evaluate_file(path: str | Path, meta: dict, cfg: dict, out_dir: str | Path) -> tuple[dict, pd.DataFrame]:
    """Score one decode file; write per_trial/<slug>.csv; upsert metrics.csv and error_analysis.csv."""
    out_dir = Path(out_dir)
    trials = load_or_score(path, cfg)
    settings = bertscore_settings(cfg)
    ident = {k: meta[k] for k in META_KEYS} | {"config": run_config(meta), "evaluator": EVALUATOR}
    per_trial = out_dir / "per_trial" / f"{slug(meta)}.csv"
    per_trial.parent.mkdir(parents=True, exist_ok=True)
    trials.assign(**ident, bertscore_hash=settings["bertscore_hash"]).to_csv(per_trial, index=False)
    row = ident | aggregate(trials) | {"per": meta.get("per", np.nan)} | settings | {
        "decodes": repo_path(path), "per_trial": repo_path(per_trial)}
    row = {k: row[k] for k in LEGACY_COLS} | row
    upsert_csv(out_dir / "metrics.csv", pd.DataFrame([row]), RUN_KEY)
    ea = error_analysis(trials, cfg["error_analysis"]["tau_grid"], cfg["error_analysis"]["tau"])
    ea = pd.concat([pd.DataFrame([ident] * len(ea)), ea], axis=1).assign(bertscore_hash=settings["bertscore_hash"])
    upsert_csv(out_dir / "error_analysis.csv", ea, RUN_KEY + ("tau",))
    return row, ea


# ---------------------------------------------------------------- across seeds

def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean ± SD (ddof=1) over seeds per (config, arm, split, decoding, condition); only this evaluator's rows."""
    m = metrics[metrics.evaluator == EVALUATOR] if "evaluator" in metrics else metrics.iloc[:0]
    if m.bertscore_hash.nunique() > 1:
        raise ValueError(f"metrics.csv mixes BERTScore configurations: {sorted(m.bertscore_hash.unique())}")
    m = m.assign(config=m["config"].fillna(m["arm"]) if "config" in m else m["arm"])
    rows = []
    for (config, arm, split, dec, cond), g in m.groupby(["config", "arm", "split", "decoding", "condition"], sort=True):
        row = {"config": config, "arm": arm, "split": split, "decoding": dec, "condition": cond,
               "n_seeds": g.seed.nunique(),
               "seeds": " ".join(str(int(s)) for s in sorted(g.seed.unique())), "n_trials": int(g.n_trials.iloc[0])}
        if g.seed.duplicated().any():
            raise ValueError(f"duplicate seeds for {config}/{arm}/{split}/{dec}/{cond}")
        for c in ("wer", "mean_sent_wer", "ser", "bertscore_f1"):
            row[f"{c}_mean"], row[f"{c}_sd"] = g[c].mean(), g[c].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def summary_markdown(summary: pd.DataFrame) -> str:
    def pm(mu, sd, scale=1.0, nd=2):
        return f"{mu * scale:.{nd}f}" + ("" if np.isnan(sd) else f" ± {sd * scale:.{nd}f}")
    lines = ["| Config | Arm | Split | Decoding | Condition | Seeds | WER (%) | Sentence WER (%) | SER (%) | BERTScore F1 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary.itertuples():
        lines.append(f"| {_md(r.config)} | {r.arm} | {r.split} | {_md(r.decoding)} | {r.condition} | {r.n_seeds} | "
                     f"{pm(r.wer_mean, r.wer_sd, 100)} | {pm(r.mean_sent_wer_mean, r.mean_sent_wer_sd, 100)} | "
                     f"{pm(r.ser_mean, r.ser_sd, 100, 1)} | {pm(r.bertscore_f1_mean, r.bertscore_f1_sd, 1, 3)} |")
    return ("Mean ± SD over seeds (SD omitted for a single seed). WER = total edits / total reference words "
            "(BIT); BERTScore F1 rescaled with baseline.\n\n" + "\n".join(lines) + "\n")


def _md(s) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


# ---------------------------------------------------------------- side-by-side examples

def pick_examples(tables: dict[str, pd.DataFrame], n: int = 20, seed: int = 0) -> np.ndarray:
    """`n` trial ids present in every table, stratified by the first table's sentence WER: common trials
    are sorted easy -> hard (ties by trial id), cut into n equal chunks and one trial is drawn per chunk."""
    first = next(iter(tables.values()))
    common = set(first.trial)
    for t in tables.values():
        common &= set(t.trial)
    ref = first.set_index("trial").loc[sorted(common)]
    for name, t in tables.items():
        other = t.set_index("trial").loc[ref.index, "reference"].map(normalize)
        if not other.equals(ref.reference.map(normalize)):
            raise ValueError(f"{name}: references differ from the first file on shared trials")
    order = ref.assign(sw=ref.word_errors / ref.ref_words.clip(lower=1)).reset_index().sort_values(
        ["sw", "trial"], kind="stable").trial.to_numpy()
    rng = np.random.default_rng(seed)
    return np.array([rng.choice(c) for c in np.array_split(order, min(n, len(order))) if len(c)])


def examples_markdown(tables: dict[str, pd.DataFrame], trials: np.ndarray, title: str = "") -> str:
    idx = {k: t.set_index("trial") for k, t in tables.items()}
    first = next(iter(idx.values()))
    lines = [f"# Side-by-side decodes{': ' + title if title else ''}", "",
             f"{len(trials)} trials shared by all {len(tables)} files, stratified by sentence WER of "
             f"`{next(iter(tables))}` (easy → hard).", ""]
    for tr in trials:
        lines += [f"### Trial {tr}", "", "| | Sentence | WER (%) |", "|---|---|---|",
                  f"| **reference** | {_md(first.loc[tr, 'reference'])} | |"]
        for name, t in idx.items():
            r = t.loc[tr]
            lines.append(f"| {_md(name)} | {_md(r.decoded) or '*(empty)*'} | "
                         f"{100 * r.word_errors / max(r.ref_words, 1):.1f} |")
        lines.append("")
    return "\n".join(lines)


def with_word_errors(df: pd.DataFrame) -> pd.DataFrame:
    if {"word_errors", "ref_words"} <= set(df.columns):
        return df
    errs = [word_errors(r, h) for r, h in zip(df.reference, df.decoded, strict=True)]
    return df.assign(word_errors=[e for e, _ in errs], ref_words=[n for _, n in errs])


# ---------------------------------------------------------------- null controls

def _ci(boot: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def _point(e: np.ndarray, w: np.ndarray, f: np.ndarray) -> dict:
    return {"wer": e.sum() / max(w.sum(), 1), "bertscore_f1": float(f.mean())}


def null_controls(real: pd.DataFrame, nulls: dict[str, pd.DataFrame], n_boot: int = 10000,
                  seed: int = 0, chunk: int = 1000) -> pd.DataFrame:
    """Paired bootstrap over trials (shared resamples for every null and metric) of
    WER(null) - WER(real) and F1(real) - F1(null); positive = real is better. `real_better` when the
    95% CI excludes 0; p_one_sided = fraction of resamples with difference <= 0."""
    trials = set(real.trial)
    for t in nulls.values():
        trials &= set(t.trial)
    trials = sorted(trials)
    if len(trials) < len(real):
        warnings.warn(f"null controls: using {len(trials)} of {len(real)} real trials (shared by all files)")
    tabs = {"real": real.set_index("trial").loc[trials]} | {k: v.set_index("trial").loc[trials]
                                                            for k, v in nulls.items()}
    n = len(trials)
    rng = np.random.default_rng(seed)
    cols = {k: (t.word_errors.to_numpy(float), t.ref_words.to_numpy(float), t.bs_f1.to_numpy(float))
            for k, t in tabs.items()}
    boot = {(k, m): [] for k in nulls for m in ("wer", "bertscore_f1")}
    for start in range(0, n_boot, chunk):
        idx = rng.integers(0, n, size=(min(chunk, n_boot - start), n))
        e, w, f = cols["real"]
        r_wer, r_f1 = e[idx].sum(1) / np.maximum(w[idx].sum(1), 1), f[idx].mean(1)
        for k in nulls:
            e2, w2, f2 = cols[k]
            boot[(k, "wer")].append(e2[idx].sum(1) / np.maximum(w2[idx].sum(1), 1) - r_wer)
            boot[(k, "bertscore_f1")].append(r_f1 - f2[idx].mean(1))
    rows = []
    for (k, metric), b in boot.items():
        b = np.concatenate(b)
        rv, nv = (_point(*cols[x])[metric] for x in ("real", k))
        lo, hi = _ci(b)
        rows.append({"null_condition": k, "metric": metric, "n_trials": n, "real": rv, "null": nv,
                     "diff": (nv - rv) if metric == "wer" else (rv - nv), "ci_low": lo, "ci_high": hi,
                     "p_one_sided": float(np.mean(b <= 0)), "real_better": bool(lo > 0),
                     "n_resamples": n_boot, "bootstrap_seed": seed})
    out = pd.DataFrame(rows)
    missing = [c for c in NULLS if c not in nulls]
    out["beats_all_nulls"] = out.groupby("metric").real_better.transform("all") & (not missing)
    out["nulls_missing"] = " ".join(missing)
    return out

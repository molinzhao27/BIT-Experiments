#!/usr/bin/env python
"""Phase 2: perception -> intent transfer. Train a word decoder on one behavior, test on another.

    python scripts/phase2_transfer.py [--config configs/phase2.yaml] [--n-perm 1000] [--device auto]
                                      [--datasets isolated interleaved] [--arrays T12-i6v ...]
                                      [--prepare-only] [--out results/phase2] [--tag corrected]

For every array x (train behavior, test behavior):
  * the 500 ms window is chosen by 10-fold CV on the TRAINING behavior only, then moved to the
    test behavior keeping its offset from the behavior's event (stimulus onset for listening /
    reading, go cue otherwise); the decoder is fit on all training trials and tested once on all
    test-behavior trials;
  * the diagonal (train = test) is the Phase 1 nested-CV accuracy;
  * Clopper-Pearson 95% CI and permutation p = (1 + #null >= observed) / (1 + n_perm),
    Benjamini-Hochberg within each (dataset, normalization) family. Transfer null: the decoder
    trained on the true training labels is kept fixed and the test behavior's labels are permuted
    within recording blocks (every block is word-balanced). Diagonal null: the whole nested CV is
    re-run on permuted labels, with its folds re-stratified on them, as for the true labels;
  * isolated blocks: 'block' (paper) and 'strict' (train-statistics-only) normalization;
    interleaved blocks: behaviors share blocks, so the paper normalization cannot single out
    one behavior (drift control).
--prepare-only builds the feature caches (CPU) so the GPU lock is held only for the permutations.
Output names carry --tag (default 'corrected'): the untagged files in results/phase2 are the first run,
whose null shuffled the training labels and kept the diagonal's folds fixed (audit 2026-10-19, item 5).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kunzdecoder import batched, models  # noqa: E402
from kunzdecoder.decoding import clopper_pearson, nested_folds, stratified_folds  # noqa: E402
from kunzdecoder.features import load_features, raw_block_means, window_means  # noqa: E402
from kunzdecoder.io import parse_cue  # noqa: E402
from kunzdecoder.plotting import matrix_grid, tgm_grid  # noqa: E402
from kunzdecoder.stats import bh  # noqa: E402
from kunzdecoder.windows import (interleaved_starts_event, isolated_event_ms, isolated_starts_ms,  # noqa: E402
                                 session_file)

WORDS = ["ban", "choice", "day", "feel", "kite", "though", "were"]
K = len(WORDS)
CHANCE = 1 / K


@dataclass
class Behavior:
    """Word trials of one behavior plus the features of the file they come from."""

    name: str
    participant: str
    bin_ms: int
    y: np.ndarray  # (n,) word index
    go: np.ndarray  # (n,) go-onset bin
    block: np.ndarray  # (T,) block id per bin (whole file)
    starts_ms: np.ndarray  # candidate window starts, rel. go
    event_ms: int  # behavior event, rel. go
    feats: np.ndarray  # (T, 2C) paper-normalized features
    n_ch: int
    arrays: dict
    smooth: np.ndarray | None = None  # (T, 2C) smoothed raw features (strict mode)
    block_ids: np.ndarray | None = None
    block_table: np.ndarray | None = None
    grand: np.ndarray | None = None

    def cols(self, array: str) -> np.ndarray:
        ch = self.arrays[array]
        return np.concatenate([ch, ch + self.n_ch])


def word_trials(s, participant: str, behavior: str | None = None):
    """(mask, word index) for word trials; behavior filters interleaved cues."""
    parsed = [parse_cue(participant, s.cue_names[c - 1]) for c in s.cues]
    mask = np.array([w is not None and (behavior is None or b == behavior) for w, b in parsed])
    y = np.array([WORDS.index(w) if w is not None else -1 for w, _ in parsed])
    return mask, y[mask]


def load_isolated(cfg1: dict, participant: str, behavior: str, strict: bool) -> Behavior:
    path = session_file(ROOT / "data" / "raw" / cfg1["task"], participant, behavior)
    sigma, trunc = cfg1["preprocess"]["smoothing_sigma_ms"], cfg1["preprocess"]["smoothing_truncate"]
    f = load_features(path, ROOT / "data" / "processed", sigma, trunc)
    s = f.session
    mask, y = word_trials(s, participant)
    b = Behavior(behavior, participant, s.bin_ms, y, s.go[mask, 0], s.block,
                 isolated_starts_ms(cfg1, participant, behavior), isolated_event_ms(cfg1, participant, behavior),
                 f.x, s.tx.shape[1], s.arrays)
    if strict:
        b.smooth = load_features(path, ROOT / "data" / "processed", sigma, trunc, center_blocks=False,
                                 zscore=False).x
        b.block_ids, b.block_table, b.grand = raw_block_means(s)
    return b


def load_interleaved(cfg1: dict, cfg: dict, participant: str) -> dict[str, Behavior]:
    icfg = cfg["interleaved"]["sessions"][participant]
    task_dir = ROOT / "data" / "raw" / cfg["interleaved"]["task"]
    path = next(task_dir.glob(f"{participant.lower()}.*_raw.mat"))
    f = load_features(path, ROOT / "data" / "processed", cfg1["preprocess"]["smoothing_sigma_ms"],
                      cfg1["preprocess"]["smoothing_truncate"])
    s = f.session
    present = {parse_cue(participant, s.cue_names[c - 1])[1] for c in s.cues}
    out = {}
    for beh in cfg1["behaviors"]:
        if beh not in present:
            continue
        mask, y = word_trials(s, participant, beh)
        starts, event = interleaved_starts_event(cfg1, icfg, beh)
        out[beh] = Behavior(beh, participant, s.bin_ms, y, s.go[mask, 0], s.block, starts, event, f.x,
                            s.tx.shape[1], s.arrays)
    return out


def windows(F: np.ndarray, beh: Behavior, starts_ms: np.ndarray, length_ms: int) -> np.ndarray:
    return window_means(F, beh.go, beh.block, starts_ms // beh.bin_ms, length_ms // beh.bin_ms)


def strict_features(a: Behavior, cols: np.ndarray):
    """Train features centered per training block; test features centered with the training
    data's raw grand mean; both scaled by training statistics. Test data is never used."""
    Ta = a.smooth[:, cols].astype(np.float64) - a.block_table[:, cols][np.searchsorted(a.block_ids, a.block)]
    c, sd = Ta.mean(0), Ta.std(0)
    sd[sd == 0] = 1.0
    train = ((Ta - c) / sd).astype(np.float32)
    return train, lambda b: ((b.smooth[:, cols] - a.grand[cols] - c) / sd).astype(np.float32)


def perm_rows(y: np.ndarray, n_perm: int, seed: int, key: str, groups: np.ndarray | None = None) -> np.ndarray:
    rng = np.random.default_rng([seed, zlib.crc32(key.encode())])
    return batched.permuted_labels(y, n_perm, rng, groups)


def chunked(fn, rows: np.ndarray, chunk: int, device) -> dict:
    """Run fn over permutation rows in chunks; concatenate tensor outputs as numpy."""
    parts = []
    for i in range(0, len(rows), chunk):
        out = fn(torch.as_tensor(rows[i:i + chunk], device=device))
        parts.append({k: v.cpu().numpy() for k, v in out.items()})
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def to_t(x, device):
    return torch.as_tensor(np.asarray(x, np.float64), device=device)


def fold_t(folds, device):
    return [(torch.as_tensor(tr, device=device), torch.as_tensor(va, device=device)) for tr, va in folds]


def p_value(null: np.ndarray, obs: float) -> float:
    return (1 + int((null >= obs).sum())) / (1 + len(null))


def run_array(dataset: str, label: str, priority: bool, array: str, behs: dict[str, Behavior], cfg: dict,
              cfg1: dict, norms: list[str], n_perm: int, device, nulls: dict, checks: dict) -> list[dict]:
    """All train x test cells (and the within-behavior diagonal) for one array."""
    L = cfg1["window"]["length_ms"]
    seed, chunk = cfg["seed"], cfg["perm_chunk"]
    rows = []
    for a_name, a in behs.items():
        cols = a.cols(array)
        rows_a = perm_rows(a.y, n_perm, seed, f"{dataset}|{label}|{a_name}")
        inner = fold_t(stratified_folds(a.y, cfg["inner_folds"], seed), device)
        y_a = torch.as_tensor(a.y[None], device=device)
        for norm in norms:
            if norm == "block":
                Fa, test_feats = a.feats[:, cols], (lambda b, cols=cols: b.feats[:, cols])
            else:
                Fa, test_feats = strict_features(a, cols)
            Xa = to_t(windows(Fa, a, a.starts_ms, L), device)
            tests, meta = {}, {}
            for b_name, b in behs.items():
                if b_name == a_name:
                    continue
                starts_b = a.starts_ms - a.event_ms + b.event_ms  # keep the offset from the event
                tests[b_name] = to_t(windows(test_feats(b), b, starts_b, L), device)
                meta[b_name] = (b, starts_b)
            out = batched.transfer_predict(Xa, y_a, K, inner, tests)
            w = int(out["windows"][0])
            for b_name, (b, starts_b) in meta.items():
                pred = out[b_name][0].cpu().numpy()
                rows_b = perm_rows(b.y, n_perm, seed, f"{dataset}|{label}|{norm}|{a_name}|{b_name}",
                                   groups=b.block[b.go])
                hits = (rows_b == pred[None]).sum(1)
                obs, null = int(hits[0]), hits[1:]
                n = len(b.y)
                lo, hi = clopper_pearson(obs, n)
                nulls[f"{dataset}|{label}|{norm}|{a_name}|{b_name}"] = null.astype(np.int16)
                rows.append({"dataset": dataset, "participant": a.participant, "array": array, "label": label,
                             "priority": priority, "norm": norm, "kind": "transfer", "train": a_name,
                             "test": b_name, "n_test": n, "n_correct": obs, "accuracy": obs / n, "ci_low": lo,
                             "ci_high": hi, "null_mean": null.mean() / n,
                             "null_p95": np.percentile(null, 95) / n, "p_perm": p_value(null, obs),
                             "train_window_start_ms": int(a.starts_ms[w]),
                             "offset_from_event_ms": int(a.starts_ms[w] - a.event_ms),
                             "test_window_start_ms": int(starts_b[w])})
        # within-behavior diagonal: Phase 1 nested CV (paper normalization); its null re-runs the
        # nested CV on permuted labels with folds re-stratified on them
        Xw = to_t(windows(a.feats[:, cols], a, a.starts_ms, L), device)
        folds_of = lambda yy: nested_folds(yy, cfg1["cv"]["outer_folds"], cfg1["cv"]["inner_folds"], seed)  # noqa: E731
        outer, inner_n = folds_of(a.y)
        out = batched.nested_cv(Xw, y_a, K, fold_t(outer, device), [fold_t(f, device) for f in inner_n])
        obs = int(out["correct"][0])
        refold = chunked(lambda lab: batched.nested_cv_refold(Xw, lab, K, folds_of), rows_a, chunk, device)
        null = refold["correct"][1:]
        checks["refold_row0_matches"] = checks.get("refold_row0_matches", 0) + int(refold["correct"][0] == obs)
        checks["diagonal_cells"] = checks.get("diagonal_cells", 0) + 1
        n = len(a.y)
        lo, hi = clopper_pearson(obs, n)
        nulls[f"{dataset}|{label}|within|{a_name}|{a_name}"] = null.astype(np.int16)
        rows.append({"dataset": dataset, "participant": a.participant, "array": array, "label": label,
                     "priority": priority, "norm": "within", "kind": "within", "train": a_name, "test": a_name,
                     "n_test": n, "n_correct": obs, "accuracy": obs / n, "ci_low": lo, "ci_high": hi,
                     "null_mean": null.mean() / n, "null_p95": np.percentile(null, 95) / n,
                     "p_perm": p_value(null, obs),
                     "train_window_start_ms": int(np.median(a.starts_ms[out["windows"][0]])),
                     "offset_from_event_ms": None, "test_window_start_ms": None})
    return rows


def tgm(a: Behavior, b: Behavior, array: str, L: int) -> dict:
    """Diagnostic train-time x test-time accuracy map (paper normalization, no selection)."""
    Xa = windows(a.feats[:, a.cols(array)], a, a.starts_ms, L)
    Xb = windows(b.feats[:, b.cols(array)], b, b.starts_ms, L).astype(np.float64)
    m = models.fit(Xa, a.y, np.arange(K), "diag_lda")
    acc = np.empty((Xa.shape[1], Xb.shape[1]))
    for wa in range(Xa.shape[1]):
        mu, inv = m["means"][wa], 1.0 / m["var"][wa, 0]
        ll = -((Xb[:, :, None, :] - mu[None, None]) ** 2 * inv).sum(-1)  # (n_b, W_b, K)
        acc[wa] = (ll.argmax(-1) == b.y[:, None]).mean(0)
    return {"acc": 100 * acc, "t_train": (a.starts_ms + L / 2 - a.event_ms) / 1000,
            "t_test": (b.starts_ms + L / 2 - b.event_ms) / 1000}


def matrices(df: pd.DataFrame, dataset: str, label: str, norm: str, order: list[str]):
    acc = np.full((len(order), len(order)), np.nan)
    sig = np.zeros_like(acc, bool)
    d = df[(df.dataset == dataset) & (df.label == label) & ((df.norm == norm) | (df.kind == "within"))]
    for r in d.itertuples():
        if r.train in order and r.test in order:
            i, j = order.index(r.train), order.index(r.test)
            acc[i, j], sig[i, j] = 100 * r.accuracy, r.significant
    return acc, sig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "phase2.yaml")
    ap.add_argument("--n-perm", type=int)
    ap.add_argument("--device")
    ap.add_argument("--datasets", nargs="+", default=["isolated", "interleaved"])
    ap.add_argument("--arrays", nargs="+", help="array labels to run (default: all in config)")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--figures-only", action="store_true", help="redraw figures/tables from saved results")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "phase2")
    ap.add_argument("--tag", default="corrected", help="suffix of every output file name ('' = none)")
    args = ap.parse_args()
    TAG[0] = f"_{args.tag}" if args.tag else ""
    cfg = yaml.safe_load(args.config.read_text())
    cfg1 = yaml.safe_load((ROOT / cfg["phase1_config"]).read_text())
    n_perm = cfg["n_permutations"] if args.n_perm is None else args.n_perm
    device = batched.device_of(args.device or cfg["device"])
    arrays = [a for a in cfg["arrays"] if not args.arrays or a[2] in args.arrays]
    participants = sorted({a[0] for a in arrays})
    t0 = time.time()

    if args.figures_only:
        df = pd.read_csv(args.out / tagged("transfer_results.csv"))
        L = cfg1["window"]["length_ms"]
        part = {a[2]: a[0] for a in cfg["arrays"]}
        tgms = {}
        for key, acc in np.load(args.out / tagged("tgm.npz")).items():
            label, a, b = key.split("|")
            p = part[label]
            t = {x: (isolated_starts_ms(cfg1, p, x) + L / 2 - isolated_event_ms(cfg1, p, x)) / 1000 for x in (a, b)}
            tgms[key] = {"acc": acc, "t_train": t[a], "t_test": t[b]}
        make_figures(df, tgms, cfg, cfg1, args.out)
        write_tables(df, cfg, cfg1, args.out)
        return 0

    if args.prepare_only:
        for p in participants:
            for beh in cfg1["behaviors"]:
                load_isolated(cfg1, p, beh, strict=True)
            load_interleaved(cfg1, cfg, p)
            print(f"cached {p} ({time.time() - t0:.0f}s)", flush=True)
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"device={device} n_perm={n_perm} arrays={[a[2] for a in arrays]}", flush=True)
    rows, nulls, tgms, checks = [], {}, {}, {}
    for p in participants:
        p_arrays = [a for a in arrays if a[0] == p]
        if "isolated" in args.datasets:
            behs = {b: load_isolated(cfg1, p, b, strict="strict" in cfg["norm_modes"]) for b in cfg1["behaviors"]}
            for _, arr, label, prio in p_arrays:
                t1 = time.time()
                rows += run_array("isolated", label, prio, arr, behs, cfg, cfg1, cfg["norm_modes"], n_perm,
                                  device, nulls, checks)
                print(f"isolated   {label:10s} {time.time() - t1:6.1f}s", flush=True)
                if label in cfg["tgm"]["arrays"]:
                    for a in cfg["tgm"]["train"]:
                        for b in cfg["tgm"]["test"]:
                            tgms[f"{label}|{a}|{b}"] = tgm(behs[a], behs[b], arr, cfg1["window"]["length_ms"])
            del behs
        if "interleaved" in args.datasets:
            behs = load_interleaved(cfg1, cfg, p)
            for _, arr, label, prio in p_arrays:
                if arr not in next(iter(behs.values())).arrays:
                    continue
                t1 = time.time()
                rows += run_array("interleaved", label, prio, arr, behs, cfg, cfg1, ["block"], n_perm, device,
                                  nulls, checks)
                print(f"interleaved {label:10s} {time.time() - t1:6.1f}s  ({', '.join(behs)})", flush=True)
            del behs

    df = pd.DataFrame(rows)
    df["family"] = df.dataset + "|" + df.norm
    df["q_bh"] = df.groupby("family").p_perm.transform(bh)
    df["significant"] = df.q_bh < cfg["alpha"]
    df["significant_ci"] = df.ci_low > CHANCE
    within = df[df.kind == "within"].set_index(["dataset", "label", "test"]).accuracy
    df["within_test_accuracy"] = [within.get((r.dataset, r.label, r.test), np.nan) for r in df.itertuples()]
    df["transfer_index"] = (df.accuracy - CHANCE) / (df.within_test_accuracy - CHANCE)
    df.to_csv(args.out / tagged("transfer_results.csv"), index=False)
    np.savez_compressed(args.out / tagged("null_distributions.npz"), **nulls)
    np.savez_compressed(args.out / tagged("tgm.npz"), **{k: v["acc"] for k, v in tgms.items()})
    changes = significance_changes(df, args.out / "transfer_results.csv") if TAG[0] else None
    if changes is not None:
        changes.to_csv(args.out / tagged("significance_changes.csv"), index=False)

    # within-behavior diagonal must reproduce Phase 1 (same folds, same model, float64)
    p1 = ROOT / "results" / "phase1" / "accuracy_primary.csv"
    check = {}
    if p1.exists() and (df.dataset == "isolated").any():
        ref = pd.read_csv(p1).set_index(["label", "behavior"]).n_correct
        w = df[(df.dataset == "isolated") & (df.kind == "within")]
        same = [int(r.n_correct) == int(ref.get((r.label, r.train), -1)) for r in w.itertuples()]
        check = {"within_cells": len(same), "identical_to_phase1": int(sum(same))}

    make_figures(df, tgms, cfg, cfg1, args.out)
    write_tables(df, cfg, cfg1, args.out)
    summary = {"n_permutations": n_perm, "device": str(device), "runtime_s": round(time.time() - t0, 1),
               "phase1_check": check, "refold_check": checks, "n_cells": len(df),
               "n_significant": df.groupby("family").significant.sum().astype(int).to_dict(),
               "n_significance_changes": None if changes is None else len(changes),
               "torch": torch.__version__, "config": cfg}
    (args.out / tagged("summary.json")).write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: summary[k] for k in ("runtime_s", "phase1_check", "refold_check", "n_significant",
                                              "n_significance_changes")}, indent=1))
    return 0


TAG = [""]  # output-name suffix, set from --tag


def tagged(name: str) -> str:
    """'transfer_results.csv' -> 'transfer_results_corrected.csv' (for --tag corrected)."""
    stem, ext = name.split(".", 1)
    return f"{stem}{TAG[0]}.{ext}"


def significance_changes(df: pd.DataFrame, original: Path) -> pd.DataFrame | None:
    """Cells whose q < alpha call differs from the untagged first run (same cells, same accuracies)."""
    if not original.exists():
        return None
    key = ["dataset", "label", "norm", "train", "test"]
    old = pd.read_csv(original)[key + ["n_correct", "p_perm", "q_bh", "significant"]]
    m = df[key + ["participant", "priority", "kind", "accuracy", "n_correct", "p_perm", "q_bh", "significant"]].merge(
        old, on=key, suffixes=("", "_old"), validate="1:1")
    if not (m.n_correct == m.n_correct_old).all():
        raise ValueError("point accuracies differ from the original run")
    return m[m.significant != m.significant_old].drop(columns="n_correct_old")


def make_figures(df: pd.DataFrame, tgms: dict, cfg: dict, cfg1: dict, out: Path) -> None:
    names = cfg1["behaviors"]
    order = list(names)
    short = {"attempted": "Attempted", "mouthed": "Mimed", "imagined1stMotor": "Motoric inner",
             "imagined1stAuditory": "Auditory inner", "imaginedListening": "Imag. listening",
             "listening": "Listening", "reading": "Reading"}
    labels = [a[2] for a in cfg["arrays"] if a[2] in set(df.label)]
    note = ("Diagonal (outlined) = within-behavior nested CV. Off-diagonal: window chosen on the train behavior only, "
            "moved to the test behavior at the same offset from its event. Hatched + n.s. = permutation q ≥ 0.05 (BH).")
    iso = df[df.dataset == "isolated"]
    for prio, fname in ((True, "phase2_isolated_transfer_priority.png"), (False, "phase2_isolated_transfer_other.png")):
        panels = []
        for label in labels:
            if not ((iso.label == label) & (iso.priority == prio)).any():
                continue
            for norm in cfg["norm_modes"]:
                acc, sig = matrices(iso, "isolated", label, norm, order)
                panels.append({"acc": acc, "sig": sig, "title": f"{label} · {norm} normalization"})
        if panels:
            matrix_grid(panels, [short[b] for b in order], [short[b] for b in order], ROOT / "figures" / tagged(fname),
                        f"Isolated blocks: train behavior → test behavior word accuracy "
                        f"({'i6v / 55b arrays' if prio else 'other significant arrays'})", ncols=4, note=note)
    inter = df[df.dataset == "interleaved"]
    panels = []
    for label in labels:
        d = inter[inter.label == label]
        if len(d):
            o = [b for b in order if b in set(d.train)]
            acc, sig = matrices(inter, "interleaved", label, "block", o)
            panels.append({"acc": acc, "sig": sig, "title": label, "labels": [short[b] for b in o]})
    if panels:
        matrix_grid(panels, [], [], ROOT / "figures" / tagged("phase2_interleaved_transfer.png"),
                    "Interleaved blocks (3 behaviors mixed within every block): train → test word accuracy",
                    ncols=5, note=note)
    for label in cfg["tgm"]["arrays"]:
        ps = []
        for a in cfg["tgm"]["train"]:
            for b in cfg["tgm"]["test"]:
                key = f"{label}|{a}|{b}"
                if key not in tgms:
                    continue
                r = iso[(iso.label == label) & (iso.norm == "block") & (iso.train == a) & (iso.test == b)]
                used = None
                if len(r):
                    t = (r.offset_from_event_ms.iloc[0] + cfg1["window"]["length_ms"] / 2) / 1000
                    used = (t, t)
                ps.append(tgms[key] | {"title": f"{short[a]} → {short[b]}", "used": used})
        if ps:
            tgm_grid(ps, ROOT / "figures" / tagged(f"phase2_tgm_{label}.png"),
                     f"{label}: train-time × test-time generalization (diagnostic only; paper normalization)",
                     ncols=len(cfg["tgm"]["test"]),
                     note="Dashed: same offset from each behavior's event (where the transfer protocol evaluates). "
                          "Ring: window the protocol chose on the training behavior.")


def write_tables(df: pd.DataFrame, cfg: dict, cfg1: dict, out: Path) -> None:
    short = {"attempted": "Att", "mouthed": "Mim", "imagined1stMotor": "Motor", "imagined1stAuditory": "Aud",
             "imaginedListening": "ImagL", "listening": "Listen", "reading": "Read"}

    def cell(r):
        return f"{100 * r.accuracy:.1f}{'*' if r.significant else ''}"

    lines = ["Accuracy % on the test behavior; * = permutation q < 0.05 (BH within dataset × normalization). "
             "Chance 14.3%. Columns: paper normalization / strict (train-statistics only).", ""]
    iso = df[(df.dataset == "isolated") & (df.priority)]
    labels = [a[2] for a in cfg["arrays"] if a[3] and a[2] in set(iso.label)]
    lines.append("| train → test | " + " | ".join(labels) + " |")
    lines.append("|---|" + "---|" * len(labels))
    for a, b in cfg["key_pairs"]:
        cells = []
        for label in labels:
            d = iso[(iso.label == label) & (iso.train == a) & (iso.test == b)].set_index("norm")
            cells.append(" / ".join(cell(d.loc[n]) for n in cfg["norm_modes"] if n in d.index) or "–")
        lines.append(f"| {short[a]} → {short[b]} | " + " | ".join(cells) + " |")
    lines += ["", "Within-behavior (diagonal, nested CV):", "",
              "| behavior | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for b in cfg1["behaviors"]:
        cells = [cell(iso[(iso.label == lb) & (iso.kind == "within") & (iso.test == b)].iloc[0]) for lb in labels]
        lines.append(f"| {short[b]} | " + " | ".join(cells) + " |")
    inter = df[(df.dataset == "interleaved") & (df.kind == "transfer")]
    if len(inter):
        lines += ["", "Interleaved sessions (shared blocks), all transfer cells:", "",
                  "| array | train → test | accuracy % [95% CI] | p (perm) | q |", "|---|---|---|---|---|"]
        for r in inter.sort_values(["label", "train", "test"]).itertuples():
            lines.append(f"| {r.label} | {short[r.train]} → {short[r.test]} | {100 * r.accuracy:.1f} "
                         f"[{100 * r.ci_low:.0f}–{100 * r.ci_high:.0f}] | {r.p_perm:.3f} | {r.q_bh:.3f} |")
    (out / tagged("key_transfers.md")).write_text("\n".join(lines) + "\n")

    # every matrix, every cell: accuracy [95% CI] and permutation q
    m = ["Rows = train behavior, columns = test behavior. Cell: accuracy % [Clopper-Pearson 95% CI] q=BH-adjusted "
         "permutation p (1000 permutations; * = q < 0.05). Diagonal = within-behavior nested CV. Null: off-diagonal "
         "= fixed decoder, test labels permuted within blocks; diagonal = nested CV re-run on permuted labels with "
         "re-stratified folds.", ""]
    for (dataset, label), g in df.groupby(["dataset", "label"], sort=False):
        order = [b for b in cfg1["behaviors"] if b in set(g.train)]
        for norm in sorted(set(g.norm) - {"within"}):
            m += [f"### {dataset} · {label} · {norm} normalization", "",
                  "| train \\ test | " + " | ".join(short[b] for b in order) + " |", "|---|" + "---|" * len(order)]
            for a in order:
                cells = []
                for b in order:
                    r = g[(g.train == a) & (g.test == b) & ((g.norm == norm) | (g.kind == "within"))].iloc[0]
                    cells.append(f"{100 * r.accuracy:.1f} [{100 * r.ci_low:.0f}–{100 * r.ci_high:.0f}] "
                                 f"q={r.q_bh:.3f}{'*' if r.significant else ''}")
                m.append(f"| {short[a]} | " + " | ".join(cells) + " |")
            m.append("")
    (out / tagged("matrices.md")).write_text("\n".join(m) + "\n")


if __name__ == "__main__":
    sys.exit(main())

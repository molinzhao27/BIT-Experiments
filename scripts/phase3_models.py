#!/usr/bin/env python
"""Phase 3: compare decoders against the paper's baseline on the Fig 1E task (84 conditions).

    python scripts/phase3_models.py --models lda_shrinkage logreg svm [--n-jobs 24]   # CPU
    python scripts/phase3_models.py --models cnn gru [--device cuda]                  # GPU
    python scripts/phase3_models.py --compare-only                                    # table + figure
    python scripts/phase3_models.py --uncertainty-only                                # clustered intervals only

Every model sees the same word trials, the same outer folds (seed 42) and the same inner
folds as Phase 1. Window-average models (shrinkage LDA, logistic regression, linear SVM)
choose the 500 ms window and their regularization strength in the inner folds. Sequence
models (1D-CNN, GRU) read the binned time series over the span the window search covers and
choose weight decay and training length in the inner folds. Outputs go to results/phase3/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kunzdecoder.decoding import clopper_pearson, nested_folds  # noqa: E402
from kunzdecoder.features import load_features, window_means  # noqa: E402
from kunzdecoder.stats import bh  # noqa: E402
from kunzdecoder.windows import isolated_starts_ms, session_file  # noqa: E402

OUT = ROOT / "results" / "phase3"


FILTER: dict = {"arrays": None, "behaviors": None}  # set from --arrays / --behaviors (smoke tests)


def conditions(cfg1: dict):
    """Yield (participant, array, label, behavior, Features) for the 84 Phase 1 conditions."""
    task_dir = ROOT / "data" / "raw" / cfg1["task"]
    by_p: dict[str, list] = {}
    for p, arr, label in cfg1["arrays"]:
        if FILTER["arrays"] is None or label in FILTER["arrays"]:
            by_p.setdefault(p, []).append((arr, label))
    for p, arrays in by_p.items():
        for beh in cfg1["behaviors"]:
            if FILTER["behaviors"] is not None and beh not in FILTER["behaviors"]:
                continue
            f = load_features(session_file(task_dir, p, beh), ROOT / "data" / "processed",
                              cfg1["preprocess"]["smoothing_sigma_ms"], cfg1["preprocess"]["smoothing_truncate"])
            for arr, label in arrays:
                yield p, arr, label, beh, f


def trial_data(f, cfg1: dict, p: str, beh: str):
    s = f.session
    words = s.cues >= 2
    return s, words, s.cues[words], np.flatnonzero(words), isolated_starts_ms(cfg1, p, beh)


def save(model: str, rows: list, preds: list, extra: list) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / f"accuracy_{model}.csv", index=False)
    pd.concat(preds).to_csv(OUT / f"predictions_{model}.csv.gz", index=False)
    pd.DataFrame(extra).to_csv(OUT / f"selection_{model}.csv", index=False)


def summary_row(p, arr, label, beh, y, pred, cfg1):
    k, n = int((pred == y).sum()), len(y)
    lo, hi = clopper_pearson(k, n)
    return {"participant": p, "array": arr, "label": label, "behavior": beh, "n_trials": n, "n_correct": k,
            "accuracy": k / n, "ci_low": lo, "ci_high": hi, "significant": lo > cfg1["chance"]}


def run_window_models(models: list[str], cfg: dict, cfg1: dict, n_jobs: int) -> None:
    from joblib import Parallel, delayed

    from kunzdecoder.linear import outer_fold_job

    L = cfg1["window"]["length_ms"]
    for model in models:
        grid = cfg["window_models"][model]["grid"]
        t0 = time.time()
        jobs, meta = [], []
        for p, arr, label, beh, f in conditions(cfg1):
            s, words, y, trial_idx, starts_ms = trial_data(f, cfg1, p, beh)
            X = window_means(f.x[:, f.array_columns(arr)], s.go[words, 0], s.block, starts_ms // s.bin_ms,
                             L // s.bin_ms).astype(np.float64)
            outer, inner = nested_folds(y, cfg1["cv"]["outer_folds"], cfg1["cv"]["inner_folds"], cfg["seed"])
            for k, (tr, te) in enumerate(outer):
                jobs.append(delayed(outer_fold_job)(model, grid, X, y, tr, te, inner[k]))
                meta.append((p, arr, label, beh, k, y, trial_idx, starts_ms))
        print(f"[{model}] {len(jobs)} outer-fold jobs on {n_jobs} workers", flush=True)
        results = Parallel(n_jobs=n_jobs, verbose=5)(jobs)
        rows, preds, extra, by_cond = [], [], [], {}
        for (p, arr, label, beh, k, y, trial_idx, starts_ms), r in zip(meta, results):
            c = by_cond.setdefault((p, arr, label, beh), {"y": y, "trial": trial_idx, "pred": np.empty_like(y),
                                                           "fold": np.empty(len(y), int), "win": np.empty(len(y), int)})
            c["pred"][r["te"]], c["fold"][r["te"]], c["win"][r["te"]] = r["pred"], k, starts_ms[r["window_idx"]]
            extra.append({"participant": p, "label": label, "behavior": beh, "fold": k,
                          "window_start_ms": int(starts_ms[r["window_idx"]]), "hp": r["hp"]})
        for (p, arr, label, beh), c in by_cond.items():
            rows.append(summary_row(p, arr, label, beh, c["y"], c["pred"], cfg1))
            preds.append(pd.DataFrame({"participant": p, "array": arr, "behavior": beh, "trial": c["trial"],
                                       "fold": c["fold"], "y_true": c["y"], "y_pred": c["pred"],
                                       "window_start_ms": c["win"]}))
        save(model, rows, preds, extra)
        print(f"[{model}] done in {time.time() - t0:.0f}s; mean accuracy "
              f"{100 * np.mean([r['accuracy'] for r in rows]):.1f}%", flush=True)


def run_sequence_models(models: list[str], cfg: dict, cfg1: dict, device_name: str) -> None:
    import torch

    from kunzdecoder import batched, seqmodels

    device = batched.device_of(device_name)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    st = cfg["seq_train"]
    J, wds, ckpts = st["jitter_bins"], st["weight_decay"], st["checkpoints"]
    for model in models:
        mcfg = cfg["sequence_models"][model]
        t0 = time.time()
        rows, preds, extra = [], [], []
        for p, arr, label, beh, f in conditions(cfg1):
            t1 = time.time()
            s, words, y, trial_idx, starts_ms = trial_data(f, cfg1, p, beh)
            lo = starts_ms.min() - J * st["bin_ms"]
            hi = starts_ms.max() + cfg1["window"]["length_ms"] + J * st["bin_ms"]
            steps = lo + st["bin_ms"] * np.arange((hi - lo) // st["bin_ms"])
            Xs = window_means(f.x[:, f.array_columns(arr)], s.go[words, 0], s.block, steps // s.bin_ms,
                              st["bin_ms"] // s.bin_ms)
            X = torch.as_tensor(Xs, device=device)
            yc = y - 2  # word index 0..6
            yt = torch.as_tensor(yc, device=device)
            n = len(y)
            outer, inner = nested_folds(y, cfg1["cv"]["outer_folds"], cfg1["cv"]["inner_folds"], cfg["seed"])
            key = zlib.crc32(f"{model}|{label}|{beh}".encode())

            # A: every outer fold x inner fold x weight decay, trained together; pick (wd, epochs) per outer fold
            specs = [(k, i, c) for k in range(len(outer)) for i in range(len(inner[k])) for c in range(len(wds))]
            correct = np.zeros((len(outer), len(ckpts), len(wds)))
            for b0 in range(0, len(specs), mcfg["models_per_batch"]):
                batch = specs[b0:b0 + mcfg["models_per_batch"]]
                tm = torch.zeros(len(batch), n, device=device)
                vals = []
                for j, (k, i, c) in enumerate(batch):
                    tr = outer[k][0]
                    tm[j, tr[inner[k][i][0]]] = 1
                    vals.append(tr[inner[k][i][1]])
                logits = seqmodels.train_eval(model, mcfg["arch"], X, yt, tm,
                                              torch.tensor([wds[c] for _, _, c in batch], device=device), ckpts,
                                              seed=key + b0, lr=st["lr"], jitter=J, noise=st["input_noise"])
                for j, (k, i, c) in enumerate(batch):
                    correct[k, :, c] += (logits[:, j, vals[j]].argmax(-1) == yc[vals[j]]).sum(-1)
            choice = [np.unravel_index(np.argmax(correct[k]), correct[k].shape) for k in range(len(outer))]

            # B: refit each outer fold's model on its whole training set with the chosen settings
            tm = torch.zeros(len(outer), n, device=device)
            for k, (tr, _) in enumerate(outer):
                tm[k, tr] = 1
            logits = seqmodels.train_eval(model, mcfg["arch"], X, yt, tm,
                                          torch.tensor([wds[c] for _, c in choice], device=device), ckpts,
                                          seed=key + 99991, lr=st["lr"], jitter=J, noise=st["input_noise"])
            pred, fold = np.empty(n, int), np.empty(n, int)
            for k, (_, te) in enumerate(outer):
                ck, c = choice[k]
                pred[te], fold[te] = logits[ck, k, te].argmax(-1) + 2, k
                extra.append({"participant": p, "label": label, "behavior": beh, "fold": k,
                              "epochs": ckpts[ck], "weight_decay": wds[c],
                              "inner_acc": correct[k, ck, c] / len(outer[k][0])})
            rows.append(summary_row(p, arr, label, beh, y, pred, cfg1))
            preds.append(pd.DataFrame({"participant": p, "array": arr, "behavior": beh, "trial": trial_idx,
                                       "fold": fold, "y_true": y, "y_pred": pred}))
            print(f"[{model}] {label:10s} {beh:20s} {100 * rows[-1]['accuracy']:5.1f}%  "
                  f"T={X.shape[1]} {time.time() - t1:5.1f}s", flush=True)
        save(model, rows, preds, extra)
        print(f"[{model}] done in {time.time() - t0:.0f}s; mean accuracy "
              f"{100 * np.mean([r['accuracy'] for r in rows]):.1f}%", flush=True)


def compare(cfg: dict, cfg1: dict) -> None:
    from scipy.stats import binomtest, wilcoxon

    from kunzdecoder.plotting import delta_strip

    key = ["participant", "array", "behavior", "trial"]
    base = pd.read_csv(ROOT / cfg["baseline"])
    labels = {"baseline": "Gaussian NB, pooled variance (paper)"}
    labels |= {m: v["label"] for m, v in cfg["window_models"].items()}
    labels |= {m: v["label"] for m, v in cfg["sequence_models"].items()}
    inputs = {"baseline": "500 ms window mean"} | {m: "500 ms window mean" for m in cfg["window_models"]}
    inputs |= {m: "binned time series" for m in cfg["sequence_models"]}
    paper_rows = {a[2] for a in cfg1["arrays"][:cfg1["n_paper_rows"]]}
    inner_speech = {"imagined1stMotor", "imagined1stAuditory", "imaginedListening"}
    rng = np.random.default_rng(cfg["seed"])
    per_cond, table, deltas = [], [], {}
    for model in labels:
        if model == "baseline":
            pr = base[key + ["y_true", "y_pred"]]
        else:
            path = OUT / f"predictions_{model}.csv.gz"
            if not path.exists():
                continue
            pr = pd.read_csv(path)[key + ["y_true", "y_pred"]]
        m = base[key + ["y_pred"]].rename(columns={"y_pred": "base_pred"}).merge(pr, on=key, validate="1:1")
        if model != "baseline" and len(m) != len(pr):
            raise ValueError(f"{model}: trials do not line up with the baseline")
        if len(m) != len(base):
            print(f"note: {model} covers {len(m)} of {len(base)} baseline trials (subset run)")
        m["ok"], m["base_ok"] = m.y_pred == m.y_true, m.base_pred == m.y_true
        rows = []
        for (p, arr, beh), g in m.groupby(["participant", "array", "behavior"], sort=False):
            k, n = int(g.ok.sum()), len(g)
            lo, _ = clopper_pearson(k, n)
            b = int((g.base_ok & ~g.ok).sum())  # baseline right, model wrong
            c = int((~g.base_ok & g.ok).sum())  # model right, baseline wrong
            mcn = binomtest(c, b + c, 0.5).pvalue if b + c else 1.0
            label = next(a[2] for a in cfg1["arrays"] if a[0] == p and a[1] == arr)
            rows.append({"model": model, "participant": p, "array": arr, "label": label, "behavior": beh,
                         "n": n, "accuracy": k / n, "significant": lo > cfg1["chance"],
                         "delta": k / n - g.base_ok.mean(), "mcnemar_p": mcn})
        d = pd.DataFrame(rows)
        if model != "baseline":
            d["mcnemar_q"] = bh(d.mcnemar_p.to_numpy())
        per_cond.append(d)
        boot = [d.delta.to_numpy()[rng.integers(0, len(d), len(d))].mean() for _ in range(cfg["compare"]["bootstrap"])]
        deltas[model] = d.delta.to_numpy()
        a = cfg["compare"]["alpha"]
        table.append({
            "model": labels[model], "input": inputs[model],
            "mean_acc_all_84": d.accuracy.mean(),
            "mean_acc_paper_arrays": d[d.label.isin(paper_rows)].accuracy.mean(),
            "mean_acc_inner_speech": d[d.behavior.isin(inner_speech)].accuracy.mean(),
            "n_significant": int(d.significant.sum()),
            "delta_vs_baseline_pp": 100 * d.delta.mean(),
            "delta_ci_low_pp": 100 * np.percentile(boot, 2.5), "delta_ci_high_pp": 100 * np.percentile(boot, 97.5),
            "n_better": int(((d.get("mcnemar_q", 1) < a) & (d.delta > 0)).sum()) if model != "baseline" else 0,
            "n_worse": int(((d.get("mcnemar_q", 1) < a) & (d.delta < 0)).sum()) if model != "baseline" else 0,
            "wilcoxon_p": wilcoxon(d.delta).pvalue if model != "baseline" and np.any(d.delta != 0) else np.nan,
        })
    tab = pd.DataFrame(table)
    pc = pd.concat(per_cond)
    tab.to_csv(OUT / "model_comparison.csv", index=False)
    pc.to_csv(OUT / "per_condition.csv", index=False)
    pc.pivot_table(index=["label", "behavior"], columns="model", values="accuracy").to_csv(OUT / "accuracy_wide.csv")
    lines = ["Same 84 conditions, trials and nested 10×10 CV as Phase 1. Accuracy in %, chance 14.3%.",
             "Δ = mean per-condition accuracy difference vs the baseline [95% bootstrap CI over conditions]; "
             "better/worse = conditions where McNemar q < 0.05 (BH within model); Wilcoxon = signed-rank test of "
             "the 84 paired differences. Both treat the 84 conditions as independent; intervals that respect their "
             "dependence: model_comparison_clustered.md.", "",
             "| Model | Input | Mean acc. (84) | Paper's 9 arrays | Inner-speech conditions | Significant conditions "
             "| Δ vs baseline (pp) | Better / worse | Wilcoxon p |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in tab.itertuples():
        base_row = r.model == labels["baseline"]
        lines.append(
            f"| {r.model} | {r.input} | {100 * r.mean_acc_all_84:.1f} | {100 * r.mean_acc_paper_arrays:.1f} | "
            f"{100 * r.mean_acc_inner_speech:.1f} | {r.n_significant}/84 | "
            + ("— | — | — |" if base_row else
               f"{r.delta_vs_baseline_pp:+.1f} [{r.delta_ci_low_pp:+.1f}, {r.delta_ci_high_pp:+.1f}] | "
               f"{r.n_better} / {r.n_worse} | {r.wilcoxon_p:.2g} |"))
    (OUT / "model_comparison.md").write_text("\n".join(lines) + "\n")
    delta_strip({labels[m]: 100 * v for m, v in deltas.items() if m != "baseline"},
                {labels[m]: pc[(pc.model == m)].behavior.isin(inner_speech).to_numpy()
                 for m in deltas if m != "baseline"},
                ROOT / "figures" / "phase3_model_comparison.png",
                "Accuracy change vs the paper's Gaussian NB, per condition (84 conditions, same nested CV)")
    print("\n".join(lines))
    clustered_uncertainty(cfg)


def clustered_uncertainty(cfg: dict) -> None:
    """Mean Δ vs the baseline with intervals that respect how the 84 conditions depend on each other.

    The conditions are 12 arrays x 7 behaviors from 4 participants, and a participant's arrays record
    the same trials, so they are not 84 independent replicates. Percentile bootstrap intervals that
    resample:
      condition         the 84 conditions (what model_comparison.csv does; for reference)
      array             the 12 arrays, each with all 7 behaviors
      session           the 28 participant x behavior recordings, each with every array that recorded it
      array x behavior  arrays and behaviors independently (two-way / pigeonhole bootstrap)
    and each participant's mean Δ (4 independent units: a sign test cannot reach p < 0.125).
    Writes model_comparison_clustered.{csv,md} and per_participant.csv from per_condition.csv.
    """
    from scipy.stats import binomtest

    pc = pd.read_csv(OUT / "per_condition.csv")
    pc = pc[pc.model != "baseline"]
    labels = {m: v["label"] for m, v in (cfg["window_models"] | cfg["sequence_models"]).items()}
    B = cfg["compare"]["bootstrap"]
    rows, parts = [], []
    for model, d in pc.groupby("model", sort=False):
        D = d.pivot(index="label", columns="behavior", values="delta")  # arrays x behaviors
        if D.isna().any().any():
            raise ValueError(f"{model}: incomplete array x behavior grid")
        D = D.to_numpy()
        na, nb = D.shape
        ses = d.groupby(["participant", "behavior"]).delta.agg(["sum", "size"])
        rng = {k: np.random.default_rng([cfg["seed"], zlib.crc32(f"{model}|{k}".encode())])
               for k in ("condition", "array", "session", "array_x_behavior")}
        flat = d.delta.to_numpy()
        boot = {
            "condition": flat[rng["condition"].integers(0, len(flat), (B, len(flat)))].mean(1),
            "array": rng["array"].multinomial(na, np.full(na, 1 / na), B) @ D.mean(1) / na,
            "session": ((w := rng["session"].multinomial(len(ses), np.full(len(ses), 1 / len(ses)), B))
                        @ ses["sum"].to_numpy()) / (w @ ses["size"].to_numpy()),
            "array_x_behavior": np.einsum("ra,ab,rb->r", rng["array_x_behavior"].multinomial(na, np.full(na, 1 / na), B),
                                          D, rng["array_x_behavior"].multinomial(nb, np.full(nb, 1 / nb), B)) / (na * nb),
        }
        by_p = d.groupby("participant").delta.mean().round(12) + 0.0  # exact ties stay 0 (not ±1e-16)
        n_pos, n_tied = int((by_p > 0).sum()), int((by_p == 0).sum())
        row = {"model": labels[model], "delta_vs_baseline_pp": 100 * d.delta.mean()}
        for k, v in boot.items():
            row[f"ci_{k}_low_pp"], row[f"ci_{k}_high_pp"] = 100 * np.percentile(v, 2.5), 100 * np.percentile(v, 97.5)
        row |= {"participants_improved": n_pos, "participants_tied": n_tied, "n_participants": len(by_p),
                "sign_test_p_participants": binomtest(n_pos, len(by_p) - n_tied).pvalue}
        rows.append(row)
        parts += [{"model": labels[model], "participant": p, "n_arrays": int((d.participant == p).sum() // nb),
                   "delta_pp": 100 * v} for p, v in by_p.items()]
    tab, pp = pd.DataFrame(rows), pd.DataFrame(parts)
    tab.to_csv(OUT / "model_comparison_clustered.csv", index=False)
    pp.to_csv(OUT / "per_participant.csv", index=False)
    wide = pp.pivot(index="model", columns="participant", values="delta_pp").loc[tab.model]
    ci = lambda r, k: f"[{r[f'ci_{k}_low_pp']:+.1f}, {r[f'ci_{k}_high_pp']:+.1f}]"  # noqa: E731
    lines = ["Δ = mean per-condition accuracy difference vs the paper's Gaussian NB (pp), 84 conditions = "
             "12 arrays × 7 behaviors from 4 participants. 95% percentile bootstrap intervals "
             f"({B} resamples) over: conditions (treats them as independent, as model_comparison.md does), "
             "arrays (12), sessions (28 participant × behavior recordings, each with every array that recorded "
             "it), and arrays × behaviors resampled independently (two-way). The independent units are the 4 "
             "participants: with 4/4 improved, a two-sided sign test gives p = 0.125, its minimum (ties, i.e. "
             "Δ = 0 exactly, are dropped from the sign test).", "",
             "| Model | Δ (pp) | by condition | by array | by session | array × behavior | participants improved | "
             "sign test p |", "|---|---|---|---|---|---|---|---|"]
    for r in tab.to_dict("records"):
        tied = f" ({r['participants_tied']} tied)" if r["participants_tied"] else ""
        lines.append(f"| {r['model']} | {r['delta_vs_baseline_pp']:+.1f} | {ci(r, 'condition')} | {ci(r, 'array')} | "
                     f"{ci(r, 'session')} | {ci(r, 'array_x_behavior')} | {r['participants_improved']}/"
                     f"{r['n_participants']}{tied} | {r['sign_test_p_participants']:.3g} |")
    lines += ["", "Mean Δ per participant (pp; arrays per participant: "
              + ", ".join(f"{p} {n}" for p, n in pp.groupby("participant").n_arrays.first().items()) + "):", "",
              "| Model | " + " | ".join(wide.columns) + " |", "|---|" + "---|" * len(wide.columns)]
    for m, r in wide.iterrows():
        lines.append(f"| {m} | " + " | ".join(f"{v:+.1f}" for v in r) + " |")
    (OUT / "model_comparison_clustered.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "phase3.yaml")
    ap.add_argument("--models", nargs="+", default=[])
    ap.add_argument("--n-jobs", type=int)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--compare-only", action="store_true")
    ap.add_argument("--uncertainty-only", action="store_true",
                    help="only the clustered intervals, from the saved per_condition.csv")
    ap.add_argument("--arrays", nargs="+", help="array labels (smoke tests)")
    ap.add_argument("--behaviors", nargs="+", help="behaviors (smoke tests)")
    ap.add_argument("--out", type=Path, help="output directory (default results/phase3)")
    args = ap.parse_args()
    global OUT
    OUT = args.out or OUT
    FILTER.update(arrays=args.arrays, behaviors=args.behaviors)
    cfg = yaml.safe_load(args.config.read_text())
    cfg1 = yaml.safe_load((ROOT / cfg["phase1_config"]).read_text())
    t0 = time.time()
    if args.uncertainty_only:
        clustered_uncertainty(cfg)
        return 0
    if not args.compare_only:
        win = [m for m in args.models if m in cfg["window_models"]]
        seq = [m for m in args.models if m in cfg["sequence_models"]]
        unknown = set(args.models) - set(win) - set(seq)
        if unknown:
            raise SystemExit(f"unknown models: {unknown}")
        if win:
            run_window_models(win, cfg, cfg1, args.n_jobs or cfg["n_jobs"])
        if seq:
            run_sequence_models(seq, cfg, cfg1, args.device)
    compare(cfg, cfg1)
    (OUT / "run_info.json").write_text(json.dumps({"models": args.models, "runtime_s": round(time.time() - t0, 1),
                                                   "config": cfg}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

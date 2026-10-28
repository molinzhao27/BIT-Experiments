#!/usr/bin/env python
"""Arm B, BERTScore vs WER: where does diffusion's WER advantage come from? (CPU, reads results/per_trial/.)

    <.venv>/bin/python scripts/bertscore_compare.py ["Aero diff=B_ssl_aero-1-audio-diffusion" ...] ["Arm A=A_ssl__A"] ["Arm A|Aero diff"]

Run `scripts/evaluate_decodes.py score <decodes>/test_greedy_real.csv ...` first (configs/eval.yaml: roberta-large,
layer 17, rescaled). Per run (BIT-test, greedy, real input): corpus WER, % exactly right, mean BERTScore F1 (all
sentences, and sentences with >= 1 word error), % of error sentences that keep the meaning (F1 >= 0.75, the
evaluator's tau), % derailed (sentence WER >= 1), and decoded / reference length. Paired bootstraps over trials
(10,000 resamples, seed 0) for F1 and for the share of sentences with an error. Then the WER gap of each pair, split
by the second run's sentence WER. Extra runs: "label=config" (the per_trial file prefix; arm B unless it names one,
as in "A_ssl__A"), compared with Aero AR. Extra pairs: "label a|label b".
"""
import sys
import numpy as np, pandas as pd
from pathlib import Path
D = str(Path(__file__).resolve().parents[1] / "results" / "per_trial") + "/"
R = {"Aero AR": "B_ssl", "Instruct AR": "B_ssl_qwen25-1.5b-instruct", "Base AR (no prompt)": "B_ssl_qwen25-1.5b-noprompt",
     "Fast-dLLM v2 (diff)": "B_ssl_fast-dllm-v2-1.5b", "Instruct diff (control)": "B_ssl_qwen25-1.5b-instruct-diffusion"}
extra = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)  # e.g. "Aero diff=B_ssl_aero-1-audio-diffusion"
R |= extra
d = {k: pd.read_csv(f"{D}{v}{'' if '__' in v else '__B'}__real__brandman_2024_text_test__greedy__seed42.csv").sort_values("trial").reset_index(drop=True) for k, v in R.items()}
rows = []
for k, t in d.items():
    err = t.word_errors > 0
    dl = t.decoded.fillna("").str.split().str.len(); rl = t.reference.str.split().str.len()
    rows.append(dict(run=k, WER=100 * t.word_errors.sum() / t.ref_words.sum(), exact=100 * (~err).mean(),
                     F1=t.bs_f1.mean(), F1_err=t.bs_f1[err].mean(),
                     keep_meaning=100 * (t.bs_f1[err] >= 0.75).mean(),  # among sentences with >=1 error
                     sWER_ge_1=100 * (t.sent_wer >= 1).mean(), sWER_med_err=t.sent_wer[err].median(),
                     len_ratio=(dl / rl).mean(), F1_lt_0=100 * (t.bs_f1 < 0).mean()))
print(pd.DataFrame(rows).round(3).to_string(index=False))
rng = np.random.default_rng(0); idx = rng.integers(0, 713, (10000, 713))
def boot(a, b, col, f=np.mean):
    x, y = d[a][col].to_numpy(float), d[b][col].to_numpy(float)
    assert (d[a].trial.to_numpy() == d[b].trial.to_numpy()).all()
    diff = x[idx].mean(1) - y[idx].mean(1)
    return f"{x.mean() - y.mean():+.4f} [{np.percentile(diff, 2.5):+.4f}, {np.percentile(diff, 97.5):+.4f}]"
pairs = [("Instruct diff (control)", "Instruct AR"), ("Instruct diff (control)", "Aero AR"), ("Fast-dLLM v2 (diff)", "Aero AR")]
pairs += [(k, "Aero AR") for k in extra] + [tuple(a.split("|")) for a in sys.argv[1:] if "|" in a and "=" not in a]
for a, b in pairs:
    for col in ("bs_f1",):
        print(f"{a} - {b}: F1 {boot(a, b, col)}")
    ea, eb = (d[a].word_errors > 0).astype(float), (d[b].word_errors > 0).astype(float)
    diff = ea.to_numpy()[idx].mean(1) - eb.to_numpy()[idx].mean(1)
    print(f"   sentences with >=1 error: {100*(ea.mean()-eb.mean()):+.1f} pp [{100*np.percentile(diff,2.5):+.1f}, {100*np.percentile(diff,97.5):+.1f}]")
# where does the WER gap come from? split the word-error difference by the AR model's sentence WER
for a, b in pairs:
    A, B = d[a], d[b]
    bins = pd.cut(B.sent_wer, [-0.01, 0, 0.5, 1.0, 99], labels=["exact", "<=50%", "50-100%", ">100%"])
    g = pd.DataFrame({"bin": bins, "dErr": A.word_errors - B.word_errors}).groupby("bin", observed=False).agg(n=("dErr", "size"), dErr=("dErr", "sum"))
    g["share_of_gap_%"] = (100 * g.dErr / g.dErr.sum()).round(1)
    print(f"\n{a} vs {b}: word-error change by {b}'s sentence WER (total {int(g.dErr.sum())} words of {int(B.ref_words.sum())})")
    print(g.to_string())

# "Keeps the meaning" (F1 >= tau among error sentences) depends on how many errors a run makes and on which
# sentences it gets wrong. Matched comparison: the same trial, both runs wrong, with the SAME number of word errors
# (so the same sentence WER); does one run's output keep more meaning? Paired bootstrap over those trials.
print("\nMatched on trial and word-error count (both runs wrong on that trial with equal errors):")
for a, b in pairs:
    A, B = d[a], d[b]
    m = (A.word_errors > 0) & (A.word_errors == B.word_errors)
    x, y = A.bs_f1[m].to_numpy(), B.bs_f1[m].to_numpy()
    j = np.random.default_rng(0).integers(0, len(x), (10000, len(x)))
    diff = x[j].mean(1) - y[j].mean(1)
    k = A.word_errors[m]
    keep = 100 * ((x >= 0.75).mean() - (y >= 0.75).mean())
    print(f"  {a} - {b}: n={m.sum()} (errors 1/2/3+: {(k == 1).sum()}/{(k == 2).sum()}/{(k >= 3).sum()}), "
          f"F1 {x.mean() - y.mean():+.4f} [{np.percentile(diff, 2.5):+.4f}, {np.percentile(diff, 97.5):+.4f}], "
          f"keeps meaning {keep:+.1f} pp")
# Unmatched but stratified: F1 of error sentences by number of word errors, per run.
print("\nMean F1 of error sentences by word-error count (n in brackets):")
tab = {}
for name, t in d.items():
    e = t[t.word_errors > 0]
    g = e.groupby(pd.cut(e.word_errors, [0, 1, 2, 3, 5, 99], labels=["1", "2", "3", "4-5", "6+"]), observed=False).bs_f1
    tab[name] = [f"{mu:.3f} [{n}]" for mu, n in zip(g.mean(), g.size())]
print(pd.DataFrame(tab, index=["1", "2", "3", "4-5", "6+"]).T.to_string())

"""scripts/ablation_report.py (docs/RETRAINING_ABLATIONS.md §9) on planted decodes, CPU: corpus WER per condition,
the grounding gap, a pair's cost and grounding kept, the difference-in-differences between two pairs, and the
refusal to compare runs decoded on different trials."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ablation_report", ROOT / "scripts" / "ablation_report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)

REFS = [f"w{i} x{i} y{i} z{i}" for i in range(200)]  # 4 words each


def _write(root, name, wrong_real, wrong_null, refs=REFS):
    """Each trial's decode gets `wrong` of its 4 words replaced (an array of per-trial counts per condition)."""
    d = root / name / "decodes"
    d.mkdir(parents=True)
    for cond, wrong in (("real", wrong_real), ("null_shuffled", wrong_null), ("null_prompt_only", wrong_null)):
        hyps = [" ".join(["bad"] * k + r.split()[k:]) for r, k in zip(refs, wrong)]
        pd.DataFrame({"trial": range(len(refs)), "reference": refs, "decoded": hyps}).to_csv(
            d / f"test_greedy_{cond}.csv", index=False)


def test_pairs_and_difference_in_differences(tmp_path):
    g = np.random.default_rng(0)
    null = np.full(200, 4)  # every word wrong under the null
    ar_c, ar_a = g.binomial(4, 0.25, 200), g.binomial(4, 0.75, 200)  # AR: erasure pushes it most of the way to the null
    df_c, df_a = g.binomial(4, 0.25, 200), g.binomial(4, 0.35, 200)  # diffusion: keeps following the signal
    for name, real in (("ar_c", ar_c), ("ar_a", ar_a), ("d_c", df_c), ("d_a", df_a)):
        _write(tmp_path, name, real, null)
    out = tmp_path / "rep"
    assert report.main(["--pair", "AR", "ar_c", "ar_a", "--pair", "diffusion", "d_c", "d_a", "--compare", "AR",
                        "diffusion", "--results", str(tmp_path), "--out", str(out), "--n-boot", "500"]) == 0
    r = json.loads(out.with_suffix(".json").read_text())
    assert r["runs"]["ar_c"]["real"] == pytest.approx(ar_c.sum() / 800)
    assert r["runs"]["ar_c"]["null_shuffled"] == 1.0
    assert r["runs"]["ar_c"]["gap"] == pytest.approx(1 - ar_c.sum() / 800)
    ar = r["pairs"]["AR"]
    assert ar["cost"] == pytest.approx((ar_a.sum() - ar_c.sum()) / 800)
    assert ar["cost_ci"][0] < ar["cost"] < ar["cost_ci"][1]
    assert ar["grounding_kept"] == pytest.approx((1 - ar_a.sum() / 800) / (1 - ar_c.sum() / 800))
    cmp = r["compare"]["diffusion - AR"]
    assert cmp["cost_diff"] < 0 and cmp["cost_diff_ci"][1] < 0  # diffusion loses less
    assert cmp["grounding_kept_diff"] > 0 and cmp["grounding_kept_diff_ci"][0] > 0
    assert r["runs"]["ar_a"]["derailed"] == pytest.approx(float((ar_a == 4).mean()))
    assert "| AR | ar_c | ar_a |" in out.with_suffix(".md").read_text()


def test_runs_on_different_trials_are_refused(tmp_path):
    _write(tmp_path, "a", np.zeros(200, int), np.full(200, 4))
    _write(tmp_path, "b", np.zeros(200, int), np.full(200, 4), refs=REFS[::-1])
    with pytest.raises(ValueError, match="references differ"):
        report.main(["--pair", "P", "a", "b", "--results", str(tmp_path), "--out", str(tmp_path / "o")])
    other = tmp_path / "elsewhere"
    _write(other, "c", np.ones(200, int), np.full(200, 4))
    assert report.main(["--pair", "P", "a", "c", "--results", str(tmp_path), "--run-dir", f"c={other / 'c'}",
                        "--out", str(tmp_path / "o"), "--n-boot", "50"]) == 0

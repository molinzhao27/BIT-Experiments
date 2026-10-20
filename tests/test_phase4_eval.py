"""Checks for the Phase 4 shared evaluator (BERTScore mocked; set KUNZ_BERTSCORE_IT=1 for the real one)."""

import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kunzdecoder import phase4_eval as ev  # noqa: E402
from kunzdecoder.textmetrics import normalize, wer  # noqa: E402

REFS = ["The cat sat on the mat.", "I would like some water, please.", "How are you doing today?",
        "We went to the store.", "It's a lovely day outside.", "Can you help me with this?"]


def fake_bertscore(cands, refs, cfg):
    """Deterministic stand-in: word-set Jaccard (0 for empty decodes), same for P/R/F1."""
    f = []
    for c, r in zip(cands, refs, strict=True):
        a, b = set(normalize(c).split()), set(normalize(r).split())
        f.append(len(a & b) / len(a | b) if a and b else 0.0)
    f = np.array(f)
    return f, f, f


@pytest.fixture
def mocked(monkeypatch):
    monkeypatch.setattr(ev, "bertscore", fake_bertscore)


def _decodes(path, hyps, refs=REFS, trials=None):
    pd.DataFrame({"trial": trials if trials is not None else range(len(hyps)), "reference": refs[:len(hyps)],
                  "decoded": hyps}).to_csv(path, index=False)
    return path


def _meta(**kw):
    return {"model": "m", "arm": "direct", "seed": 0, "split": "t15:val", "decoding": "greedy",
            "condition": "real"} | kw


def test_bertscore_wrapper_normalizes_and_skips_empty(monkeypatch):
    seen = {}

    def score(cands, refs, **kw):
        seen.update(cands=cands, refs=refs, kw=kw)
        return (np.full(len(cands), 0.5),) * 3

    monkeypatch.setitem(sys.modules, "bert_score", types.SimpleNamespace(score=score))
    cfg = ev.load_config()
    cfg["bertscore"]["device"] = "cpu"
    P, R, F = ev.bertscore(["The Cat, sat!", "", "   ...  "], ["the cat sat.", "Hello there.", "Hi."], cfg)
    assert seen["cands"] == ["the cat sat"] and seen["refs"] == ["the cat sat"]
    assert seen["kw"]["model_type"] == "roberta-large"
    assert seen["kw"]["rescale_with_baseline"] and not seen["kw"]["idf"] and seen["kw"]["num_layers"] == 17
    assert np.allclose(F, [0.5, 0, 0]) and np.allclose(P, F) and np.allclose(R, F)


def test_bertscore_all_empty_never_loads_model(monkeypatch):
    monkeypatch.setitem(sys.modules, "bert_score", None)  # import would fail
    P, R, F = ev.bertscore(["", "!!"], ["a b", "c"], ev.load_config())
    assert np.all(F == 0) and np.all(P == 0) and np.all(R == 0)


def test_score_and_aggregate_match_bit_wer(mocked):
    hyps = ["the cat sat on the mat", "I would like water please", "", "we went to a store",
            "it's a lovely day outside", "can you help me with this"]
    df = pd.DataFrame({"trial": range(6), "reference": REFS, "decoded": hyps})
    t = ev.score_trials(df, ev.load_config())
    agg = ev.aggregate(t)
    assert agg["n_trials"] == 6
    assert np.isclose(agg["wer"], wer(REFS, hyps))
    assert np.isclose(agg["mean_sent_wer"], np.mean(t.word_errors / t.ref_words))
    assert np.isclose(agg["ser"], 3 / 6)
    assert t.loc[2, "bs_f1"] == 0 and t.loc[2, "word_errors"] == 5


def test_error_analysis_threshold_grid():
    t = pd.DataFrame({"word_errors": [0, 1, 2, 1, 3], "ref_words": [5] * 5, "bs_f1": [0.99, 0.95, 0.72, 0.65, 0.1]})
    ea = ev.error_analysis(t, [0.6, 0.7, 0.75, 0.8, 0.9], 0.75).set_index("tau")
    assert list(ea.index) == [0.6, 0.7, 0.75, 0.8, 0.9]
    assert ea.n_error_sentences.unique().tolist() == [4]
    assert ea.n_meaning_preserving.tolist() == [3, 2, 1, 1, 1]
    assert np.isclose(ea.loc[0.75, "frac_meaning_changing"], 0.75) and ea.loc[0.75, "is_default_tau"]
    assert ea.is_default_tau.sum() == 1


def test_metrics_schema_growth_and_rerun(tmp_path, mocked):
    out = tmp_path / "phase4"
    out.mkdir()
    legacy = pd.DataFrame([{"model": "BIT", "arm": "bit_baseline", "seed": 42, "split": "x:val",
                            "decoding": "greedy", "n_trials": 3, "wer": 0.2, "per": 0.1}])
    legacy.to_csv(out / "metrics.csv", index=False)
    f = _decodes(tmp_path / "d.csv", ["the cat sat", "", "how are you doing today"])
    cfg = ev.load_config()
    for _ in range(2):  # second run replaces the first instead of duplicating it
        ev.evaluate_file(f, _meta(), cfg, out)
    m = pd.read_csv(out / "metrics.csv")
    assert list(m.columns[:8]) == list(ev.LEGACY_COLS)
    assert {"condition", "bertscore_f1", "mean_sent_wer", "ser", "bertscore_hash", "evaluator"} <= set(m.columns)
    assert len(m) == 2 and m.loc[0, "arm"] == "bit_baseline" and m.loc[0, "per"] == 0.1
    assert (out / "per_trial" / f"{ev.slug(_meta())}.csv").exists()
    ea = pd.read_csv(out / "error_analysis.csv")
    assert len(ea) == 5 and set(ea.tau) == {0.6, 0.7, 0.75, 0.8, 0.9}
    # run_bit_baseline.py appends headerless 8-column rows; they must land in the right columns
    legacy.to_csv(out / "metrics.csv", mode="a", header=False, index=False)
    m = pd.read_csv(out / "metrics.csv")
    assert m.iloc[-1].arm == "bit_baseline" and m.iloc[-1].per == 0.1 and pd.isna(m.iloc[-1].condition)


def test_configs_of_one_arm_and_seed_kept_apart(tmp_path, mocked):
    """Stage 2 runs of one arm and seed with different LLMs (e.g. B_ssl_s42 vs B_ssl_qwen25-1.5b-instruct_s42) must
    neither replace each other nor pool as seeds; seeds of one config pool."""
    out = tmp_path / "phase4"
    cfg = ev.load_config()
    f = _decodes(tmp_path / "d.csv", ["the cat sat", "how are you doing today"])
    for name, seed in (("B_ssl", 42), ("B_ssl", 43), ("B_ssl_qwen25-1.5b-instruct", 42), ("B_rand_bit4a", 42)):
        model = f"stage2 {name}_s{seed}" if name != "B_rand_bit4a" else f"stage2 {name}"
        meta = ev.run_meta(f, overrides={"model": model, "arm": "B", "seed": seed, "split": "t15:test",
                                         "decoding": "greedy"})
        ev.evaluate_file(f, meta, cfg, out)
    m = pd.read_csv(out / "metrics.csv")
    assert sorted(m.config) == ["B_rand_bit4a", "B_ssl", "B_ssl", "B_ssl_qwen25-1.5b-instruct"]
    assert len(list((out / "per_trial").glob("*.csv"))) == 4
    s = ev.summarize(m).set_index("config")
    assert s.loc["B_ssl", "n_seeds"] == 2 and s.loc["B_ssl_qwen25-1.5b-instruct", "n_seeds"] == 1


def test_run_meta_sources(tmp_path):
    f = _decodes(tmp_path / "run.csv", ["a"])
    with pytest.raises(ValueError, match="missing run metadata"):
        ev.run_meta(f)
    (tmp_path / "run.meta.yaml").write_text("model: m\narm: direct\nseed: 1\nsplit: s\ndecoding: greedy\n")
    meta = ev.run_meta(f)
    assert meta["seed"] == 1 and meta["condition"] == "real"
    assert ev.run_meta(f, overrides={"seed": 2, "condition": "null_noise", "arm": None})["seed"] == 2
    with pytest.raises(ValueError, match="condition"):
        ev.run_meta(f, overrides={"condition": "bogus"})


def test_summarize_mean_sd(tmp_path, mocked):
    out = tmp_path / "phase4"
    cfg = ev.load_config()
    hyps = [["the cat sat on the mat", "I would like water", "how are you"],
            ["the cat sat", "I would like some water please", "how are you doing today"],
            ["", "I would like some water please", "how are you doing today"]]
    for seed, h in enumerate(hyps):
        ev.evaluate_file(_decodes(tmp_path / f"s{seed}.csv", h), _meta(seed=seed), cfg, out)
    ev.evaluate_file(_decodes(tmp_path / "null.csv", ["x", "y", "z"]), _meta(condition="null_noise"), cfg, out)
    m = pd.read_csv(out / "metrics.csv")
    s = ev.summarize(m)
    real = s[s.condition == "real"].iloc[0]
    wers = [wer(REFS[:3], h) for h in hyps]
    assert real.n_seeds == 3 and real.seeds == "0 1 2"
    assert np.isclose(real.wer_mean, np.mean(wers)) and np.isclose(real.wer_sd, np.std(wers, ddof=1))
    assert np.isnan(s[s.condition == "null_noise"].iloc[0].wer_sd)
    md = ev.summary_markdown(s)
    assert "±" in md and md.count("| direct |") == 2
    m.loc[0, "bertscore_hash"] = "other"
    with pytest.raises(ValueError, match="mixes BERTScore"):
        ev.summarize(m)


def test_pick_examples_shared_stratified_deterministic():
    rng = np.random.default_rng(1)
    n = 100
    base = pd.DataFrame({"trial": range(n), "reference": [f"w{i} a b c" for i in range(n)],
                         "word_errors": rng.integers(0, 5, n), "ref_words": 4})
    a, b = base, base.drop(index=[3, 7]).assign(word_errors=0)
    c = base.iloc[::-1].assign(decoded="x")
    t1 = ev.pick_examples({"a": a, "b": b, "c": c}, 20, seed=0)
    t2 = ev.pick_examples({"a": a, "b": b, "c": c}, 20, seed=0)
    assert np.array_equal(t1, t2) and len(t1) == 20 and len(set(t1)) == 20
    assert not {3, 7} & set(t1)
    sw = a.set_index("trial").loc[t1].word_errors.to_numpy()
    assert np.all(np.diff(sw) >= 0) and sw.min() == 0 and sw.max() == 4  # easy -> hard, full range
    bad = b.assign(reference="different words")
    with pytest.raises(ValueError, match="references differ"):
        ev.pick_examples({"a": a, "bad": bad}, 5)
    md = ev.examples_markdown({"a": a.assign(decoded=""), "b": b.assign(decoded="w")}, t1[:2])
    assert md.count("### Trial") == 2 and "*(empty)*" in md and "| **reference** |" in md


def _trials(err, f1, n_words=8):
    return pd.DataFrame({"trial": range(len(err)), "word_errors": err, "ref_words": n_words, "bs_f1": f1})


def test_null_controls_bootstrap():
    rng = np.random.default_rng(0)
    n = 200
    real = _trials(rng.integers(0, 3, n), rng.uniform(0.6, 1.0, n))
    nulls = {c: _trials(rng.integers(5, 9, n), rng.uniform(-0.2, 0.3, n)) for c in ev.NULLS}
    r = ev.null_controls(real, nulls, n_boot=2000, seed=0)
    assert len(r) == 6 and r.real_better.all() and r.beats_all_nulls.all() and (r.ci_low > 0).all()
    assert r.nulls_missing.iloc[0] == ""
    assert r.equals(ev.null_controls(real, nulls, n_boot=2000, seed=0))  # fixed seed -> identical
    w = r[(r.metric == "wer") & (r.null_condition == "null_noise")].iloc[0]
    assert np.isclose(w["diff"], nulls["null_noise"].word_errors.sum() / (8 * n) - real.word_errors.sum() / (8 * n))
    assert w.ci_low <= w["diff"] <= w.ci_high
    same = ev.null_controls(real, {"null_shuffled": real.copy()}, n_boot=500, seed=0)
    assert not same.real_better.any() and np.allclose(same[["diff", "ci_low", "ci_high"]], 0)
    assert not same.beats_all_nulls.any() and same.nulls_missing.iloc[0] == "null_noise null_prompt_only"


def test_cli_end_to_end(tmp_path, mocked, monkeypatch):
    spec = importlib.util.spec_from_file_location("evaluate_decodes", ROOT / "scripts/phase4/evaluate_decodes.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli.ev, "bertscore", fake_bertscore)
    out = tmp_path / "out"

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["evaluate_decodes.py", "--out-dir", str(out), *argv])
        assert cli.main() == 0

    good = ["the cat sat on the mat", "I would like some water", "how are you doing", "we went to the store",
            "it's a day outside", "can you help me with this"]
    files = []
    for seed in range(3):
        files.append(_decodes(tmp_path / f"direct_s{seed}.csv", good))
        (tmp_path / f"direct_s{seed}.meta.yaml").write_text(
            f"model: m\narm: direct\nseed: {seed}\nsplit: t15:val\ndecoding: greedy\n")
    run("score", *map(str, files))
    for cond in ev.NULLS:
        f = _decodes(tmp_path / f"{cond}.csv", ["um"] * 6)
        run("score", str(f), "--model", "m", "--arm", "direct", "--seed", "0", "--split", "t15:val",
            "--decoding", "greedy", "--condition", cond)
    run("summarize", "--metrics", str(out / "metrics.csv"))
    run("examples", str(files[0]), str(tmp_path / "null_noise.csv"), "--labels", "direct", "noise", "--n", "4")
    per_trial = {c: out / "per_trial" / f"{ev.slug(_meta(condition=c))}.csv" for c in ev.CONDITIONS}
    run("nulls", "--real", str(per_trial["real"]), "--null-shuffled", str(per_trial["null_shuffled"]),
        "--null-noise", str(per_trial["null_noise"]), "--null-prompt-only", str(per_trial["null_prompt_only"]))
    assert len(pd.read_csv(out / "metrics.csv")) == 6
    assert len(pd.read_csv(out / "error_analysis.csv")) == 30
    s = pd.read_csv(out / "metrics_summary.csv")
    assert len(s) == 4 and s[s.condition == "real"].n_seeds.iloc[0] == 3
    assert (out / "metrics_summary.md").read_text().count("| m | direct |") == 4
    assert (out / "examples_side_by_side.md").read_text().count("### Trial") == 4
    nc = pd.read_csv(out / "null_controls.csv")
    assert len(nc) == 6 and nc.beats_all_nulls.all() and (nc.arm == "direct").all()


@pytest.mark.skipif(os.environ.get("KUNZ_BERTSCORE_IT") != "1", reason="set KUNZ_BERTSCORE_IT=1 (downloads a model)")
def test_real_bertscore_tiny_model():
    cfg = ev.load_config()
    cfg["bertscore"] |= {"model_type": "distilbert-base-uncased", "num_layers": 5, "device": "cpu", "batch_size": 4}
    P, R, F = ev.bertscore(["The cat sat on the mat.", "a dog ran", ""],
                           ["the cat sat on the mat", "the cat sat on the mat", "hello"], cfg)
    assert np.isclose(F[0], 1, atol=1e-3) and F[1] < F[0] and F[2] == 0
    assert np.all(np.isfinite(P)) and np.all(np.isfinite(R))

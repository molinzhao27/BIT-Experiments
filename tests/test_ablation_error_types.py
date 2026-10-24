"""scripts/ablation_error_types.py, CPU only (project .venv: pronouncing, wordfreq; distilbert from the local cache).

    .venv/bin/python -m pytest tests/test_ablation_error_types.py

- Config: every condition is one of eval_ablations.yaml's ablations, the baseline is among them, contrasts name them.
- End to end on synthetic decode sets of two runs (no embeddings): rows in config order, a missing condition and a
  missing run skipped, the reference decodes added, the baseline check (= / ≠ saved), WER = textmetrics.wer with its
  CI around it, a contrast of identical decode sets exactly 0, contrasts antisymmetric, seeded and reproducible.
- The phonetic numbers equal eval_ablations.py's own hook (same resamples) on the same decodes.
- With distilbert (skipped if it is not in the local HF cache): ctx / related / sem present; substitutes related in
  meaning give a larger related and sem excess than unrelated substitutes.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("pronouncing")
pytest.importorskip("wordfreq")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


aet = _load("ablation_error_types", ROOT / "scripts" / "ablation_error_types.py")
from bit_experiments.textmetrics import wer  # noqa: E402

REFS = ["the cat sat on the mat", "i want to go home now", "they read a good book", "we ate bread for lunch",
        "my car is very fast", "she likes the blue sky", "he drank cold water today", "our dog ran in the park"]
# Per condition: reference word -> substitute (the first occurrence in each sentence is replaced).
SOUND = {"cat": "hat", "for": "four", "read": "red", "fast": "last", "sky": "high", "cold": "gold", "park": "bark",
         "home": "comb"}
MEANING = {"cat": "dog", "car": "truck", "water": "milk", "bread": "rice", "book": "story", "lunch": "dinner",
           "home": "house", "park": "garden", "sky": "sun"}
UNRELATED = {"cat": "idea", "car": "seven", "water": "pencil", "bread": "window", "book": "river", "lunch": "money",
             "home": "paper", "park": "doctor", "sky": "table"}
N_REPEAT = 4  # each sentence appears this many times (trials 0 .. 31)


def substitute(sentence: str, table: dict) -> str:
    words, done = sentence.split(), False
    for k, w in enumerate(words):
        if w in table and not done:
            words[k], done = table[w], True
    return " ".join(words)


def frame(table: dict | None, drop_word_in_trial: int | None = None) -> pd.DataFrame:
    rows = []
    for t in range(len(REFS) * N_REPEAT):
        ref = REFS[t % len(REFS)]
        hyp = ref if table is None else substitute(ref, table)
        if t == drop_word_in_trial:
            hyp = " ".join(hyp.split()[:-1])
        rows.append({"trial": t, "reference": ref, "decoded": hyp, "seen_in_train": t % 3 == 0})
    return pd.DataFrame(rows)


def write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


@pytest.fixture
def results(tmp_path):
    """Two runs: run_a with every condition but one and matching saved decodes, run_b with fewer conditions and
    saved decodes that differ from its baseline in one trial; run_c has no ablations at all."""
    root = tmp_path / "stage2"
    sets_a = {"none": frame(SOUND), "drop_v6v": frame(MEANING), "drop_55b": frame(UNRELATED),
              "copy_none": frame(SOUND)}
    for cond, df in sets_a.items():
        write(df, root / "run_a" / "ablations" / cond / "test_greedy.csv")
    write(sets_a["none"], root / "run_a" / "decodes" / "test_greedy_real.csv")
    shuffled = frame(SOUND).assign(decoded=lambda d: list(d["decoded"])[1:] + list(d["decoded"])[:1])
    write(shuffled, root / "run_a" / "decodes" / "test_greedy_null_shuffled.csv")
    write(frame(MEANING), root / "run_b" / "ablations" / "none" / "test_greedy.csv")
    write(frame(UNRELATED), root / "run_b" / "ablations" / "drop_55b" / "test_greedy.csv")
    write(frame(MEANING, drop_word_in_trial=5), root / "run_b" / "decodes" / "test_greedy_real.csv")
    (root / "run_c" / "decodes").mkdir(parents=True)
    return root


def config(tmp_path, embed: str = "", **over) -> Path:
    cfg = {"split": "test", "seed": 3, "n_boot": 300, "min_null": 3, "embed_model": embed,
           "runs": ["run_a", "run_b", "run_c"], "baseline": "none",
           "conditions": ["none", "drop_v6v", "drop_55b", "copy_none", "not_decoded"],
           "reference_decodes": ["null_shuffled"], "contrasts": [["drop_55b", "drop_v6v"]],
           "slices": ["all", "mostly_right"], "out": str(tmp_path / "unused")} | over
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def run_main(results, cfg_path, out) -> dict:
    assert aet.main(["--config", str(cfg_path), "--results", str(results), "--out", str(out)]) == 0
    return json.loads((out / "summary.json").read_text())


def test_config_names_eval_ablations_conditions():
    cfg = aet.load_config()
    ablations = yaml.safe_load((ROOT / "configs" / "eval_ablations.yaml").read_text())["ablations"]
    assert set(cfg["conditions"]) <= set(ablations) and cfg["baseline"] in cfg["conditions"]
    assert all(a in cfg["conditions"] and b in cfg["conditions"] for a, b in cfg["contrasts"])
    labels = aet.condition_labels(cfg)
    assert labels["drop_55b"] == ablations["drop_55b"]["label"] and "null_shuffled" in labels


def test_end_to_end(results, tmp_path):
    s = run_main(results, config(tmp_path), tmp_path / "out")
    rows = [(r["run"], r["condition"]) for r in s["runs"]]
    assert rows == [("run_a", c) for c in ("none", "drop_v6v", "drop_55b", "copy_none", "null_shuffled")] + [
        ("run_b", "none"), ("run_b", "drop_55b")]  # config order; not_decoded, run_c and run_b's missing sets skipped
    checks = s["meta"]["baseline_checks"]
    assert checks["run_a"]["matches_saved"] is True
    assert checks["run_b"]["matches_saved"] is False and checks["run_b"]["trials_differing_from_saved"] == 1
    assert "run_c" not in checks

    sets = {"none": frame(SOUND), "drop_v6v": frame(MEANING), "drop_55b": frame(UNRELATED)}
    for r in s["runs"]:
        if r["run"] == "run_a" and r["condition"] in sets:
            df = sets[r["condition"]]
            assert r["wer"] == pytest.approx(wer(list(df["reference"]), list(df["decoded"])))
        assert r["wer_ci"][0] <= r["wer"] + 1e-12 and r["wer"] <= r["wer_ci"][1] + 1e-12
        a = r["slices"]["all"]
        assert a["alike"]["obs_ci"][0] <= a["alike"]["obs"] + 1e-12 <= a["alike"]["obs_ci"][1] + 2e-12
        assert "ctx" not in a and "sem" not in a  # no embeddings in this config

    by = {(r["run"], r["condition"]): r for r in s["runs"]}
    sound, meaning = by[("run_a", "none")]["slices"]["all"], by[("run_a", "drop_v6v")]["slices"]["all"]
    assert sound["alike"]["obs"] > 0.8 and sound["phon"]["excess"] > meaning["phon"]["excess"]

    cs = {(c["run"], c["a"], c["b"], c["slice"]): c for c in s["contrasts"]}
    assert ("run_a", "drop_55b", "drop_v6v", "all") in cs and ("run_b", "drop_55b", "none", "all") in cs
    assert ("run_b", "drop_55b", "drop_v6v", "all") not in cs  # run_b has no drop_v6v
    same = cs[("run_a", "copy_none", "none", "all")]
    for m in ("wer", "alike_obs", "phon"):
        assert same[m]["diff"] == 0 and same[m]["ci"] == [0, 0]
    assert "wer" not in cs[("run_a", "copy_none", "none", "mostly_right")]  # WER only for slice all

    md = (tmp_path / "out" / "tables.md").read_text()
    assert "= saved decodes" in md and "≠ saved decodes (1 trials)" in md
    assert "| run_a | drop_55b − drop_v6v |" in md

    run_main(results, config(tmp_path), tmp_path / "out2")  # seeded: identical output (text: NaN != NaN)
    assert (tmp_path / "out2" / "summary.json").read_text() == (tmp_path / "out" / "summary.json").read_text()


def test_contrast_antisymmetric(results, tmp_path):
    cfg = aet.load_config(config(tmp_path))
    sets, _ = aet.collect(cfg, ["run_a"], results)
    pea = aet.pea
    loaded = [(run, cond, df, *pea.extract(df)) for run, cond, df in sets]
    words = {p[k] for *_, pairs, _, _ in loaded for p in pairs for k in ("ref", "hyp")}
    words |= {w for *_, rw in loaded for ws in rw.values() for w in ws}
    lex, costs = pea.Lexicon(words, None), pea.substitution_costs()
    ras = {cond: pea.RunAnalysis(run, cond, df, pairs, ops, rw, lex, costs, 3)
           for run, cond, df, pairs, ops, rw in loaded}
    stats = pea.Stats(sorted(range(len(REFS) * N_REPEAT)), 200, 1, 0)
    for ra in ras.values():
        ra.build(None)
    w = {c: aet.wer_boot(stats, ra.df) for c, ra in ras.items()}
    ab = aet.contrast(stats, ras["drop_55b"], ras["none"], w["drop_55b"], w["none"], "all")
    ba = aet.contrast(stats, ras["none"], ras["drop_55b"], w["none"], w["drop_55b"], "all")
    for m in ("wer", "alike_obs", "phon"):
        assert ab[m]["diff"] == pytest.approx(-ba[m]["diff"])
        assert ab[m]["ci"] == pytest.approx([-ba[m]["ci"][1], -ba[m]["ci"][0]])


def test_phon_equals_eval_ablations_hook(results, tmp_path):
    """Same decodes, resamples and seed: the phonetic numbers equal eval_ablations.py's phonetic() (its --phonetic-only
    step), so this table and ablations_summary.md agree."""
    pytest.importorskip("torch")
    ea = sys.modules.get("eval_ablations") or _load("eval_ablations", ROOT / "scripts" / "eval_ablations.py")
    s = run_main(results, config(tmp_path, runs=["run_a"], reference_decodes=[]), tmp_path / "out")
    frames = {c: aet.read_decodes(results / "run_a" / "ablations" / c / "test_greedy.csv")
              for c in ("none", "drop_v6v", "drop_55b", "copy_none")}
    hook = ea.phonetic("run_a", frames, {"n_boot": 300, "seed": 3, "min_null": 3})
    for r in s["runs"]:
        h, a = hook[r["condition"]], r["slices"]["all"]
        for m in ("alike", "phon", "phon|orth"):
            assert a[m]["excess"] == pytest.approx(h[m]["excess"]) and a[m]["ci"] == pytest.approx(h[m]["ci"])


def _distilbert_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache("distilbert-base-uncased", "config.json"), str)


@pytest.mark.skipif(not _distilbert_cached(), reason="distilbert-base-uncased is not in the local HF cache")
def test_language_measures_with_distilbert(results, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    s = run_main(results, config(tmp_path, embed="distilbert-base-uncased", runs=["run_a"], n_boot=200),
                 tmp_path / "out")
    by = {r["condition"]: r["slices"]["all"] for r in s["runs"]}
    for cond in ("none", "drop_v6v", "drop_55b"):
        for m in ("ctx", "related", "sem", "ctx>r"):
            assert np.isfinite(by[cond][m]["excess"]) and by[cond][m]["n"] > 0
    meaning, unrelated = by["drop_v6v"], by["drop_55b"]
    assert meaning["related"]["obs"] > unrelated["related"]["obs"]
    assert meaning["sem"]["excess"] > unrelated["sem"]["excess"]
    md = (tmp_path / "out" / "tables.md").read_text()
    row = next(line for line in md.splitlines() if line.startswith("| drop_v6v"))
    assert "–" not in row  # every column filled

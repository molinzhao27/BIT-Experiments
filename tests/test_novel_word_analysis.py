"""scripts/novel_word_analysis.py, CPU only (project .venv: editdistance, h5py, wordfreq; pronouncing for the sound part).

    .venv/bin/python -m pytest tests/test_novel_word_analysis.py

- Training sentences: BIT's processing (punctuation but apostrophes removed, lower-cased), read from a synthetic
  hdf5_data_final/ directory and from the same files zipped.
- Alignment: edit count = editdistance for random word sequences; the most hits among fewest-edit alignments
  (brute force over every alignment of short sequences); ref "a b" / hyp "b c" keeps the hit.
- Counting on a hand-made decode set: novel / in-vocab recall, novel sub and del rates, fallback to known words,
  novel outputs, inflection split, bag-of-words count, WER = textmetrics.wer.
- Bootstrap: seeded and reproducible; identical runs give contrasts of exactly 0; standardization of a pool with
  the same recall in every cell returns that recall.
- seen_in_train flags that disagree with the training sentences stop the analysis.
- Meaning (stand-in contextual embedder, no model): a related substitute scores a higher contextual similarity
  than an unrelated one; seeded; per-trial F1 that depends only on the error count gives a matched gap of 0;
  examples are the same tokens for every run; per-trial files are found by run name.
"""

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import editdistance
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("wordfreq")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


nwa = _load("novel_word_analysis", ROOT / "scripts" / "novel_word_analysis.py")
from bit_experiments.textmetrics import wer  # noqa: E402

TRAIN = ["The cat sat.", "A dog ran home!", "the cat ran", "Don't stop, X-ray it.", "dogs bark"]


def _decodes(rows, train=TRAIN):
    _, train_set = nwa.train_counts(train)
    from bit_experiments.textmetrics import normalize

    return pd.DataFrame({"trial": range(len(rows)), "reference": [r for r, _ in rows], "decoded": [h for _, h in rows],
                         "seen_in_train": [normalize(r) in train_set for r, _ in rows]})


# ---------------------------------------------------------------- training sentences

def test_bit_sentence_and_counts():
    assert nwa.bit_sentence("Don't stop, X-ray it.") == "don't stop xray it"
    counts, sentences = nwa.train_counts(TRAIN)
    assert counts["cat"] == 2 and counts["ran"] == 2 and counts["don't"] == 1 and counts["xray"] == 1
    assert counts["x-ray"] == 0 and "the cat sat" in sentences


def _write_h5(path: Path, labels):
    h5py = pytest.importorskip("h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for k, s in enumerate(labels):
            g = f.create_group(f"trial_{k:04d}")
            g.attrs["sentence_label"] = s
            g.create_dataset("input_features", data=np.zeros((2, 4), dtype=np.float32))


def test_read_train_labels_dir_and_zip(tmp_path):
    base = tmp_path / "hdf5_data_final"
    _write_h5(base / "t15.2023.08.11" / "data_train.hdf5", ["Bring it closer.", "Hello there"])
    _write_h5(base / "t15.2023.08.13" / "data_train.hdf5", ["Third one"])
    _write_h5(base / "t15.2023.08.13" / "data_val.hdf5", ["not training"])
    labels, n = nwa.read_train_labels(tmp_path)
    assert labels == ["Bring it closer.", "Hello there", "Third one"] and n == 2
    zpath = tmp_path / "release.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for p in base.rglob("*.hdf5"):
            z.write(p, p.relative_to(tmp_path).as_posix())
    assert nwa.read_train_labels(zpath) == (labels, 2)
    cache = tmp_path / "cache" / "train.txt"
    got, prov = nwa.load_train_labels(cache, zpath)
    assert got == labels and cache.is_file() and prov["n_train_trials"] == 3
    again, prov2 = nwa.load_train_labels(cache, None)
    assert again == labels and prov2["sha256_labels"] == prov["sha256_labels"]


# ---------------------------------------------------------------- alignment

def _all_alignments(r, h):
    """Every alignment of r and h as (edits, hits)."""
    if not r and not h:
        yield 0, 0
        return
    if r and h:
        for e, k in _all_alignments(r[1:], h[1:]):
            yield e + (r[0] != h[0]), k + (r[0] == h[0])
    if r:
        for e, k in _all_alignments(r[1:], h):
            yield e + 1, k
    if h:
        for e, k in _all_alignments(r, h[1:]):
            yield e + 1, k


def _summary(ops):
    return sum(op != "ok" for op, _, _ in ops), sum(op == "ok" for op, _, _ in ops)


def test_align_fewest_edits_most_hits():
    rng = np.random.default_rng(0)
    for _ in range(300):
        r = list(rng.choice(list("abcd"), rng.integers(0, 5)))
        h = list(rng.choice(list("abcd"), rng.integers(0, 5)))
        ops = nwa.align(r, h)
        edits, hits = _summary(ops)
        assert edits == editdistance.eval(r, h)
        best = max(k for e, k in _all_alignments(r, h) if e == edits)
        assert hits == best
        assert [i for _, i, _ in ops if i is not None] == list(range(len(r)))
        assert [j for _, _, j in ops if j is not None] == list(range(len(h)))
    assert nwa.align(["a", "b"], ["b", "c"]) == [("del", 0, None), ("ok", 1, 0), ("ins", None, 1)]


# ---------------------------------------------------------------- counting

def test_counting_on_hand_made_decodes():
    counts, _ = nwa.train_counts(TRAIN)
    # novel reference words: zebra (correct), lion (-> dog, a known word), tiger (deleted), barks (-> barks; an
    # inflection of bark), purple (-> mauve, unknown); in-vocab: everything else.
    rows = [("the zebra sat", "the zebra sat"),
            ("a lion ran home", "a dog ran home"),
            ("the tiger ran", "the ran"),
            ("dog barks", "dog barks loudly"),
            ("purple cat", "mauve cat"),
            ("the cat sat", "the cat sat")]
    df = _decodes(rows)
    rec, hyp, pairs = nwa.word_records(df, counts)
    boot = nwa.Boot(len(df), 50, 0)
    res, _ = nwa.analyse_run(rec, hyp, boot)
    rc = res["recall"]
    assert rc["novel"]["n"] == 5 and rc["novel"]["ok"] == 2 and rc["novel"]["est"] == pytest.approx(2 / 5)
    n_inv = len(rec) - 5
    assert rc["inv"]["n"] == n_inv and rc["inv"]["ok"] == n_inv  # every in-vocab word decoded correctly
    assert res["novel_ops"]["n_sub"] == 2 and res["novel_ops"]["n_del"] == 1
    assert res["novel_ops"]["sub_rate"] == pytest.approx(2 / 5) and res["novel_ops"]["del_rate"] == pytest.approx(1 / 5)
    assert res["fallback_known"]["novel"]["known"] == 1 and res["fallback_known"]["novel"]["n_sub"] == 2
    # decoded tokens outside the training vocabulary: zebra (ok), barks (ok), loudly (ins), mauve (sub)
    assert res["novel_outputs"]["n_novel"] == 4 and res["novel_outputs"]["correct"] == 2
    assert res["novel_in_decode_anywhere"] == 2
    assert rc["novel_inflection"]["n"] == 1 and rc["novel_inflection"]["ok"] == 1  # barks / bark
    assert sorted(p["ref"] for p in pairs if p["novel"]) == ["lion", "purple"]
    # unseen-sentence trials only in inv_unseen: the last trial is a training sentence
    assert bool(df["seen_in_train"].iloc[-1]) and rc["inv_unseen"]["n"] == n_inv - 3


def test_inflections():
    vocab = {"fox", "raid", "run", "try", "delivered", "corner", "comments"}
    for w in ("foxes", "raiding", "running", "tried", "trying", "deliver", "comment"):
        assert nwa.inflection_of_train(w, vocab), w
    for w in ("corn", "zebra", "foxy"):
        assert not nwa.inflection_of_train(w, vocab), w


# ---------------------------------------------------------------- bootstrap and the whole analysis

def _synthetic_runs():
    rows_a = [("the zebra sat", "the zebra sat"), ("a lion ran home", "a dog ran home"),
              ("the tiger ran", "the ran"), ("dog barks", "dog barks loudly"), ("purple cat", "mauve cat"),
              ("the cat sat", "the cat sat"), ("a dog sat", "a cat sat")]
    rows_b = [(r, r) if k % 2 else (r, "the cat") for k, (r, _) in enumerate(rows_a)]
    return {"run_a": _decodes(rows_a), "run_b": _decodes(rows_b), "run_a_copy": _decodes(rows_a)}


def test_analyse_contrasts_and_seed(monkeypatch):
    monkeypatch.setattr(nwa, "CONTRASTS", [("a - b", "run_a", "run_b"), ("a - copy", "run_a", "run_a_copy")])
    monkeypatch.setattr(nwa, "DID", [("did", ("run_a", "run_b"), ("run_a_copy", "run_b"))])
    runs = _synthetic_runs()
    res = nwa.analyse(runs, TRAIN, 300, 0, False)
    again = nwa.analyse(runs, TRAIN, 300, 0, False)
    assert json.dumps(res["contrasts"], default=float) == json.dumps(again["contrasts"], default=float)
    for name, df in runs.items():
        assert res["runs"][name]["wer"] == pytest.approx(wer(list(df.reference), list(df.decoded)))
    by = {c["label"]: c for c in res["contrasts"]}
    for m in ("novel", "inv", "inv_unseen"):
        assert by["a - copy"][m]["est"] == 0 and by["a - copy"][m]["ci"] == [0.0, 0.0]
    assert by["a - b"]["novel"]["est"] == pytest.approx(
        res["runs"]["run_a"]["recall"]["novel"]["est"] - res["runs"]["run_b"]["recall"]["novel"]["est"])
    assert by["did"]["novel"]["est"] == pytest.approx(by["a - b"]["novel"]["est"])
    v = res["vocab"]
    assert v["n_novel_tokens"] == 5 and v["n_novel_types"] == 5
    assert res["checks"]["seen_matches"]
    md = nwa.markdown({**res, "train": {"n_train_trials": len(TRAIN), "n_unique_sentences": 5}, "caveats": ["x"]})
    assert "Novel recall" in md and "a - b" in md


def test_seen_mismatch_stops():
    runs = _synthetic_runs()
    runs["run_b"].loc[0, "seen_in_train"] = not runs["run_b"].loc[0, "seen_in_train"]
    with pytest.raises(SystemExit):
        nwa.analyse(runs, TRAIN, 10, 0, False)


def test_matched_constant_pool():
    # pool words: recall 0.5 in every cell; target words in two cells
    rec = pd.DataFrame({"trial": [0, 0, 1, 1, 2, 2, 3, 3], "cell": [1, 1, 2, 2, 1, 2, 1, 2]})
    target = np.array([False] * 4 + [True] * 4)
    pool = ~target
    hit = np.array([True, False, True, False, True, True, False, False])
    boot = nwa.Boot(4, 20, 0)
    pt, bt, bm = boot.matched(rec, target, pool, hit, "cell")
    assert pt["matched"] == pytest.approx(0.5) and pt["target"] == pytest.approx(0.5) and pt["coverage"] == 1.0
    assert np.allclose(bm[~np.isnan(bm)], 0.5) and (~np.isnan(bm)).any()


def test_phonetic_part_runs():
    pytest.importorskip("pronouncing")
    runs = _synthetic_runs()
    res = nwa.analyse({k: runs[k] for k in ("run_a", "run_b")}, TRAIN, 50, 0, True)
    ph = res["phonetic"]["run_a"]
    assert ph["novel"]["n_sub"] == 2 and ph["inv"]["n_sub"] == 1  # lion->dog, purple->mauve; dog->cat
    assert {"exc_phon", "obs_alike", "null_alike"} <= set(ph["novel"])


# ---------------------------------------------------------------- meaning

CLUSTERS = {"animal": {"cat", "dog", "lion", "zebra", "tiger"}, "colour": {"purple", "mauve", "red"}}


def _stub_embedder(calls):
    """Contextual vector = the slot word's cluster direction (plus a small word-specific part); records calls."""
    dims = {name: k for k, name in enumerate(CLUSTERS)}

    def vec(word):
        v = np.random.default_rng(abs(hash(word)) % 2**32).normal(0, 0.1, 8)
        for name, words in CLUSTERS.items():
            if word in words:
                v[dims[name]] += 1.0
        return v

    def embed(items):
        calls.append(len(items))
        return {nwa.ContextEmbedder.key(w, p): vec(w[p]) for w, p in items}

    return embed


def _meaning_runs():
    rows_a = [("a lion ran home", "a dog ran home"),        # novel lion -> dog: related
              ("the zebra sat", "the red sat"),             # novel zebra -> red: unrelated
              ("purple cat", "mauve cat"),                  # novel purple -> mauve: related
              ("the cat ran", "the home ran"),              # in-vocab cat -> home: unrelated
              ("a dog sat", "a cat sat"),                   # in-vocab dog -> cat: related
              ("the tiger ran", "the tiger ran")]
    rows_b = [(r, r) for r, _ in rows_a]
    return {"run_a": _decodes(rows_a), "run_b": _decodes(rows_b)}


def _per_trial(df):
    from bit_experiments.textmetrics import word_errors

    errs = [word_errors(r, h)[0] for r, h in zip(df.reference, df.decoded)]
    return pd.DataFrame({"trial": df.trial, "reference": df.reference, "decoded": df.decoded, "word_errors": errs,
                         "bs_f1": [1.0 - 0.2 * e for e in errs]})


def test_meaning_section_with_stub_embedder(monkeypatch):
    pytest.importorskip("pronouncing")
    monkeypatch.setattr(nwa, "CONTRASTS", [("a - b", "run_a", "run_b")])
    monkeypatch.setattr(nwa, "EXAMPLE_RUNS", ["run_a", "run_b"])
    runs = _meaning_runs()
    calls = []
    meaning = {"embed_model": None, "contextual": _stub_embedder(calls), "k": 2,
               "per_trial": {k: _per_trial(v) for k, v in runs.items()}, "examples": 3}
    res = nwa.analyse(runs, TRAIN, 200, 0, False, meaning)
    again = nwa.analyse(runs, TRAIN, 200, 0, False, {**meaning, "contextual": _stub_embedder([])})
    assert json.dumps(res["meaning"], default=float) == json.dumps(again["meaning"], default=float)
    assert res["contextual_info"]["k_null"] == 2 and calls
    m = res["meaning"]["run_a"]
    assert m["novel"]["n_sub"] == 3 and m["inv"]["n_sub"] == 2
    assert np.isnan(m["novel"]["exc_sem"]["est"])  # no embed model: sem left out
    # per-pair: related substitutes score higher than unrelated ones
    rec, _, pairs = nwa.word_records(runs["run_a"], nwa.train_counts(TRAIN)[0])
    ref_words = {int(t): r.split() for t, r in zip(runs["run_a"].trial, runs["run_a"].reference)}
    ras = nwa.pair_analyses({"run_a": pairs}, ref_words, None)
    nwa.contextual_similarity(ras, ref_words, _stub_embedder([]), 2, 0)
    sim = {(p["ref"], p["hyp"]): p["obs_csim"] for p in ras["run_a"].pairs}
    assert sim[("lion", "dog")] > sim[("zebra", "red")] and sim[("dog", "cat")] > sim[("cat", "home")]
    assert sim[("purple", "mauve")] > sim[("zebra", "red")]
    # sentence level: F1 depends only on the error count, so the errors-matched gap is 0
    sm = res["sentence_meaning"]["run_a"]
    assert sm["matched_errors"]["gap"]["est"] == pytest.approx(0.0, abs=1e-12)
    assert sm["f1_missed"]["n"] == 3
    # examples: the same novel tokens for both runs
    ex = res["examples"]
    assert len(ex) == 3 and all(set(r["outputs"]) == {"run_a", "run_b"} for r in ex)
    assert all(r["outputs"]["run_b"] == "= (correct)" for r in ex)
    assert all(r["sentence"].split()[r["pos"]] == r["word"].upper() for r in ex)
    md = nwa.markdown({**res, "train": {"n_train_trials": 5, "n_unique_sentences": 5}, "caveats": nwa.caveats(res)})
    assert "Meaning of the wrong word" in md and "Sentence meaning when a novel word is missed" in md


def test_per_trial_path_naming():
    path = nwa.per_trial_path("B_ssl_s42")
    if path is None:
        pytest.skip("no local per-trial file for B_ssl_s42")
    assert path.name == "B_ssl__B__real__brandman_2024_text_test__greedy__seed42.csv"
    assert nwa.per_trial_path("no_such_run_s42") is None

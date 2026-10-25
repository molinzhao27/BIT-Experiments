"""N-best rescoring side test (docs/RESCORING.md), CPU only, tiny stubs, no downloads.

    .venv/bin/python -m pytest tests/test_rescoring.py          # the sampler test is skipped without peft
    ~/envs/bit-mac/bin/python -m pytest tests/test_rescoring.py   # BIT's versions: everything runs

- tuning sees BIT-val only, and picks the val optimum whatever BIT-test holds;
- the oracle is a lower bound for every rule; the greedy rule returns the greedy decode;
- the paired bootstrap is deterministic for a seed;
- candidate files are resume-safe (torn lines and incomplete trials are dropped; a rerun decodes only the rest);
- AR decoder scores stop at the first EOS; OPT scoring skips its BOS, ignores padding, and scores "" as 0;
- the diffusion sampler's commit_logprobs option is off by default and changes no decode.
- report, end to end on a synthetic run (stub OPT scores, no BERTScore): the oracle bounds every rule, and runs that
  differ only in BIT-test get the same tuned weights.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "bit"))

from bit_experiments import rescoring as rs  # noqa: E402

GRID = {"alpha": [0, 0.5, 1, 2, 4], "beta": [-1, 0, 1], "gamma": [0, 1]}


def cand_frame(trials, seed=0, n=6):
    """Candidates for `trials` = [(reference, [candidate texts...]), ...]; the first candidate is the greedy one.
    Random decoder / OPT scores."""
    rng = np.random.default_rng(seed)
    rows = []
    for t, (ref, cands) in enumerate(trials):
        for k, c in enumerate(cands):
            rows.append({"trial": t, "k": k, "n": len(cands), "reference": ref, "candidate": c, "is_greedy": k == 0,
                         "seen_in_train": t % 2 == 0, "dec_sum": float(rng.normal(-5, 2)), "dec_mean": 0.0})
    df = pd.DataFrame(rows)
    df["norm"] = df.candidate.map(rs.normalize)
    opt = {rs.opt_text(c, {}): {"opt_sum": float(rng.normal(-30, 10)), "opt_n_tokens": len(c.split()) + 1,
                                "opt_eos": -2.0} for c in df.candidate}
    return df, opt


def random_trials(n_trials, seed):
    rng = np.random.default_rng(seed)
    words = "the a cat dog sat ran on mat log fast slow".split()
    out = []
    for _ in range(n_trials):
        ref = " ".join(rng.choice(words, size=rng.integers(3, 7)))
        cands = [" ".join(rng.choice(words, size=rng.integers(2, 8))) for _ in range(5)]
        cands.insert(int(rng.integers(0, 3)), ref)  # sometimes right
        out.append((ref, cands))
    return out


def table(trials, seed=0):
    df, opt = cand_frame(trials, seed)
    return rs.trial_table(df, opt, {"text": "normalized"})


# ---------------------------------------------------------------- tables and rules

def test_duplicates_merge_and_greedy_rule_returns_greedy():
    t = table([("the cat sat", ["The cat sat.", "the cat sat", "a cat sat", "the cat"])])
    assert t["count"].tolist() == [2, 1, 1] and t.is_greedy.tolist() == [True, False, False]
    assert t.errors.tolist() == [0, 1, 1] and t.n_samples.tolist() == [4, 4, 4]
    sel = rs.Selector(t)
    assert sel.select(rs.rule_weights("greedy")).tolist() == [0]


def test_oracle_is_a_lower_bound_for_every_rule():
    t = table(random_trials(60, 1), seed=1)
    sel = rs.Selector(t)
    g = t.groupby("trial")
    oracle = g.errors.min().sum() / g.ref_words.first().sum()
    rng = np.random.default_rng(0)
    for rule in ("greedy", "opt", "opt_norm", "decoder"):
        assert sel.wer(rs.rule_weights(rule)) >= oracle - 1e-12
    for _ in range(50):
        w = {"dec_sum": 1.0, "opt_sum": float(rng.normal()), "n_words": float(rng.normal())}
        assert sel.wer(w) >= oracle - 1e-12


def test_tuning_uses_val_only():
    val, test = table(random_trials(80, 2), seed=2), table(random_trials(80, 3), seed=3)
    cfg = {"grid": GRID}
    params, wer = rs.tune("dec_opt", rs.Selector(val), cfg)
    brute = min(((rs.Selector(val).wer(rs.rule_weights("dec_opt", {"alpha": a, "beta": b})), i, {"alpha": a, "beta": b})
                 for i, (a, b) in enumerate((a, b) for a in GRID["alpha"] for b in GRID["beta"])),
                key=lambda x: (x[0], x[1]))
    assert params == brute[2] and wer == brute[0]
    # whatever BIT-test holds (here: its best weights differ), the tuned weights are val's
    test_best = rs.tune("dec_opt", rs.Selector(test), cfg)[0]
    poisoned = test.assign(errors=test.errors[::-1].to_numpy())
    for other in (test, poisoned):
        assert rs.tune("dec_opt", rs.Selector(val), cfg)[0] == params
        rs.Selector(other).wer(rs.rule_weights("dec_opt", params))  # applied unchanged
    assert isinstance(test_best, dict)


def test_paired_bootstrap_is_deterministic():
    rng = np.random.default_rng(4)
    err, base, words = rng.integers(0, 5, 200).astype(float), rng.integers(0, 5, 200).astype(float), np.full(200, 6.0)
    a = rs.paired_bootstrap(err, base, words, rs.bootstrap_indices(200, 2000, 0))
    b = rs.paired_bootstrap(err, base, words, rs.bootstrap_indices(200, 2000, 0))
    c = rs.paired_bootstrap(err, base, words, rs.bootstrap_indices(200, 2000, 1))
    assert a == b and a != c
    assert a["delta"] == pytest.approx((err.sum() - base.sum()) / words.sum())
    assert a["ci_low"] <= a["delta"] <= a["ci_high"]


# ---------------------------------------------------------------- candidate files and resume

def rows_for(trial, n=3):
    return [{"split": "val", "trial": trial, "k": k, "n": n, "reference": "r", "candidate": f"c{k}",
             "is_greedy": k == 0, "scores": {"dec_sum": -1.0}} for k in range(n)]


def test_candidate_file_drops_torn_lines_and_incomplete_trials(tmp_path):
    f = tmp_path / "candidates_val.jsonl"
    cf = rs.CandidateFile(f)
    cf.append(rows_for(0))
    cf.append(rows_for(1))
    with open(f, "a") as h:  # a crash mid-trial: one complete line of trial 2, then a torn one
        h.write(json.dumps(rows_for(2)[0]) + "\n" + json.dumps(rows_for(2)[1])[:20])
    again = rs.CandidateFile(f)
    assert again.done == {0, 1} and len(again.rows) == 6
    assert len(f.read_text().splitlines()) == 6  # rewritten without the partial trial
    with pytest.raises(ValueError):
        again.append(rows_for(3)[:2])  # not a whole trial


def load_nbest():
    spec = importlib.util.spec_from_file_location("nbest_decode", ROOT / "scripts" / "nbest_decode.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class StubDecoder:
    def __init__(self):
        self.calls = []

    def inputs(self, ds, i):
        self.calls.append(i)
        return i

    def greedy(self, i):
        return {"candidate": f"greedy {i}", "scores": {"dec_sum": -1.0}}

    def samples(self, i, seed):
        return [{"candidate": f"sample {i} {k}", "scores": {"dec_sum": -2.0}, "decoding": "nucleus"} for k in range(2)]


def test_resume_skips_existing_trials(tmp_path):
    nb = load_nbest()
    f, ds = tmp_path / "candidates_test.jsonl", list(range(8))
    refs, seen = [f"ref {i}" for i in ds], [False] * 8
    dec = StubDecoder()
    nb.run_split(dec, ds, "test", refs, seen, f, seed=42, limit=5)
    assert dec.calls == [0, 1, 2, 3, 4]
    lines = f.read_text().splitlines()
    with open(f, "w") as h:  # lose the last trial's last line
        h.write("\n".join(lines[:-1]) + "\n")
    dec2 = StubDecoder()
    timing = nb.run_split(dec2, ds, "test", refs, seen, f, seed=42, limit=None)
    assert dec2.calls == [4, 5, 6, 7] and timing["rows"] == 8 * 3
    rows = rs.CandidateFile(f).rows
    assert sorted({r["trial"] for r in rows}) == list(range(8))
    assert all(r["seed"] == 42 * 100003 + r["trial"] for r in rows)
    assert [r["is_greedy"] for r in rows if r["trial"] == 6] == [True, False, False]


# ---------------------------------------------------------------- decoder and OPT scores

def test_ar_scores_stop_at_the_first_eos():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 7)
    ids = torch.tensor([[1, 2, 6, 3, 3], [4, 4, 4, 4, 4]])  # eos 6 at position 2, then padding; row 2 never ends
    out = rs.ar_sequence_scores(logits, ids, eos_id=6)
    lp = torch.log_softmax(logits, -1)
    assert out[0]["dec_n_tokens"] == 3 and out[0]["ended_eos"]
    assert out[0]["dec_sum"] == pytest.approx(float(lp[0, 0, 1] + lp[0, 1, 2] + lp[0, 2, 6]), abs=1e-5)
    assert out[1]["dec_n_tokens"] == 5 and not out[1]["ended_eos"]
    assert out[1]["dec_mean"] == pytest.approx(out[1]["dec_sum"] / 5)


class BigramLM(torch.nn.Module):
    """log p(next | previous) from a fixed table; ignores everything else (position-free, so padding is harmless)."""

    def __init__(self, v=12):
        super().__init__()
        torch.manual_seed(1)
        self.table = torch.nn.Parameter(torch.randn(v, v))

    def forward(self, input_ids, attention_mask=None):
        return type("Out", (), {"logits": self.table[input_ids]})()


class CharTokenizer:
    """OPT-like: prepends BOS (id 2) to every text; letters a..i are ids 3..11; pads with 1 on the right."""
    bos_token_id, eos_token_id, pad_token_id = 2, 2, 1

    def __call__(self, texts, return_tensors="pt", padding=True):
        seqs = [[2] + [3 + "abcdefghi".index(ch) for ch in t if ch != " "] for t in texts]
        n = max(map(len, seqs))
        ids = torch.tensor([s + [1] * (n - len(s)) for s in seqs])
        mask = torch.tensor([[1] * len(s) + [0] * (n - len(s)) for s in seqs])
        return {"input_ids": ids, "attention_mask": mask}


def test_opt_scoring_skips_bos_ignores_padding_and_scores_empty_as_zero():
    lm, tok = BigramLM(), CharTokenizer()
    lp = torch.log_softmax(lm.table, -1)
    alone = rs.opt_logprobs(["abc"], lm, tok, "cpu")[0]
    batch = rs.opt_logprobs(["ab", "abc", "abcdefghi", ""], lm, tok, "cpu", batch_size=4)
    want = float(lp[2, 3] + lp[3, 4] + lp[4, 5])  # BOS -> a -> b -> c: BOS itself is not scored
    assert alone["opt_sum"] == pytest.approx(want, abs=1e-5) and alone["opt_n_tokens"] == 3
    assert batch[1]["opt_sum"] == pytest.approx(want, abs=1e-5)
    assert alone["opt_eos"] == pytest.approx(float(lp[5, 2]), abs=1e-5)
    assert batch[3]["opt_sum"] == 0 and batch[3]["opt_n_tokens"] == 0
    assert batch[3]["opt_eos"] == pytest.approx(float(lp[2, 2]), abs=1e-5)


def test_score_cache_round_trip(tmp_path):
    c = rs.ScoreCache(tmp_path / "cache.jsonl")
    c.add({"a b": {"opt_sum": -1.5}})
    assert rs.ScoreCache(tmp_path / "cache.jsonl").data == {"a b": {"opt_sum": -1.5}}
    assert c.missing(["a b", "c"]) == ["c"]


# ---------------------------------------------------------------- the diffusion sampler's new option

def test_commit_logprobs_off_by_default_and_decodes_unchanged():
    pytest.importorskip("peft")
    spec = importlib.util.spec_from_file_location("tdr", ROOT / "tests" / "test_diffusion_revision.py")
    tdr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tdr)
    llm = tdr.tiny_llm()
    llm.eval()
    m = tdr.bci(llm)
    assert m.commit_logprobs is None
    g = torch.Generator().manual_seed(0)
    for n in (5, 31, 32, 33, 64, 90):
        prefix = torch.randn(n, tdr.D, generator=g)
        for kw in (tdr.GREEDY, tdr.NUCLEUS, tdr.GREEDY | {"max_new_tokens": 70}):
            torch.manual_seed(n)
            plain = m._denoise(prefix, eos=tdr.EOS, **kw)
            m.commit_logprobs, m.trace = [], []
            torch.manual_seed(n)
            scored = m._denoise(prefix, eos=tdr.EOS, **kw)
            rec, trace = m.commit_logprobs[-1], m.trace[-1]
            m.commit_logprobs = m.trace = None
            assert scored == plain and rec["ids"] == plain and len(rec["logprobs"]) == len(plain)
            assert all(x <= 1e-6 for x in rec["logprobs"])
            if not kw["do_sample"]:  # greedy commits the argmax: its log-prob is log(the trace's probability)
                last = {}
                for e in trace["events"]:
                    for pos, tokn, prob, committed in e.get("m", []):
                        if committed:
                            last[pos] = (tokn, prob)
                for pos, (tokn, prob) in last.items():
                    if pos < len(plain):
                        assert plain[pos] == tokn
                        assert np.exp(rec["logprobs"][pos]) == pytest.approx(prob, abs=1e-4)


# ---------------------------------------------------------------- report, end to end (synthetic run, no BERTScore)

def load_rescore(monkeypatch, out):
    spec = importlib.util.spec_from_file_location("rescore_nbest", ROOT / "scripts" / "rescore_nbest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "OUT", out)
    return mod


def write_run(out, run, seed, test_refs_shift=0):
    rng = np.random.default_rng(seed)
    texts = set()
    for split, trials in (("val", random_trials(40, seed)), ("test", random_trials(40, seed + 1))):
        cf = rs.CandidateFile(out / run / f"candidates_{split}.jsonl")
        for t, (ref, cands) in enumerate(trials):
            ref = " ".join(ref.split()[test_refs_shift:]) if split == "test" else ref
            cf.append([{"split": split, "trial": t, "k": k, "n": len(cands), "reference": ref, "seen_in_train": t % 3 == 0,
                        "candidate": c, "is_greedy": k == 0, "decoding": "greedy" if k == 0 else "nucleus",
                        "seed": t, "scores": {"dec_sum": float(rng.normal(-5, 2)), "dec_mean": -1.0}}
                       for k, c in enumerate(cands)])
            texts |= {rs.normalize(c) for c in cands}
        (out / run / f"candidates_{split}.meta.json").write_text(json.dumps(
            {"model_class": "stub", "llm": "stub", "seed": 42, "timing": {"seconds_this_session": 1.0,
                                                                          "sec_per_candidate": 0.1}}))
    return texts


def test_report_end_to_end_tunes_on_val_only(tmp_path, monkeypatch):
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "rescoring.yaml").read_text())
    cfg["metrics"]["bootstrap"]["n_resamples"] = 500
    texts = write_run(tmp_path, "R1", 5) | write_run(tmp_path, "R2", 5, test_refs_shift=1)
    rng = np.random.default_rng(0)
    mod = load_rescore(monkeypatch, tmp_path)
    rs.ScoreCache(mod.opt_cache_path(cfg)).add({t: {"opt_sum": float(rng.normal(-20, 5)), "opt_n_tokens": 5,
                                                    "opt_eos": -1.0} for t in sorted(texts)})
    s1, s2 = mod.report("R1", cfg, bertscore=False), mod.report("R2", cfg, bertscore=False)
    assert set(s1["rules"]) == {"greedy", "opt", "dec_opt"}
    for s in (s1, s2):
        for r in s["rules"].values():
            assert s["oracle"]["test"]["wer"] <= r["test"]["wer"] + 1e-12
    # same val candidates, different test references: same tuned weights and val WER
    assert s1["rules"]["dec_opt"]["params"] == s2["rules"]["dec_opt"]["params"]
    assert s1["rules"]["dec_opt"]["val_wer_tuned"] == s2["rules"]["dec_opt"]["val_wer_tuned"]
    assert s1["rules"]["greedy"]["vs_greedy"]["delta"] == 0
    assert (tmp_path / "R1" / "summary.md").exists() and (tmp_path / "R1" / "selected" / "test_dec_opt.csv").exists()
    assert "R2" in mod.write_table().read_text()

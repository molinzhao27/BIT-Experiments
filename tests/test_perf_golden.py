"""CPU tests for the golden harness's comparison logic (perf/golden): numeric comparisons, near-tie classification,
block-diffusion trace alignment, compare.py on synthetic run directories, noise_floor.py and the verdict, and the
runner's pure helpers (fixed batches, shortest-trial selection, sampler copies, code-root checks)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

GOLDEN = Path(__file__).resolve().parents[1] / "perf" / "golden"
sys.path.insert(0, str(GOLDEN))

import compare  # noqa: E402
import golden_lib as gl  # noqa: E402
import noise_floor  # noqa: E402
import run as runner  # noqa: E402


# ---------------------------------------------------------------- numeric comparisons

def test_compare_tensors_exact_and_differences():
    a = {"w": torch.tensor([1.0, -2.0, 4.0]), "b": torch.tensor([0.5])}
    assert gl.compare_tensors(a, {k: v.clone() for k, v in a.items()})["exact"]
    b = {"w": torch.tensor([1.0, -2.0, 4.001]), "b": torch.tensor([0.5])}
    s = gl.compare_tensors(a, b)
    assert not s["exact"] and s["comparable"]
    assert s["max_abs"] == pytest.approx(0.001, rel=1e-3)
    assert s["max_rel"] == pytest.approx(0.001 / 4.0, rel=1e-3)  # normwise: |diff|_inf / |base|_inf
    assert s["worst"] == "w" and s["n_tensors_differ"] == 1


def test_compare_tensors_bitwise_signed_zero_and_nan():
    z = gl.compare_tensors({"x": torch.tensor([0.0])}, {"x": torch.tensor([-0.0])})
    assert not z["exact"] and z["max_abs"] == 0.0  # numerically equal, bitwise different
    nan = torch.tensor([1.0, float("nan")])
    assert gl.compare_tensors({"x": nan}, {"x": nan.clone()})["exact"]
    other = gl.compare_tensors({"x": nan}, {"x": torch.tensor([1.0, 2.0])})
    assert math.isinf(other["max_abs"]) and other["problems"]


def test_compare_tensors_shape_and_missing():
    s = gl.compare_tensors({"x": torch.zeros(2)}, {"x": torch.zeros(3)})
    assert not s["comparable"] and not s["exact"]
    s = gl.compare_tensors({"x": torch.zeros(2)}, {"y": torch.zeros(2)})
    assert not s["comparable"] and len(s["problems"]) == 2


def test_compare_tensors_integers_count_mismatches():
    s = gl.compare_tensors({"i": torch.tensor([1, 2, 3])}, {"i": torch.tensor([1, 5, 3])})
    assert s["discrete_mismatches"] == 1 and not s["exact"]


def test_compare_scalars_elementwise_relative():
    s = gl.compare_scalars([2.0, 10.0], [2.0, 10.5])
    assert s["max_abs"] == pytest.approx(0.5) and s["max_rel"] == pytest.approx(0.05)
    assert gl.compare_scalars([[1.0, 0.9]], [[1.0, 0.9]])["exact"]
    assert not gl.compare_scalars([1.0], [1.0, 2.0])["comparable"]


def test_nested_digest_bitwise_and_callables():
    assert gl.nested_digest({"a": torch.tensor([1.0])}) != gl.nested_digest({"a": torch.tensor([1.0 + 1e-7])})
    assert gl.nested_digest({"f": print}) == gl.nested_digest({"f": print})
    assert gl.nested_digest((1, np.arange(3))) == gl.nested_digest((1, np.arange(3)))


# ---------------------------------------------------------------- near ties

def test_classify_ar():
    base = {"ids": [5, 6, 7, 2], "margins": [3.0, 0.125, 4.0, 9.0]}
    assert gl.classify_ar(base, {"ids": [5, 6, 7, 2]}) is None
    r = gl.classify_ar(base, {"ids": [5, 9, 7, 2]})
    assert r["pos"] == 1 and r["near_tie"] and r["margin"] == 0.125
    r = gl.classify_ar(base, {"ids": [5, 6, 8]})
    assert r["pos"] == 2 and not r["near_tie"]
    r = gl.classify_ar(base, {"ids": [5, 6, 7, 2, 4]})  # baseline is a strict prefix: no margin there
    assert r["pos"] == 4 and not r["near_tie"]


def _events(tokens, margin=3.0, prob=0.95):
    """One block, one fill pass committing every token: m entries [pos, token, prob, committed, margin]."""
    return [{"b": 0, "m": [[i, t, prob, 1, margin] for i, t in enumerate(tokens)]}]


def test_classify_diffusion_token_near_tie_and_not():
    base = {"ids": [10, 11, 2], "events": _events([10, 11, 2], margin=0.25)}
    cand = {"ids": [10, 12, 2], "events": _events([10, 12, 2])}
    r = gl.classify_diffusion(base, cand, 0.9)
    assert r["how"] == "trace" and r["near_tie"] and r["decisions"][0]["kind"] == "token"
    base["events"] = _events([10, 11, 2], margin=2.0)
    assert not gl.classify_diffusion(base, cand, 0.9)["near_tie"]
    assert gl.classify_diffusion(base, {"ids": [10, 11, 2]}, 0.9) is None


def test_classify_diffusion_commit_decision():
    # pass 1: position 0 committed (p 0.95), position 1 not (p 0.88, just under 0.9) in the baseline; the candidate
    # commits position 1 too. Then the runs fill different tokens later.
    base_ev = [{"b": 0, "m": [[0, 10, 0.95, 1, 3.0], [1, 11, 0.88, 0, 2.0]]}, {"b": 0, "m": [[1, 13, 0.97, 1, 3.0]]}]
    cand_ev = [{"b": 0, "m": [[0, 10, 0.95, 1, 3.0], [1, 11, 0.91, 1, 2.0]]}]
    base, cand = {"ids": [10, 13], "events": base_ev}, {"ids": [10, 11], "events": cand_ev}
    r = gl.classify_diffusion(base, cand, 0.9)
    assert r["event"] == 0 and r["decisions"][0]["kind"] == "commit" and r["near_tie"]
    base_ev[0]["m"][1][2] = 0.5  # far below the threshold: not a near-tie
    assert not gl.classify_diffusion(base, cand, 0.9)["near_tie"]


def test_classify_diffusion_ignores_uncommitted_proposals_and_falls_back():
    # an uncommitted proposal differs (no effect on the state); the first commit difference comes later
    base_ev = [{"b": 0, "m": [[0, 10, 0.95, 1, 3.0], [1, 11, 0.5, 0, 0.1]]}, {"b": 0, "m": [[1, 12, 0.97, 1, 1.5]]}]
    cand_ev = [{"b": 0, "m": [[0, 10, 0.95, 1, 3.0], [1, 14, 0.5, 0, 0.1]]}, {"b": 0, "m": [[1, 15, 0.97, 1, 1.5]]}]
    r = gl.classify_diffusion({"ids": [10, 12], "events": base_ev}, {"ids": [10, 15], "events": cand_ev}, 0.9)
    assert r["event"] == 1 and not r["near_tie"]
    # no candidate trace: the output-level fallback uses the baseline's commit margin
    r = gl.classify_diffusion({"ids": [10, 12], "events": base_ev, "margins": [3.0, 0.2], "commit_probs": [0.95, 0.97]},
                              {"ids": [10, 15]}, 0.9)
    assert r["how"] == "output" and r["near_tie"]


def test_classify_argmax():
    r = gl.classify_argmax([1, 2, 3, 4], [1, 9, 3, 8], [5.0, 0.1, 5.0, 3.0])
    assert r["mismatches"] == 2 and r["near_tie"] == 1 and r["other"] == 1


# ---------------------------------------------------------------- block-diffusion trace alignment

def _call(n, top1, margins):
    return {"L": n, "top1": list(top1), "margin": list(margins)}


def test_align_diffusion_trace_token_shift_and_prefill():
    bs, tail = 4, 1  # prefix: 4 full-block tokens + 1 tail token; block 0 has 3 slots
    rec = {"prefix_len": 5, "tail": tail, "out": [7, 8, 9, 2],
           "events": [{"b": 0, "m": [[0, 7, 0.95, 1], [1, 8, 0.5, 0], [2, 9, 0.6, 0]]},
                      {"b": 0, "m": [[1, 8, 0.97, 1], [2, 9, 0.96, 1]]},
                      {"b": 1, "first": [3, 2]}]}
    # pass input = [tail, 3 slots] (L 4); slot s reads raw position max(tail + s - 1, 0) = s
    calls = [_call(4, [0, 0, 0, 0], [0.0] * 4),  # the partial-prefix prefill: no event
             _call(4, [7, 8, 9, 5], [1.0, 2.0, 3.0, 4.0]),
             _call(4, [7, 8, 9, 5], [1.5, 2.5, 3.5, 4.5]),
             _call(4, [1, 1, 1, 2], [0.0, 0.0, 0.0, 6.0])]  # commit forward: the next block's first token is at -1
    a = gl.align_diffusion_trace(rec, calls, bs)
    assert not a["problems"]
    assert a["events"][0]["m"][0][4] == 1.0 and a["events"][1]["m"][1][4] == 3.5
    assert a["margins"] == [1.0, 2.5, 3.5, 6.0]
    assert a["commit_probs"] == [0.95, 0.97, 0.96, None]


def test_align_diffusion_trace_detects_misalignment():
    rec = {"prefix_len": 4, "tail": 0, "out": [7], "events": [{"b": 0, "first": [0, 7]}]}
    a = gl.align_diffusion_trace(rec, [_call(4, [0, 0, 0, 8], [1.0] * 4)], 4)
    assert a["problems"] and a["margins"] == [None]
    a = gl.align_diffusion_trace(rec, [_call(4, [0, 0, 0, 8], [0.0] * 4)], 4)  # exact tie: either token may win
    assert not a["problems"] and a["margins"] == [0.0]
    a = gl.align_diffusion_trace(rec, [], 4)
    assert a["problems"]


# ---------------------------------------------------------------- compare.py on synthetic runs

def make_run(d: Path, machine="m1", gpu="GPU A", loss_eps=0.0, grad_eps=0.0, decode_ids=None, resume_exact=True,
             case="b_ar", code="c0", harness="h0", family=None, train_args=None):
    d.mkdir(parents=True)
    shape = "small"
    gl.write_json(d / "manifest.json", {"machine": machine, "case": case, "shapes": [shape], "status": "ok",
                                        "case_config": {"family": family or case}, "train_args": train_args or [],
                                        "code_root": {"training_files_sha256": code, "git": None},
                                        "harness": {"combined_sha256": harness, "git": None}})
    sel = {"train": [{"indices": [0, 1], "data_sha256": "x"}, {"indices": [2, 3], "data_sha256": "y"}],
           "val": [{"indices": [0, 1], "data_sha256": "v"}]}
    ev = {"avg_loss": 1.0 + loss_eps, "batches": [{"loss": 8.0 + loss_eps, "n": 8, "argmax": [3, 4], "margin": [2.0, 0.1]}]}
    main = {"status": "ok", "env": {"gpu": gpu, "torch": "2.7.1", "cuda": "12.8"},
            "initial_digests": {"trainable": "t0", "rng_torch": "t", "rng_python": str(d)},  # python: per process
            "rng_fingerprint": {"rand": "r"},
            "shape_results": {shape: {"selection": sel,
                                      "train": {"loss": [5.0, 4.0 + loss_eps], "n": [10, 12],
                                                "shapes": [{"spikes": [2, 9, 4]}] * 2, "lr": [[1e-5], [2e-5]],
                                                "grad_norm": [3.0, 2.0 + grad_eps]},
                                      "scheduler_end": {"last_epoch": 2}, "digests_k": {"trainable": "k"},
                                      "digests_end": {"trainable": "e" if not loss_eps else "e2"},
                                      "grads_none": [], "eval_after_train": ev}},
            "best": {"digest": "b", "eval": {shape: ev}}}
    gl.write_json(d / "main" / "phase.json", main)
    sd = d / "main" / shape
    sd.mkdir()
    torch.save({"w": torch.tensor([1.0, 2.0 + grad_eps])}, sd / "grads_step1.pt")
    torch.save({"w": torch.tensor([1e-3, 1e-3 + grad_eps * 1e-3])}, sd / "updates.pt")
    torch.save({"state": {"w": {"step": torch.tensor(2.0), "exp_avg": torch.tensor([0.1, 0.2]),
                                "exp_avg_sq": torch.tensor([0.01, 0.02])}}, "groups": [{"lr": 2e-5}]},
               sd / "optimizer.pt")
    for key in ("rng_k", "rng_end"):
        torch.save({"torch": torch.zeros(4, dtype=torch.uint8)}, sd / f"{key}.pt")
    gl.write_json(d / "resume" / "phase.json", {"status": "ok", "shape_results": {shape: {
        "train": {"loss": [4.0 + loss_eps]}, "digests_end": {"trainable": "e"}, "exact": resume_exact,
        "load_matches_step_k": {"trainable": True}, "vs_uninterrupted": {"loss": {"exact": resume_exact}}}}})
    ids = decode_ids or [[5, 6, 2], [7, 8, 2]]
    gl.write_json(d / "main" / "decodes.json", {
        "trials": [3, 9], "diffusion": False, "threshold": None, "score": {"wer": 0.25 if decode_ids is None else 0.3},
        "rows": [{"trial": t, "text": " ".join(map(str, i)), "ids": i, "margins": [2.0, 0.2, 5.0]}
                 for t, i in zip([3, 9], ids)]})
    return d


def test_compare_identical_runs_all_exact(tmp_path):
    a, b = make_run(tmp_path / "a"), make_run(tmp_path / "b")
    res = compare.compare_runs(a, b)
    assert res["comparable"] and res["summary"]["all_exact"], res["summary"]
    assert res["same_machine"] and res["same_code"]


def test_compare_reports_differences_and_near_ties(tmp_path):
    a = make_run(tmp_path / "a")
    b = make_run(tmp_path / "b", machine="m2", gpu="GPU B", loss_eps=1e-5, grad_eps=1e-4,
                 decode_ids=[[5, 9, 2], [7, 8, 2]], resume_exact=False)
    res = compare.compare_runs(a, b)
    o = res["outputs"]
    assert o["small/train/loss"]["max_abs"] == pytest.approx(1e-5, rel=1e-3)
    assert o["small/grads_step1"]["max_abs"] == pytest.approx(1e-4, rel=1e-2)
    ids = o["best/decode/ids"]
    assert ids["mismatches"] == 1 and ids["near_tie"] == 1 and ids["details"][0]["pos"] == 1
    assert o["small/resume/self_exact"]["mismatches"] == 1
    assert not res["same_machine"]
    text = compare.human(res)
    assert "best/decode/ids" in text and "near-tie" in text


def test_compare_refuses_different_cases(tmp_path):
    res = compare.compare_runs(make_run(tmp_path / "a"), make_run(tmp_path / "b", case="b_diff"))
    assert not res["comparable"]


def test_compare_variant_in_family_and_informational_rng(tmp_path):
    a = make_run(tmp_path / "a", case="b_diff")
    b = make_run(tmp_path / "b", case="b_diff_fast", family="b_diff", train_args=["--runtime", "optimized"])
    res = compare.compare_runs(a, b)
    assert res["comparable"] and not res["same_case"] and not res["same_train_args"]
    assert res["outputs"]["init/digests/rng_python"]["kind"] == "info"
    assert res["summary"]["all_exact"] and res["summary"]["informational"] == {"init/digests/rng_python": False}
    assert compare.verdict(res, {"floor": {}})["pass"]


def test_noise_floor_and_verdict(tmp_path):
    a, b = make_run(tmp_path / "a"), make_run(tmp_path / "b")
    x = make_run(tmp_path / "x", machine="m2", gpu="GPU B", loss_eps=1e-5, grad_eps=1e-4,
                 decode_ids=[[5, 9, 2], [7, 8, 2]])
    same = tmp_path / "same.json"
    cross = tmp_path / "cross.json"
    gl.write_json(same, compare.compare_runs(a, b))
    gl.write_json(cross, compare.compare_runs(a, x))
    floor = tmp_path / "floor.json"
    assert noise_floor.main([str(same), "--out", str(floor), "--pair", "m1~m1", "--kind", "same_machine",
                             "--provisional", "--baseline-note", "trees"]) == 0
    doc = json.loads(floor.read_text())
    assert doc["provisional"] and doc["floor_basis"].startswith("same-machine")
    assert all(v["all_exact"] for v in doc["floor"].values())
    # a candidate that differs fails a bit-exact floor
    v = compare.verdict(compare.compare_runs(a, x), doc)
    assert not v["pass"]
    # add the cross-machine pair: the floor now admits those differences
    assert noise_floor.main([str(cross), "--out", str(floor), "--pair", "m1~m2", "--kind", "cross_machine",
                             "--merge"]) == 0
    doc = json.loads(floor.read_text())
    assert set(doc["pairs"]) == {"m1~m1", "m1~m2"} and doc["floor_basis"].startswith("cross-machine")
    assert doc["floor"]["b_ar/best/decode/ids"]["near_tie"] == 1
    assert compare.verdict(compare.compare_runs(a, x), doc)["pass"]
    # a larger difference than the floor fails
    y = make_run(tmp_path / "y", loss_eps=1e-3)
    assert not compare.verdict(compare.compare_runs(a, y), doc)["pass"]


def test_noise_floor_rejects_mixed_code(tmp_path):
    a, b = make_run(tmp_path / "a"), make_run(tmp_path / "b", code="c1")
    cmp_path = tmp_path / "c.json"
    gl.write_json(cmp_path, compare.compare_runs(a, b))
    with pytest.raises(SystemExit):
        noise_floor.main([str(cmp_path), "--out", str(tmp_path / "f.json"), "--pair", "p", "--kind", "same_machine"])


def test_noise_floor_one_baseline_tree_per_case(tmp_path):
    cmps = []
    for case, code in (("b_ar", "ar-tree"), ("b_diff", "diffusion-tree")):
        a = make_run(tmp_path / f"{case}1", case=case, code=code)
        b = make_run(tmp_path / f"{case}2", case=case, code=code)
        cmps.append(tmp_path / f"{case}.json")
        gl.write_json(cmps[-1], compare.compare_runs(a, b))
    floor = tmp_path / "f.json"
    assert noise_floor.main([*map(str, cmps), "--out", str(floor), "--pair", "m1~m1", "--kind", "same_machine"]) == 0
    doc = json.loads(floor.read_text())
    assert {k: v["training_files_sha256"] for k, v in doc["baseline_code"].items()} == {"b_ar": "ar-tree",
                                                                                         "b_diff": "diffusion-tree"}


# ---------------------------------------------------------------- runner helpers (no GPU)

class _DS:
    def __init__(self, lengths):
        self.dataset = [{"spikes": np.zeros((n, 2), np.float32), "sentence": f"s{i}"} for i, n in enumerate(lengths)]

    def __len__(self):
        return len(self.dataset)


def test_fixed_batches_and_shortest_selection():
    fb = runner.FixedBatches([[3, 1], [2, 0]])
    assert list(fb) == [[3, 1], [2, 0]] and len(fb) == 2 and fb.batch_size == 2
    ds = _DS([9, 3, 7, 3, 1, 8, 2, 5])
    b = runner.shortest_batches(ds, 2, 2, seed=0)
    assert sorted(i for x in b for i in x) == [1, 3, 4, 6]  # lengths 1, 2, 3, 3 (ties by index)
    assert b == runner.shortest_batches(ds, 2, 2, seed=0)
    info = runner.batch_info(ds, b)
    assert info[0]["spike_bins"] == [len(ds.dataset[i]["spikes"]) for i in b[0]]
    assert info[0]["data_sha256"] == runner.batch_info(ds, b)[0]["data_sha256"]


class _Sampler:
    """Like BIT's GroupBatchSampler: a numpy Generator shuffles on every epoch."""

    def __init__(self):
        self.rng = np.random.default_rng(42)

    def __iter__(self):
        idx = np.arange(10)
        self.rng.shuffle(idx)
        return iter([idx[i:i + 2].tolist() for i in range(0, 10, 2)])


class _Loader:
    def __init__(self):
        self.batch_sampler = _Sampler()


def test_real_loader_switches_only_the_sampler_instance():
    loader = _Loader()
    sampler = loader.batch_sampler
    prod, _ = runner.sampler_batches(loader, 2)
    assert runner.real_loader(loader, [[7, 8], [1, 2]]) is loader and loader.batch_sampler is sampler
    assert list(iter(sampler)) == [[7, 8], [1, 2]] and len(sampler) == 2
    assert isinstance(sampler, _Sampler)  # a subclass of the loader's own sampler class
    again, _ = runner.sampler_batches(loader, 2)  # copies still run the original class, from the untouched seed
    assert again == prod
    runner.real_loader(loader, [[3, 4]])
    assert list(iter(sampler)) == [[3, 4]]
    assert type(sampler).__mro__[1] is _Sampler  # no subclass of a subclass


def test_sampler_batches_does_not_advance_the_sampler():
    loader = _Loader()
    first, kind = runner.sampler_batches(loader, 3)
    again, _ = runner.sampler_batches(loader, 3)
    assert first == again and len(first) == 3 and kind.endswith("_Sampler")
    assert next(iter(loader.batch_sampler)) == first[0]  # the real epoch 1 starts with the same batch


def test_detect_older_layout(tmp_path):
    root = tmp_path / "old"
    (root / "scripts" / "phase4").mkdir(parents=True)
    (root / "scripts/phase4/train_stage2.py").write_text("from kunzdecoder import stage1, stage2  # noqa: E402\n")
    (root / "src" / "kunzdecoder").mkdir(parents=True)
    (root / "src/kunzdecoder/stage2.py").write_text("")
    for rel in ("external/bit/models/trainer.py", "external/bit/configs/finetune/sentence/ndt/trainer.yaml"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("")
    assert runner.detect_layout(root) == {"script": "scripts/phase4/train_stage2.py", "package": "kunzdecoder"}
    assert runner.check_code_root(root) == []
    (root / "results" / "phase4" / "stage2" / "golden_main_y").mkdir(parents=True)
    assert [p.name for p in runner.run_dirs(root, "golden_*")] == ["golden_main_y"]


def test_check_code_root_and_links(tmp_path):
    root = tmp_path / "code"
    for rel in ("scripts/train_stage2.py", "src/bit_experiments/stage2.py", "external/bit/models/trainer.py",
                "external/bit/configs/finetune/sentence/ndt/trainer.yaml"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("")
    (root / "scripts/train_stage2.py").write_text("import sys\nfrom bit_experiments import stage1, stage2  # noqa\n")
    assert runner.detect_layout(root) == {"script": "scripts/train_stage2.py", "package": "bit_experiments"}
    assert runner.check_code_root(root) == []
    (root / "results" / "stage2" / "B_ssl_s42").mkdir(parents=True)  # tracked results: fine
    assert runner.check_code_root(root) == []
    (root / "checkpoints" / "stage2" / "B_ssl_s42").mkdir(parents=True)  # a trained run: a project copy
    assert any("trained runs" in p for p in runner.check_code_root(root))
    (root / "checkpoints" / "stage2" / "B_ssl_s42").rmdir()
    (root / "checkpoints" / "stage2" / "golden_main_x").mkdir()
    assert runner.check_code_root(root) == []
    (root / "data" / "raw").mkdir(parents=True)
    assert any("data/raw" in p for p in runner.check_code_root(root))
    (root / "data" / "raw").rmdir()
    target = tmp_path / "hf"
    target.mkdir()
    assert runner.ensure_link(root / "checkpoints" / "hf", target) == "created"
    assert runner.ensure_link(root / "checkpoints" / "hf", target) == "existing"
    with pytest.raises(SystemExit):
        runner.ensure_link(root / "checkpoints" / "hf", tmp_path)
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "checkpoints").symlink_to(target)
    assert any("symlink" in p for p in runner.check_code_root(linked))

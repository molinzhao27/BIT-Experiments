"""CPU tests for the CUDA benchmark's state restoration, gates and isolation."""

import copy
import importlib.util
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("profile_training_runtime", ROOT / "scripts/profile_training_runtime.py")
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def test_comparison_distinguishes_exact_rounding_and_corruption():
    ref = {"x": torch.tensor([1.0, -0.0]), "count": torch.tensor(3)}
    rounding = {"x": torch.tensor([1.000001, 0.0]), "count": torch.tensor(3)}
    check = profile.compare_values(ref, rounding)
    assert check["close"] and not check["exact"] and check["max_abs"] > 0
    rounding["count"] += 1
    check = profile.compare_values(ref, rounding)
    assert not check["close"] and ".count" in check["failures"]
    assert not profile.compare_values(torch.tensor(float("nan")), torch.tensor(float("nan")))["close"]
    assert profile.compare_values(np.array(["sentence"]), np.array(["sentence"]))["exact"]


def test_state_round_trip_restores_training_and_all_rngs(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Dropout(0.3))
    model.register_buffer("saved", torch.tensor([4.0]))
    model[0].bias.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    trainer = NS(model=model, optimizer=optimizer, lr_scheduler=scheduler,
                 accelerator=NS(unwrap_model=lambda x: x, scaler=None))
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    scheduler.step()
    state = profile.capture_state(trainer)
    expected_draws = (torch.rand(4), random.random(), np.random.rand(4))
    optimizer.zero_grad()
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    scheduler.step()
    model.saved.fill_(99)
    model.eval()
    profile.restore_state(trainer, state)
    check = profile.compare_values(state, profile.capture_state(trainer), rtol=0, atol=0)
    assert check["exact"] and check["close"]
    actual_draws = (torch.rand(4), random.random(), np.random.rand(4))
    assert profile.compare_values(expected_draws, actual_draws)["exact"]


def test_run_gate_requires_same_counts_rng_and_small_updates():
    initial = {"parameters": {"p": torch.tensor([100.0])}}
    run = {"losses": [2.0], "counts": [2], "state": {
        "parameters": {"p": torch.tensor([100.01])}, "optimizer": {}, "scheduler": {},
        "buffers": {}, "scaler": None, "rng": {"torch": torch.tensor([1, 2], dtype=torch.uint8)}}}
    candidate = copy.deepcopy(run)
    assert profile.compare_runs(run, candidate, initial, 1e-4, 1e-7)["passed"]
    candidate["state"]["parameters"]["p"] = torch.tensor([100.0])
    assert not profile.compare_runs(run, candidate, initial, 1e-4, 1e-7)["passed"]
    candidate = copy.deepcopy(run)
    candidate["state"]["rng"]["torch"][0] = 3
    assert not profile.compare_runs(run, candidate, initial, 1e-4, 1e-7)["passed"]


@pytest.mark.parametrize("raises", [False, True])
def test_interception_restores_hooks_argv_cwd_and_only_own_dirs(tmp_path, raises):
    class Trainer:
        def train(self):
            return "original"

    old_train, old_argv, old_cwd = Trainer.train, sys.argv, Path.cwd()
    unrelated = tmp_path / "results/stage2/production"
    unrelated.mkdir(parents=True)
    (unrelated / "summary.json").write_text("keep")
    command = ["train_stage2.py", "--runtime", "reference", "--name", "runtime_profile_test"]
    def execute():
        with profile.intercepted_training(NS(), Trainer, command, lambda _: "profiled", root=tmp_path):
            assert Trainer().train() == "profiled" and sys.argv is command
            os.chdir(tmp_path)
            if raises:
                raise RuntimeError("inside")
    if raises:
        with pytest.raises(RuntimeError, match="inside"):
            execute()
    else:
        execute()
    assert Trainer.train is old_train and sys.argv is old_argv and Path.cwd() == old_cwd
    assert (unrelated / "summary.json").read_text() == "keep"
    assert not (tmp_path / "results/stage2/runtime_profile_test").exists()
    assert not (tmp_path / "checkpoints/stage2/runtime_profile_test").exists()


def test_interception_refuses_existing_directory_without_deleting_it(tmp_path):
    existing = tmp_path / "checkpoints/stage2/collision"
    existing.mkdir(parents=True)
    (existing / "model.bin").write_bytes(b"keep")
    class Trainer:
        def train(self):
            pass
    command = ["train_stage2.py", "--name", "collision"]
    with pytest.raises(FileExistsError):
        with profile.intercepted_training(NS(), Trainer, command, lambda _: None, root=tmp_path):
            pass
    assert (existing / "model.bin").read_bytes() == b"keep"
    assert not (tmp_path / "results/stage2/collision").exists()


def test_main_intercepts_reference_and_omits_default_llm(tmp_path, monkeypatch):
    class Trainer:
        def train(self):
            raise AssertionError("would have performed ordinary training")

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "train_stage2.py").write_text(
        "import sys\nfrom models.trainer import Trainer\n"
        "def main():\n"
        "    assert sys.argv[sys.argv.index('--runtime')+1] == 'reference'\n"
        "    assert '--llm' not in sys.argv\n"
        "    Trainer().train()\n"
        "    raise AssertionError('would save checkpoints')\n")
    monkeypatch.setattr(profile, "ROOT", tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setitem(sys.modules, "models.trainer", NS(Trainer=Trainer))
    def fake_profile(trainer, args):
        assert args.selective_ar_head
        raise profile.ProfileDone({"passed": True, "speedup": 1.1, "modes": {}})
    monkeypatch.setattr(profile, "profile_train", fake_profile)
    original = Trainer.train
    report = tmp_path / "report.json"
    assert profile.main(["--llm", "default", "--selective-ar-head", "--output", str(report)]) == 0
    assert report.exists() and Trainer.train is original
    assert not list((tmp_path / "results/stage2").iterdir())
    assert not list((tmp_path / "checkpoints/stage2").iterdir())


@pytest.mark.parametrize("option", ["--rtol", "--atol"])
def test_nonfinite_tolerances_rejected_before_gpu_initialization(option, monkeypatch):
    def unexpected():
        raise AssertionError("should validate tolerance first")
    monkeypatch.setattr(torch.cuda, "is_available", unexpected)
    with pytest.raises(SystemExit) as caught:
        profile.main([option, "nan"])
    assert caught.value.code == 2

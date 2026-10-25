#!/usr/bin/env python
"""BIT golden harness runner (PerfLab CONTRACT.md): a short, seeded slice of a real Stage 2 job, recorded for compare.py.

    <bit env>/python -u run.py --code-root <BIT copy> --case b_ar --shapes small --out <new dir> --machine <label> \
        --init <init ckpt dir> --best <trained BEST dir> --hf-dir <dir holding Aero-1-Audio> [--ablations-dir <dir>]

The training code is imported from --code-root (its scripts/train_stage2.py, src/ and external/bit), never from here,
so a candidate branch cannot change the harness. Nothing in the code root's recipe is changed: the job is the code
root's own train_stage2.main() with the case's real arguments (reference runtime, bf16 autocast, BIT's optimizer,
scheduler, dropout and RNG). The harness takes over at the point where main() calls trainer.train():

1. per shape, a fixed list of real train batches (recorded by trial index and content hash):
   - small: the 8 x N shortest train trials (stable order), shuffled into batches with a fixed seed; meant to fit
     a 32 GB GPU with the reference runtime;
   - prod: the first N batches of the production sampler's first epoch (BIT's GroupBatchSampler, seed 42);
2. N optimizer steps through the code root's own Trainer.train (one epoch over the fixed batches), saved by
   Trainer.train's own checkpoint code after step k (LAST = BIT checkpoint + accelerate state), then the remaining
   steps, uninterrupted, in the same process;
3. a resume in a FRESH process: a new model, optimizer and scheduler built by main(), accelerate load_state of the
   step-k checkpoint, the remaining steps; compared with the uninterrupted run inside the job;
4. a teacher-forced eval loss (BIT's Trainer.evaluate) on a few val batches, after training;
5. with --best: the trained checkpoint loaded with stage2.load_model_state, the eval again, and a greedy decode of
   seeded val trials with stage2.generate (the run's own decode settings), recording text, token ids and the
   top-1 / top-2 logit margin of every generated token (block diffusion: of every sampler decision, via its trace).

Recorded: per-step loss, supervised count, learning rate, gradient norm and batch shapes; every trainable gradient of
step 1; parameter updates and optimizer state after N steps; RNG states after steps k and N; digests; eval losses and
argmaxes; decodes; peak CUDA memory and timings; versions, GPU, code and harness hashes (manifest.json).
A later shape in the same job restarts from the captured initial state (parameters, buffers, optimizer, scheduler,
module modes, RNG), as scripts/profile_training_runtime.py's replays do; the first shape starts from the fresh build.

Writes only under --out (and two temporary run dirs, results/stage2/golden_* and checkpoints/stage2/golden_*, that
main() creates in the code root and the harness removes). Refuses a code root whose checkpoints/ or results/ is a
symlink or holds other runs (a project copy). Large temporary checkpoints go to <out>/tmp and are deleted at the end.
Exit 0: every phase ran; 1: a phase failed (its error is in manifest.json).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib
import json
import os
import re
import shlex
import platform
import random
import resource
import shutil
import socket
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))  # after everything else: never shadows the code root's modules
import golden_lib as gl  # noqa: E402

SEED = 42  # every case's run seed (the real runs')
# family: runs of cases in one family are comparable (same run, init, batches, eval and decode); a variant's
# baseline is the family's plain case.
CASES = {
    "b_ar": {"family": "b_ar", "run": "B_ssl_s42", "arm": "B", "llm": None, "ablation": None, "decode": True,
             "about": "Aero-1-Audio AR, arm B, SSL-core init (B_ssl_s42); decode from its BEST"},
    "b_diff": {"family": "b_diff", "run": "B_ssl_aero-1-audio-diffusion_s42", "arm": "B",
               "llm": "aero-1-audio-diffusion", "ablation": None, "decode": True,
               "about": "Aero block diffusion, plain step, arm B (B_ssl_aero-1-audio-diffusion_s42); decode from its "
                        "BEST with its sampler"},
    "b_diff_fast": {"family": "b_diff", "run": "B_ssl_aero-1-audio-diffusion_s42 (packed fast step)", "arm": "B",
                    "llm": "aero-1-audio-diffusion-fast", "ablation": None, "decode": True,
                    "about": "b_diff with the packed fast training step (--llm aero-1-audio-diffusion-fast): a "
                             "candidate variant, compared against b_diff's plain baseline"},
    "c_ar": {"family": "c_ar", "run": "C_ssl_l14_b1024_s42", "arm": "C", "llm": None, "ablation": None,
             "decode": False,
             "about": "arm C (Stage 1 contrastive init: encoder + projector), Aero AR; no trained checkpoint yet"},
    "eraser_ar": {"family": "eraser_ar", "run": "B_ssl_phonerased_s42", "arm": "B", "llm": None,
                  "ablation": "phoneme-erasure", "decode": True,
                  "about": "frozen B_ssl_s42 encoder + LEACE phoneme eraser, Aero AR; decode: B_ssl_s42 BEST with the "
                           "eraser inserted at decode time (the eraser prep's check b)"},
}
TRAIN_SCRIPTS = ("scripts/train_stage2.py", "scripts/phase4/train_stage2.py")  # current layout, older layout
INFO_RNG = ("rng_python", "rng_numpy")  # BIT seeds torch only: Python's and numpy's global RNGs are per-process
SHAPES = ("small", "prod")
CODE_GLOBS = ("scripts/train_stage2.py", "scripts/phase4/train_stage2.py", "src/*/**/*.py", "external/bit/**/*.py",
              "external/bit/configs/**/*.yaml", "configs/*.yaml", "configs/phase4/*.yaml")
RUNNER_FILES = ("run.py", "golden_lib.py")  # what determines a run's outputs: their hash is the harness identity
HARNESS_FILES = RUNNER_FILES + ("compare.py", "noise_floor.py", "README.md", "HARNESS_SOURCE.json")
GIB = 2 ** 30


class HarnessDone(Exception):
    """Raised from inside main()'s trainer.train() once the harness has recorded everything."""


# ================================================================ parent (no torch, no CUDA)

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code-root", required=True, type=Path, help="BIT code copy under test")
    ap.add_argument("--case", required=True, choices=sorted(CASES))
    ap.add_argument("--shapes", nargs="+", default=["small"], choices=SHAPES, help="in this order, one job")
    ap.add_argument("--out", required=True, type=Path, help="new (or empty) output directory")
    ap.add_argument("--machine", required=True, help="machine label for the manifest (e.g. the gq machine name)")
    ap.add_argument("--init", required=True, type=Path, help="absolute: the case's --init checkpoint dir")
    ap.add_argument("--best", type=Path, help="absolute: the trained checkpoint to evaluate and decode")
    ap.add_argument("--hf-dir", required=True, type=Path, help="absolute: the dir holding Aero-1-Audio (checkpoints/hf)")
    ap.add_argument("--ablations-dir", type=Path, help="absolute: the ablation artifacts dir (checkpoints/ablations)")
    ap.add_argument("--steps", type=int, default=8, help="optimizer steps per shape")
    ap.add_argument("--save-at", type=int, default=4, help="checkpoint after this step; resume from it")
    ap.add_argument("--small-batch", type=int, default=8, help="batch size of the small shape (prod: the config's)")
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--decode-trials", type=int, default=32)
    ap.add_argument("--harness-seed", type=int, default=0, help="seed of the small-shape order and decode trials")
    ap.add_argument("--train-args", default="", help="extra train_stage2.py flags (a candidate's), appended to the "
                    "case's arguments and recorded, e.g. \"--runtime optimized\"")
    ap.add_argument("--train-script", help="the code root's train_stage2.py, relative (default: found by layout)")
    ap.add_argument("--loader", choices=("real", "fresh"), default="real",
                    help="real (default): batches come through the trainer's own prepared loaders, their batch "
                         "sampler set to the selected batches; fresh: new loaders copying their settings (debugging)")
    ap.add_argument("--no-resume", action="store_true", help="debugging only: skip the fresh-process resume")
    ap.add_argument("--keep-tmp", action="store_true", help="debugging only: keep the temporary checkpoints")
    ap.add_argument("--phase", choices=("main", "resume"), help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    case = CASES[args.case]
    if not 1 <= args.save_at < args.steps:
        ap.error("need 1 <= --save-at < --steps")
    for name in ("init", "best", "hf_dir", "ablations_dir"):
        p = getattr(args, name)
        if p is not None and not p.is_absolute():
            ap.error(f"--{name.replace('_', '-')} must be an absolute path")
    if case["decode"] and args.best is None:
        ap.error(f"case {args.case} decodes a trained checkpoint: give --best")
    if case["ablation"] and args.ablations_dir is None:
        ap.error(f"case {args.case} needs --ablations-dir")
    if len(set(args.shapes)) != len(args.shapes):
        ap.error("--shapes: each shape once")
    args.out = args.out.absolute()  # main() changes the working directory
    args.code_root = args.code_root.absolute()
    return args


def detect_layout(root: Path, script: str | None = None) -> dict:
    """The code root's train_stage2.py and the package it imports stage2 from (bit_experiments, or the older
    layout's scripts/phase4 + kunzdecoder)."""
    scripts = [script] if script else [s for s in TRAIN_SCRIPTS if (root / s).is_file()]
    if not scripts or not (root / scripts[0]).is_file():
        return {"script": None, "package": None}
    m = re.search(r"^from (\w+) import [^\n]*\bstage2\b", (root / scripts[0]).read_text(), re.M)
    return {"script": scripts[0], "package": m.group(1) if m else None}


def check_code_root(root: Path, script: str | None = None) -> list[str]:
    problems = []
    lay = detect_layout(root, script)
    if lay["script"] is None:
        problems.append(f"no train_stage2.py (looked for {script or ', '.join(TRAIN_SCRIPTS)})")
    elif lay["package"] is None or not (root / "src" / lay["package"] / "stage2.py").is_file():
        problems.append(f"{lay['script']}: its stage2 package is not under src/")
    for rel in ("external/bit/models/trainer.py", "external/bit/configs/finetune/sentence/ndt/trainer.yaml"):
        if not (root / rel).is_file():
            problems.append(f"missing {rel}")
    if (root / "external" / "bit").is_symlink():
        problems.append("external/bit is a symlink: copy the tree it points to (BIT's patched YAMLs included)")
    # A project copy links or holds checkpoints and data; a clone of the repo holds neither (results/ is tracked).
    for rel in ("checkpoints", "results", "data", "checkpoints/stage2", "results/stage2", "results/phase4",
                "results/phase4/stage2"):
        if (root / rel).is_symlink():
            problems.append(f"{rel} is a symlink (a project copy?): main() writes temporary run dirs there")
    d = root / "checkpoints" / "stage2"
    if d.is_dir() and not d.is_symlink():
        other = [p.name for p in d.iterdir() if not p.name.startswith("golden_")]
        if other:
            problems.append(f"checkpoints/stage2 holds trained runs ({', '.join(sorted(other)[:3])}): a project copy?")
    if (root / "data" / "raw").exists():
        problems.append("data/raw exists: a project copy? (the harness reads data by the configs' absolute paths)")
    return problems


def run_dirs(root: Path, pattern: str) -> list:
    """Run directories main() creates: results/[phase4/]stage2/<name> and checkpoints/stage2/<name>."""
    out = []
    for rel in ("results/stage2", "results/phase4/stage2", "checkpoints/stage2"):
        d = root / rel
        if d.is_dir() and not d.is_symlink():
            out += [p for p in d.glob(pattern) if p.is_dir()]
    return out


def ensure_link(link: Path, target: Path) -> str:
    """code root path -> absolute target (read-only inputs that BIT resolves relative to the code root)."""
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise SystemExit(f"{link} points to {os.readlink(link)}, not {target}")
        return "existing"
    if link.exists():
        raise SystemExit(f"{link} exists and is not a symlink to {target}")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)
    return "created"


def file_hashes(root: Path, globs) -> dict:
    files = sorted({p for g in globs for p in root.glob(g) if p.is_file() and "__pycache__" not in p.parts})
    return {str(p.relative_to(root)): gl.sha256_file(p) for p in files}


def combined(hashes: dict) -> str:
    return gl.sha256_bytes("".join(f"{k}\0{v}\n" for k, v in sorted(hashes.items())).encode())


def git_commit(path: Path):
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain", "--", "."], capture_output=True,
                               text=True, timeout=10)
        if out.returncode == 0:
            return {"commit": out.stdout.strip(), "dirty": bool(dirty.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def harness_info() -> dict:
    files = {f: gl.sha256_file(HERE / f) for f in HARNESS_FILES if (HERE / f).is_file()}
    info = {"files_sha256": files, "combined_sha256": combined({f: files[f] for f in RUNNER_FILES}),
            "combined_of": list(RUNNER_FILES), "git": git_commit(HERE)}
    src = HERE / "HARNESS_SOURCE.json"
    if src.is_file():
        info["source"] = gl.read_json(src)
    return info


def parent(args) -> int:
    t0 = time.time()
    root = args.code_root.resolve()
    out = args.out
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"--out {out} exists and is not empty")
    problems = check_code_root(root, args.train_script)
    if problems:
        raise SystemExit("refusing code root " + str(root) + ":\n  " + "\n  ".join(problems))
    for d in run_dirs(root, "golden_*"):  # leftovers of a killed harness job (the GPU queue is serial)
        shutil.rmtree(d, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    links = {"checkpoints/hf": ensure_link(root / "checkpoints" / "hf", args.hf_dir)}
    if args.ablations_dir is not None:
        links["checkpoints/ablations"] = ensure_link(root / "checkpoints" / "ablations", args.ablations_dir)
    code = file_hashes(root, CODE_GLOBS)
    manifest = {"harness": harness_info(), "status": "running", "case": args.case, "case_config": CASES[args.case],
                "shapes": args.shapes, "machine": args.machine, "host": socket.gethostname(),
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
                "argv": sys.argv, "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "train_args": shlex.split(args.train_args),
                "code_root": {"path": str(root), "git": git_commit(root), "training_files_sha256": combined(code),
                              "layout": detect_layout(root, args.train_script), "files": code, "links": links,
                              "link_targets": {k: str((root / k).resolve()) for k in links}},
                "inputs": {k: str(getattr(args, k).resolve()) for k in ("init", "best", "hf_dir", "ablations_dir")
                           if getattr(args, k) is not None},
                "phases": {}}
    gl.write_json(out / "manifest.json", manifest)
    rc = 0
    phases = ["main"] + ([] if args.no_resume else ["resume"])
    for phase in phases:
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), *sys.argv[1:], "--phase", phase]
        t = time.time()
        print(f"[golden] phase {phase}: start", flush=True)
        proc = subprocess.run(cmd)
        info = {"rc": proc.returncode, "seconds": round(time.time() - t, 1)}
        pj = out / phase / "phase.json"
        if pj.is_file():
            info["status"] = gl.read_json(pj).get("status")
        manifest["phases"][phase] = info
        gl.write_json(out / "manifest.json", manifest)
        print(f"[golden] phase {phase}: rc {proc.returncode} in {info['seconds']} s", flush=True)
        if proc.returncode != 0:
            rc = 1
            break
    tmp = out / "tmp"
    if tmp.exists():
        manifest["tmp_listing"] = sorted(f"{p.stat().st_size} {p.relative_to(out)}" for p in tmp.rglob("*") if p.is_file())
        if not args.keep_tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    manifest.update(status="ok" if rc == 0 else "failed", seconds=round(time.time() - t0, 1),
                    ended_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    gl.write_json(out / "manifest.json", manifest)
    print(f"[golden] {args.case} {' '.join(args.shapes)}: {manifest['status']} in {manifest['seconds']} s -> {out}",
          flush=True)
    return rc


# ================================================================ child: shared machinery

def env_info() -> dict:
    import accelerate
    import numpy as np
    import peft
    import torch
    import transformers

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        smi = None
    return {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "transformers": transformers.__version__,
            "accelerate": accelerate.__version__, "peft": peft.__version__, "numpy": np.__version__,
            "gpu": props.name, "gpu_capability": [props.major, props.minor], "gpu_sm_count": props.multi_processor_count,
            "gpu_memory_gib": round(props.total_memory / GIB, 2), "nvidia_smi": smi,
            "flags": {"cudnn_deterministic": torch.backends.cudnn.deterministic,
                      "cudnn_benchmark": torch.backends.cudnn.benchmark,
                      "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                      "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                      "float32_matmul_precision": torch.get_float32_matmul_precision(),
                      "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
            "env": {k: os.environ.get(k) for k in ("CC", "CUDA_HOME", "DS_SKIP_CUDA_CHECK", "HF_HUB_OFFLINE",
                                                   "WANDB_MODE", "CUBLAS_WORKSPACE_CONFIG", "PYTORCH_CUDA_ALLOC_CONF",
                                                   "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")}}


def rng_state():
    import numpy as np
    import torch

    return {"torch": torch.get_rng_state().clone(), "cuda": [s.clone() for s in torch.cuda.get_rng_state_all()],
            "python": random.getstate(), "numpy": np.random.get_state()}


def restore_rng(state) -> None:
    import numpy as np
    import torch

    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])


def seed_all(seed: int) -> None:
    import numpy as np
    import torch

    torch.manual_seed(seed)  # CPU and every CUDA device
    random.seed(seed)
    np.random.seed(seed)


def rng_fingerprint() -> dict:
    """Whether this GPU draws the same CUDA random streams as another: dropout masks and uniforms of a few sizes,
    from a fixed seed. Large draws depend on the kernel's grid (the GPU's SM count). The global RNG is restored."""
    import torch
    import torch.nn.functional as F

    saved = rng_state()
    out = {}
    try:
        for n in (1_000, 100_000, 4_000_000):
            torch.manual_seed(1234)
            x = torch.ones(n, device="cuda")
            out[f"dropout_p0.2_n{n}"] = gl.tensor_digest(F.dropout(x, 0.2, training=True).cpu())
            torch.manual_seed(1234)
            out[f"rand_n{n}"] = gl.tensor_digest(torch.rand(n, device="cuda").cpu())
    finally:
        restore_rng(saved)
    return out


def map_tensors(value, fn):
    import torch

    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, dict):
        return {k: map_tensors(v, fn) for k, v in value.items()}
    if isinstance(value, list):
        return [map_tensors(v, fn) for v in value]
    if isinstance(value, tuple):
        return tuple(map_tensors(v, fn) for v in value)
    return copy.deepcopy(value)


def cpu_clone(t):
    return t.detach().cpu().clone()


def load_train_module(code_root: Path, layout: dict):
    import importlib.util

    spec = importlib.util.spec_from_file_location("golden_train_stage2", code_root / layout["script"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # its own sys.path setup: code_root/src and code_root/external/bit first
    pkg = importlib.import_module(layout["package"])
    import models.trainer

    paths = {layout["package"]: str(Path(pkg.__file__).resolve()),
             "models.trainer": str(Path(models.trainer.__file__).resolve()),
             "train_stage2": str(Path(module.__file__).resolve())}
    for key, p in paths.items():
        if not p.startswith(str(code_root.resolve()) + os.sep):
            raise RuntimeError(f"{key} imported from {p}, outside the code root {code_root}")
    return module, models.trainer.Trainer, paths


@contextlib.contextmanager
def intercepted(trainer_cls, callback, code_root: Path, name: str, argv: list):
    """Run the code root's main() with `argv`; when main() calls trainer.train(), call callback(trainer) instead
    (an instance attribute set right after Trainer.__init__, so it works whatever the runtime's train method).
    Restores the class, argv and cwd, and removes the run dirs main() created."""
    old_init, old_argv, old_cwd = trainer_cls.__init__, sys.argv, os.getcwd()

    def init(self, *a, **k):
        old_init(self, *a, **k)
        self.train = lambda: callback(self)

    try:
        trainer_cls.__init__ = init
        sys.argv = argv
        yield
    finally:
        trainer_cls.__init__ = old_init
        sys.argv = old_argv
        os.chdir(old_cwd)
        for d in run_dirs(code_root, name):
            shutil.rmtree(d, ignore_errors=True)


def main_locals(need=("collate", "splits", "tokenizer", "refs", "seen", "out")) -> dict:
    """The locals of the code root's main() at its trainer.train() call: its generation collate, val/test datasets,
    tokenizer, references and seen masks (what main() itself decodes with)."""
    f = sys._getframe(1)
    while f is not None:
        if f.f_code.co_name == "main" and all(k in f.f_locals for k in need):
            return {k: f.f_locals[k] for k in need}
        f = f.f_back
    raise RuntimeError(f"main()'s locals {need} not found above trainer.train()")


def case_argv(args, name: str) -> list:
    case = CASES[args.case]
    argv = ["train_stage2.py", "--name", name, "--arm", case["arm"], "--init", str(args.init), "--seed", str(SEED)]
    if case["llm"]:
        argv += ["--llm", case["llm"]]
    if case["ablation"]:
        argv += ["--ablation", case["ablation"]]
    return argv + shlex.split(args.train_args)


class FixedBatches:
    """A batch sampler over a fixed list of index batches (torch DataLoader batch_sampler protocol)."""

    def __init__(self, batches):
        self.batches = [[int(i) for i in b] for b in batches]
        self.batch_size = max(len(b) for b in self.batches)  # accelerate's split_batches check reads it

    def __iter__(self):
        return iter([list(b) for b in self.batches])

    def __len__(self):
        return len(self.batches)


_GOLDEN_SAMPLERS = {}


def golden_sampler_class(cls):
    """A subclass of the loader's batch-sampler class that yields the instance's _golden_batches, in order."""
    sub = _GOLDEN_SAMPLERS.get(cls)
    if sub is None:
        def __iter__(self):
            return iter([list(b) for b in self._golden_batches])

        def __len__(self):
            return len(self._golden_batches)

        sub = _GOLDEN_SAMPLERS[cls] = type(f"Golden{cls.__name__}", (cls,), {"__iter__": __iter__, "__len__": __len__})
    return sub


def real_loader(ref, batches):
    """The trainer's own prepared loader (BIT's DataLoader after accelerate's prepare: its dataset, collate_fn,
    workers, pinning, wrapper and device placement), with its batch sampler instance switched to yield `batches`.
    The sampler's own state (its seeded generator) is left untouched."""
    bs = ref.batch_sampler
    inner = getattr(bs, "batch_sampler", bs)
    if inner is not bs:
        raise RuntimeError(f"sharded batch sampler {type(bs).__name__}: one process only")
    if not hasattr(inner, "_golden_class"):
        inner._golden_class = type(inner)
    inner.__class__ = golden_sampler_class(inner._golden_class)
    inner._golden_batches = [[int(i) for i in b] for b in batches]
    return ref


def batches_loader(trainer, ref, batches, mode: str):
    return real_loader(ref, batches) if mode == "real" else fixed_loader(trainer, ref, batches)


def expected_digests(ref, batches) -> list:
    """Digests of what the loader's collate_fn makes of each selected batch: what the forward must receive."""
    out = []
    for b in batches:
        inputs = ref.collate_fn([ref.dataset[i] for i in b])
        inputs = inputs[0] if isinstance(inputs, tuple) else inputs
        out.append(input_digest(inputs))
    return out


def input_digest(inputs: dict) -> str:
    import torch

    return gl.nested_digest({k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)})


def fixed_loader(trainer, ref, batches):
    """A loader like the trainer's prepared `ref` (same dataset, collate, pinning, workers), over fixed batches,
    prepared by the trainer's accelerator as Trainer.prepare_for_distributed_training prepares BIT's."""
    from torch.utils.data import DataLoader

    kw = {k: getattr(ref, k) for k in ("num_workers", "collate_fn", "pin_memory", "timeout", "worker_init_fn",
                                         "multiprocessing_context", "generator", "persistent_workers",
                                         "pin_memory_device") if hasattr(ref, k)}
    if kw.get("num_workers", 0) > 0:
        kw["prefetch_factor"] = ref.prefetch_factor
    else:
        kw.pop("persistent_workers", None)
    return trainer.accelerator.prepare(DataLoader(ref.dataset, batch_sampler=FixedBatches(batches), **kw))


def sampler_batches(loader, n: int) -> tuple[list, str]:
    """The first n batches the loader's own batch sampler would give in its next epoch, without advancing it."""
    bs = loader.batch_sampler
    inner = getattr(bs, "batch_sampler", bs)
    s = copy.copy(inner)
    if hasattr(inner, "_golden_class"):  # a sampler run_loader switched: copy its original behaviour
        s.__class__ = inner._golden_class
    if hasattr(inner, "rng"):
        s.rng = copy.deepcopy(inner.rng)
    out = []
    for b in s:
        out.append([int(i) for i in b])
        if len(out) == n:
            break
    return out, f"{type(inner).__module__}.{type(inner).__name__}"


def trial_lengths(ds):
    import numpy as np

    key = getattr(ds, "spikes_name", "spikes")
    return np.array([ds.dataset[i][key].shape[0] for i in range(len(ds))])


def shortest_batches(ds, n_batches: int, batch: int, seed: int) -> list:
    import numpy as np

    lengths = trial_lengths(ds)
    order = np.lexsort((np.arange(len(lengths)), lengths))[: n_batches * batch]
    order = order[np.random.default_rng(seed).permutation(len(order))]
    return [[int(i) for i in order[j * batch:(j + 1) * batch]] for j in range(n_batches)]


def batch_info(ds, batches) -> list:
    lengths = trial_lengths(ds)
    return [{"indices": b, "spike_bins": [int(lengths[i]) for i in b],
             "data_sha256": gl.nested_digest([ds.dataset[i] for i in b])} for b in batches]


class Recorder:
    """Forward and optimizer hooks that only read: per-step loss, supervised count, batch shapes, learning rate,
    gradient norm, the step-1 gradients; per eval batch the loss, count and argmax / margin at supervised positions."""

    def __init__(self, model, optimizer, names):
        self.names, self.window, self.eval = names, None, None
        self.trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
        self.handles = [model.register_forward_pre_hook(self.pre, with_kwargs=True),
                        model.register_forward_hook(self.post),
                        optimizer.register_step_pre_hook(self.step)]

    def open_window(self, want_grads: bool):
        self.window = {"loss": [], "n": [], "shapes": [], "inputs": [], "lr": [], "betas": [], "grad_norm": [],
                       "t_step": [], "want_grads": want_grads, "grads_step1": None, "grads_none": None}

    def close_window(self):
        w, self.window = self.window, None
        w["loss"] = [float(x) for x in w["loss"]]
        w["n"] = [int(x) for x in w["n"]]
        return w

    def pre(self, module, args, kwargs):
        import torch

        if self.window is not None and module.training:
            self.window["shapes"].append({k: list(v.shape) for k, v in kwargs.items()
                                          if isinstance(v, torch.Tensor) and k in ("spikes", "input_ids")})
            self.window["inputs"].append(input_digest(kwargs))
        elif self.eval is not None and not module.training:
            self._eval_input = input_digest(kwargs)

    def post(self, module, args, output):
        import torch

        if self.window is not None and module.training:
            self.window["loss"].append(output.loss.detach().float().item())
            self.window["n"].append(int(output.n_examples))
        elif self.eval is not None and not module.training:
            row = {"loss": output.loss.detach().float().item(), "n": int(output.n_examples),
                   "inputs": getattr(self, "_eval_input", None)}
            preds, targets = output.preds, output.targets
            if isinstance(preds, torch.Tensor) and isinstance(targets, torch.Tensor):
                if preds.dim() == 3:  # AR: logits [B, L, V] predict the next target
                    preds, targets = preds[:, :-1], targets[:, 1:]
                sup = targets != -100
                sel = preds[sup].float()
                if sel.numel():
                    top = sel.topk(2, dim=-1)
                    row.update(argmax=top.indices[:, 0].int().cpu().tolist(),
                               margin=(top.values[:, 0] - top.values[:, 1]).cpu().tolist(),
                               target=targets[sup].int().cpu().tolist())
                del sel
            self.eval.append(row)

    def step(self, optimizer, args, kwargs):
        w = self.window
        if w is None:
            return
        import torch

        grads = [(self.names.get(id(p), f"?{id(p)}"), p.grad) for g in optimizer.param_groups for p in g["params"]
                 if p.grad is not None]
        norms = torch.stack([torch.linalg.vector_norm(g.detach(), dtype=torch.float64) for _, g in grads])
        w["grad_norm"].append(float(torch.linalg.vector_norm(norms)))
        w["lr"].append([float(g["lr"]) for g in optimizer.param_groups])
        w["betas"].append([list(g["betas"]) if "betas" in g else None for g in optimizer.param_groups])
        w["t_step"].append(time.perf_counter())
        if w["want_grads"] and w["grads_step1"] is None:
            w["grads_step1"] = {n: cpu_clone(g) for n, g in grads}
            have = {n for n, _ in grads}
            w["grads_none"] = sorted(self.trainable_names - have)

    def remove(self):
        for h in self.handles:
            h.remove()


def trainable(model) -> dict:
    return {n: cpu_clone(p) for n, p in model.named_parameters() if p.requires_grad}


def optimizer_by_name(trainer, names) -> tuple[dict, list]:
    opt = getattr(trainer.optimizer, "optimizer", trainer.optimizer)
    sd = opt.state_dict()
    idx = {}
    for live, saved in zip(opt.param_groups, sd["param_groups"]):
        for p, i in zip(live["params"], saved["params"]):
            idx[i] = names.get(id(p), f"param{i}")
    state = {idx[i]: {k: cpu_clone(v) if hasattr(v, "detach") else v for k, v in s.items()} for i, s in sd["state"].items()}
    groups = [{k: v for k, v in g.items() if k != "params"} for g in sd["param_groups"]]
    return state, groups


def full_state(trainer, model) -> dict:
    """Everything a fresh build has before training (for a later shape's restart), as profile_training_runtime."""
    return {"params": trainable(model), "buffers": {n: cpu_clone(b) for n, b in model.named_buffers()},
            "optimizer": map_tensors(trainer.optimizer.state_dict(), cpu_clone),
            "scheduler": copy.deepcopy(trainer.lr_scheduler.state_dict()),
            "training": {n: m.training for n, m in model.named_modules()}, "rng": rng_state()}


def restore_full(trainer, model, st) -> None:
    import torch

    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in st["params"]:
                p.copy_(st["params"][n])
                p.grad = None
        for n, b in model.named_buffers():
            b.copy_(st["buffers"][n])
    trainer.optimizer.load_state_dict(copy.deepcopy(st["optimizer"]))
    trainer.lr_scheduler.load_state_dict(copy.deepcopy(st["scheduler"]))
    for n, m in model.named_modules():
        m.training = st["training"][n]
    restore_rng(st["rng"])


def rng_digests(state=None) -> dict:
    """Per generator: torch CPU, CUDA, Python, numpy (the last two are not seeded by BIT: per-process)."""
    state = rng_state() if state is None else state
    return {f"rng_{k}": gl.nested_digest(v) for k, v in state.items()}


def digests(trainer, model, names) -> dict:
    opt_state, groups = optimizer_by_name(trainer, names)
    return {"trainable": gl.nested_digest(trainable(model)), "optimizer_state": gl.nested_digest(opt_state),
            "optimizer_groups": gl.nested_digest(groups), "scheduler": gl.nested_digest(trainer.lr_scheduler.state_dict()),
            "buffers": gl.nested_digest({n: b for n, b in model.named_buffers()}), **rng_digests()}


def call_train(trainer, batches, ckpt_dir: Path) -> float:
    """One epoch of the code root's own train() over `batches`; its checkpoint save at the end goes to ckpt_dir/LAST."""
    trainer.train_dataloader = batches_loader(trainer, trainer._golden_ref_train, batches, trainer._golden_loader)
    trainer.config["training"]["num_epochs"] = 1  # one pass over the fixed batches; no eval (eval_every 5)
    trainer.start_epoch = 1
    trainer.checkpoint_dir = str(ckpt_dir)
    t = time.perf_counter()
    type(trainer).train(trainer)
    return time.perf_counter() - t


def listing(d: Path) -> list:
    return sorted(f"{p.stat().st_size} {p.relative_to(d)}" for p in d.rglob("*") if p.is_file()) if d.exists() else []


def memory() -> dict:
    import torch

    return {"max_allocated_gib": round(torch.cuda.max_memory_allocated() / GIB, 3),
            "max_reserved_gib": round(torch.cuda.max_memory_reserved() / GIB, 3),
            "host_max_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2)}


def segment(ctx, name: str) -> dict:
    """Peak CUDA memory since the last segment (recorded under `name`), folded into the phase maximum; then reset."""
    import torch

    m = memory()
    ctx.rec["memory"][name] = m
    top = ctx.rec["memory"].setdefault("max", {k: 0.0 for k in m})
    for k, v in m.items():
        top[k] = max(top[k], v)
    torch.cuda.reset_peak_memory_stats()
    return m


def run_eval(trainer, recorder, batches, seed: int) -> dict:
    """BIT's teacher-forced evaluation (the base Trainer.evaluate of the code root) over fixed val batches."""
    from models.trainer import Trainer as BaseTrainer

    trainer.val_dataloader = batches_loader(trainer, trainer._golden_ref_val, batches, trainer._golden_loader)
    rng = {"before_seed": rng_digests()}
    seed_all(seed)  # the diffusion objective draws its masks from the global CUDA RNG in eval too
    rng["after_seed"] = rng_digests()
    recorder.eval = []
    t = time.perf_counter()
    try:
        loss, metrics, gt, preds = BaseTrainer.evaluate(trainer)
        del gt, preds
    finally:
        rows, recorder.eval = recorder.eval, None
    rng["after_eval"] = rng_digests()
    return {"avg_loss": float(loss), "metrics": metrics, "batches": rows, "seconds": round(time.perf_counter() - t, 2),
            "seed": seed, "rng": rng}


class Subset:
    def __init__(self, ds, idx):
        self.ds, self.idx = ds, list(idx)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return self.ds[self.idx[i]]


@contextlib.contextmanager
def decode_capture(model):
    """Reads what each decode returns and the logits behind its decisions; changes nothing decoded.
    Every model: the ids model.generate returns. AR: the generation scores (HF output_scores; the processed logits
    greedy takes the argmax of). Block diffusion: the sampler's own trace plus every lm_head output."""
    import torch

    cap = {"ids": [], "ar": [], "calls": [], "diffusion": hasattr(model, "_denoise")}
    orig = model.generate

    def generate(*a, **k):
        ids = orig(*a, **k)
        cap["ids"].append(ids[0].tolist())
        return ids

    model.generate = generate
    handle, llm_patched = None, False
    if cap["diffusion"]:
        model.trace = []
        head = model.llm.get_base_model().lm_head
        mask_id = int(model.mask_id)

        def hook(module, args, output):
            lg = output[0].detach().to(torch.float32, copy=True)
            lg[:, mask_id] = -torch.inf  # as the sampler does before proposing
            top = lg.topk(2, dim=-1)
            cap["calls"].append({"trial": len(model.trace) - 1, "L": int(lg.shape[0]),
                                 "top1": top.indices[:, 0].tolist(),
                                 "margin": (top.values[:, 0] - top.values[:, 1]).tolist()})

        handle = head.register_forward_hook(hook)
    else:
        llm_generate = model.llm.generate

        def scored(*a, **k):
            out = llm_generate(*a, **dict(k, output_scores=True, return_dict_in_generate=True))
            scores = torch.stack([s[0] for s in out.scores]).float() if out.scores else None
            row = {"n_scores": 0 if scores is None else int(scores.shape[0])}
            if scores is not None:
                top = scores.topk(2, dim=-1)
                row.update(top1=top.indices[:, 0].tolist(), top2=top.indices[:, 1].tolist(),
                           margins=(top.values[:, 0] - top.values[:, 1]).tolist())
            cap["ar"].append(row)
            return out.sequences

        model.llm.generate = scored
        llm_patched = True
    try:
        yield cap
    finally:
        del model.generate
        if handle is not None:
            handle.remove()
        if cap["diffusion"]:
            cap["trace"] = model.trace
            model.trace = None
        if llm_patched:
            del model.llm.generate


def run_decode(trainer, loc, trials, case_name, pkg: str) -> dict:
    import torch

    stage2 = importlib.import_module(f"{pkg}.stage2")
    model = trainer.accelerator.unwrap_model(trainer.model)
    ds = Subset(loc["splits"]["val"], trials)
    refs = [loc["refs"]["val"][i] for i in trials]
    seen = loc["seen"]["val"][trials]
    torch.cuda.synchronize()
    t = time.perf_counter()
    with decode_capture(model) as cap:
        hyps = stage2.generate(model, ds, loc["collate"], loc["tokenizer"], "greedy",
                               device=trainer.accelerator.device)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t
    rows, problems = [], []
    threshold = getattr(model, "sampler", {}).get("threshold") if cap["diffusion"] else None
    for i, (trial, ref, hyp) in enumerate(zip(trials, refs, hyps)):
        row = {"trial": int(trial), "reference": ref, "text": hyp,
               "ids": cap["ids"][i] if i < len(cap["ids"]) else None}
        if cap["diffusion"]:
            trace = cap["trace"][i] if i < len(cap.get("trace") or []) else None
            calls = [c for c in cap["calls"] if c["trial"] == i]
            if trace is not None:
                aligned = gl.align_diffusion_trace(trace, calls, int(model.block_size))
                row.update(events=aligned["events"], margins=aligned["margins"], commit_probs=aligned["commit_probs"],
                           trace_out=trace.get("out"))
                if aligned["problems"]:
                    problems.append({"trial": int(trial), "problems": aligned["problems"][:5]})
        else:
            ar = cap["ar"][i] if i < len(cap["ar"]) else {}
            row.update(margins=ar.get("margins"), top1=ar.get("top1"), top2=ar.get("top2"))
            if row["ids"] is not None and ar.get("top1") is not None:
                if len(ar["top1"]) != len(row["ids"]):
                    problems.append({"trial": int(trial), "problems": ["scores and ids differ in length"]})
                else:
                    bad = [j for j, (a, b, m) in enumerate(zip(ar["top1"], row["ids"], ar["margins"]))
                           if a != b and m != 0]
                    if bad:
                        problems.append({"trial": int(trial), "problems": [f"top-1 != chosen token at {bad[:5]}"]})
        rows.append(row)
    return {"case": case_name, "decoding": "greedy", "trials": [int(t) for t in trials], "rows": rows,
            "score": stage2.score(refs, hyps, seen), "seconds": round(seconds, 2), "threshold": threshold,
            "diffusion": cap["diffusion"], "sampler": getattr(model, "sampler", None) if cap["diffusion"] else None,
            "capture_problems": problems}


# ================================================================ child: phases

class Ctx:
    def __init__(self, args, phase):
        self.args, self.phase = args, phase
        self.case = CASES[args.case]
        self.out = args.out / phase
        self.tmp = args.out / "tmp"
        self.code_root = args.code_root.resolve()
        self.layout = detect_layout(self.code_root, args.train_script)
        self.t0 = time.time()
        self.rec = {"phase": phase, "status": "running", "case": args.case, "shapes": args.shapes,
                    "layout": self.layout, "train_args": shlex.split(args.train_args),
                    "steps": args.steps, "save_at": args.save_at, "seed": SEED, "harness_seed": args.harness_seed,
                    "timings": {}, "memory": {}}

    def save(self):
        gl.write_json(self.out / "phase.json", self.rec)


def setup_callback(ctx: Ctx, trainer):
    """Common start of both phases' callbacks: main()'s locals, the unwrapped model, names, hooks, references."""
    import torch

    ctx.rec["timings"]["build_seconds"] = round(time.time() - ctx.t0, 1)
    segment(ctx, "build")
    model = trainer.accelerator.unwrap_model(trainer.model)
    names = {id(p): n for n, p in model.named_parameters()}
    trainer._golden_ref_train, trainer._golden_ref_val = trainer.train_dataloader, trainer.val_dataloader
    trainer._golden_loader = ctx.args.loader
    ctx.rec["loader"] = {"mode": ctx.args.loader,
                         "train": f"{type(trainer.train_dataloader).__module__}.{type(trainer.train_dataloader).__name__}",
                         "train_batch_sampler": type(trainer.train_dataloader.batch_sampler).__name__,
                         "num_workers": getattr(trainer.train_dataloader, "num_workers", None),
                         "pin_memory": getattr(trainer.train_dataloader, "pin_memory", None)}
    run_json = Path(ctx.loc["out"]) / "run.json"
    if run_json.is_file():
        shutil.copy(run_json, ctx.out / "run.json")
    ctx.rec["model"] = {"class": f"{type(model).__module__}.{type(model).__name__}",
                        "n_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
                        "n_trainable_tensors": sum(1 for p in model.parameters() if p.requires_grad),
                        "n_parameters": sum(p.numel() for p in model.parameters()),
                        "trainable_dtypes": sorted({str(p.dtype) for p in model.parameters() if p.requires_grad}),
                        "mixed_precision": trainer.config.training.mixed_precision,
                        "runtime_report": getattr(trainer, "runtime_report", None),
                        "padding_report": getattr(trainer, "padding_report", None)}
    opt = getattr(trainer.optimizer, "optimizer", trainer.optimizer)
    recorder = Recorder(model, opt, names)
    return model, names, recorder


def select_shape(ctx, trainer, shape):
    a = ctx.args
    n = a.steps
    if shape == "prod":
        train, sampler = sampler_batches(trainer._golden_ref_train, n)
        val, val_sampler = sampler_batches(trainer._golden_ref_val, a.eval_batches)
        how = {"train": f"first {n} batches of the production sampler's first epoch ({sampler})",
               "val": f"first {a.eval_batches} batches of the production val sampler ({val_sampler})"}
    else:
        train = shortest_batches(trainer.train_dataset, n, a.small_batch, a.harness_seed)
        val = shortest_batches(trainer.val_dataset, a.eval_batches, a.small_batch, a.harness_seed)
        how = {"train": f"the {n * a.small_batch} shortest train trials, shuffled (seed {a.harness_seed}) into batches "
                        f"of {a.small_batch}",
               "val": f"the {a.eval_batches * a.small_batch} shortest val trials, shuffled (seed {a.harness_seed})"}
    if len(train) < n:
        raise RuntimeError(f"only {len(train)} train batches for {n} steps")
    return {"how": how, "train": batch_info(trainer.train_dataset, train), "val": batch_info(trainer.val_dataset, val)}


def main_callback(ctx: Ctx, trainer):
    import numpy as np
    import torch

    a = ctx.args
    loc = ctx.loc = main_locals()
    model, names, recorder = setup_callback(ctx, trainer)
    initial = full_state(trainer, model)
    ctx.rec["initial_digests"] = digests(trainer, model, names)
    ctx.rec["shape_results"] = {}
    ctx.save()
    for si, shape in enumerate(a.shapes):
        if si:
            restore_full(trainer, model, initial)
            if digests(trainer, model, names) != ctx.rec["initial_digests"]:
                raise RuntimeError("restoring the initial state did not reproduce its digests")
        sel = select_shape(ctx, trainer, shape)
        batches = [b["indices"] for b in sel["train"]]
        d = ctx.out / shape
        d.mkdir(parents=True, exist_ok=True)
        res = {"selection": sel}
        ctx.rec["shape_results"][shape] = res
        k = a.save_at
        # steps 1..k, then Trainer.train's own checkpoint save (LAST) = the resume point
        recorder.open_window(want_grads=True)
        res["train_seconds_1"] = round(call_train(trainer, batches[:k], ctx.tmp / shape / "ckpt_k"), 2)
        w1 = recorder.close_window()
        res["ckpt_k_listing"] = listing(ctx.tmp / shape / "ckpt_k")
        res["digests_k"] = digests(trainer, model, names)
        torch.save(rng_state(), d / "rng_k.pt")
        # steps k+1..N, uninterrupted
        recorder.open_window(want_grads=False)
        res["train_seconds_2"] = round(call_train(trainer, batches[k:], ctx.tmp / shape / "ckpt_end"), 2)
        w2 = recorder.close_window()
        res["ckpt_end_listing"] = listing(ctx.tmp / shape / "ckpt_end")
        if not a.keep_tmp:
            shutil.rmtree(ctx.tmp / shape / "ckpt_end", ignore_errors=True)
        res["train"] = {key: w1[key] + w2[key] for key in ("loss", "n", "shapes", "inputs", "lr", "betas", "grad_norm")}
        expect = expected_digests(trainer._golden_ref_train, batches)
        res["delivered_as_selected"] = res["train"]["inputs"] == expect  # what the forward received = the selection
        t_steps = w1["t_step"] + w2["t_step"]
        res["train"]["step_intervals_s"] = [round(y - x, 4) for x, y in zip(t_steps, t_steps[1:])]
        res["grads_none"] = w1["grads_none"]
        torch.save(w1["grads_step1"], d / "grads_step1.pt")
        final = trainable(model)
        torch.save({n: final[n] - initial["params"][n] for n in final}, d / "updates.pt")
        opt_state, groups = optimizer_by_name(trainer, names)
        torch.save({"state": opt_state, "groups": groups}, d / "optimizer.pt")
        torch.save(rng_state(), d / "rng_end.pt")
        res["scheduler_end"] = gl.jsonable(trainer.lr_scheduler.state_dict())
        res["digests_end"] = digests(trainer, model, names)
        res["memory_train"] = segment(ctx, f"train_{shape}")
        del final, opt_state
        ctx.save()
        res["eval_after_train"] = run_eval(trainer, recorder, [b["indices"] for b in sel["val"]], SEED)
        res["memory_eval"] = segment(ctx, f"eval_{shape}")
        ctx.save()
        print(f"[golden] {shape}: losses {res['train']['loss']}, eval {res['eval_after_train']['avg_loss']:.6f}",
              flush=True)
    if ctx.case["decode"]:
        stage2 = importlib.import_module(f"{ctx.layout['package']}.stage2")
        best = {"dir": str(a.best)}
        t = time.perf_counter()
        best["load"] = stage2.load_model_state(model, a.best)
        if ctx.case["ablation"]:
            neural_ablations = importlib.import_module(f"{ctx.layout['package']}.neural_ablations")
            best["ablation_verify"] = neural_ablations.verify(model)
        best["load_seconds"] = round(time.perf_counter() - t, 2)
        best["digest"] = gl.nested_digest(trainable(model))
        best["eval"] = {}
        for shape in a.shapes:
            sel = ctx.rec["shape_results"][shape]["selection"]
            best["eval"][shape] = run_eval(trainer, recorder, [b["indices"] for b in sel["val"]], SEED)
        n_val = len(loc["splits"]["val"])
        rng = np.random.default_rng(a.harness_seed + 1)
        trials = sorted(int(i) for i in rng.choice(n_val, size=min(a.decode_trials, n_val), replace=False))
        dec = run_decode(trainer, loc, trials, a.case, ctx.layout["package"])
        gl.write_json(ctx.out / "decodes.json", dec)
        best["decode"] = {k: v for k, v in dec.items() if k != "rows"}
        best["memory"] = segment(ctx, "best")
        ctx.rec["best"] = best
        print(f"[golden] decode {len(trials)} val trials: WER {dec['score']['wer']:.4f} in {dec['seconds']} s",
              flush=True)
    recorder.remove()
    raise HarnessDone()


def resume_callback(ctx: Ctx, trainer):
    import torch

    a = ctx.args
    main = gl.read_json(a.out / "main" / "phase.json")
    ctx.loc = main_locals()
    model, names, recorder = setup_callback(ctx, trainer)
    init_digests = digests(trainer, model, names)
    initial = trainable(model)
    ctx.rec["initial_digests"] = init_digests
    ctx.rec["fresh_build_matches_main"] = {k: init_digests[k] == main["initial_digests"][k] for k in init_digests}
    ctx.rec["shape_results"] = {}
    for shape in a.shapes:
        mres = main["shape_results"][shape]
        d_main, d = a.out / "main" / shape, ctx.out / shape
        d.mkdir(parents=True, exist_ok=True)
        res = {}
        ctx.rec["shape_results"][shape] = res
        ckpt = ctx.tmp / shape / "ckpt_k" / "LAST"
        t = time.perf_counter()
        trainer.accelerator.load_state(str(ckpt))
        res["load_seconds"] = round(time.perf_counter() - t, 2)
        got = digests(trainer, model, names)
        res["load_matches_step_k"] = {key: got[key] == mres["digests_k"][key] for key in got}
        rng_k = rng_digests(torch.load(d_main / "rng_k.pt", weights_only=False))
        res["accelerate_restored_rng"] = {k: got[k] == rng_k[k] for k in rng_k}
        if not all(res["accelerate_restored_rng"].values()):
            # accelerate's load_state reads random_states_0.pkl with torch.load's default weights_only=True, which
            # fails for its numpy / Python states under torch >= 2.6, and it skips the restore silently. Restore
            # the same file's states the way it would have, so the remaining checks test everything else.
            st = torch.load(ckpt / "random_states_0.pkl", map_location="cpu", weights_only=False)
            random.setstate(st["random_state"])
            __import__("numpy").random.set_state(st["numpy_random_seed"])
            torch.set_rng_state(st["torch_manual_seed"])
            torch.cuda.set_rng_state_all(st["torch_cuda_manual_seed"])
            after = rng_digests()
            res["rng_repaired_from_checkpoint_file"] = {k: after[k] == rng_k[k] for k in rng_k}
        batches = [b["indices"] for b in mres["selection"]["train"]][a.save_at:]
        recorder.open_window(want_grads=False)
        res["train_seconds"] = round(call_train(trainer, batches, ctx.tmp / shape / "ckpt_resume_end"), 2)
        w = recorder.close_window()
        if not a.keep_tmp:
            shutil.rmtree(ctx.tmp / shape / "ckpt_resume_end", ignore_errors=True)
        res["train"] = {key: w[key] for key in ("loss", "n", "shapes", "inputs", "lr", "betas", "grad_norm")}
        res["delivered_as_selected"] = res["train"]["inputs"] == expected_digests(trainer._golden_ref_train, batches)
        final = trainable(model)
        updates = {n: final[n] - initial[n] for n in final}
        opt_state, groups = optimizer_by_name(trainer, names)
        main_updates = torch.load(d_main / "updates.pt", weights_only=False)
        main_opt = torch.load(d_main / "optimizer.pt", weights_only=False)
        exp = lambda st, key: {n: s[key] for n, s in st.items() if key in s}  # noqa: E731
        mtrain = mres["train"]
        res["vs_uninterrupted"] = {
            "loss": gl.compare_scalars(mtrain["loss"][a.save_at:], res["train"]["loss"]),
            "n": gl.compare_discrete(mtrain["n"][a.save_at:], res["train"]["n"]),
            "lr": gl.compare_scalars(mtrain["lr"][a.save_at:], res["train"]["lr"]),
            "grad_norm": gl.compare_scalars(mtrain["grad_norm"][a.save_at:], res["train"]["grad_norm"]),
            "updates": gl.compare_tensors(main_updates, updates),
            "exp_avg": gl.compare_tensors(exp(main_opt["state"], "exp_avg"), exp(opt_state, "exp_avg")),
            "exp_avg_sq": gl.compare_tensors(exp(main_opt["state"], "exp_avg_sq"), exp(opt_state, "exp_avg_sq")),
            "optimizer_step": gl.compare_tensors(exp(main_opt["state"], "step"), exp(opt_state, "step")),
        }
        rng_end_main = rng_digests(torch.load(d_main / "rng_end.pt", weights_only=False))
        now = rng_digests()
        res["vs_uninterrupted"]["rng_end"] = {k: now[k] == rng_end_main[k] for k in now}
        got_end = digests(trainer, model, names)
        res["digests_end"] = got_end
        res["end_matches_uninterrupted"] = {key: got_end[key] == mres["digests_end"][key] for key in got_end}
        res["exact"] = all(res["end_matches_uninterrupted"].values()) and res["vs_uninterrupted"]["loss"]["exact"]
        res["memory"] = segment(ctx, f"resume_{shape}")
        torch.save(rng_state(), d / "rng_end.pt")
        ctx.save()
        print(f"[golden] resume {shape}: load matches step k {res['load_matches_step_k']}, "
              f"end exact {res['exact']}, losses {res['train']['loss']}", flush=True)
    recorder.remove()
    raise HarnessDone()


def child(args) -> int:
    ctx = Ctx(args, args.phase)
    ctx.out.mkdir(parents=True, exist_ok=True)
    rc = 0
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("the harness needs CUDA")
        ctx.rec["env"] = env_info()
        ctx.rec["rng_fingerprint"] = rng_fingerprint()
        module, trainer_cls, paths = load_train_module(ctx.code_root, ctx.layout)
        ctx.rec["imported_from"] = paths
        ctx.name = f"golden_{args.phase}_{uuid.uuid4().hex[:12]}"
        argv = case_argv(args, ctx.name)
        ctx.rec["train_stage2_argv"] = argv
        ctx.save()
        callback = main_callback if args.phase == "main" else resume_callback
        with intercepted(trainer_cls, lambda trainer: callback(ctx, trainer), ctx.code_root, ctx.name, argv):
            try:
                module.main()
            except HarnessDone:
                pass
            else:
                raise RuntimeError("train_stage2.main() returned without calling trainer.train()")
        ctx.rec["status"] = "ok"
    except BaseException as e:  # noqa: BLE001 - recorded, then a nonzero exit
        rc = 1
        ctx.rec["status"] = "oom" if type(e).__name__ == "OutOfMemoryError" else "error"
        ctx.rec["error"] = traceback.format_exc()[-8000:]
        print(ctx.rec["error"], flush=True)
    try:
        segment(ctx, "end")  # the rest (an error's peak included)
    except Exception:  # noqa: BLE001
        pass
    ctx.rec["timings"]["phase_seconds"] = round(time.time() - ctx.t0, 1)
    ctx.save()
    return rc


if __name__ == "__main__":
    _args = parse_args()
    sys.exit(child(_args) if _args.phase else parent(_args))

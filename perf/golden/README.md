# BIT golden harness (PerfLab)

A short, seeded slice of real Stage 2 jobs, recorded so that a speed-up branch can be checked against the baseline
code: "no bigger than switching machines" (PerfLab's `CONTRACT.md`). The harness is a measurement tool only. It
never changes a real run, and it never edits the code it tests.

- `run.py`: runs one case on one GPU and writes the outputs plus `manifest.json`.
- `compare.py`: compares two runs of the same case.
- `noise_floor.py`: builds the floor JSON from comparisons.
- `golden_lib.py`: shared digests, comparisons and near-tie rules.
- CPU tests: `tests/test_perf_golden.py`.

## What a run does

`run.py` imports the training code from `--code-root`: its `train_stage2.py`, `src/` and `external/bit`, with BIT's
YAMLs patched. It finds the layout itself: the current one (`scripts/train_stage2.py`, `src/bit_experiments`) or the
older one (`scripts/phase4/train_stage2.py`, `src/kunzdecoder`). It never imports training code from its own
directory, so a candidate branch cannot change the harness. It calls the code root's own `train_stage2.main()` with the case's real
arguments: reference runtime, bf16 autocast, BIT's AdamW, OneCycle schedule, LoRA dropout and seeds, and the full data
(so the schedule's length is the production one). It takes over where `main()` calls `trainer.train()`:

1. **Fixed batches, per shape.** Each batch is recorded by trial index and by content hash.
   - `small`: the 8·N shortest train trials (stable order), shuffled with a fixed seed into batches of 8 (`--small-batch`).
     It is meant to fit a 32 GB GPU with the reference runtime.
   - `prod`: the first N batches of the production sampler's first epoch (BIT's `GroupBatchSampler`, seed 42).
   - **The batches come through BIT's real loaders** (`--loader real`, the default).
     - The trainer's own prepared DataLoader is used as is: its dataset, `collate_fn`, workers, pinning, the
       accelerate wrapper and device placement. Only its batch-sampler instance is switched to a subclass that
       yields the selected batches; its seeded generator is never advanced.
     - Every forward records a digest of the tensors it received. These are checked against the loader's
       `collate_fn` applied to the selected trials (`delivered_as_selected`), and compared across runs.
     - A loader-level candidate (prefetch, collation, pinned or non-blocking transfer) therefore goes through the
       harness. `--loader fresh` builds new loaders copying those settings instead; it is for debugging only.
2. **N = 8 optimizer steps** through the code root's own `Trainer.train`, one epoch over the fixed batches.
   - After step k = 4, `Trainer.train` saves its own checkpoint (`LAST`: BIT's files plus accelerate's state).
   - The remaining steps then run in the same process, uninterrupted.
3. **Resume in a fresh process.** `main()` builds a new model, optimizer and scheduler. The step-k checkpoint is
   loaded with accelerate's `load_state`, and the remaining steps run. The result is compared with the uninterrupted
   run inside the job.
4. **Teacher-forced eval loss** with BIT's `Trainer.evaluate`, on 4 val batches, after training. The RNG is seeded
   first, because the diffusion objective draws masks in eval too.
5. **Trained checkpoint** (`--best`).
   - It is loaded with `stage2.load_model_state`, and the eval runs again.
   - Then 32 seeded val trials are decoded greedily with `stage2.generate`. Block diffusion uses its own sampler
     settings.
   - Per generated token, the run records the token id and the top-1 minus top-2 logit margin. For AR this comes from
     HF's `output_scores`. For block diffusion it comes from the sampler's own trace plus the `lm_head` outputs, so
     every commit decision has its margin.

Recorded per shape:
- per-step loss, supervised count, learning rate (and AdamW betas), gradient norm and batch shapes;
- every trainable gradient of step 1 (`grads_step1.pt`);
- the parameter updates after N steps (`updates.pt`), and AdamW's `exp_avg`, `exp_avg_sq` and `step` (`optimizer.pt`);
- the RNG states after steps k and N, and digests of the trainable state, optimizer, scheduler, buffers and RNG;
- the eval losses and argmaxes at supervised positions, with their margins;
- the decodes (`decodes.json`);
- peak CUDA memory per segment, host RSS, and timings.

`manifest.json` records:
- the harness's identity (sha256 of `run.py` + `golden_lib.py`, plus `HARNESS_SOURCE.json` written when it is staged);
- the sha256 of every training-path file in the code root (and its git state);
- the GPU, the versions, the inputs, and an RNG fingerprint of the GPU (dropout masks and uniforms of a few sizes from
  a fixed seed).

When a job runs two shapes, the second starts from the captured initial state: parameters, buffers, optimizer,
scheduler, module modes and RNG, as `scripts/profile_training_runtime.py` replays do. Its digests are checked against
the fresh build's.

## Baseline code

The baseline is the code the queued jobs actually run, not `main`. There are two trees, each copied from a project
copy on the GPU host and mirrored byte for byte to the other host:

- **AR tree** (`b_ar`, `c_ar`): the AR project copy's code, in the older layout. The queued B seeds and arm C run it.
  Its data, checkpoints, results, venv and logs are left out.
- **Diffusion tree** (`b_diff`, `b_diff_fast`, `eraser_ar`): the eraser-pair copy, the plain-step code that the queued
  diffusion and eraser jobs share. Its `checkpoints`, `data`, `results/stage2` and `external/bit` are symlinks there:
  the copy leaves out the first three and takes `external/bit` from the tree it points to.

Every run's manifest records the sha256 of each training-path file of its tree.

## Cases

| case | the real run | init (`--init`) | decode (`--best`) |
|---|---|---|---|
| `b_ar` | `B_ssl_s42` (Aero AR, arm B) | `checkpoints/ssl_human/BEST_core` | `checkpoints/stage2/B_ssl_s42/BEST` |
| `b_diff` | `B_ssl_aero-1-audio-diffusion_s42` (plain block-diffusion step) | `checkpoints/ssl_human/BEST_core` | its `BEST` |
| `c_ar` | `C_ssl_l14_b1024_s42` (arm C) | `checkpoints/stage1/C_l14_b1024_s42/best` | none: no trained run yet |
| `eraser_ar` | `B_ssl_phonerased_s42` (frozen B_ssl_s42 encoder + LEACE eraser) | `<ablations>/B_ssl_s42-encoder` | `B_ssl_s42`'s BEST with the eraser at decode time (the eraser prep's check b) |
| `b_diff_fast` | `b_diff` with the packed fast step (`--llm aero-1-audio-diffusion-fast`) | as `b_diff` | as `b_diff` |

`b_diff_fast` is a candidate variant in `b_diff`'s family: `compare.py` compares runs of one family, so a fast-step
candidate is checked against the plain `b_diff` baseline. The fast step needs more memory than the plain step.

**Candidates.** `--train-args "<flags>"` appends a candidate's `train_stage2.py` flags (e.g. `--runtime optimized`) to
the case's arguments. They are recorded in the manifest, and `compare.py` prints both runs' flags.

`eraser_ar` needs `--ablations-dir`, the directory holding `B_ssl_s42-encoder/` and `phoneme-erasure/eraser.pt`.
The harness links it as the code root's `checkpoints/ablations`, which is where `configs/ablations.yaml` points. It
also links `--hf-dir` as `checkpoints/hf`, where `configs/llms.yaml` points. Data, the AR LLM and the tokenizer come
from the absolute paths in BIT's patched YAMLs.

## RNG

- BIT seeds torch only: CPU and CUDA.
  - Python's `random` and numpy's global RNG start from OS entropy in every process.
  - Nothing in Stage 2 draws from them (BIT's samplers use their own seeded generators).
  - They are recorded per component and reported as informational, never judged.
- Eval draws from the global CUDA RNG (the diffusion objective's masks), so the order is fixed: train, then eval.
  - The RNG is seeded before each eval.
  - Its digests are recorded before the seed, after the seed, and after the eval.
- `accelerate` 1.3 under torch ≥ 2.6 cannot restore RNG on resume. `load_state` reads its `random_states_0.pkl` with
  `torch.load`'s default `weights_only=True`, which refuses that file's numpy and Python states, and skips the
  restore silently.
  - The resume records this (`accelerate_restored_rng`), then restores the states from that same file
    (`rng_repaired_from_checkpoint_file`). The remaining steps therefore test everything else the checkpoint holds.
  - No production run resumes, so no result is affected.

## Safety

- **Code root.** `run.py` refuses a code root that looks like a project copy: one whose `checkpoints`, `results` or
  `data` (or their `stage2`) is a symlink, that holds trained checkpoints in `checkpoints/stage2`, or that has
  `data/raw`.
- **Temporary run dirs.** `main()` creates `results/stage2/golden_*` and `checkpoints/stage2/golden_*` in the code
  root. The harness removes them.
- **Checkpoints.** The step-k checkpoint and the end-of-train saves are full accelerate states (about 8.5 GB for
  AR, 6 GB for diffusion). They go to `<out>/tmp` and are deleted when the job ends. If gq kills a job, remove its
  `<out>/tmp` by hand.

## Commands

Placeholders:

| placeholder | meaning |
|---|---|
| `<perf>` | the perf work dir inside the project container |
| `<proj>` | the project copy whose checkpoints and data are read, read-only |
| `<py>` | BIT's pinned interpreter |
| `<machine>` | the gq machine name |

```bash
# baseline trees, inside the container on the GPU host (read-only tars of the project copies)
tar -C <AR copy> --exclude=./data --exclude=./checkpoints --exclude=./results --exclude=./.venv --exclude=./logs \
  --exclude=./figures --exclude=./.pytest_cache --exclude=__pycache__ --exclude='*.egg-info' -cf - . \
  | tar -C <perf>/bit-golden-baseline-ar -xf -
tar -C <diffusion copy> --exclude=./checkpoints --exclude=./data --exclude=./external --exclude=./results \
  --exclude=./ops --exclude=__pycache__ --exclude='*.egg-info' -cf - . | tar -C <perf>/bit-golden-baseline-diffusion -xf -
tar -C <AR copy> --exclude=__pycache__ -cf - external/bit | tar -C <perf>/bit-golden-baseline-diffusion -xf -
# then stream both to the other host, and compare their file-hash listings
# harness: perf/golden/{run.py,golden_lib.py,compare.py,noise_floor.py,README.md} -> <perf>/bit-golden-harness

gq submit --on <machine> --project perf --name bit_golden_b_ar_small_r1 --gpu --est 15m --container <container> \
  --cwd <perf>/bit-golden-harness --env CC=<cc> --env CUDA_HOME=<cuda> --env DS_SKIP_CUDA_CHECK=1 \
  --env HF_HUB_OFFLINE=1 --env WANDB_MODE=disabled -- \
  <py> -u run.py --code-root <perf>/bit-golden-baseline-ar --case b_ar --shapes small \
    --out <perf>/bit-golden-out/<machine>/b_ar_small_r1 --machine <machine> \
    --init <proj>/checkpoints/ssl_human/BEST_core --best <proj>/checkpoints/stage2/B_ssl_s42/BEST \
    --hf-dir <proj>/checkpoints/hf
# b_diff: --code-root <perf>/bit-golden-baseline-diffusion --case b_diff \
#           --best <proj>/checkpoints/stage2/B_ssl_aero-1-audio-diffusion_s42/BEST
# c_ar:   --code-root <perf>/bit-golden-baseline-ar --case c_ar --init <C stage 1 best>        (no --best)
# eraser: --code-root <perf>/bit-golden-baseline-diffusion --case eraser_ar --init <ablations>/B_ssl_s42-encoder \
#           --ablations-dir <ablations> --best <B_ssl_s42 BEST>
# a candidate: --code-root <its copy> [--train-args "<its flags>"]; a fast-step candidate: --case b_diff_fast
# both shapes in one job (a GPU with room for prod): --shapes small prod

# on the Mac, after copying the output dirs (project .venv)
.venv/bin/python perf/golden/compare.py <runs>/<m>/b_ar_small_r1 <runs>/<m>/b_ar_small_r2 --out <cmp>/b_ar_<m>_r1_r2.json
.venv/bin/python perf/golden/noise_floor.py <cmp>/*_<m>_r1_r2.json --out <PerfLab>/noise_floor/bit.json \
  --pair <m>~<m> --kind same_machine --provisional --baseline-note "<where the trees came from>"
.venv/bin/python perf/golden/noise_floor.py <cmp>/*_<m1>_vs_<m2>.json --out <PerfLab>/noise_floor/bit.json \
  --pair <m1>~<m2> --kind cross_machine --merge
# a candidate on one machine against the baseline on the same machine:
.venv/bin/python perf/golden/compare.py <baseline run> <candidate run> --floor <PerfLab>/noise_floor/bit.json
```

The candidate's run uses `--code-root` = the candidate's copy, with the same harness and the same arguments (plus
its `--train-args`). Compare it with the baseline run of the same case family on the same machine.

## Definitions

- `max_abs`: the largest elementwise |candidate − baseline| in an output.
- `max_rel`: per tensor, ‖candidate − baseline‖∞ / ‖baseline‖∞ (normwise), then the largest over the output's
  tensors. For lists of scalars (losses) it is elementwise.
- `rel_l2`: ‖candidate − baseline‖₂ / ‖baseline‖₂ over the output.
- `exact`: bitwise equal.
- **Near-tie.** The baseline's top-1 minus top-2 logit at the first differing token is at most 0.5 logit
  (`--tau-logit`), about 4 bf16 ulps at typical top-1 logits.
  - For block diffusion, the first sampler event whose commits differ is located. The state is identical before it.
  - Every differing decision there must be a near-tie: a token margin of at most 0.5 logit, or a commit probability
    within 0.05 of the threshold (`--tau-prob`). For the forced commit of the most confident position, the two top
    probabilities must be within 0.05.
  - Proposals that were not committed do not count.

## Pass rule (`compare.py --floor`)

- **Same machine.** The candidate is run against the baseline on the same machine.
- **Outputs without a floor entry** must be bitwise exact.
- **Continuous outputs** need `max_abs` ≤ the floor's and `max_rel` ≤ the floor's.
- **Decodes and argmaxes:**
  - no more non-near-tie mismatches than the floor's (normally 0);
  - no more mismatches in all than the floor's.
- **Other discrete outputs** (batch indices, data hashes, counts, digests, RNG states) need no more mismatches than
  the floor's.
- **Which pair sets the floor.** `noise_floor.py` takes the worst case per output over the cross-machine pairs. Until
  a cross-machine pair exists, it uses the same-machine pairs, and marks the floor provisional.
- **Tolerances.** Never edit them to make a candidate pass.

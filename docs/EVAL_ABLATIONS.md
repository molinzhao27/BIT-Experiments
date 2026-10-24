# Eval-only input ablations

**Tier 2** (explains the result, next to the LLM swap, DESIGN.md §8; never in the headline table). The error analysis
(`results/stage2/phonetic_errors.md`) found sound-alike substitutions far above chance that vanish when the neural
input is shuffled. This asks which parts of the input the trained decoders use: which electrode arrays, which part of
the trial, and whether the temporal order matters. Nothing is retrained. The same weights decode greedily, and only
each trial's neural features change.

Code: `scripts/eval_ablations.py`. Config: `configs/eval_ablations.yaml`. Tests: `tests/test_eval_ablations.py`.

## Method

Each finished run is rebuilt eval-only from its `run.json`, exactly as `train_stage2.py --eval-only` and
`decoding_order_sweep.py` do it:
- BIT's sentence config, with the run's LLM entry and prompt;
- `stage2.fix_padding`;
- BEST loaded through `stage2.load_model_state`.

This works for every model class: `BCI_AudioLLM` (BIT's Aero path), `BCI_TextLLM` and `BCI_DiffusionLLM`. A diffusion
run decodes with the sampler recorded in its `run.json`. The rebuilt config is checked against `run.json` (model
class, prompt, LLM snapshot, features, channel range), and a mismatch fails that run. Only runs with a
`summary.json` are evaluated, so a run still in training is never decoded from an intermediate BEST.

The ablation is applied to `spikes` (T × 512) before BIT's collate. Everything after that is unchanged and sees the
ablated input: pad-masked Gaussian smoothing (sd 2 bins), patching into 5-bin (100 ms) patches, and the read-in.

| Name | What changes |
|---|---|
| `none` | Nothing; the trial is decoded again. This is the pairing baseline. `matches_saved` checks that it reproduces the run's own `test_greedy_real.csv`. |
| `drop_<array>` | That array's 64 threshold-crossing and 64 spike-band-power columns are set to the fill. |
| `only_<array>` | The other three arrays' columns are set to the fill. |
| `time_first_half`, `time_second_half`, `time_middle_half` | Only bins `[floor(lo·T), floor(hi·T))` are kept, with (lo, hi) = (0, ½), (½, 1) and (¼, ¾). Every feature of the other bins is set to the fill. The trial keeps its length. |
| `scramble_patches` | The trial's full 5-bin encoder patches are permuted. Trial *i* uses `default_rng([seed, i])`, so the order is the same for every run (paired). Bins keep their order inside a patch, and a trailing partial patch stays last. |
| `reverse_time` | The bins are reversed. |
| `drop_tx`, `drop_sbp` (optional, not in the default batch) | One feature type is removed on every array. |

**Fill.** The fill is each feature's mean over every training-set bin (`fill: train_mean`, the default). The released
features are already z-scored per channel against the preceding 20 trials (Dryad README), so the fill is close to 0.
It is the trial-independent value of each feature: activity at its average level, carrying no information about the
sentence.

Literal silence would mean zero threshold crossings and minimal spike power. In the normalised units that is a large
negative value that differs by channel and by normalisation window, and it can't be recovered: the raw statistics
aren't released. It would also put the input far outside anything seen in training. `fill: zero` (the z-score
centre) is the alternative; both are recorded in `summary.json` (`fill.vector`, mean |fill| per feature type).

**Caveat: the models never saw this.** A constant feature never occurred in training. The read-in also applies a
LayerNorm over each patch's 5 × 512 values, so filling some features rescales the remaining ones. An ablation's WER
therefore measures information removed *plus* a distribution shift; it is an upper bound on how much the decoder
relies on that part of the input. A swap-in control (the array taken from another trial) would keep the input in
distribution. It isn't implemented.

**No pre-speech vs speech split.** The hdf5 trials carry no event times. Their fields (`load_h5py_file` in BIT's
loader and in nejm-brain-to-text) are `input_features`, `n_time_steps`, `seq_class_ids`, `seq_len`, `transcription`,
`sentence_label`, `session`, `block_num` and `trial_num`. There is no go cue or speech onset, so the time windows are
fractions of each trial.

**Measures**, per (run, split, ablation):
- **WER** and exact-match rate (`stage2.score`), seen / unseen;
- **ΔWER = WER(ablation) − WER(none)**, from a paired trial bootstrap: 10,000 resamples, shared across one run's
  ablations, with a 95% CI and a two-sided p. Also the number of trials with more / fewer word errors.
- **The substitutions' error pattern**, computed with `scripts/phonetic_error_analysis.py`'s own functions (`extract`,
  `Lexicon`, `RunAnalysis`, `summarize`, `contrast`) and no word embeddings:
  - the sound-alike share (phoneme similarity ≥ 0.5) against its length-matched null, built from that decode set's
    own substitutes;
  - the phoneme-similarity excess, also beyond spelling (`phon|orth`);
  - the paired change in that excess vs `none`.

  This reproduces `phonetic_errors/summary.json` exactly on run 3's saved decodes (23.6% vs 2.72% null; excess
  +0.192). It needs `pronouncing` and `wordfreq`, which BIT's env lacks. In that case the GPU run marks it pending, and
  `--phonetic-only` fills it in from the saved decodes in the project `.venv` (about 1 min per run).

## Channel map

The source is the [nejm-brain-to-text README](https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text/blob/980a5b7e96392210edf976d76bdb5d15d2e0e62b/README.md)
@ `980a5b7`, lines 37–45. It describes four 64-electrode arrays: "The 512 features are ordered as follows in all data
files: 0-64: ventral 6v threshold crossings, 65-128: area 4 threshold crossings, …, 449-512: dorsal 6v spike band
power".

The ranges are written 1-based and inclusive, except the first ("0-64"). The only reading consistent with 64
electrodes per array is the 0-based, half-open one below.

BIT's loader (`external/bit/data/brandman_2024_text/prepare_data.py`, with `features: [tx1, spikePow]`, area 0–256)
concatenates `input_features[:, 0:256]` and `input_features[:, 256:512]`. So model column *j* is the file's feature
*j*. Every run here uses that data config, and the runner checks it.

| Array | Area | Threshold crossings | Spike-band power |
|---|---|---|---|
| `v6v` | ventral 6v | 0–63 | 256–319 |
| `a4` | area 4 (M1) | 64–127 | 320–383 |
| `55b` | area 55b | 128–191 | 384–447 |
| `d6v` | dorsal 6v | 192–255 | 448–511 |

The Kunz data notes give the same four areas in the same order for T15's isolated-behaviour export (`i6v · M1 · 55b ·
s6v`, 0–63 … 192–255; `KunzInnerSpeechReproduction/docs/DATA_NOTES.md`). That is a different export, so it is
supporting evidence only. `load_config` checks that the arrays cover the 512 columns exactly once.

## GPU time and commands

The runner prints this estimate first: `--estimate-only`, from each run's own final evaluation. That is summary.json
minutes minus the last log.jsonl entry, over its decode sets, nulls and nucleus included. Add 2 min per run for
loading. The batch is 14 conditions × 713 BIT-test trials per run.

| Run | s / trial | Estimate |
|---|---|---|
| `B_ssl_fast-dllm-v2-1.5b_s42` (run 3) | 0.127 (greedy only: 0.103, decoding-order sweep) | 23 min (≈ 19 at the sweep's rate) |
| `B_ssl_qwen25-1.5b-instruct-diffusion_s42` (control) | 0.13 (class default until its summary.json exists) | 24 min |
| `B_ssl_s42` (Aero AR) | 0.176 | 31 min |
| Subtotal, first batch | | **≈ 66–78 min** |
| `B_ssl_qwen25-1.5b-noprompt_s42` | 0.137 | 25 min |
| `B_ssl_qwen25-1.5b-instruct_s42` | 0.130 | 24 min |

The first three runs always run (`budget_exempt: 3`). The Qwen AR runs run only if the batch, rescaled by its own
measured speed, would stay within `budget_minutes: 75`, so normally they are skipped. Adding `--splits val test`
doubles everything. The raw dataset is loaded once per process (host RAM). One model is on the GPU at a time.
Eval-only peaks so far: 6.7 GB for run 3's decoding-order sweep and 9.0 GB for Aero's eval-only 4a run, so ~10 GB.
`summary.json` records `max_mem_gb`.

Run from the host's code copy. It needs `scripts/eval_ablations.py`, `configs/eval_ablations.yaml`,
`tests/test_eval_ablations.py` and `scripts/phonetic_error_analysis.py` (for labels and the phonetic step). The runner
reads `results/stage2/<run>/{run.json,summary.json,decodes/}` and `checkpoints/stage2/<run>/BEST` relative to that
copy.

```bash
<bit env>/python -m pytest tests/test_eval_ablations.py -q            # CPU, seconds: 24 passed, 1 skipped
<bit env>/python scripts/eval_ablations.py --estimate-only             # the plan, no GPU
<bit env>/python scripts/eval_ablations.py --runs B_ssl_fast-dllm-v2-1.5b_s42 --limit 8 --device cuda   # smoke: data + model load, then seconds
<bit env>/python scripts/eval_ablations.py --device cuda 2>&1 | tee logs/eval_ablations.log             # the batch
```

The batch's default `--runs` is the config's list in priority order. A run that fails is logged and skipped, and the
exit code is 0. A full run replaces the smoke's results, because a different `--limit` starts a fresh summary.

Later runs:

```bash
<bit env>/python scripts/eval_ablations.py --runs B_ssl_qwen25-1.5b-noprompt_s42 B_ssl_qwen25-1.5b-instruct_s42 --budget-minutes 0 --device cuda
<bit env>/python scripts/eval_ablations.py --runs B_ssl_aero-1-audio-diffusion-fast_s42 --device cuda   # or ..._aero-1-audio-diffusion_s42
```

Afterwards, on the Mac with the results pulled:

```bash
.venv/bin/python scripts/eval_ablations.py --phonetic-only
```

`--phonetic-only` defaults to the config's runs; pass `--runs` for others.

**Log lines:**
- `[ablation] <run> <name>: {json}`: one per decode set;
- `[ablation] <run> <name> phonetic: {json}`;
- `[ablation] <run> <where>: ERROR …`, with the traceback in `results/stage2/<run>/ablations/error.log`.

**Outputs:**
- `results/stage2/<run>/ablations/<name>/<split>_greedy.csv`, plus `.meta.json` with condition `ablation_<name>`;
- `results/stage2/<run>/ablations/summary.json`: every measure, the fill vector, the config used, the load report;
- `results/stage2/ablations_summary.{json,md}`: one table per run and split, rewritten after each run.

## Reading it

The table is one seed per run, so contrasts between runs are suggestive, not tests. Within a run, ΔWER and its CI
are paired and solid; every ablation is compared with the same re-decoded baseline.

What each pattern would mean:
- **A large ΔWER for `drop_<array>`:** the decoder depends on that array.
- **`only_<array>` far better than `null_shuffled`:** that array alone carries usable information.
- **`scramble_patches` / `reverse_time` near `none`:** the decoder reads a bag of 100 ms snapshots rather than their
  order.
- **A time half near `none`:** the information is concentrated there.
- **The sound-alike excess falling more than WER rises under an ablation:** that part of the input carries the
  phonetic signal.

## Tests (CPU)

`tests/test_eval_ablations.py`: 22 tests pass in the project `.venv`, and 24 in BIT's env (the local CPU copy). They
cover:
- **The channel map**, including the overlap check.
- **Every transform** on small synthetic trials: shapes and dtypes, exactly which columns and bins are filled, the
  input never modified, seeded determinism, a permutation of whole patches, reversal as an involution.
- **The fill and the paired bootstrap.**
- **`main()` end to end with a stub decoder** whose words depend on which arrays and which half are active:
  - every decode equals the stub's rule applied to the ablated trial, with the expected ΔWERs;
  - a failing run is logged while the others continue, and the exit code is 0;
  - the budget skip, the summaries, and the phonetic step or its pending note.
- **BIT's env only: the real build path on 3 trials for all three model classes**, using tiny random Qwen2 /
  Qwen2-Audio snapshots behind the registry entries: BIT's sentence config, `use_llm`, BIT's Trainer, `fix_padding`,
  a slim BEST, the full NDT encoder. Each checks that:
  - the rebuilt model reproduces the saved decodes, and they aren't degenerate;
  - the ablated features are exactly what reaches the encoder.

The real data and checkpoints are only on the host, so the first real-weight check is the GPU smoke. There,
`matches_saved` must be true for `none`.

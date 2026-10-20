# SSL pretraining on human data (plan and implementation notes)

Status 2026-10-18: this is a code reading plus the data layer. **Nothing has been trained and no GPU was used.**
BIT = `external/bit` (github.com/yzhang511/bit @ `f107ae9`, read-only). Paper: Zhang et al., arXiv:2511.21740v5,
App. Q Table 10. The dataset inventory is in `docs/PRETRAIN_DATA.md`.

Deliverables of this step:

| File | Role |
|---|---|
| `src/bit_experiments/ssl_data.py` | Adapters (B2T'25, B2T'24, Kunz), normalization, corpus writer/reader, BIT stitch tables |
| `scripts/build_ssl_corpus.py` | CLI. Builds `data/processed/ssl_human/` from `configs/ssl_human.yaml` |
| `configs/ssl_human.yaml` | Every corpus choice, plus the planned pretraining run (`pretrain:`) |
| `tests/test_ssl_data.py` | Synthetic fixtures in each documented format. Checks B2T splits against BIT's own loaders |

---

## 1. What BIT already supports

Short answer: **the model and trainer support masked-modeling SSL, but no SSL run is wired up.** There is
no SSL config, no SSL entry point, and only one dataset can be loaded per run.

### 1.1 Present and usable unchanged

| Piece | Where | Notes |
|---|---|---|
| `method_name: ssl` in NDT | `models/ndt.py:523-525, 541-549, 568-570, 583-592` | Targets are `spikes.clone()`, taken **before** smoothing and noise (568-570). MSE (or Poisson NLL) is summed over bins that are **masked and not padding** (`targets_mask & spikes_mask & targets != -100`, 585-592). `n_examples` = number of such bins |
| Patching | `models/ndt.py:179-247` | Right-pads T to a multiple of `patch_size` with −100 (`pad_for_patching`). Reshapes (B, T, C) → (B, T/5, 5·C). `spikes_timestamp[:, 0::5]` (245): **RoPE positions are bin indices, not patch indices** |
| Masking (`embedder.masker`) | `models/ndt.py:166-172, 249-293` | One span length per **batch**, `span ~ U{1..max_timespan}` (if `expand_prob` = 1). `int(ratio/span · n_patches)` spans with random, possibly overlapping starts, so the effective ratio is ≤ `ratio`. Masked **embeddings** (after read-in) are replaced by a learned `mask_token`. `randperm(n_patches − span + 1)` needs ≥ 14 patches (70 bins) per batch. For `method ssl` masking is also applied at eval (569: `force_mask = True`) |
| Default encoder config | `configs/models/encoder/ndt.yaml` | Already matches Table 10: hidden 384, 7 layers, 6 heads, inter 512, RoPE, patch 5, masker active / ratio 0.5 / expand_prob 1.0 / max_timespan 15, `max_F` 2500, dropout 0.4 / 0.2, fixup init, smoothing σ = 2 bins + white noise 0.2 + offset 0.05 (`models/model_utils.py:49-76`) |
| Per-dataset read-in / read-out ("stitching") | `models/ndt.py:36-123` | `ReadIn`: LN → Linear(5·C → 384) → LN. `ReadOut`: LN → Linear(384 → 5·C) → LN. One module per entry of `DATASET_CHANNELS[features]`, **keyed by position** (`enumerate(channel_dict.values())`, 52-54 / 95-101). `torch.unique(dataset_idx).item()` (75, 122) ⇒ **every batch must come from a single dataset** |
| Group batching | `utils/datasets.py:119-159` | `GroupBatchSampler` groups indices by `dataset[idx]["dataset_name"]` and shuffles batches across groups. It already handles any number of groups. It calls `__getitem__` (a deepcopy) on every record once at construction |
| Collate / padding | `utils/datasets.py:162-301` | `pad_collate_fn` pads keys listed in `pad_dict` (right, −100 for spikes; 0 for `spikes_mask` / `spikes_timestamp`). Other arrays of equal shape are stacked; everything else becomes a list in `unused_inputs`. Model inputs = the parameter names of `NDT.forward` (`models/trainer.py:174-177`): `spikes, spikes_mask, spikes_timestamp, targets, targets_lengths, dataset_name` |
| SSL metric | `train.py:72-75`, `utils/eval_spike.py:117-171` | `r2` if loss = mse, else `bps` |
| Trainer | `models/trainer.py` | AdamW + cosine `OneCycleLR` (229-254). Eval every `eval_every`. The best checkpoint is chosen by the **max** of `test_avg_metrics[metric_kwargs.metric_name]` for ssl (311-312, 334-339). `metric_fns` can be injected through the `Trainer(...)` argument (86-87). Checkpoints: `encoder.bin`, `encoder_config.pth`, `decoder.bin` (`models/ndt.py:614-618`) |

### 1.2 Missing or broken for SSL (why `train.py` can't run it as is)

1. **No SSL config.** `ConfigBuilder` reads `configs/{mode}/{task}/{encoder}/trainer.yaml` (`utils/config.py:169-171`).
   The only task choices are `none | phoneme | sentence` (`train.py:27-29`), and `configs/train_from_scratch/` has only
   `phoneme/`. There is **no `configs/train_from_scratch/ssl`** (and no `none/`).
2. **`--dataset none` crashes.** `dataset_defaults.get("none")` returns `None`, then `dataset_cfg["features"] = …` fails
   (`utils/config.py:186-197`).
3. **Dataset defaults switch masking off.** With a real `--dataset`, `_apply_dataset_config` replaces the
   embedder's `masker` with `active: false, ratio: 0.0` (`registry.py:51-56, 68-74` via `utils/config.py:204-205`).
   An SSL run through `train.py` would therefore train with no masked tokens.
4. **One dataset per run.** `load_brain2text_dataset` loads a single registry entry (`train.py:47-63`).
   `SpikingDataset` stamps **one** `dataset_name` on every record of a split (`utils/datasets.py:48, 65`).
5. **Hard-coded stitch tables.** `DATASET_TO_IDX` and `DATASET_CHANNELS` list only `brandman_2024_text` (0, 256 TX ch)
   and `willett_2023_text` (1, 128 TX ch) (`utils/datasets.py:16-34`). `NDT` reads them at construction
   (`models/ndt.py:15, 497`).
6. `embedder.stitch`, `max_channels` and `input_dim` are **never read** by the model. Only `utils/config.py:192-195`
   touches `max_channels`. Read-in/out is always per dataset; there is nothing to switch on.
7. Relative paths: `configs/trainer.yaml` (`utils/config.py:9`) and `configs/models/encoder/ndt.yaml`
   (`models/ndt.py:25`) mean the run must use **cwd = `external/bit`**.

### 1.3 Behaviors to know about (not bugs we fix, but they affect the setup)

- **Padding vs smoothing.** `SmoothAndNoise` runs on the padded batch, including the −100 pad values, and it also runs at
  eval (`models/model_utils.py:61-68`, called at `models/ndt.py:454`). In shorter sequences of a mixed-length batch,
  the last ~6 valid input bins get pulled toward −100. Only inputs are affected; targets are clean. Fixed-length Kunz
  chunks avoid this. B2T trials are variable length, as in BIT's own runs.
- **Variable length.** `spikes_mask` marks real bins. Patches that are entirely padding are excluded from attention
  keys (`models/ndt.py:463-466`) and from the loss. The patch at the end of a sequence that is partly padding still
  attends and still feeds the loss for its real bins.
- **RoPE / context length.** Positions are bin indices, so each sequence must be ≤ `max_F` = 2500 bins (50 s).
  All B2T'25 train/val trials are ≤ 2475 bins (measured), and all B2T'24 trials are ≤ 919.
- **BIT's R² is not "masked reconstruction R²".** `eval_neurons_metric` takes all bins (masked **and** unmasked,
  padding zeroed in both), keeps the 1,000 channels with the largest |sum|, and computes one pooled `r2_score` over
  the flattened batch (`utils/eval_spike.py:156-170`). The trainer then averages over batches (`models/trainer.py:418`).
  Unmasked bins are never trained, so this number mixes two things. Eval masks are random each epoch, so the
  selection metric is noisy.
- **bf16 in eval.** Float inputs are cast to bf16 for eval (`models/trainer.py:389-391`). Attention always runs in
  bf16 (`models/model_utils.py:136-137`).
- **Normalization in BIT's loaders.** B2T'24: optional z-score per block, then per day using train statistics
  (`data/willett_2023_text/prepare_data.py:41-56, 85-99`; registry default: both on). B2T'25: none; the code asserts
  the release is already normalized (`data/brandman_2024_text/prepare_data.py:31`).
- **BIT's splits.** Official val (B2T'25 `data_val`; B2T'24 `test/` folder) is permuted with `RandomState(42)`. The first
  half is "val" (selection), the second half is "test" (held out) (`split_val`, `…/prepare_data.py:131-144 / 170-183`).
  The official test/holdout is renamed `holdout` and dropped for supervised runs (`train.py:55-56`).
- **Fine-tuning loads read-ins by index.** `from_pt` loads `encoder.bin` with `strict=False`
  (`models/ndt.py:500-513`). `strict=False` ignores extra keys (our Kunz read-ins) but **not shape mismatches**.
  BIT's decoding runs use `--features all` (TX+SBP: 512 / 256 inputs per patch bin). A TX-only pretrained read-in
  therefore can't be loaded into them (see open question Q9).

---

## 2. What we need for multi-dataset human pretraining

| Need | Solution |
|---|---|
| Records for 3 sources × 4 participants in one split dict | `ssl_data.build_corpus` → `load_corpus` (done) |
| One read-in/out per stitch group | Patch `utils.datasets.DATASET_TO_IDX` / `DATASET_CHANNELS` **in place** from `ssl_data.bit_stitch_tables(manifest)` before building `NDT`. `models.ndt` imported the same dict objects, so in-place updates are seen |
| Per-record group id | A 10-line `SpikingDataset` subclass that sets `dataset_name = record["stitch_idx"]`, registered as `trainer.NAME2DATASET["ssl_multi"]` |
| Masking on | Build the config dict ourselves; don't go through `ConfigBuilder._apply_dataset_config` |
| Clean val metric | Inject `metric_fns` (masked-bin R², per group) and seed the eval masks (Trainer subclass overriding `evaluate`) |

Stitch index order is fixed by `corpus.stitch_order`: **0 = `b2t25_T15`, 1 = `b2t24_T12`**, the same as BIT's own
`brandman_2024_text` = 0 and `willett_2023_text` = 1. Then come `kunz_T12`, `kunz_T15`, `kunz_T16`, `kunz_T17`.
A TX-only fine-tune with BIT's unmodified loader therefore finds the B2T read-ins in the right slots.

---

## 3. Plan

### 3.1 Reuse unchanged from BIT

`models/ndt.py` (NDT, ReadIn/ReadOut, masker, loss), `models/model_utils.py`, `models/trainer.py` (`Trainer`,
`train`, checkpointing), `utils/datasets.py` (`SpikingDataset` base, `GroupBatchSampler`, `pad_collate_fn`),
`utils/config.py` (`update_config`), `configs/models/encoder/ndt.yaml`, `configs/trainer.yaml`. The BIT
environment is the Phase 4a one (`configs/bit_baseline.yaml`).

### 3.2 Thin wrappers in our repo

Done in this step: the data layer (`ssl_data.py`, `build_ssl_corpus.py`, `ssl_human.yaml`).

Next step (not written yet; ~120 lines): `scripts/run_ssl_pretrain.py`. Run in BIT's env with cwd `external/bit`:

```python
data, manifest = ssl_data.load_corpus(corpus_dir)                 # {"train": [...], "val": [...]}
to_idx, channels = ssl_data.bit_stitch_tables(manifest, "tx1")
import utils.datasets as D, models.trainer as T
D.DATASET_TO_IDX.clear(); D.DATASET_TO_IDX.update(to_idx)         # in place: models.ndt holds the same objects
D.DATASET_CHANNELS.clear(); D.DATASET_CHANNELS.update(channels)
class MultiSpikingDataset(D.SpikingDataset):                      # per-record read-in index
    def __getitem__(self, i):
        out = super().__getitem__(i); out["dataset_name"] = np.asarray(out.pop("stitch_idx")); return out
T.NAME2DATASET["ssl_multi"] = MultiSpikingDataset
cfg = update_config(default_trainer_config(), <dict below>)
Trainer(cfg, dataset=data, metric_fns={"r2_masked": masked_r2}, extra_model_kwargs={"features": "tx1"}).train()
```

The config dict comes from `ssl_human.yaml: pretrain`:

```yaml
seed: 42
savestring: ssl-ndt-human-tx                 # checkpoint dir name
log_to_wandb: false
dirs: {checkpoint_dir: /workspace/KunzDecoder/checkpoints/bit_ssl, log_dir: /workspace/KunzDecoder/logs/bit_ssl}
training: {num_epochs: 400, train_batch_size: 64, test_batch_size: 64, eval_every: 1, save_every: 50,
           shuffle_test_dataloader: false, drop_last_train_dataloader: false, mixed_precision: bf16}
model: include:configs/models/encoder/ndt.yaml   # unchanged; masker active, ratio 0.5, max_timespan 15, max_F 2500
data: {dataset_class: ssl_multi, train_name: train, val_name: val, train_len: null, val_len: null}
method:
  model_kwargs: {method_name: ssl, loss: mse}
  dataset_kwargs: {dataset_name: b2t25_T15}      # only to satisfy SpikingDataset.__init__; overridden per record
  metric_kwargs: {metric_name: r2_masked}
  dataloader_kwargs:
    pad_dict:
      spikes:           [{dim: 0, side: right, value: -100, truncate: null, min_length: null, max_length: null}]
      spikes_mask:      [{dim: 0, side: right, value: 0,    truncate: null, min_length: null, max_length: null}]
      spikes_timestamp: [{dim: 0, side: right, value: 0,    truncate: null, min_length: null, max_length: null}]
optimizer: {gradient_accumulation_steps: 1, lr: 5.e-4, wd: 1.e-5, eps: 1.e-8, scheduler: cosine,
            warmup_pct: 0.0, div_factor: 25}
```

Estimated size: ~70.6 h training data in ~22.7k sequences at batch 64 → ~360 steps per epoch, ~144k steps over
400 epochs. Measure throughput on 1 epoch before committing the GPU.

Selection metric (`r2_masked`, proposal): for each val group (`b2t24_T12`, `b2t25_T15`), take R² over **masked,
unpadded** bins and all channels, accumulating SS_res/SS_tot over the whole val set. Average the two groups.
Eval masks come from a fixed seed. BIT's built-in `r2` is also logged, for comparability with the paper.

---

## 4. The corpus (`data/processed/ssl_human/`)

### 4.1 Sources and splits

| Dataset | Subject | Stitch group | Input | Train | Val (R² selection) | Never read as neural data |
|---|---|---|---|---|---|---|
| B2T'25 (Card 2024) | T15 | `b2t25_T15` (256 ch) | `input_features[:, 0:256]` (released, already normalized TX) | all `data_train` trials: **8,072 trials, 39.23 h** | BIT `split_val` half of `data_val`: **713 of 1,426** trials (~3.7 h) | `data_test` (no labels); the other `data_val` half (= BIT "test") |
| B2T'24 (Willett 2023) | T12 | `b2t24_T12` (128 ch) | `tx1[:, 0:128]` (area 6v, −3.5 RMS counts) | all `train/` trials: **8,800 trials, 15.38 h** | BIT `split_val` half of `test/`: **440 of 880** (~0.7 h) | `competitionHoldOut/`; the other `test/` half |
| Kunz 2025 | T12, T15, T16, T17 | `kunz_T12` (128), `kunz_T15/16/17` (256) | `binnedTX`, summed to 20 ms within blocks | everything below | none (by design) | `*_allBehaviors_raw.mat` (duplicates); blocks also in B2T'25 |

The B2T counts and hours come from the local copies (`data/t15_copyTask_neuralData.zip`, `data/raw/willett_2023_text/`),
which arrived during this step. The adapters were checked on real files: shapes, dtypes and the block metadata scan.
The full B2T corpus build has **not** been run yet. The `val` halves are reproduced exactly: `tests/test_ssl_data.py`
runs BIT's own `load_data` on synthetic files and asserts identical train/val membership and order, for both the
directory and the zip / tar.gz inputs. It also asserts that the competition holdout is never opened (a corrupt holdout
file doesn't break our loader).

### 4.2 Kunz details and de-duplication

- **Collections:** isolated per-behavior files (28), interleaved (4), `sentenceData.zip` members (15, read directly from
  the zip with `zipfile` + `scipy.io.loadmat`; no extraction).
- **Rebinning:** 10 ms files sum adjacent pairs starting at each block's first bin. A trailing odd bin is dropped. Never
  across blocks.
- **Mode `blocks` (default):** each recorded block is a continuous stream, including delay, go and inter-trial periods.
  It is cut into ⌈L/500⌉ near-equal chunks (250–500 bins = 5–10 s). Mode `trials` ([delay onset, go end)) is also
  implemented.
- **Duplicate of B2T'25 (found here):** `sentenceData/t15.2024.03.03_attempted_50words.mat` blocks 3/6/11/13 **are**
  B2T'25 `t15.2024.03.03/data_train.hdf5` blocks 3/6/11/13. The first trials have the same sentences, and the go
  lengths are exactly 2× at 10 ms (1114 vs 557 bins). Those blocks are dropped from Kunz: the static list in the
  config, re-derived from the B2T'25 files whenever they are present. That file contributes 0 segments.
- **Channel order (found here):** per-channel TX correlations between that Kunz trial (rebinned) and the matching
  B2T'25 trial put the maximum on the same channel index for **256/256 channels** (mean r = 0.50 vs 0.006
  off-diagonal). Kunz T15 and B2T'25 T15 share the electrode order. T12 (Kunz vs B2T'24) can't be checked:
  there are no shared trials (2022 vs 2023–24).
- **Same day, different blocks:** Kunz T15 interleaved 2024.06.14 (blocks 3/4/5/7/8) is the same day as B2T'25
  2024.06.14 (train 1/9/11, val+test 12). No trials are shared. It is kept (Q6).
- **Skipped:** `t16.2024.04.29_imagined_50words_eval.mat` has `binSize` = **33 ms**, which can't be summed to 20 ms
  (≈ 12 min, Q7).
- **Within Kunz,** a (participant, day, block) seen in an earlier file is dropped from later files. None occurred in
  the real data. T17 isolated (blocks 1–14) and interleaved (16–19) share 2024-12-09 but have distinct blocks.

### 4.3 Normalization ("z-scored across days")

Implemented: per feature, z-score **per (dataset, participant, day)**, using that day's **training** bins. B2T val
trials therefore use their day's train statistics. Every Kunz bin is train. This is BIT's `zscore_day` for B2T'24
without the extra per-block step. It is applied uniformly, including the already-normalized B2T'25 release, where it
is a small affine correction. The sd floor is 0.01 and there is no clipping. Alternatives already in the code:
`scope: block` (self-normalize each block) and `scope: subject` (one set of statistics per group, across all days).
Statistics are saved per group in `norm_<dataset>_<subject>.npz` and summarized in `manifest.json`.

### 4.4 Local build (Kunz only; `python scripts/build_ssl_corpus.py --only kunz`, 12 s)

| Stitch group | Days | Segments (≤ 10 s) | Hours | Max abs z | Fraction abs z > 10 |
|---|---:|---:|---:|---:|---:|
| `kunz_T12` (128 ch) | 3 | 1,406 | 3.81 | 42.1 | 1.0e-5 |
| `kunz_T15` (256 ch) | 5 | 2,412 | 6.67 | 62.6 | 1.8e-4 |
| `kunz_T16` (256 ch) | 4 | 1,253 | 3.41 | 136.1 | 4.9e-4 |
| `kunz_T17` (256 ch) | 1 | 758 | 2.10 | 29.5 | 1.4e-5 |
| **Total** | 13 | **5,829** | **15.98** | | |

By collection: isolated 5.85 h, interleaved 2.93 h, sentences 7.20 h. The isolated and interleaved figures match
the inventory's 5.85 / 2.93 h. Per-day z-scores were checked after reading back: |mean| ≤ 4e-4, sd 0.998–1.002.
Output is 0.8 GB (lzf HDF5).

**Expected full corpus:** about **70.6 h train** (B2T'25 39.2 + B2T'24 15.4 + Kunz 16.0), ~22.7k sequences.
Val is ~1,150 trials. For comparison, BIT reports ~98 human hours, which includes the holdout splits and other
datasets.

---

## 5. Decisions made vs open questions

**Made (all in `configs/ssl_human.yaml`; easy to change):**

TX only, 20 ms bins. One read-in per (dataset, participant), 6 groups. B2T val = BIT's selection half only; the
other half and the competition test/holdout are never read. Kunz uses whole blocks in ≤ 10 s chunks. Kunz
duplicates of B2T'25 are dropped. Per-day z-score with train statistics. `min_bins` 75 (the masker's minimum;
drops no real B2T trial). `max_bins` 2500 (= `max_F`; cuts no real trial). Seed 42.

**Open, for you:**

1. **Q1. Kunz data is in pretraining *and* in our downstream evaluation (transductive).** Phases 1–3 evaluate on
   these same isolated/interleaved trials. Pretraining on them is label-free, but it is test-domain exposure
   (BIT §A.4 allows it; you excluded it for the competitions). Options: (a) keep all Kunz (current default);
   (b) a strict variant without the isolated + interleaved collections (set `isolated: null`,
   `interleaved: null`; −8.8 h); (c) hold out only the evaluation days. *Recommendation:* (b) as the main
   pretraining for any Kunz word-decoding claim, and (a) as an ablation.
2. **Q2. Normalization.** Per-day (default) vs per-block (`block`, absorbs within-day drift; in Kunz isolated data a
   block = one behavior, though that only matters downstream) vs one set of statistics across all days
   (`subject`). Also: should the already-normalized B2T'25 release be re-z-scored? Default yes (uniform), but BIT's
   own B2T'25 fine-tune feeds it un-re-normalized.
3. **Q3. Outliers.** Near-silent channels give |z| up to 136 (T16). That is 5e-4 of values for T16 and ≤ 2e-4
   elsewhere. MSE is sensitive to these. Options: leave as is (BIT does the same on B2T'24), `clip: 10`, or a
   higher `min_sd`.
4. **Q4. Kunz blocks vs trials.** Default: whole blocks in 10 s chunks. This adds the inter-trial periods:
   15.98 h vs 13.69 h for `mode: trials` (delay onset → go end, measured), and the chunks are fixed-length, so
   there is no padding. The chunk length (500) is a free choice. B2T trials are median 5.9 s (T12) and 16.9 s
   (T15).
5. **Q5. Stitching.** Separate read-ins for Kunz T15 vs B2T'25 T15 (default), or shared per participant
   (`stitch_by: subject`)? The T15 channel order is verified identical, but thresholds (and B2T'25's rolling
   normalization) differ. T12 is unverified.
6. **Q6. Same-day exposure.** Kunz T15 2024.06.14 shares a day (different blocks) with B2T'25 val/test block 12.
   Keep (default) or drop that file?
7. **Q7.** The 33 ms file (T16 2024.04.29 eval, ~12 min): skip (default) or resample?
8. **Q8. More Kunz data.** `chittychittybangbang`, `conjunctiveCounting` and the 6 `seqRecall*` archives (all local,
   human, `binnedTX` + `blockNum`) aren't included because they weren't on the requested list. The same adapter
   would read them. Hours not measured; the seqRecall README says block means were already subtracted.
9. **Q9. Downstream compatibility.** TX-only pretraining vs BIT's `--features all` (TX+SBP) decoding. The pretrained
   read-in can't be loaded there (shape mismatch); only the transformer core transfers. Options: fine-tune TX-only,
   or re-initialize the read-in for TX+SBP and load the core.
10. **Q10. Selection metric.** Masked-bin R² per group (proposal) vs BIT's all-bin batch-mean R². Also: equal
    weight for T12 and T15?

---

## 6. Implementation notes (pretraining wrapper; CPU smoke tests only, nothing pretrained yet)

Tier 1 (DESIGN.md §0): this is the shared SSL step that every arm starts from.

| File | Role |
|---|---|
| `src/bit_experiments/bit_ssl.py` | Stitch-table patch, per-record dataset, fixed-mask R², training loop, checkpoint export / `load_core` |
| `scripts/pretrain_ssl.py` | CLI: `--config --corpus --out --seed --device auto --max-epochs N --limit-records N` |
| `tests/test_bit_ssl.py` | 6 CPU tests on a tiny synthetic corpus built by `ssl_data.build_corpus` (smooth low-rank latents) |

### 6.1 What is BIT's and what is ours

**Used unchanged from BIT:** `NDT` with `method_name: ssl` / `loss: mse` (masker, summed MSE on masked unpadded bins,
per-group read-in/out), `configs/models/encoder/ndt.yaml` merged with `pretrain.model`, `SpikingDataset`,
`GroupBatchSampler`, `pad_collate_fn`, `update_config` / `default_trainer_config`, `NDT.save_checkpoint`.

- **Stitch tables** (`patch_stitch_tables`): `DATASET_TO_IDX` is cleared and refilled in place, and so is
  `DATASET_CHANNELS[feature]`. The other features' tables (`all`, `spikePow`) are kept. `models.ndt` holds the same
  object (asserted). The patch raises if `b2t25_T15` is not at slot 0 or `b2t24_T12` is not at slot 1 (BIT's fixed
  `brandman_2024_text` / `willett_2023_text` slots).
- **Dataset** (`multi_spiking_dataset`): `SpikingDataset` items with `dataset_name = record["stitch_idx"]`. Records are
  copied shallowly, not deep-copied: `GroupBatchSampler` reads every item once when it is built, which would
  otherwise copy the whole corpus (~11 GB, estimated), and nothing downstream writes into the arrays. Batches come from BIT's
  `GroupBatchSampler` (one stitch group per batch, seeded), so `ReadIn`'s `torch.unique(...).item()` holds.
- **Selection metric** (`evaluate`): masks come from BIT's own masker algorithm (`models/ndt.py:249-277`, copied as
  `_bit_masker`), but drawn from a CPU `torch.Generator` that is reseeded from (`eval_mask_seed`, group, batch
  position). Val batches are in fixed order (`shuffle: false`). So the masks are identical every epoch and do not
  depend on the global RNG or the device. Per group, over the bins the SSL loss uses:
  `r2_masked_<g> = 1 − Σ(y−ŷ)² / Σ_c Σ_b (y − ȳ_c)²` (variance-weighted over channels, pooled over the whole val
  half), plus `mse_masked_<g>`. `r2_masked` is the equal-weight mean over `selection.val_groups`
  (`b2t24_T12`, `b2t25_T15`). BEST = maximum of `r2_masked`.
- **BIT's own metric**, logged for reference as `bit_r2` / `bit_r2_<g>`: this is the value of `eval_neurons_metric`
  (pooled r2_score over all bins, padding zeroed, top-1000 channels), averaged over batches as BIT's Trainer does.
  It uses the same fixed-mask forward passes. BIT computes this identical pooled score once per channel on the CPU
  and averages the copies (256× the work). We compute it once, in float64 on the device. A test checks that the
  value equals BIT's function (|Δ| < 1e-5).

### 6.2 Why not BIT's `Trainer`

`SSLPretrainer` re-implements `Trainer.train` (`models/trainer.py:229-369`) for one device:

- the same AdamW(lr, wd, eps) over all parameters;
- the same `OneCycleLR` (max_lr = lr, pct_start = warmup_pct, div_factor, total = num_epochs × batches), stepped
  every batch;
- the same summed loss, backward per batch, `torch.autocast(bf16)` forward, and eval / BEST / `EPOCH<n>` / LAST logic.

BIT's Trainer is not driven directly, for four reasons:

1. `models/trainer.py` imports deepspeed, wandb, accelerate and `models.bci` (peft, Qwen2-Audio) at import time,
   although NDT SSL needs none of them. That ties the run to the LLM environment and rules out CPU tests in `.venv`.
2. Its selection metric is the mean of per-batch `metric_fns` values. A pooled-per-group R² needs to accumulate over
   the whole val set, so `evaluate` would have to be overridden.
3. Its eval masks come from the global RNG and change every epoch.
4. The per-epoch train loss exists only as a local variable in `train()` (it goes to stdout / wandb). Logging it
   would mean copying `train()` as well.

With both `evaluate` and `train` replaced, a subclass would just be a copy.

**Other deviations:**

- Eval inputs stay float32 under autocast. BIT casts them to bf16 (`trainer.py:389-391`), which also quantizes the
  targets.
- bf16 autocast runs only on CUDA. CPU runs use fp32. BIT's attention runs in bf16 either way
  (`model_utils.py:136-137`).
- Only `gradient_accumulation_steps: 1` and `scheduler: cosine` are supported (the Table-10 run uses exactly these).
- No wandb and no raster plots.
- `--max-epochs` replaces `num_epochs`, so the cosine schedule spans the shortened run.

### 6.3 Checkpoint format and fine-tuning

Each of `BEST/`, `LAST/` and `EPOCH<n>/` contains BIT's files:

- `encoder.bin`, `encoder_config.pth`, `decoder.bin` (from `NDT.save_checkpoint`);
- `trainer_config.pth`;
- `model.safetensors`: the whole NDT state dict, i.e. the model file of `accelerator.save_state`, which
  `BCI_AudioLLM.load_safetensors` reads (`models/bci.py:161-186`);
- `epoch.pth`.

Ours on top: `stitch_tables.json` (the patched tables, needed to rebuild the model), `metrics.json`, and in LAST
`optimizer.bin` / `scheduler.bin`. All `.pth` / `.bin` files load with `torch.load(weights_only=True)`.

**`BEST_core/`** is rewritten every time BEST improves. It is BEST with every `embedder.read_in.*` and `read_out.*`
tensor removed (from `encoder.bin`, `decoder.bin` and `model.safetensors`); `core.json` lists the removed keys.

How BIT's loading behaves (each case is verified in `test_checkpoint_loads_with_bit_from_pt`):

- **TX-only fine-tune** (`--features tx1`, BIT's own tables): `--ft_ckpt <out>/BEST` loads every core tensor and read-ins
  0 / 1 (b2t25_T15 → brandman slot, b2t24_T12 → willett slot). Our extra read-in (`kunz_T12`, slot 2) is only an
  unexpected key under `strict=False`.
- **TX+SBP fine-tune** (`--features all`, 512 / 256 ch): NDT's `from_pt` (`models/ndt.py:500-513`) **raises**
  `RuntimeError: size mismatch` on `BEST`, because `strict=False` does not cover shape mismatches. Use
  `--ft_ckpt <out>/BEST_core` instead. BIT's unmodified loader then loads the transformer core plus `mask_token`
  and leaves the read-ins and read-outs freshly initialized (they are missing keys).
- **In our own code:** `bit_ssl.load_core(model, <out>/BEST)` loads every tensor whose shape matches. A read-in slot
  is loaded whole or not at all (a slot with any mismatching tensor keeps its initialization). The function reports
  `reinit_read_in`, `unexpected`, `skipped_shape` and `missing`. `exclude_read_in=True` re-initializes all read-ins.
- **Caveat (BIT behavior, not changed here):** `from_pt` also merges `encoder_config.pth` over the fine-tuning encoder
  config (`ndt.py:502-503`). The pretraining masker settings (`active: true`, ratio 0.5) therefore carry over, and
  the CTC path masks during fine-tuning training (`force_mask = self.training`). `BCI_AudioLLM` switches masking
  off explicitly (`bci.py:154-159`); a CTC arm would need to do the same.

### 6.4 Smoke tests (CPU, this Mac)

`pytest tests`: 37 passed, 1 skipped. That includes the 6 new tests and every existing one.

**Tiny model** (hidden 32, 2 layers, batch 4, lr 5e-3, 2 epochs, 58 train records): masked train MSE with fixed
masks fell in every group.

| Group | Before | After |
|---|---:|---:|
| b2t25_T15 | 1.589 | 1.461 |
| b2t24_T12 | 1.986 | 1.877 |
| kunz_T12 | 2.009 | 1.920 |

Initial MSE is about 2 because BIT's read-out ends in a LayerNorm over the patch, so predictions start at unit
variance. Val R² is still negative after these 30 steps.

**Full-size Table-10 model** through the CLI (`--max-epochs 2 --limit-records 16`, CPU): 8.89M parameters,
**6.91M core** (paper: ~7M). About 0.5 s per batch of 16 short trials. Every checkpoint directory and both log
files were written.

None of these numbers says anything about real data.

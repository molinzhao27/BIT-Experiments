# Results

Design and decisions: `docs/DESIGN.md`. All metrics: `results/metrics.csv`.

## 4a. BIT baseline reproduction (BIT's released code, as documented), done 2026-10-19

Code: BIT @ `f107ae9` (`external/bit`), run by `scripts/run_bit_baseline.py` with
`configs/bit_baseline.yaml` (overrides are paths, no W&B, and the paper's MLP projector, `inter_size` 1024).
Data: B2T '25 (T15). BIT splits the official val set in half (seed 42): **BIT-val** (713 trials) and **BIT-test** (713).
The holdout set has no labels and is never used.
LLM: `lmms-lab/Aero-1-Audio` @ `2efba6e` (Qwen2.5-1.5B LM), LoRA r 8 / α 32 / dropout 0.2.

**Two properties of BIT's released code change what these numbers mean** (found 2026-10-19, after the run):
1. **The sentence stage does not load the phoneme encoder.** README step 2 passes the phoneme checkpoint with
   `--ft_ckpt`. `ConfigBuilder` writes it to `model.encoder.from_pt`, but `BCI_AudioLLM.load_neural_encoder` (and
   NDT) read `model.encoder.encoder.from_pt` (or `method.model_kwargs.load_ndt_from_pt`), which stay null. Evidence:
   - the config trace;
   - no "Load NDT from" line in the sentence log;
   - the trained sentence-stage encoder's weights are uncorrelated with the phoneme encoder's (|r| < 0.01 for
     every matrix checked).

   So the end-to-end model below never saw phoneme supervision: it is a **randomly initialized encoder trained
   directly on sentences** (CE + BIT's in-batch contrastive term, batch 8), i.e. arm B from random init.
2. **The default evaluation is teacher-forced.** `eval_llm.py` without `--nbest` runs one forward pass over prompt +
   reference and takes the argmax at every position, so each word is predicted from the true preceding words. Its
   greedy / ≤20-token settings are never used. 97% of decodes have exactly the reference's word count. BIT's
   checkpoint selection uses the same teacher-forced WER.

| Split | PER (separate phoneme model) | Teacher-forced WER (end-to-end) | Exact sentences | Seen in train (n=81) | Unseen (n=632) |
|---|---|---|---|---|---|
| BIT-val | 8.91% | 19.06% | 35.2% | 4.6% | 20.9% |
| BIT-test | 9.01% | 20.21% | 33.8% | 2.2% | 22.5% |

- **The PER reproduces the paper.** Table 4 gives BIT-TFS on T15 as 8.87% (val PER); we get 8.91% on BIT-val.
- **The WER is not a decoding result.** Free-generation WER for this checkpoint will be measured by the Stage 2
  evaluator (`train_stage2.py --eval-only`). The paper's end-to-end number (11.06%, T15 holdout, Tables 2 and 8b)
  uses the SSL encoder, nucleus sampling and the competition holdout, and is not comparable.
- **A phoneme-first model as the paper describes** (phoneme encoder actually loaded) is arm A, trained by the
  Stage 2 runner, which loads every initialization explicitly and verifies it.
- **Sentence memorization.** 11.4% of BIT-test trials repeat a training sentence; they are decoded almost perfectly
  (2.2%). Every arm reports seen and unseen sentences separately.

Compute (RTX PRO 6000 Blackwell, 96 GB, one job at a time):

| Stage | Wall time | Peak GPU memory |
|---|---|---|
| Phoneme CTC, 800 epochs | 5.9 h | 8.0 GB |
| Sentence (LLM + LoRA), 150 epochs | 7.8 h | 94.8 GB (reported by nvidia-smi) |
| Evaluation (val + test) | 0.05 h | 11.1 GB |

Outputs: `results/bit_baseline/decodes_{val,test}.csv` (trial, reference, decoded, word errors) and
`results/bit_baseline/summary.json`.
Selection follows BIT (teacher-forced val WER). The arms in 4c select by free-generation WER on BIT-val instead.

## 4b. SSL pretraining on human data (threshold crossings only), done 2026-10-19

Code: `src/bit_experiments/bit_ssl.py` + `scripts/pretrain_ssl.py` (BIT's NDT masked modeling, Table 10 settings:
400 epochs, batch 64, lr 5e-4 cosine, 50% of 100 ms patches masked in spans of up to 1.5 s, MSE on masked bins).
Config: `configs/ssl_human.yaml`; decisions and evidence: `docs/DESIGN.md` §7.

Corpus (`data/processed/ssl_human`, speech-sentence recordings only, threshold crossings (TX) only):

| Source | Train hours | Notes |
|---|---|---|
| B2T '25 (T15) | 39.2 | official train split; used as released (already z-scored, clipped at ±10) |
| B2T '24 (T12) | 15.4 | official train split; z-scored per block, then per day; clipped at ±10 |
| Kunz sentence sessions (T12, T15, T16) | 5.4 | T15 shares the B2T '25 read-in; blocks duplicated in B2T '25 excluded |
| **Total** | **60.0** (18,846 sequences) | val: 4.3 h (1,153 sequences) from the official B2T val splits |

No holdout data and no phoneme labels are used. The val sequences are BIT's val halves (BIT-val), never BIT-test.

Result: checkpoint `checkpoints/ssl_human/BEST` (epoch 363 of 400; `BEST_core` = the same without read-in/read-out),
selected by masked-bin val R², equal weight T12 / T15, fixed eval masks. Figure: `figures/ssl_human_curve.png`.

| Masked-bin val R² (same bins for every row) | T15 | T12 |
|---|---|---|
| SSL encoder (BEST), BIT's input path (as trained and selected; **leaks**, see below) | 0.052 | 0.050 |
| SSL encoder (BEST), masked raw bins zeroed before smoothing (**leak-free**) | **0.042** | **0.033** |
| SSL encoder (BEST), other leak-free inputs: masked bins resampled / smoothing over visible bins only | 0.040 / 0.041 | 0.029 / 0.031 |
| No learning: each channel's mean over the trial's unmasked bins | −0.001 | −0.015 |
| No learning: mean of the nearest unmasked 100 ms on each side of the gap | −0.123 | −0.154 |
| BIT's own R² metric (all bins, top-1000 channels), for reference | 0.046 | 0.059 |

Reading: the target is a single raw 20 ms bin of threshold crossings, most of whose variance is spiking noise, so the
attainable R² is small. Without the leak, the encoder explains about 4% (T15) / 3% (T12) of it, while copying
nearby activity makes predictions worse than the mean (`results/ssl_baselines.json`,
`scripts/ssl_baselines.py`). The curve is still rising very slowly at the end (mean 0.0508 at epoch 363 vs
0.0479 at epoch 150, both with the leak), so a longer run would add little.
Whether the encoder helps decoding is tested downstream in 4c: SSL vs random initialization for arms A and C.

**Smoothing leak (found in an audit, re-measured 2026-10-19).** BIT's NDT Gaussian-smooths the raw input
(sd 2 bins, ±6 bins) before patching and masking. So the visible bins next to a masked span carry 18 / 12 / 6.5 / 2.7%
of the masked raw values at offsets 1–4. The no-learning rows see raw unmasked bins only.
`scripts/ssl_leakfree_eval.py` re-scores BEST without retraining. It uses the same masks, bins and R². The
masked raw bins are removed before smoothing in three ways: zeroed, replaced by random visible bins of the same trial,
or smoothing renormalized over the visible bins only. The model then has the same information as the no-learning
rows. Results: `results/ssl_leakfree.json`.
- **Size.** The leak accounts for 0.010–0.012 of the T15 R² (~20%) and 0.017–0.020 of the T12 R² (~35–40%). The
  selection metric (mean of the two) drops from 0.051 to 0.034–0.037.
- **Most of the R² is real.** About 76–77% of masked bins are ≥ 7 bins from any visible bin, where no input can leak.
  On them, BIT's own path scores 0.040 (T15) / 0.034 (T12), and the leak-free inputs score 0.038–0.039 / 0.030–0.031. On
  the remaining edge bins, R² falls from 0.089 / 0.104 to 0.046–0.052 / 0.027–0.037.
- **The leak could have explained everything.** A least-squares reader of the exact leaked signal at the nearest
  visible bin on each side, plus the trial mean, reaches 0.056 / 0.045. The reported number alone therefore did not
  show learning; the leak-free numbers do.
- **Caveat.** The model was trained on leaky inputs, so leak-free inputs are slightly out of distribution for it
  (the interior bins also drop by 0.002–0.004). A model trained without the leak might score a little higher.
  BEST was selected with the leaky metric.
- **Downstream comparisons are unaffected.** SSL and random initialization in 4c use the same smoothing.
- **Padding (SSL_PLAN.md §1.3).** Every fixed-mask eval batch mixes lengths: 12/12 T15 batches (297–2,169 bins)
  and 7/7 T12 batches (128–826 bins). 701/713 and 432/440 trials are shorter than their batch maximum, so smoothing
  pulls their last 6 input bins toward −100 (0.7% / 2.0% of valid input bins). Smoothing each trial as if it were
  alone changes R² by ≤ 0.0002 (0.0520 → 0.0520, 0.0496 → 0.0498), which is negligible here.

Compute: 2.6 h on the GPU (23.5 s per epoch), peak 6.4 GB allocated; 9.9 M parameters (6.9 M transformer core).

## 4c. Arms (in progress)

Fixes that apply to every arm (`src/bit_experiments/stage2.py`, `scripts/train_stage2.py`):
- each arm's starting checkpoint is loaded explicitly and verified tensor by tensor;
- checkpoints are selected by free-generation WER on BIT-val;
- every result uses free generation (greedy and nucleus) plus null controls on BIT-test;
- BEST is slimmed to everything except the frozen base LLM (94 MB instead of 8.5 GB).

Arm A from SSL fine-tunes from `checkpoints/ssl_human/BEST_core_ft`, with the SSL masker off as in the
from-scratch run. BIT's `from_pt` would otherwise merge in the SSL encoder config and mask 50% of the input
during CTC training.

**Arm C, Stage 1** (contrastive alignment, no LLM, no phonemes; SSL core init, layer-14 target, batch 1024 by
GradCache, 51 min): best epoch 84 by val top-1.

| Sentence retrieval (rank the trial's sentence among all distinct sentences of the split) | Top-1 | Top-5 | Candidates |
|---|---|---|---|
| BIT-val, all trials | 36.2% | 64.1% | 706 |
| BIT-val, sentences unseen in training | 36.9% | 64.4% | 706 |
| BIT-test (report only), all trials | 35.8% | 65.2% | 704 |
| BIT-test (report only), unseen in training | 36.1% | 65.7% | 704 |

Chance top-1 is 0.14%. Retrieval is no better on sentences seen in training, so the alignment generalizes to new
sentences rather than memorizing training ones.
Log: `checkpoints/stage1/C_l14_b1024_s42/log.jsonl`.

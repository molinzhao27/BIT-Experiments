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

## 4d. Sentence decoding (arm B, Stage 2), done 2026-10-25

Free-generation greedy WER on BIT-test, selected on BIT-val. Every run starts from the same
SSL-pretrained encoder (`checkpoints/ssl_human/BEST_core`) with no phoneme stage. All results
are one seed (42) unless noted.

### WER

| Run name | LLM | Objective | Test WER | Val WER | Exact | Seen / unseen |
|---|---|---|---|---|---|---|
| B_ssl_aero-1-audio-diffusion_s42 | Aero-1-Audio | block diffusion | **21.76%** | 20.77% | 35.1% | 10.4 / 23.2 |
| B_ssl_qwen25-1.5b-instruct-diffusion_s42 | Qwen2.5-1.5B-Instruct | block diffusion | 23.65% | 22.97% | 30.0% | 15.4 / 24.7 |
| B_ssl_fast-dllm-v2-1.5b_s42 | Fast-dLLM v2 1.5B | block diffusion | 25.22% | 24.43% | 26.4% | 15.2 / 26.5 |
| B_rand_bit4a | Aero-1-Audio | autoregressive | 33.6% | 33.5% | — | 11.5 / 36.4 |
| B_ssl_s42 | Aero-1-Audio | autoregressive | 34.23% | 31.64% | 32.5% | 19.7 / 36.1 |
| B_ssl_qwen25-1.5b-noprompt_s42 | Qwen2.5-1.5B base, no prompt | autoregressive | 43.75% | — | — | 21.3 / 46.6 |
| B_ssl_qwen25-1.5b-instruct_s42 | Qwen2.5-1.5B-Instruct | autoregressive | 55.66% | — | — | 26.7 / 59.3 |

`B_rand_bit4a` is BIT's released checkpoint (4a), re-evaluated with free generation instead of
teacher forcing.

**Null controls.** Shuffled-input, noise and prompt-only decodes give 99–142% WER across
all runs (stored in each run's `summary.json`), confirming the output depends on the neural input.

Outputs: `results/stage2/<run>/summary.json`, `results/per_trial/`.

### Comparisons (paired trial bootstrap, 95% CI on BIT-test)

Training objective, same LLM (Aero):
- Diffusion − AR: −12.47 pp [−14.52, −10.54]

LLM swaps under diffusion:
- Aero diffusion − Instruct diffusion: −1.89 [−2.95, −0.83]
- Aero diffusion − Fast-dLLM v2: −3.46 [−4.58, −2.33]

Aero's audio post-training helps ~21 pp under autoregressive training (34.2 vs 55.7) but only
~1.9 pp under diffusion, so the objective absorbs most of the LLM advantage.

**Decoding order barely matters.** Strict left-to-right decoding of the Fast-dLLM v2 diffusion
model costs only +1.2 pp on test (not significant on val), so the gain is from training, not from
the non-autoregressive generation order.
Details: `results/stage2/B_ssl_fast-dllm-v2-1.5b_s42/decoding_order/`.

### Error analysis

Scripts: `scripts/phonetic_error_analysis.py`, `scripts/bertscore_compare.py`,
`scripts/ablation_error_types.py` (all CPU).

**Sound-alike errors scale with decoder quality.** The share of in-dictionary substitutions
that are phonemically similar (≥ 0.5) to the reference word, and the phoneme-similarity excess
over length-matched null substitutions:

| Run | Sound-alike obs / null | Phoneme excess [95% CI] |
|---|---|---|
| Aero + diffusion | 25.9% / 2.7% | +0.200 [+0.185, +0.217] |
| Qwen Instruct + diffusion | 25.6% / 2.7% | +0.203 [+0.189, +0.219] |
| Fast-dLLM v2 (diffusion) | 23.6% / 2.7% | +0.192 [+0.178, +0.206] |
| Aero AR | 16.1% / 2.7% | +0.118 [+0.105, +0.131] |
| Qwen base, no prompt | 10.4% / 2.5% | +0.074 [+0.065, +0.084] |
| Qwen Instruct | 6.9% / 2.5% | +0.048 [+0.040, +0.056] |

The signal is from speech-motor cortex, so diffusion decoders — which, on our reading, lean more on the
neural signal and less on the language model's prior — make errors that sound like the target.
Shuffled-input controls show zero excess.
Details: `results/stage2/phonetic_errors.md`, `results/stage2/phonetic_errors/tables.md`.

**BERTScore** (roberta-large L17, rescaled; `scripts/bertscore_compare.py`). Aero diffusion vs
Aero AR on BIT-test:
- Mean F1: 0.682 vs 0.633, +0.049 [+0.031, +0.069]
- Exact sentences: 35.1% vs 32.5%
- Derailed (sentence WER ≥ 100%): 2.0% vs 9.3%

Of the WER gap between them, 67% comes from sentences where Aero AR's sentence WER is 50–100%,
28% from sentences at ≤ 50%, and 20% from sentences where it derails past 100%; diffusion gives
back 15% on sentences Aero AR got exactly right. Reading: autoregressive decoding is all-or-nothing; diffusion fails
more gracefully.

**Matched meaning per error.** The raw "keeps the meaning" share is confounded by how many
errors each run makes. Matched on the same trial and the same number of word errors,
autoregressive errors keep slightly more meaning:

| Diffusion run | AR run | ΔF1 [95% CI] | n |
|---|---|---|---|
| Aero diffusion | Aero AR | −0.041 [−0.065, −0.018] | 131 |
| Instruct diffusion | Instruct AR | −0.038 [−0.069, −0.008] | 70 |
| Instruct diffusion | Aero AR | −0.052 [−0.075, −0.030] | 146 |
| Fast-dLLM v2 | Aero AR | −0.038 [−0.059, −0.018] | 157 |

Diffusion's errors lean on the signal (sound-based slips), while AR's lean on the language
prior (plausible words that keep more meaning per error).

**Ablation error types** (`results/stage2/ablation_error_types.md`). Removing input never makes
the wrong words more language-driven. Dropping one array cuts the sound-alike excess (−0.02 to
−0.07), leaves the related-meaning excess unchanged (±1.6 pp, n.s.), and keeps context fit at
baseline on mostly-right trials. With nearly all input gone, everything falls to the shuffled-
input level.

### Eval-time input ablations

Same weights, greedy decoding; only the neural input is changed at eval time. Full tables with
all ablations and phonetic columns: `results/stage2/ablations_summary.md`.

Key drop-one-area results (ΔWER in pp, test):

| Ablation | Fast-dLLM v2 | Instruct diffusion | Aero AR | Aero diffusion |
|---|---|---|---|---|
| drop v6v (ventral 6v) | +19.9 [+18.2, +21.6] | +23.3 [+21.6, +25.0] | +24.8 [+22.3, +27.4] | +21.9 [+20.2, +23.6] |
| drop 55b | +22.2 [+20.6, +23.9] | +22.0 [+20.5, +23.6] | +20.8 [+18.5, +23.0] | +21.0 [+19.5, +22.5] |
| drop d6v (dorsal 6v) | +2.1 [+1.3, +2.9] | +0.9 [+0.2, +1.6] | +0.8 [−0.5, +2.1] | +1.3 [+0.5, +2.0] |
| drop a4 (area 4) | +12.2 [+10.9, +13.4] | +11.2 [+10.0, +12.4] | +11.9 [+9.8, +14.0] | +10.6 [+9.4, +11.8] |

Ventral 6v and 55b carry most of the information; dorsal 6v barely matters. At matched damage,
dropping v6v reduces the sound-alike excess more than dropping 55b (phoneme excess 0.134 vs 0.162
for Fast-dLLM v2, CIs don't overlap), so v6v carries more of the phonetic detail.

### Comparison with BIT

BIT's end-to-end result (11.06% WER, arXiv 2511.21740v5) uses the competition holdout, nucleus
sampling, and the BIT-All encoder (366.8 hours of human + monkey pretraining). Ours uses BIT-test,
greedy decoding, and 60 hours of human pretraining. The remaining ~10 pp gap mixes the evaluation
split, the decoding method, and the pretraining scale. Arm A (phoneme-first) on our setup will
measure how much of the gap comes from the phoneme stage.

### Still running

- Arm A (SSL → phoneme CTC → sentence), autoregressive and block-diffusion decoders.
- Arm C (SSL → contrastive → sentence), both decoders.
- Additional seeds (43, 44) for arm B.
- A 50% training-data run.
- A phoneme-eraser retraining ablation (`docs/RETRAINING_ABLATIONS.md`).

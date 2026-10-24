# Do the wrong words turn into language-driven guesses when brain input is removed?

**In short: no.** Taking brain input away makes the decoders' wrong words less sound-alike, but no more likely to fit
the true sentence or to mean something similar. With nearly all input gone, both kinds of signal fall to the
shuffled-input level. At matched WER damage, dropping v6v and dropping 55b differ clearly only on the phonetic side; a
language-side hint for 55b on mostly-right trials is significant in one run of three. These ablations measure how much
the trained models lean on each input, not what the input encodes (the models never saw missing arrays in training).
There is one seed per run, and context fit is scored against the true sentence, so guesses that fit the decoder's own
wrong sentence go unseen.

Test split, 713 trials. Three runs, one seed each: run 3 (Fast-dLLM v2, block diffusion), the control (Qwen Instruct
+ the diffusion objective) and Aero (autoregressive). Each run has the 14 eval-only ablations from
`scripts/eval_ablations.py`, plus its shuffled-input decodes (`null_shuffled`, another trial's neural data) as a
reference row. `scripts/ablation_error_types.py` (`configs/ablation_error_types.yaml`) runs
`scripts/phonetic_error_analysis.py`'s measures on these decodes. No new metric is involved.

- **Excess** = observed − a length-matched null drawn from the same decode set's own substitutes.
- **ctx** = how well the wrong word fits the slot of the true sentence (masked-LM log-prob, nats).
- **related** = share related in meaning (sem ≥ 0.4).
- **CIs** come from one trial-level cluster bootstrap (5000 resamples, seed 0) shared by every run and condition, so
  every contrast is paired.
- **Baselines:** all three `none` decodes match the saved decodes exactly.
- **Mostly right** = each condition's own trials decoded with sentence WER ≤ 0.5, where the decoded sentence frame is
  still mostly correct.

Full tables (every contrast, the mostly-right slice per condition, sem, ctx > ref, phon beyond spelling):
`ablation_error_types/tables.md`.

## What it shows

1. **Removing one array doesn't make the wrong words more language-driven; they just become less sound-alike.**
   - Dropping v6v, 55b or a4 lowers the phon excess by 0.02–0.07.
   - The related-in-meaning excess doesn't move: every Δ is within ±1.6 pp, and every CI includes 0.
   - Context fit falls by 0.37–0.60 nats for v6v and 55b. On mostly-right trials, though, it holds at baseline after
     dropping v6v (+0.03, −0.24, +0.19 nats; none significant), while sound-alikeness still falls there. So most of
     the overall drop comes from whole sentences going wrong.
2. **Less input lowers every signal, and none rises.** No condition raises the ctx, related or sem excess above the
   unablated decodes, in either slice: no CI against `none` lies above 0.
   - With one array alone or half the trial (WER 56–98%), phon excess falls to 0.03–0.15 and ctx to 0.6–1.2 nats.
     Related falls less: +2 to +7 pp, against +5 to +8 unablated.
   - With nearly no usable input (only d6v, scrambled patches, reversed time; WER ≥ 100%), everything sits near the
     shuffled-input level: phon ≈ 0.01, ctx +0.2 to +0.3 nats (`null_shuffled` +0.17 to +0.20), related +0.8 to
     +1.5 pp (`null_shuffled` +0.5 to +1.0).
   - There the LLM still writes sentences, but its wrong words neither sound like the true words nor fit the true
     sentence. A prior with no input can't know this trial's context.
3. **By array, at matched damage (ΔWER drop_55b − drop_v6v: +2.4, −1.3, −4.0 pp), the clear difference is phonetic.**
   - Dropping v6v costs 0.027–0.028 more phon excess than dropping 55b in all three runs, with every CI excluding 0.
   - Context fit and relatedness differ by little or nothing: ctx −0.05 to −0.15 nats, related +0.1 to +0.4 pp, all
     n.s.
   - A hint on the language side: dropping 55b lowers context fit on mostly-right trials (−0.10, −0.45, −0.51 nats;
     significant in the control and Aero), where dropping v6v doesn't. But the direct 55b − v6v contrast there is
     significant only for Aero (−0.70 [−1.16, −0.26]).
   - The post-hoc only_55b − only_v6v contrast changes sign across runs (ctx +0.20, +0.03, −0.31 nats). On
     mostly-right trials, run 3's 55b-alone errors fit context and meaning better (ctx +0.82, related +7.2 pp), the
     control points the same way (n.s.), and Aero doesn't. So there is no consistent array-specific language signal.

## What it doesn't show

- **Reliance, not information content.** The models never saw a missing array or a cut-off trial in training. An
  ablation shows how much the trained decoder leans on that input, not what the input encodes. The retraining
  ablations (`docs/RETRAINING_ABLATIONS.md`) address the latter.
- **ctx scores the wrong word in the true sentence.** A guess that fits the decoder's own wrong sentence counts as a
  miss, so ctx can't see prior-driven guessing once the sentence frame is lost. The mostly-right slice is a partial
  fix only: it is a subset each condition selects for itself (0–806 pairs), not the same trials across conditions.
- **Each null is its own decode set's substitutes.** Ablated decoders draw on a different vocabulary of substitutes:
  the related null rises from 3–4% at baseline to about 8–9% at null level. So each excess compares a set with its own
  pool.
- **Other limits.**
  - One seed per run.
  - This is the error pattern only, not the §5 probes.
  - The only_55b − only_v6v contrast was added after seeing the first table.
  - One lexicon centres the embeddings for every decode set here. So sem and related differ slightly from
    `phonetic_errors.md` (sem ≤ 0.002, related ≤ 0.6 pp on the unablated rows); phon and ctx are identical.

## Per run × condition (slice all; 95% CIs)

| Condition | Run | WER | sound-alike [95% CI] / null | phon excess | ctx excess (nats) | related excess (pp) |
|---|---|---|---|---|---|---|
| none | Run 3 | 25.2% [23.6, 26.9] | 23.6% [21.1, 26.2] / 2.7% | +0.192 [+0.178, +0.206] | +1.90 [+1.71, +2.10] | +7.0 [+5.4, +8.6] |
|  | Control | 23.6% [22.1, 25.3] | 25.6% [23.1, 28.1] / 2.7% | +0.203 [+0.189, +0.219] | +2.02 [+1.81, +2.24] | +7.9 [+6.2, +9.7] |
|  | Aero | 34.2% [31.6, 36.9] | 16.1% [14.1, 18.2] / 2.7% | +0.118 [+0.105, +0.131] | +1.82 [+1.60, +2.05] | +5.4 [+3.9, +7.0] |
| drop_v6v | Run 3 | 45.1% [43.0, 47.2] | 17.6% [16.0, 19.4] / 3.3% | +0.134 [+0.123, +0.145] | +1.54 [+1.39, +1.69] | +7.2 [+5.9, +8.5] |
|  | Control | 47.0% [44.9, 49.1] | 18.2% [16.5, 20.1] / 3.4% | +0.135 [+0.124, +0.146] | +1.58 [+1.43, +1.73] | +8.1 [+6.7, +9.5] |
|  | Aero | 59.0% [56.4, 61.8] | 10.9% [9.5, 12.3] / 3.1% | +0.068 [+0.059, +0.077] | +1.29 [+1.14, +1.45] | +5.1 [+4.0, +6.2] |
| drop_a4 | Run 3 | 37.4% [35.4, 39.4] | 20.2% [18.2, 22.2] / 3.1% | +0.162 [+0.151, +0.174] | +1.76 [+1.60, +1.92] | +8.2 [+6.8, +9.7] |
|  | Control | 34.8% [33.0, 36.7] | 21.0% [18.8, 23.2] / 3.0% | +0.166 [+0.153, +0.177] | +1.90 [+1.73, +2.08] | +9.5 [+7.9, +11.2] |
|  | Aero | 46.1% [43.4, 48.9] | 12.3% [10.8, 14.0] / 2.8% | +0.087 [+0.077, +0.097] | +1.58 [+1.40, +1.76] | +4.9 [+3.7, +6.1] |
| drop_55b | Run 3 | 47.4% [45.5, 49.5] | 20.6% [18.8, 22.5] / 3.0% | +0.162 [+0.152, +0.172] | +1.43 [+1.30, +1.57] | +7.5 [+6.2, +8.8] |
|  | Control | 45.7% [43.7, 47.7] | 20.2% [18.5, 22.0] / 3.1% | +0.162 [+0.152, +0.172] | +1.43 [+1.30, +1.57] | +8.2 [+6.9, +9.5] |
|  | Aero | 55.0% [52.4, 57.6] | 13.3% [11.8, 14.9] / 2.8% | +0.095 [+0.085, +0.105] | +1.25 [+1.10, +1.39] | +5.1 [+4.0, +6.4] |
| drop_d6v | Run 3 | 27.3% [25.6, 29.0] | 24.1% [21.7, 26.5] / 2.6% | +0.200 [+0.187, +0.214] | +1.96 [+1.76, +2.18] | +8.3 [+6.8, +10.0] |
|  | Control | 24.5% [22.9, 26.2] | 24.6% [22.0, 27.2] / 2.7% | +0.198 [+0.184, +0.212] | +2.03 [+1.83, +2.23] | +7.9 [+6.2, +9.7] |
|  | Aero | 35.1% [32.5, 37.7] | 14.4% [12.5, 16.4] / 2.5% | +0.113 [+0.100, +0.126] | +1.83 [+1.64, +2.04] | +5.8 [+4.5, +7.2] |
| only_v6v | Run 3 | 70.9% [69.1, 72.7] | 13.2% [11.9, 14.4] / 3.5% | +0.101 [+0.093, +0.109] | +0.91 [+0.81, +1.02] | +4.8 [+3.8, +5.8] |
|  | Control | 69.3% [67.4, 71.2] | 13.1% [11.9, 14.4] / 3.3% | +0.101 [+0.093, +0.109] | +0.92 [+0.82, +1.03] | +4.7 [+3.8, +5.7] |
|  | Aero | 77.8% [75.5, 80.0] | 8.6% [7.5, 9.6] / 3.1% | +0.056 [+0.049, +0.063] | +0.89 [+0.78, +1.01] | +4.2 [+3.3, +5.2] |
| only_a4 | Run 3 | 89.0% [87.4, 90.7] | 9.0% [8.0, 10.0] / 3.2% | +0.056 [+0.050, +0.062] | +0.62 [+0.52, +0.72] | +4.1 [+3.2, +5.1] |
|  | Control | 93.4% [91.4, 95.5] | 9.2% [8.2, 10.2] / 3.2% | +0.056 [+0.049, +0.062] | +0.61 [+0.51, +0.71] | +3.6 [+2.7, +4.5] |
|  | Aero | 96.3% [94.5, 98.2] | 5.7% [4.9, 6.5] / 3.4% | +0.026 [+0.020, +0.031] | +0.61 [+0.51, +0.70] | +2.4 [+1.6, +3.2] |
| only_55b | Run 3 | 82.3% [80.7, 83.8] | 10.1% [9.0, 11.1] / 4.3% | +0.058 [+0.051, +0.064] | +1.11 [+1.00, +1.23] | +4.5 [+3.6, +5.5] |
|  | Control | 81.8% [80.3, 83.3] | 9.8% [8.8, 10.9] / 4.3% | +0.059 [+0.053, +0.066] | +0.96 [+0.85, +1.07] | +4.7 [+3.7, +5.7] |
|  | Aero | 97.5% [95.6, 99.5] | 5.9% [5.1, 6.7] / 3.5% | +0.027 [+0.021, +0.032] | +0.58 [+0.48, +0.68] | +3.0 [+2.2, +3.9] |
| only_d6v | Run 3 | 110.6% [109.0, 112.3] | 4.0% [3.4, 4.6] / 3.2% | +0.011 [+0.007, +0.016] | +0.19 [+0.11, +0.27] | +1.1 [+0.3, +1.9] |
|  | Control | 118.8% [116.8, 120.8] | 4.4% [3.8, 5.0] / 3.7% | +0.008 [+0.004, +0.012] | +0.21 [+0.12, +0.29] | +0.8 [+0.0, +1.5] |
|  | Aero | 120.3% [118.0, 122.6] | 3.9% [3.4, 4.5] / 3.4% | +0.006 [+0.002, +0.010] | +0.18 [+0.10, +0.26] | +0.9 [+0.2, +1.7] |
| time_first_half | Run 3 | 59.6% [58.3, 61.0] | 12.5% [10.4, 14.7] / 2.4% | +0.092 [+0.078, +0.106] | +0.92 [+0.73, +1.12] | +2.7 [+1.1, +4.4] |
|  | Control | 57.4% [56.1, 58.7] | 11.0% [9.0, 13.1] / 2.2% | +0.081 [+0.067, +0.096] | +1.10 [+0.91, +1.30] | +5.2 [+3.4, +6.9] |
|  | Aero | 59.9% [58.2, 61.6] | 8.4% [6.9, 10.1] / 2.8% | +0.050 [+0.040, +0.061] | +0.93 [+0.75, +1.11] | +3.7 [+2.2, +5.2] |
| time_second_half | Run 3 | 74.1% [72.7, 75.5] | 17.4% [15.1, 19.6] / 3.0% | +0.132 [+0.118, +0.145] | +0.84 [+0.69, +1.01] | +4.5 [+3.1, +5.9] |
|  | Control | 72.4% [71.0, 73.9] | 17.9% [15.6, 20.2] / 2.9% | +0.136 [+0.122, +0.150] | +0.94 [+0.76, +1.12] | +6.7 [+5.1, +8.4] |
|  | Aero | 78.5% [77.0, 80.0] | 14.0% [12.1, 16.1] / 2.8% | +0.101 [+0.088, +0.113] | +0.90 [+0.75, +1.07] | +4.1 [+2.8, +5.4] |
| time_middle_half | Run 3 | 58.2% [56.8, 59.6] | 15.4% [13.5, 17.2] / 2.8% | +0.126 [+0.114, +0.138] | +1.03 [+0.89, +1.18] | +6.0 [+4.5, +7.4] |
|  | Control | 56.0% [54.6, 57.4] | 18.9% [16.6, 21.1] / 2.7% | +0.147 [+0.134, +0.160] | +1.22 [+1.05, +1.39] | +6.6 [+5.0, +8.2] |
|  | Aero | 67.0% [65.1, 69.0] | 12.4% [10.9, 13.9] / 2.8% | +0.080 [+0.071, +0.091] | +1.13 [+0.98, +1.29] | +6.0 [+4.7, +7.3] |
| scramble_patches | Run 3 | 105.7% [104.5, 106.9] | 4.3% [3.7, 4.9] / 3.0% | +0.016 [+0.012, +0.021] | +0.30 [+0.22, +0.39] | +1.3 [+0.5, +2.1] |
|  | Control | 111.4% [110.0, 113.0] | 4.4% [3.8, 5.1] / 3.2% | +0.017 [+0.012, +0.022] | +0.30 [+0.23, +0.39] | +1.5 [+0.8, +2.3] |
|  | Aero | 114.3% [112.2, 116.4] | 3.5% [3.0, 4.1] / 3.1% | +0.010 [+0.005, +0.014] | +0.34 [+0.26, +0.42] | +1.4 [+0.6, +2.1] |
| reverse_time | Run 3 | 99.7% [99.0, 100.4] | 4.0% [3.4, 4.6] / 3.1% | +0.010 [+0.006, +0.015] | +0.31 [+0.23, +0.39] | +1.4 [+0.7, +2.2] |
|  | Control | 103.2% [102.3, 104.1] | 4.1% [3.5, 4.7] / 3.2% | +0.007 [+0.003, +0.011] | +0.29 [+0.22, +0.37] | +1.5 [+0.8, +2.3] |
|  | Aero | 108.9% [107.4, 110.5] | 3.7% [3.1, 4.3] / 3.2% | +0.007 [+0.003, +0.011] | +0.34 [+0.25, +0.43] | +1.5 [+0.7, +2.3] |
| null_shuffled | Run 3 | 114.7% [112.8, 116.8] | 3.0% [2.5, 3.6] / 3.3% | +0.000 [-0.004, +0.004] | +0.19 [+0.11, +0.27] | +1.0 [+0.2, +1.7] |
|  | Control | 115.0% [113.0, 117.2] | 3.2% [2.7, 3.8] / 3.3% | -0.001 [-0.005, +0.004] | +0.20 [+0.12, +0.28] | +0.9 [+0.1, +1.7] |
|  | Aero | 115.6% [113.3, 117.9] | 3.3% [2.7, 3.8] / 3.3% | +0.000 [-0.004, +0.005] | +0.17 [+0.08, +0.26] | +0.5 [-0.3, +1.2] |

## Key paired contrasts (a − b; 95% CIs; all others in `ablation_error_types/tables.md`)

| a − b | slice | Run | ΔWER (pp) | Δ phon excess | Δ ctx excess (nats) | Δ related excess (pp) |
|---|---|---|---|---|---|---|
| drop_v6v − none | all | Run 3 | +19.9 [+18.2, +21.6] | -0.058 [-0.073, -0.044] | -0.37 [-0.59, -0.14] | +0.2 [-1.7, +2.0] |
|  |  | Control | +23.3 [+21.6, +25.1] | -0.068 [-0.084, -0.053] | -0.45 [-0.69, -0.21] | +0.2 [-1.8, +2.2] |
|  |  | Aero | +24.8 [+22.3, +27.3] | -0.050 [-0.064, -0.036] | -0.53 [-0.78, -0.29] | -0.4 [-2.2, +1.4] |
| drop_v6v − none | mostly_right | Run 3 | – | -0.048 [-0.069, -0.027] | +0.03 [-0.28, +0.34] | -0.3 [-2.9, +2.4] |
|  |  | Control | – | -0.054 [-0.077, -0.030] | -0.24 [-0.57, +0.09] | +1.2 [-2.0, +4.4] |
|  |  | Aero | – | -0.056 [-0.085, -0.029] | +0.19 [-0.24, +0.62] | -1.4 [-5.1, +2.5] |
| drop_55b − none | all | Run 3 | +22.2 [+20.6, +23.9] | -0.030 [-0.045, -0.015] | -0.47 [-0.69, -0.26] | +0.6 [-1.4, +2.4] |
|  |  | Control | +22.0 [+20.5, +23.6] | -0.041 [-0.056, -0.027] | -0.60 [-0.82, -0.36] | +0.3 [-1.8, +2.3] |
|  |  | Aero | +20.8 [+18.4, +23.0] | -0.023 [-0.037, -0.009] | -0.57 [-0.82, -0.34] | -0.3 [-2.1, +1.4] |
| drop_55b − none | mostly_right | Run 3 | – | -0.017 [-0.038, +0.004] | -0.10 [-0.42, +0.20] | +2.3 [-0.4, +5.1] |
|  |  | Control | – | -0.043 [-0.066, -0.020] | -0.45 [-0.75, -0.14] | -1.3 [-4.3, +1.6] |
|  |  | Aero | – | -0.017 [-0.042, +0.008] | -0.51 [-0.92, -0.12] | -0.4 [-4.0, +3.2] |
| drop_55b − drop_v6v | all | Run 3 | +2.4 [+0.4, +4.2] | +0.028 [+0.014, +0.042] | -0.11 [-0.30, +0.08] | +0.4 [-1.5, +2.2] |
|  |  | Control | -1.3 [-3.3, +0.6] | +0.027 [+0.014, +0.040] | -0.15 [-0.33, +0.04] | +0.1 [-1.7, +1.9] |
|  |  | Aero | -4.0 [-6.7, -1.3] | +0.027 [+0.015, +0.039] | -0.05 [-0.25, +0.16] | +0.1 [-1.5, +1.6] |
| drop_55b − drop_v6v | mostly_right | Run 3 | – | +0.030 [+0.008, +0.053] | -0.14 [-0.46, +0.18] | +2.6 [-0.3, +5.6] |
|  |  | Control | – | +0.011 [-0.010, +0.031] | -0.21 [-0.53, +0.11] | -2.6 [-5.7, +0.6] |
|  |  | Aero | – | +0.039 [+0.010, +0.068] | -0.70 [-1.16, -0.26] | +1.0 [-3.2, +5.1] |
| only_55b − only_v6v | all | Run 3 | +11.4 [+9.4, +13.4] | -0.044 [-0.053, -0.034] | +0.20 [+0.05, +0.35] | -0.2 [-1.6, +1.1] |
|  |  | Control | +12.5 [+10.4, +14.6] | -0.042 [-0.051, -0.032] | +0.03 [-0.12, +0.18] | +0.0 [-1.4, +1.4] |
|  |  | Aero | +19.7 [+17.0, +22.4] | -0.029 [-0.037, -0.020] | -0.31 [-0.46, -0.16] | -1.2 [-2.5, +0.1] |
| only_55b − only_v6v | mostly_right | Run 3 | – | -0.038 [-0.070, -0.006] | +0.82 [+0.16, +1.43] | +7.2 [+1.7, +13.0] |
|  |  | Control | – | -0.030 [-0.064, +0.006] | +0.34 [-0.31, +0.97] | +4.5 [-1.7, +10.7] |
|  |  | Aero | – | -0.060 [-0.108, -0.009] | -0.12 [-1.28, +1.09] | +5.3 [-4.0, +15.0] |

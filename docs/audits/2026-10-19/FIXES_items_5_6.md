# Fixes for audit items 5 and 6 (19 October 2026)

Only significance and uncertainty were revised. No point accuracy, CI or decoder changed, and nothing
was retrained. Original result files are kept; the corrected ones sit next to them.

## Item 6: Phase 3 uncertainty

**Problem.** `compare()` bootstrapped the 84 per-condition Δs and ran a Wilcoxon signed-rank test on them
as if they were independent. They are 12 arrays × 7 behaviors from 4 participants, and a participant's
arrays record the same trials.

**Change.** `scripts/phase3_models.py` has a new `clustered_uncertainty()`. It runs with
`--uncertainty-only` and at the end of `--compare-only`. It reads the saved `per_condition.csv` and writes:

- `results/phase3/model_comparison_clustered.{csv,md}`: the mean Δ with 95% percentile bootstrap intervals
  (10,000 resamples). The intervals resample conditions (the old method, kept for reference), arrays (12),
  sessions (28 participant × behavior recordings), and arrays × behaviors (two-way). It also gives
  participants improved and a participant-level sign test.
- `results/phase3/per_participant.csv`.

`model_comparison.{csv,md}` were not regenerated. Its header text in the code now points to the
clustered file.

| Model | Δ (pp) | old CI (by condition) | by array | by session | array × behavior | participants improved |
|---|---|---|---|---|---|---|
| Shrinkage LDA | +3.7 | [+2.5, +5.0] | [+1.9, +5.6] | [+2.1, +5.5] | [+1.5, +6.3] | 4/4 |
| Logistic regression | +2.9 | [+1.8, +4.0] | [+1.2, +4.6] | [+1.5, +4.3] | [+0.8, +5.3] | 3/4 (T17 tie) |
| Linear SVM | +3.1 | [+2.1, +4.2] | [+1.3, +5.0] | [+1.7, +4.6] | [+1.0, +5.5] | 3/4 (T12 −0.1) |
| 1D-CNN | +5.5 | [+3.8, +7.3] | [+3.0, +8.0] | [+3.3, +7.9] | [+2.0, +9.5] | 4/4 |
| GRU | −0.7 | [−2.5, +1.1] | [−2.6, +1.4] | [−3.2, +1.8] | [−3.7, +2.2] | 2/4 |

CNN per participant: T12 +3.7, T15 +7.4, T16 +6.9, T17 +2.6 pp. The pre-computed reference was array-clustered
[2.9, 8.1]; [3.0, 8.0] here differs only by bootstrap noise.

**What changes in the write-up.** No conclusion changes direction. Every interval except the GRU's still
excludes zero, but the two-way intervals are 1.6–2.1× as wide as the old ones. The condition-level Wilcoxon
p-values (1e-7 to 2e-6) are withdrawn: the independent units are the 4 participants, and a two-sided sign
test on four cannot go below p = 0.125. The README Phase 3 table now shows the by-array and two-way
intervals and participants improved in place of the Wilcoxon column. A new finding states the gains as
benchmark gains on this dataset, not evidence for new participants.

## Item 5: Phase 2 permutation significance

**Problem.** Transfer cells shuffled the *training* labels and refit the decoder. That tests whether
the training behavior has word information, not whether the decoder predicts the test words; the
audit's counterexample gave a 36% false-positive rate at α = 0.05. Diagonal cells permuted labels
inside folds that stayed stratified on the true labels. The observed pipeline re-stratifies on its
labels, so the null was a different procedure; the earlier re-check's simulation
(`within_null_sim.py`) found it about 2× anti-conservative.

**Change.**
- `src/kunzdecoder/batched.py`:
  - `transfer_predict` returns the fixed decoder's predictions.
  - `permuted_labels(..., groups=)` permutes within groups (blocks).
  - `nested_cv_refold` re-runs the nested CV with folds re-stratified on each permuted row.
  - `transfer` and `nested_cv` are unchanged in behavior.
- `scripts/phase2_transfer.py`:
  - Transfer null: the decoder fit on the true training labels is kept fixed, and its predictions are
    scored against 1000 permutations of the test labels within recording blocks. Every block, in both
    isolated and interleaved sessions, holds each word equally often, so this is the experiment's own
    randomization.
  - Diagonal null: the same 1000 label permutations as before, now with folds re-stratified per
    permutation.
  - BH within dataset × normalization as before. Outputs carry `--tag` (default `corrected`), so the
    original files are not overwritten.
- `tests/test_batched.py`: two new tests. One checks that `nested_cv_refold` equals the fixed-fold
  pipeline run on each row's own folds. The other checks the within-block permutations.

**Verification.**
- 70/70 diagonal accuracies are identical to Phase 1, and all 1000 cells' accuracies are identical to the
  first run (checked in `significance_changes`).
- The re-stratified null run on the true labels reproduces the observed count in 100/100 diagonal cells.
- The audit counterexample under the new transfer null gives a false-positive rate of 0.046 at α = 0.05
  and 0.008 at α = 0.01 (old null: 0.361). See `transfer_null_fixed_check.{py,json}`.
- Test suite: 49 passed, 1 skipped (CUDA hidden).
- The run took 41 min on the Mac CPU; no GPU was used.

**Corrected files.**
- `results/phase2/{transfer_results,null_distributions,matrices,key_transfers,summary,tgm}_corrected.*`
- `results/phase2/significance_changes_corrected.csv`
- `figures/phase2_*_corrected.png`

The TGM files are recomputed and unchanged in content.

**Significant cells per family (q < 0.05), first run → corrected:**

| Family | Cells | Old | Corrected | Gained / lost |
|---|---|---|---|---|
| isolated · block (transfer) | 420 | 230 | 248 | +18 / −0 |
| isolated · strict (transfer) | 420 | 202 | 216 | +17 / −3 |
| isolated · within (diagonal) | 70 | 54 | 51 | +0 / −3 |
| interleaved · block (transfer) | 60 | 31 | 31 | +1 / −1 |
| interleaved · within (diagonal) | 30 | 25 | 25 | 0 |

Why most changes are gains: on these data the new transfer null is usually narrower than the old one
(median SD ratio 0.90 across transfer cells). The old null's spread came from refitting decoders on
shuffled labels, which is not the variability that matters for the claim. The new null is the exact
randomization distribution for fixed predictions and block-balanced labels. It is also narrower than a
binomial approximation when predictions are unevenly spread over the words. That is why 43 of the 900
transfer calls differ from the earlier binomial re-check (`phase2_binom.csv`); the other 857 agree. The
diagonal null got slightly wider (median SD ratio 1.03), as the re-check predicted.

**Cells named in the request:**
- Interleaved T15-55b listening → imagined listening (Finding 3): q 0.048 → 0.031, still significant.
  Finding 3's interleaved claim stands.
- T15-i6v strict, listening → inner: auditory q 0.042 → 0.047 (still significant, narrowly), imagined
  listening 0.003 → 0.003, motoric 0.351 → 0.313 (n.s.). Still 2/3. The other T15-i6v strict change:
  reading → attempted gains significance.
- T17-55b block, listening → motoric inner: q 0.054 → 0.022, now significant (README table 0/3 → 1/3).
  Listening → imagined listening stays n.s. (0.077 → 0.056).
- T17-55b strict, listening → imagined listening: still significant (0.040 → 0.016). Also gained:
  auditory → reading and listening → reading; lost: mimed → reading.
- T17-55b diagonal: motoric inner speech and mimed lose permutation significance (q 0.025/0.028 → 0.054).

**All 43 changes** (for 19 of them, the old or new q is within 0.01 of 0.05, so with 1000 permutations
(Monte Carlo SE ≈ 0.007 near p = 0.05) these could go either way):

| Dataset · norm | Array | Train → test | Acc. % | q old → new | Now |
|---|---|---|---|---|---|
| interleaved · block | T15-55b | ImagL → Att | 19.5 | 0.100 → 0.009 | sig. |
| interleaved · block | T15-s6v | ImagL → Listen | 19.5 | 0.048 → 0.074 | n.s. |
| isolated · block | T12-i6v | Mim → Read | 20.4 | 0.091 → 0.010 | sig. |
| isolated · block | T12-s6v | Motor → Mim | 20.4 | 0.063 → 0.043 | sig. |
| isolated · block | T12-s6v | Read → Att | 20.4 | 0.071 → 0.030 | sig. |
| isolated · block | T15-4 | Att → Listen | 17.9 | 0.167 → 0.046 | sig. |
| isolated · block | T15-4 | Att → Read | 19.3 | 0.071 → 0.017 | sig. |
| isolated · block | T15-4 | ImagL → Mim | 20.7 | 0.074 → 0.004 | sig. |
| isolated · block | T15-4 | Listen → Mim | 21.4 | 0.067 → 0.008 | sig. |
| isolated · block | T15-4 | Read → Att | 20.0 | 0.145 → 0.008 | sig. |
| isolated · block | T15-55b | Motor → Read | 20.7 | 0.051 → 0.038 | sig. |
| isolated · block | T15-55b | Mim → Aud | 17.1 | 0.192 → 0.038 | sig. |
| isolated · block | T15-55b | Read → Listen | 20.7 | 0.100 → 0.040 | sig. |
| isolated · block | T15-i6v | Att → Listen | 19.3 | 0.063 → 0.030 | sig. |
| isolated · block | T17-55b | Listen → Motor | 21.4 | 0.054 → 0.022 | sig. |
| isolated · block | T17-s6v-A | Att → ImagL | 20.0 | 0.064 → 0.044 | sig. |
| isolated · block | T17-s6v-A | Listen → Mim | 21.4 | 0.102 → 0.019 | sig. |
| isolated · block | T17-s6v-A | Read → Att | 22.9 | 0.101 → 0.008 | sig. |
| isolated · block | T17-s6v-B | Att → Aud | 20.7 | 0.071 → 0.037 | sig. |
| isolated · block | T17-s6v-B | Aud → Att | 20.7 | 0.115 → 0.022 | sig. |
| isolated · strict | T12-i6v | Att → Read | 19.0 | 0.084 → 0.044 | sig. |
| isolated · strict | T12-i6v | ImagL → Att | 17.0 | 0.297 → 0.009 | sig. |
| isolated · strict | T12-i6v | Mim → Listen | 19.0 | 0.069 → 0.005 | sig. |
| isolated · strict | T15-4 | Motor → Att | 19.3 | 0.142 → 0.005 | sig. |
| isolated · strict | T15-4 | ImagL → Att | 17.9 | 0.239 → 0.012 | sig. |
| isolated · strict | T15-4 | Mim → ImagL | 18.6 | 0.055 → 0.018 | sig. |
| isolated · strict | T15-55b | Aud → Listen | 19.3 | 0.121 → 0.022 | sig. |
| isolated · strict | T15-55b | Aud → Mim | 21.4 | 0.064 → 0.003 | sig. |
| isolated · strict | T15-55b | Motor → Read | 20.0 | 0.048 → 0.050 | n.s. |
| isolated · strict | T15-55b | Listen → Motor | 18.6 | 0.061 → 0.049 | sig. |
| isolated · strict | T15-55b | Mim → Motor | 17.6 | 0.089 → 0.014 | sig. |
| isolated · strict | T15-i6v | Read → Att | 22.9 | 0.064 → 0.003 | sig. |
| isolated · strict | T15-s6v | Motor → Att | 20.0 | 0.089 → 0.025 | sig. |
| isolated · strict | T17-55b | Aud → Read | 20.0 | 0.061 → 0.029 | sig. |
| isolated · strict | T17-55b | Listen → Read | 19.3 | 0.081 → 0.046 | sig. |
| isolated · strict | T17-55b | Mim → Read | 20.0 | 0.045 → 0.068 | n.s. |
| isolated · strict | T17-s6v-A | Motor → Aud | 20.7 | 0.053 → 0.031 | sig. |
| isolated · strict | T17-s6v-B | Att → Aud | 19.3 | 0.053 → 0.025 | sig. |
| isolated · strict | T17-s6v-B | Aud → Att | 20.0 | 0.099 → 0.049 | sig. |
| isolated · strict | T17-s6v-B | Motor → ImagL | 20.0 | 0.048 → 0.060 | n.s. |
| isolated · within | T17-55b | Motor → Motor | 21.4 | 0.025 → 0.054 | n.s. |
| isolated · within | T17-55b | Mim → Mim | 20.7 | 0.028 → 0.054 | n.s. |
| isolated · within | T17-s6v-A | Read → Read | 20.7 | 0.043 → 0.089 | n.s. |

**README changes (Phase 2).**
- The statistics description and the "how the nulls are built" caveat now describe the new nulls.
- File and figure pointers go to `*_corrected`.
- Table: T15-55b strict 0/3 → 1/3; T17-55b paper norm. 0/3 → 1/3; interleaved q values updated.
- Finding 1: inner → reading 10/15 → 11/15, plus the new T17-55b cell.
- Finding 3: T15-55b strict "does not survive" → "1 of 3 survives, narrowly"; T15-55b interleaved q = 0.031.
- New finding 6 summarizes these changes.
- Run time and compute rows updated.

No Phase 2 conclusion reverses.


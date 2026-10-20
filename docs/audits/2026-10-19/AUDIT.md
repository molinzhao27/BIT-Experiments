**KunzDecoder audit — 19 October 2026 (Pacific time)**

The main risks are experiment correctness and result integrity. Four defects deserve priority before relying on the affected comparisons: padded neural positions supervising the no-prompt model, evaluator identities that collide across LLMs, incorrect phoneme validation, and stale checkpoints winning over retrained ones. The earlier point accuracies line up correctly, but their significance/uncertainty needs revision.

This was an audit, not a repair. No training code, configuration, data, checkpoints, results, or GPU queues were changed. No GPU job was paused, stopped, restarted, or launched. The new files are this report and its CPU evidence.

Priority meanings: **P1** = fix before trusting the affected run or comparison; **P2** = correct the stated statistical interpretation or a specific operational failure path. Priority does not imply a request to interrupt training.

**1. P1 — The queued no-prompt model trains its first word from padding.**

[BIT's embedding merge](/Users/molin/Documents/KunzDecoder/external/bit/models/bci.py:267) inserts every padded neural position before the sentence. Its [shifted cross-entropy](/Users/molin/Documents/KunzDecoder/external/bit/models/bci.py:349) therefore predicts the first word from the last padded position for shorter examples when the prompt is empty. Attention masking removes padding as keys, not the query whose logits enter the loss. [Evaluation](/Users/molin/Documents/KunzDecoder/src/kunzdecoder/stage2.py:166) uses one trial at a time and predicts from the last real neural token. The [queued base-Qwen run](/Users/molin/Documents/KunzDecoder/scripts/phase4/queue_llmswap.sh:21) uses this no-prompt path.

CPU reproduction: four real neural tokens plus twenty pads move the first supervised query from real position 3 to masked position 23. The first-word logits change by 0.470232; the loss sends 4.027456 summed absolute gradient into padding. This establishes the indexing error, not the trained model's WER penalty. Prompted runs also retain interior position gaps, but their first-word query is real prompt text.

Fix: pack each example's valid neural and text tokens contiguously, then pad the merged sequences. Verify the first supervised query and singleton-versus-mixed-length outputs. [Reproduction](/Users/molin/Documents/KunzDecoder/docs/audits/2026-10-19/padding_repro.py), [output](/Users/molin/Documents/KunzDecoder/docs/audits/2026-10-19/padding_repro.json).

**2. P1 — The shared evaluator overwrites distinct arm-B experiments.**

[RUN_KEY](/Users/molin/Documents/KunzDecoder/src/kunzdecoder/phase4_eval.py:27), [artifact names](/Users/molin/Documents/KunzDecoder/src/kunzdecoder/phase4_eval.py:123), and [seed summaries](/Users/molin/Documents/KunzDecoder/src/kunzdecoder/phase4_eval.py:238) omit the LLM/run recipe. Aero, Qwen-Instruct, base-Qwen, and Fast-dLLM all use arm B / seed 42. Their distinct model names do not protect their outputs. A reproduction scoring two such models leaves exactly one metrics row and one per-trial file: the second replaces the first. No completed Stage 2 scoring artifacts were observed, so this is a confirmed pending failure, not evidence results have already been lost.

Related integrity gap: [null comparisons](/Users/molin/Documents/KunzDecoder/src/kunzdecoder/phase4_eval.py:326) join only on local integer trial IDs, while the [CLI](/Users/molin/Documents/KunzDecoder/scripts/phase4/evaluate_decodes.py:96) does not enforce matching run, split, seed, decoding, or references. Tables containing different reference sentences but identical row IDs were accepted and labelled as beating all nulls.

Fix: carry a stable experiment identity covering LLM, prompt, initialization, label fraction, and recipe, separately from seed. Use it consistently in keys, filenames, summaries, and null comparisons. Require identical normalized references and trial identities for paired comparisons; reject mismatches.

**3. P1 — Phoneme checkpoint selection uses an incorrect PER calculation.**

[The training PER function](/Users/molin/Documents/KunzDecoder/external/bit/utils/eval_llm.py:48) decodes the full padded prediction tensor. [NDT](/Users/molin/Documents/KunzDecoder/external/bit/models/ndt.py:595) uses valid lengths for CTCLoss but returns untrimmed predictions. With validation batches of 32, outputs beyond the end of shorter trials can therefore affect the metric that selects BEST. Two otherwise perfect synthetic predictions scored 33.33% PER instead of 0% solely because of padding. [Batch PERs are also averaged equally](/Users/molin/Documents/KunzDecoder/external/bit/models/trainer.py:418), rather than accumulating edits and reference phonemes.

Separately, [format_ctc](/Users/molin/Documents/KunzDecoder/external/bit/utils/eval_llm.py:35) fails to update the previous symbol on blanks. AA, blank, AA becomes one AA instead of two. Correct greedy CTC collapse merges consecutive repeats before removing blanks, as in [PyTorch's reference decoder](https://docs.pytorch.org/audio/2.8/tutorials/asr_inference_with_ctc_decoder_tutorial.html).

Measured impact of the blank rule alone on the completed baseline's saved logits:

| Split | Existing PER | Correct blank-collapse PER | Changed decodes |
|---|---:|---:|---:|
| BIT-val | 8.9141% | 8.9190% | 5 / 713 |
| BIT-test | 9.0145% | 9.0097% | 1 / 713 |

The blank-only effect is small. This rescore does not measure the separate padded-validation or batch-weighting effects on checkpoint selection. The active phoneme job uses the affected evaluation code; its training loss is not shown to be wrong by these findings.

Fix: decode only the valid post-patching timesteps, implement correct CTC collapse, and aggregate edit counts. Preserve the original BIT-convention numbers separately. Corrected selection can be assessed on retained checkpoints without assuming a full retraining is necessary. [Saved-output rescore](/Users/molin/Documents/KunzDecoder/docs/audits/2026-10-19/ctc_saved_output_rescore.json).

**4. P1 — Reusing a completed Stage 2 run name can discard the new model.**

[load_model_state](/Users/molin/Documents/KunzDecoder/src/kunzdecoder/stage2.py:97) always prefers slim.safetensors when it exists. Retraining the same name writes a fresh full checkpoint into BEST without removing the old slim file. [Final evaluation](/Users/molin/Documents/KunzDecoder/scripts/phase4/train_stage2.py:163) then loads the old slim weights; [cleanup](/Users/molin/Documents/KunzDecoder/scripts/phase4/train_stage2.py:183) can slim those old weights and remove the new full model. A toy checkpoint with old weight 1 and new full weight 2 loaded weight 1.

There is a separate stale-output path in the original baseline: [eval_llm.py](/Users/molin/Documents/KunzDecoder/external/bit/eval_llm.py:371) skips evaluation whenever an nbest file exists; [collection](/Users/molin/Documents/KunzDecoder/scripts/phase4/run_bit_baseline.py:115) then reads an existing historical prediction CSV without checking its checkpoint identity. No existing result was proven stale in this audit.

Fix: refuse accidental reuse of completed run directories, or explicitly select the current invocation's checkpoint format and invalidate stale artifacts. Tie prediction caches to checkpoint and generation-setting digests. Do not rely on filenames or modification order alone.

**5. P2 — Phase 2's permutation significance is not calibrated to the transfer claim.**

[Off-diagonal transfer](/Users/molin/Documents/KunzDecoder/scripts/phase2_transfer.py:171) shuffles individual source training labels and compares the resulting classifiers on fixed target labels. This destroys source word structure, even though an informative source is compatible with the null that target words are unpredictable.

A bounded counterexample using the repository's actual fitting/prediction functions retains strong source structure and gives targets independent balanced labels. Mean target accuracy is 14.2607%, essentially the 14.2857% chance level, but 361 of 1,000 target-label assignments receive p < .05. Of 1,000 shuffled-source classifiers, 850 collapse to predicting one target class, yielding an overly narrow reference distribution. This establishes a calibration failure, not that the saved large transfer effects disappear. [Script](/Users/molin/Documents/KunzDecoder/docs/audits/2026-10-19/transfer_null_repro.py), [output](/Users/molin/Documents/KunzDecoder/docs/audits/2026-10-19/transfer_null_output.json).

The diagonal null has a second issue: [labels are shuffled globally](/Users/molin/Documents/KunzDecoder/src/kunzdecoder/batched.py:106), while [nested folds stay stratified on the original labels](/Users/molin/Documents/KunzDecoder/scripts/phase2_transfer.py:204). This contradicts the documented within-partition permutation scheme. A fold with original counts [2,2,2,2,2,2,2] changes to [1,1,1,3,3,2,3]. The observed and null pipelines therefore use different label-balance regimes; the direction of the resulting distortion is sample-dependent.

Fix: for transfer, keep source selection/fitting fixed and test predictions against target-label permutations respecting the experimental randomization/block structure. For within-behavior tests, reproduce the label-dependent splitting inside every permutation or use an explicitly valid restricted design. Recalculate p/q values and update significance claims; point accuracies need not change.

**6. P2 — Phase 3's global uncertainty treats dependent conditions as independent replicates.**

[Bootstrap intervals](/Users/molin/Documents/KunzDecoder/scripts/phase3_models.py:233) and [Wilcoxon p-values](/Users/molin/Documents/KunzDecoder/scripts/phase3_models.py:246) operate on 84 array × behavior conditions from four participants, including arrays recording the same trials. These are not 84 independent biological replicates. The signed-rank test's independence assumption is described in [SciPy's documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html).

The saved CNN gains are positive for all four participants: T12 +3.7415 pp, T15 +7.4065 pp, T16 +6.9323 pp, T17 +2.5510 pp. That descriptive result remains useful. As an illustration of the limited independent sample size, a four-participant two-sided sign test cannot attain p below .125; the reported condition-level Wilcoxon p is 1.56e-7. This is not offered as a replacement full analysis.

Fix: retain descriptive benchmark gains, show participant effects, and use justified clustered/hierarchical inference or clearly limit claims to the fixed benchmark. No retraining is required to revise uncertainty.

**7. P2 — SSL masked reconstruction and its baselines receive unequal information.**

[NDT smooths raw inputs before masking](/Users/molin/Documents/KunzDecoder/external/bit/models/ndt.py:454). Gaussian smoothing carries values from masked target bins into neighboring visible embeddings; the [no-learning baselines exclude those target values](/Users/molin/Documents/KunzDecoder/src/kunzdecoder/bit_ssl.py:307).

In a CPU NDT reproduction with bins 50–54 masked and dropout/noise disabled, changing only masked bin 50/channel 0 changed visible smoothed bin 49 by 0.176213 and masked predictions by 0.093144, while the masked embedding itself stayed unchanged. Thus the reported roughly 5% raw-bin reconstruction R² does not isolate prediction from truly unseen neural values. This is inherited BIT behavior, not demonstrated split contamination, and it does not invalidate downstream WER by itself. The size of its effect on the trained model is unmeasured.

Fix: qualify the current reconstruction/baseline interpretation. A later diagnostic should mask raw targets before smoothing, or also remove contaminated neighboring context, and give all predictors equivalent information. Existing runs need not be interrupted for this audit finding.

**Other concrete failure paths, lower priority than the findings above**

| Path | Evidence and consequence | Remedy |
|---|---|---|
| [Unverified cached scores](/Users/molin/Documents/KunzDecoder/src/kunzdecoder/phase4_eval.py:180) | Cached numeric scores without a BERTScore hash are trusted, then written with the current hash. A wrong decode with stale zero errors/F1=1 was accepted as perfect. | Recompute inexpensive word errors; require matching provenance before reusing semantic scores. |
| [Diffusion progress](/Users/molin/Documents/KunzDecoder/src/kunzdecoder/diffusion_llm.py:229) | If the decoder proposes mask_id, accepting it leaves the position masked. There is no per-block iteration bound, and max_new_tokens is checked only after the block. A bounded fake-logit CPU reproduction made no progress. This has not been observed on the actual checkpoint. | Suppress mask_id proposals and enforce a bounded progress/failure condition. |
| [Fetch exit status](/Users/molin/Documents/KunzDecoder/scripts/phase4/fetch_pretrain_data.py:69) | A checksum mismatch prints MISMATCH but main returns 0, with no verified destination. Reproduced with a stubbed response; no network download. | Return failure when requested files fail verification; retain the partial file for diagnosis. |

**Coverage and verification**

| Area | What was checked |
|---|---|
| Phase 0–3 | IO, array/channel mapping, feature/window extraction, nested CV, Gaussian/linear/stacked models, statistics, saved predictions and summaries. All five Phase 3 models have 12,257 unique trial keys with labels and outer folds matching baseline. No confirmed point-accuracy alignment error. |
| Data / SSL | Adapters, deduplication, normalization, source membership, fixed-mask metrics, corpus metadata, checkpoint exports. All 24 inspected B2T'24 train/test session pairs had disjoint blocks. Remote corpus matches the intended 60.043 h train / 4.32 h val configuration. |
| Supervised models | Stage 1 targets and GradCache, initialization, objectives, checkpoint loading, Stage 2 AR/diffusion integration, decoding, null controls and result storage. |
| GPU operations | Read-only process, GPU, queue/log, storage, configuration and source checks. The shared-lock wrapper is in use. At the initial snapshot, the phoneme stage was active and later queues were waiting; approximately 119 GB was free on the volume. No resource/job changes were made. |
| Tests | 43 passed, 1 skipped in 6.60 s, with CUDA hidden and CPU thread limits. The skipped test downloads a BERTScore model. Syntax checks passed for 50 Python files and 13 shell scripts. |
| Evidence | [CPU reproductions](/Users/molin/Documents/KunzDecoder/docs/audits/2026-10-19/reproduce.py), [results](/Users/molin/Documents/KunzDecoder/docs/audits/2026-10-19/reproduction_results.json), [verification and source hashes](/Users/molin/Documents/KunzDecoder/docs/audits/2026-10-19/verification.json). |

At 2026-10-20 01:20 UTC, all 90 common source/project-config files in the captured local/remote inventory matched. The additional local C/A queue script is deployed on the host rather than in the container. The workspace has no top-level Git repository, so this audit records file hashes rather than a commit. Matching disk files does not prove which earlier source version an already-running process loaded.

Limits: this was code review, bounded CPU reproduction, artifact inspection and a CPU rescore of completed baseline logits—not full model retraining or GPU inference. Local tests use torch 2.14.1 / transformers 5.20.1; the GPU BIT environment uses a different pinned stack. The local SSL corpus is an older Kunz-only artifact; the remote manifest was checked separately. Future label-fraction curves are not yet end-to-end low-label comparisons: Stage 2 subsampling alone does not remove labels already used by an A/C initialization checkpoint.

The already documented teacher-forced baseline WER, original BIT initialization issue, padding-before-smoothing limitation, and CNN time-coverage advantage were not presented as newly discovered defects. No blanket conclusion that the system or its existing accuracy results are invalid is warranted.

# Revision run: a block-diffusion decoder that can change committed tokens (prepared, not launched)

**Status (2026-10-21).** The code and CPU tests are ready. Nothing has run on the GPU and nothing is queued. The user
decides whether and when to run this, after run 3's test results.

**Three routes to revision, cheapest first:**
- **Re-masking sampler** (section 4): decode-only, no training. Run 3's checkpoint can be decoded again with it as soon
  as run 3 is done (entry `fast-dllm-v2-1.5b-remask`).
- **One step** (sections 1–3): editing is learned inside the Stage 2 brain fine-tune.
- **Two steps** (the "One step or two" section): first post-train editing into Fast-dLLM v2 1.5B on general text
  (`scripts/posttrain_edit.py`, written and CPU-tested, not run), then run Stage 2 on that base.

The training loss is shared by both training routes.

**Question.** Run 3 (`B_ssl_fast-dllm-v2-1.5b_s42`, DESIGN.md §8) decodes with Fast-dLLM v2's sampler, which commits
each token permanently. Does WER drop if the decoder may also *revise* tokens it has already committed?

**Design.** Everything is run 3's (arm B from `checkpoints/ssl_human/BEST_core`, the same model, LoRA, prompt,
contrastive term, epochs, selection and evaluation, seed 42) except three options. Each is off by default, and with
all three off, training and decoding are bit-identical to run 3 (tested below). The new entry
`fast-dllm-v2-1.5b-edit` in `configs/llms.yaml` turns all three on. It has the same snapshot, revision and loader as
`fast-dllm-v2-1.5b`, and its `sampler` adds only the five keys below.

```
<bit env>/python scripts/train_stage2.py --name B_ssl_fast-dllm-v2-1.5b-edit_s42 --arm B \
    --init checkpoints/ssl_human/BEST_core --seed 42 --llm fast-dllm-v2-1.5b-edit
```

These are run 3's arguments (its run.json), with only `--name` and `--llm` changed. `train_stage2.py` needs no
change: `text_llm.use_llm` passes the entry's whole `sampler` dict to the model, and BIT's config keeps the new keys
(checked).

Code: `src/bit_experiments/diffusion_llm.py`:
- `_align`: the aligned layout;
- `block_diffusion_ce`: the M2T + T2T loss, at module level, used by `denoising_ce`;
- `text_block_diffusion_ce`: the same loss on token ids, for text-only post-training;
- `_denoise`: the sampler, with editing and re-masking.

Also: `scripts/posttrain_edit.py` (step 1 of two). Tests: `tests/test_diffusion_revision.py`,
`tests/test_posttrain_edit.py`.

## 1. Aligned sentence block (`align_sentence_block: true`)

**Run 3's layout.** The sentence starts right after the prefix ([prompt_a, neural tokens, prompt_b]), inside the
prefix's partly filled last block. So a sentence often straddles two 32-token blocks. Once the first block is
committed to the KV cache, its words can no longer change.

**Aligned layout.** The prefix is left-padded with `<|endoftext|>` until its *last* token opens a block. The sentence
+ EOS + EOS padding then fill positions 1–31 of that block.

**Why the boundary sits one token before the sentence.** Fast-dLLM v2 predicts position j from the output at j − 1
(token shift). A block's position 0 is therefore predicted from the previous block's last output. That output is
cached and attends only to earlier blocks, so it never sees the current block. If the sentence started exactly at a
boundary, its first word would be a fixed left-to-right guess that no pass could revise. This is the "AR first token"
that Fast-dLLM v2 places at the start of every later block. Here, the prompt's last token sits at position 0 and
attends to the whole block, so every sentence token, the first word included, is predicted from the whole block.

**Fit.** Sentences + EOS are at most 18 tokens on BIT-val/test and 26 on train (DESIGN.md §8), within the 31 slots.
Decoding stops at 25 new tokens, so a decode is always exactly one block.

**Why this padding scheme.** It is the same in training (`forward`) and decoding (`generate`), applied by one helper
(`_align`) to every prefix, including the prompt-only null control.
- **The filler is `<|endoftext|>`** (151643, the config's bos token and Qwen's document separator). A run of it
  before the prompt reads like the end of a previous packed document. It is never a target and never generated. EOS
  (`<|im_end|>`, 151645) is avoided because it is the token the model learns to emit at the end of a sentence.
- **The filler goes in front of everything,** so that [prompt_b, sentence] stay adjacent, exactly as in run 3. The
  sentence's first word is still predicted from the prompt's last token.
- **Rejected: filler between the prompt and the sentence.** It would break that adjacency.
- **Rejected: attention-masked padding.** It would need new masks in the vendored model, for both the flex training
  mask and the decode mask, plus position-id changes. That is a larger change to code the run depends on, for no gain
  in consistency.

**Side effects.**
- The absolute positions shift by 0–31. RoPE positions already vary with the number of neural tokens.
- In 31 of 32 trials, the sequence's first token becomes `<|endoftext|>` instead of the prompt. That is the
  attention-sink position. The change is identical in training and decoding.
- Each trial gets at most one extra block, about 16 tokens on average (see the cost section).

## 2. Edit objective, token-to-token (`edit_rate: 0.1`)

**What LLaDA2.1 states.** LLaDA2.1 ([arXiv 2602.08676](https://arxiv.org/abs/2602.08676), §3.1) trains a "Mixture of
M2T and T2T" objective. Mask-to-token predicts masked positions. Token-to-token recovers original tokens from random
noise perturbations.

**What it does not state.** The paper gives no noise distribution, no rate, which positions are supervised, or the
weighting, and I found no released training code. The choices below are therefore ours.

**Our version, in `denoising_ce`:**
- **Corruption.** After run 3's masks are drawn (the same calls, so the masks are unchanged), each *visible* target
  token is replaced by a uniformly random ordinary token with probability `edit_rate`. Ordinary means ids below
  `<|endoftext|>`: no special tokens, no mask.
- **One draw serves both copies.** Each target token is visible in exactly one of the complementary pair, so it is
  corrupted with probability `edit_rate` once per pair.
- **Supervision.** A corrupted token is supervised with its original token, at the same token-shifted output as a
  masked one.
- **Prefix and clean halves.** The prefix and the clean halves are never corrupted. The EOS padding is a target, so
  it can be corrupted too. That teaches the model that junk after the sentence's end goes back to EOS.
- **Weighting (run 3's convention).** The CE is summed. Every target token still counts exactly once per pair as a
  masked token, so the M2T term has run 3's per-token weight next to BIT's summed contrastive term. The corrupted
  tokens add a T2T term of about `edit_rate` × (target tokens) per pair.
- **Logged validation loss.** `val_loss_tf` in log.jsonl is therefore not comparable with run 3's: it has more
  supervised positions and harder inputs.

**Why 0.1.** No published value exists. A higher rate gives more edit examples, but it also fills the context of the
masked predictions with junk, which could hurt the drafting that makes run 3 good (~24% val WER). 0.1 means about 3
corrupted tokens per 31-token sentence block per pair, which over 8,072 trials × 150 epochs is plenty of signal.

**Supervising only corrupted tokens.** This follows the task's description and LLaDA2.1's wording, but it is a real
choice:
- The model is never told what to do with a *correct* visible token. At those positions its proposal is effectively
  a re-prediction from context.
- **Alternative: also supervise the clean visible tokens ("keep").** This is the uniform-noise diffusion loss. It
  would calibrate when to keep a token, but random corruptions are nearly always implausible. The model would learn
  "keep anything plausible", which is exactly the preservation bias that makes revision useless. See open question 1.

**Alternative recipe: SCOPE** ([arXiv 2606.01026](https://arxiv.org/abs/2606.01026)). It corrupts with the model's own
errors instead of random tokens:
- **Recipe.** A no-gradient draft pass samples the masked positions (τ = 1.5). The top 30% by confidence are
  committed. A second pass is trained with CE on the committed and the remaining masked positions, and half of the
  steps are plain masked-diffusion training.
- **Evidence.** On LLaDA-8B, the wrong committed token is kept 70.1% of the time and the truth recovered 9.2%. With
  SCOPE, those become 17.0% and 50.2%.
- **Critique of random corruption.** A follow-up ([arXiv 2606.17175](https://arxiv.org/abs/2606.17175)) makes the same
  point against LLaDA2.1's random corruption: at inference the editor meets fluent, confident draft errors, not random
  tokens.
- **Cost here.** One extra no-gradient forward per step, roughly +30–50% GPU time.

It is the next recipe to try if random-token T2T yields few useful edits.

## 3. Editing sampler (`editing_threshold: 0.5, max_post_steps: 16, max_edits: 2`)

**The rule** (LLaDA2.1 Eq. 1–3, within the current block):
- **Fill.** Every forward pass fills the current sub-block exactly as run 3 does: proposals > 0.9, and at least the
  most confident one.
- **Edit, from the same forward.** Each *committed* position of the current block is overwritten when its new
  proposal differs and has probability > `editing_threshold`. Both sets are decided on the state before the pass and
  applied together.

**Which positions can be edited.** Every committed token of the current block up to and including its first EOS.
- Positions after the first EOS are padding and are never edited, so a pass cannot keep going on junk past the end.
- A block's AR first token, when no prompt token shares its block, is never edited: it is predicted from the cached
  block, so its proposal cannot change. With the aligned layout this case never arises.
- Earlier blocks are in the KV cache and are never touched.

**Passes that only edit.** Once the block has nothing left to fill (no mask before an EOS, or no mask at all), passes
that only edit continue until one edits nothing, or `max_post_steps` of them have run (16, LLaDA2.1's default).
`max_edits` caps the edits per position per block, which stops a token from flipping back and forth. Set it to
`None` for no cap.

**Termination.** A block takes at most 31 filling passes plus `max_post_steps` passes that only edit:
- every filling pass commits at least one mask;
- edits never write a mask (the mask logit is −∞);
- the post-pass counter is never reset within a block.

**EOS handling.**
- **The first EOS is editable.** If an edit turns it into a word, the block is no longer finished, and the masks after
  it are filled again.
- **A word can be edited into an EOS,** which shortens the sentence.
- **The output is cut at the first EOS,** as before.

**Thresholds.**
- **Edit threshold 0.5** is LLaDA2.1's Quality mode, whose recommended settings are threshold 0.7, editing 0.5 and 16
  post steps ([diffusers LLaDA2 docs](https://huggingface.co/docs/diffusers/v0.39.0/api/pipelines/llada2)).
- **Fill threshold.** We keep run 3's 0.9 so that only revision changes.
- **Post-hoc tuning.** All three sampler values act only at decode time. They can be re-tuned on BIT-val by decoding
  BEST again, as the decoding-order sweep does. Selection during training uses the entry's values.

**Nucleus decoding.** Fills and edits both use the sampled proposal and its probability.

**Trace (`model.trace = []`).** The trace is now native to `_denoise`. With all options off it is identical to
`scripts/decoding_order_sweep.py`'s `traced_denoise` (tested). With editing on:
- every pass event also carries `"e": [[pos, token, proposal, probability, edited 0/1], ...]`, listing every committed
  position up to the first EOS whose proposal differs, before the pass. Declined edits are listed too, for the viewer;
- a pass that only edits has `"m": []`.

Replaying the `m` commits and `e` edits in order reproduces `out` (tested).

## 4. Re-masking sampler, decode-only (`remask_threshold: 0.9, remask_rounds: 3`)

**Why.** The revision route with a precedent on task data (Mask-Predict; Mask-CTC in speech recognition, see "Is one
step feasible?"). It needs no new objective, only the mask-to-token skill every run already has. So it applies to run
3's checkpoint as it is: an evaluation-only job (decoding, no training).

**Rule.** When a block is done (finished, or full):
- every committed token up to the first EOS whose probability *when it was committed* is below `remask_threshold` is
  masked again;
- the same sampler then re-fills the block, now with the rest of the sentence visible;
- this repeats up to `remask_rounds` times, stopping early when a round re-fills exactly what it masked.

**Why 0.9.** Every token that cleared run 3's 0.9 threshold is kept. The ones below it are exactly the sub-block's
"best guess" commits, made when nothing cleared the threshold and while later words were still masked. Those get a
second look with full context.

**Details:**
- The AR first token of a block without prompt is never re-masked: it is predicted from the cached block.
- Earlier blocks are never touched.
- Re-masking the EOS reopens the block: if the re-fill puts a word there, the sentence continues.
- The output is cut at the first EOS.
- It can be combined with editing; the edit-only pass budget is shared across rounds.

**Termination and cost.** Each round masks at most the block's committed tokens, and each re-fill pass commits at
least one, so a block takes at most (rounds + 1) × 31 filling passes. In practice a round adds at most one pass per
re-masked token, usually fewer.

**Trace.** A re-masking round adds an event `{"b": block, "r": [[pos, token, probability when committed], ...]}`. The
passes that re-fill those positions list them in `"m"` again. Replay removes them, then re-commits them (tested).

**Relation to run 3's revision-headroom check** (decoding-order sweep). That check hides *every* decoded token on its
own, offline, and re-predicts it once. This sampler is the decoding counterpart:
- it hides only the low-confidence tokens;
- it hides them all at once, then re-fills them with the usual confidence order;
- it iterates.

**On run 3.** The checkpoint, prompt and layout are unchanged; only the sampler entry differs:

```
<bit env>/python scripts/train_stage2.py --name B_ssl_fast-dllm-v2-1.5b-remask_s42 --arm B \
    --eval-only checkpoints/stage2/B_ssl_fast-dllm-v2-1.5b_s42/BEST --llm fast-dllm-v2-1.5b-remask
```

This decodes BIT-val and BIT-test, greedy and nucleus, plus the null controls, all with re-masking. Compare with run 3's
own summary.json.

**Caveats.**
- In run 3's layout, a sentence that straddles the prompt's block and the next one is re-masked one block at a time:
  the second block's words cannot revise the first block. The aligned layout (section 1) removes that, but it needs
  training.
- `remask_threshold` and `remask_rounds` are decode-time values. Choose them on BIT-val only, as with the chunk size.
- The decoding-order sweep's `traced_denoise` (a copy of run 3's sampler) ignores re-masking. Trace through the model
  instead (`model.trace = []`).

## Verification (CPU, no GPU)

**Environment.** The tests need BIT's environment: torch 2.7.1, transformers 4.48.2, peft 0.11.1 (locally, a CPU copy
of it). The project `.venv` skips them (no peft). The model is a tiny random Fast-dLLM v2 (2 layers, 64-d, 96 tokens,
block 32) in BIT's LoRA (r 8, α 32, dropout 0.2, seven projections, nonzero LoRA weights). Its head is scaled so that
about a third of the proposals clear 0.9 and some decodes end at an EOS. A scripted stand-in decoder covers the
sampler's edge cases.

**Options off = run 3,** compared against run 3's `diffusion_llm.py` (commit `0b98fa9`, imported side by side):
- **Forward.** Loss, logits, labels and every LoRA gradient are `torch.equal` over 3 seeds × 6 trials. The trials have
  prefixes of 32–128 tokens, ending anywhere in a block. LoRA dropout is on. The options were tested both absent and
  explicitly off.
- **Decodes.** Identical token for token in 120 `_denoise` comparisons: sub-blocks 1/4/8/32 × 10 prefix lengths (tail
  0, 1, 4, 5, 8, 26, 31) × greedy / nucleus / 70-token multi-block. Both EOS-terminated and capped decodes occurred.
  Eight more comparisons went through `generate`.
- **Trace.** The native trace equals the sweep's `traced_denoise` record.
- **Text path** (two-step design). With edit_rate 0, `text_block_diffusion_ce` on token ids reproduces Fast-dLLM v2's
  own released training forward on the same seed: same masks, loss within 2e-6, over 3 seeds, with a context span.
  With edit_rate it trains.

**Options on:**
- **Aligned layout.** The prefix's last token is at block offset 0, and the sentence + EOS + padding fill offsets
  1–31 of one block. The fillers are `<|endoftext|>` embeddings, and the content is otherwise run 3's.
- **T2T.**
  - The masks are identical to options off, and the clean halves are untouched.
  - Only visible target tokens change, each to an ordinary token's embedding, and each is supervised with its
    original token.
  - `n_examples` = masked + corrupted.
  - The measured corruption rate is within 0.05 of the setting.
  - Backward gives finite, nonzero LoRA gradients, and the base stays frozen.
- **Sampler, scripted.**
  - A model that never agrees stops at the cap (each position edited at most 2×, then a pass with no edit), or, with no
    cap, after exactly 4 filling + 16 edit-only passes.
  - An EOS → word edit reopens and refills the block. Proposals after the first EOS are ignored, and the output is cut
    at the first EOS.
  - The AR first token and earlier blocks are never edited.
  - With editing off there are no extra passes and no `e` events.
- **Re-masking, scripted.**
  - A best-guess token (p 0.6) is masked again and re-filled with context.
  - A round that re-fills the same tokens stops the rounds.
  - Re-masking the EOS reopens the block and the sentence continues.
  - The AR first token is never re-masked.
- **End-to-end, all options on (the registry entry).** Forward + backward run. 16 decodes through `generate`
  terminate, each in one block with tail 1, and the trace replays to the output. Edits are both made and declined.
  The same holds for the re-masking entry: decodes terminate, replay, and re-mask within the round cap.
- **Post-training script (two steps), `tests/test_posttrain_edit.py`:**
  - the leakage filter (whole sentences after normalisation; sentences under 4 words are skipped) and packing;
  - layer checkpointing gives the same loss and gradients, full and LoRA;
  - the training loop's held-out check, for full fine-tune and LoRA: held-out M2T and T2T loss fall, and the full
    fine-tune learns to recover corrupted tokens;
  - milestone snapshots reload as plain models, with LoRA merged, and give the trained model's loss;
  - `main()` end to end on a local text file, an exclusion file and a word-level tokenizer.

```
$ <bit env>/python -m pytest tests/test_diffusion_revision.py tests/test_posttrain_edit.py -v
test_options_off_forward_is_run3s[absent] PASSED
test_options_off_forward_is_run3s[explicitly-off] PASSED
test_options_off_decodes_are_run3s[1] PASSED
test_options_off_decodes_are_run3s[4] PASSED
test_options_off_decodes_are_run3s[8] PASSED
test_options_off_decodes_are_run3s[32] PASSED
test_options_off_trace_is_the_sweeps PASSED
test_entries_are_run3_plus_revision_options[fast-dllm-v2-1.5b-edit-added0] PASSED
test_entries_are_run3_plus_revision_options[fast-dllm-v2-1.5b-remask-added1] PASSED
test_option_checks PASSED
test_aligned_sentence_block_layout PASSED
test_t2t_corrupts_and_supervises_visible_targets PASSED
test_text_path_is_the_released_objective PASSED
test_editing_terminates_when_the_model_never_agrees[2] PASSED
test_editing_terminates_when_the_model_never_agrees[None] PASSED
test_editing_eos_handling PASSED
test_editing_spares_the_ar_first_token_and_earlier_blocks PASSED
test_editing_off_has_no_post_passes_or_edit_events PASSED
test_remasking_refills_low_confidence_tokens_with_context PASSED
test_remasking_stops_when_a_round_changes_nothing PASSED
test_remasking_the_eos_reopens_the_block PASSED
test_remasking_spares_the_ar_first_token PASSED
test_all_options_on_end_to_end PASSED
test_remask_entry_end_to_end PASSED
posttrain: test_count PASSED
posttrain: test_excluder_matches_whole_sentences_after_normalisation PASSED
posttrain: test_pack_drops_excluded_documents PASSED
posttrain: test_layer_checkpointing_is_exact[False] PASSED
posttrain: test_layer_checkpointing_is_exact[True] PASSED
posttrain: test_training_loop_and_snapshots[False] PASSED
posttrain: test_training_loop_and_snapshots[True] PASSED
posttrain: test_main_end_to_end PASSED
32 passed in 12.8s
tiny random model, 16 decodes, all options on: 795 edits applied, 1476 declined; passes per decode 13.6 (max 22,
bound 47; 3.0 edit-only); the same decodes with the options off: 10.4 passes
```

The tiny model's edit counts mean nothing for the real run: a random model contradicts itself. They show only that
the machinery works.

**Not covered on CPU.** The GPU training path (flex attention, bf16 autocast) is unchanged code with different input
contents and lengths that are still multiples of 128. Before the full run, a short GPU smoke should still run under
the GPU lock, e.g. `--limit 64 --epochs 1 --eval-every 1`, under a throwaway `--name`.

## Expected GPU cost

**About 30 h, like run 3.** Run 3 took ~60 min per 5 epochs including validation decoding, so ~30 h for 150 epochs.
- **Aligned layout.** At most one extra 32-token block per trial, about +16 positions on average. The 128-token length
  bucket absorbs it in most batches: expect +0–10% training time.
- **T2T.** No extra forward. Logits and CE are computed at about 10% more positions, which is negligible.
- **Editing sampler.** At least one edit-only pass per decode, more when edits fire. That makes the validation decode
  every 5 epochs, and the final evaluation, maybe 10–40% slower. Decoding is a minority of the total.
- **Total.** ~30–34 h on the same single GPU, with about run 3's memory. With SCOPE instead of random corruption, add
  ~30–50%.

## How to read the result

**Primary comparison.** Unseen-sentence WER on BIT-test, greedy, against run 3. It is one seed, so a difference counts
only if it exceeds arm B's seed spread (DESIGN.md §8).

**Decomposition.** Two eval-only decodes, no training, separate the three options:
- **(a) This run's BEST with `editing_threshold: null`:** the trained changes (layout + T2T) without revision at
  decode time.
- **(b) Run 3's BEST with only the editing sampler** (`editing_threshold: 0.5`, its own unaligned layout):
  inference-only revision on a model never trained to edit. Expect little, given the preservation bias above.

Together with the two main runs, these give a 2 × 2 of training (run 3 / revision) × sampler (commit / edit). Run 3's
revision-headroom check in the decoding-order sweep (hide each decoded token and re-predict it) already bounds (b) from
above for one round of edits.

**Edit statistics from the traces.** Edits per sentence, and how many fixed or broke a word, as in the sweep's
`revision_report`.

## One step or two: post-train editing into the base first?

**Short answer.** Yes, the two-step plan is: post-train the 1.5B diffusion LM with the editing objective on general
text, then fine-tune that base on brain data with the usual Stage 2.

**Memory.** Post-training needs about 55 GB as a full fine-tune at one 2,048-token sequence per micro-step, or about
30 GB with activation checkpointing. LoRA needs about 10–15 GB with checkpointing. All are estimates, detailed below.

**The two designs.**
- **One step** (sections 1–3): `edit_rate` acts inside Stage 2. The editor is learned only from the 8,072 training
  sentences, about 10⁵ unique sentence tokens, each seen 150 times. It is learned through LoRA r8, at the same time as
  the brain-to-text mapping.
- **Two steps:**
  - **(1) Post-train.** Train Fast-dLLM v2 1.5B on general text with M2T + T2T: `text_block_diffusion_ce(model, ids,
    mask_id, edit_rate=…)`, mean loss `ce / n`. There is no contrastive term to balance here, and Fast-dLLM v2's own
    fine-tune averages. Save the result as a new base, "Fast-dLLM v2 1.5B-edit". If step 1 uses LoRA, merge it first
    so that Stage 2's recipe is unchanged.
  - **(2) Stage 2 on that base,** exactly as the revision run: aligned block + editing sampler. Either keep
    `edit_rate: 0.1`, which adapts the editor to brain-conditioned drafts, or set 0 for pure transfer.

| | One step | Two steps |
|---|---|---|
| Editing learned from | ~10⁵ unique sentence tokens (×150 epochs), LoRA r8, entangled with the brain mapping | 10⁷–10⁹ tokens of general English, full weights or a large LoRA; Stage 2 then adapts it |
| Generalisation to unseen sentences (our metric) | Doubtful: an editor fitted to 8k sentences may mostly learn to edit towards them | Better odds: how LLaDA2.1 got a working editor (T2T in continual pre-training and SFT, never only in a small fine-tune), and how SCOPE added self-correction (post-training on FineWeb-Edu) |
| GPU time | ~30–34 h | Step 1 (~0.5 day to a week, by token budget; below) + ~30–34 h |
| What the comparison with run 3 isolates | Revision (layout + objective + sampler) | Revision **plus** extra training on corpus X, unless controlled (below) |
| Reuse | None | The base serves every later diffusion run (seeds, arms) |
| Main risks | Editor too weak, so few edits | Drift: worse drafting (forgetting); eval sentences leaking via the corpus |
| Code | Ready | Ready: the loss is tested against the released objective, and `scripts/posttrain_edit.py` is CPU-tested; needs a GPU smoke |

**Controls for two steps.** Two eval-only decodes are cheap:
- the two-step model with editing off, which isolates the sampler given the new base;
- run 3 with editing on, which is inference-only revision.

Attributing a gain to the *editing* objective rather than to the extra text training needs one more control: an
M2T-only post-train on the same tokens, followed by Stage 2. That is a second step 1 plus a second Stage 2.

**Recommendation.** First read run 3's revision-headroom check. If revision has room, go to two steps with a modest
step 1 (~100M tokens). An editor learned from 10⁵ sentence tokens through LoRA r8 is the weakest version of the idea,
and every precedent we found trains editing at text scale (next subsection). The one-step run stays available as the
cheap variant.

### Is one step feasible? What the literature says (user's question, 2026-10-21)

**No direct precedent.** Every working token editor or self-corrector in diffusion LMs that we found was trained on
general text at large scale. None was learned only inside a small task fine-tune:

| Work | How revision was learned | Data |
|---|---|---|
| LLaDA2.1 ([2602.08676](https://arxiv.org/abs/2602.08676)) | M2T + T2T (random-token noise) in continual pre-training **and** SFT | pre-training scale; 16B / 100B MoE |
| ProSeCo ([2602.11590](https://arxiv.org/abs/2602.11590)) | corrector trained on the model's own predictions, in SFT | ~40B tokens of maths and code SFT, LLaDA-8B; beat plain masked SFT on HumanEval, MBPP, GSM8K and Minerva |
| RemeDi ([2509.23653](https://arxiv.org/abs/2509.23653)) | SFT mixing masks with deliberately wrong tokens, then RL | large maths and code data; remask-SFT beat vanilla SFT in its ablation |
| SCOPE ([2606.01026](https://arxiv.org/abs/2606.01026)) | own-error post-training, LoRA | the smallest success: 500 steps on FineWeb-Edu, ~20 min on an A100, but general text, not task data |

**Closest analogue that worked on task data alone: non-autoregressive speech recognition.**
- **Mask-CTC** ([2005.08700](https://arxiv.org/abs/2005.08700)) trains a mask-predict refinement decoder only on the
  task's speech: WSJ, 81 h, a few times our 8k sentences.
- At inference it re-masks low-confidence tokens and re-predicts them from the input signal plus the rest of the
  sentence. WSJ WER fell from 17.9% (CTC) to 12.1%, against 11.3% autoregressive.
- So revision conditioned on an input signal can be learned from task-sized data. But that revision is by
  **re-masking**, which needs no editing objective, not token-to-token editing.

**Brain-to-text.** In the Brain-to-Text Benchmark '24 ([2412.17227](https://arxiv.org/abs/2412.17227)), the largest
gains came from a fine-tuned LLM that corrects and merges several decoders' outputs: the baseline was 9.7% WER and the
best entry 5.8%. Errors in brain-decoded text are correctable, and a corrector can be fine-tuned on task data. That was
a separate text model working from candidate transcripts, though, not an editor inside the decoder.

**What this means for one step.**
- **The new skill has little data.** Learning token-to-token editing from ~10⁵ sentence tokens through LoRA r8 has no
  precedent.
- **The noise mismatch** already flagged for random-token corruption (2606.17175, SCOPE) applies in full.
- **Memorisation risk.** With 150 epochs over the same 8k sentences, the editor could learn to edit towards training
  sentences. That helps seen sentences, not the unseen ones we score.

None of this rules one step out. It is untested, and the literature points to two steps, or to re-masking.

**Cheaper checks before any 30-hour run:**
1. **Run 3's revision-headroom check** (decoding-order sweep, tonight). Each decoded token is hidden and re-predicted.
   That is exactly Mask-CTC-style revision, on a model already trained for it (M2T), with no new run. If one-round
   edits fix much more than they break, the **re-masking sampler** (section 4, now implemented, entry
   `fast-dllm-v2-1.5b-remask`) may capture most of the benefit at zero training cost: one `--eval-only` decoding
   job on run 3's BEST. If even the oracle (only the edits that fix a sentence) barely lowers WER,
   there is little for any revision to fix.
2. **Short one-step probe.** Continue run 3's BEST for ~10–20 epochs with `edit_rate: 0.1` and editing on, in run 3's
   own layout. Then measure edit fix / break on BIT-val. That is a few GPU hours instead of ~30. It needs a small
   `train_stage2.py` addition, because `--init` loads only the encoder and projector, not a trained Stage 2 model.

### Corpus (step 1)

- **FineWeb-Edu (`HuggingFaceFW/fineweb-edu`).** English, quality-filtered web text, ODC-By licence, the corpus SCOPE
  post-trained on. It streams, so no full download is needed, and a 10B-token sample subset is far more than needed.
- **Possible mix-in: the Llama-Nemotron post-training data** Fast-dLLM v2 was converted on (Fast-dLLM v2 §A.1). It
  keeps the base near its own distribution (less forgetting), but it is mostly chat, maths, code and reasoning, far
  from our short spoken-style sentences.
- **Conversational corpora,** which would match our domain better, are small or have restrictive or unclear licences.
  General English is the safe default, and Stage 2 already adapts to the sentence domain.
- **Format.** Pack documents into 2,048-token rows (64 blocks), every token a target, as in Fast-dLLM v2's own
  training. Each block is then a target conditioned on clean earlier blocks, which is the geometry of our aligned
  sentence block, so step 1 needs no special sentence layout.
- **Leakage filter.** Drop any document containing a BIT train/val/test sentence, matched on normalised text as
  `stage2.seen_mask` does, so that "unseen" stays unseen, and log how many were dropped. T15's sentences are
  ordinary English and could occur on the web.

### Token budget

**Reference points:**
- **Fast-dLLM v2, the AR → block-diffusion conversion of this very model.** A much larger change than adding an edit
  skill. 1.5B: 6,000 steps × 256 × 2,048 = 3.15B tokens per its appendix; the abstract says "~1B". Full SFT on 64
  A100s with ZeRO-3, ~8 h (~512 A100-hours), lr 2e-5, the complementary pair and [noisy‖clean] as here.
- **LLaDA2.1.** T2T is mixed into the continual pre-training and SFT of 16B and 100B MoE models. No token count for
  the editing stage is given in what we read, and the scale is far beyond ours. The usable lesson is only that editing
  was trained at pre-training scale.
- **SCOPE.** Self-correction added to LLaDA-8B with 500 LoRA steps (r 64, attention only), 4 sequences per step,
  FineWeb-Edu, ~20 min on one A100. The sequence length is not stated: about 8M tokens if they were 4,096 long, which
  is our inference.
- **Self-generated T2T (2606.17175).** "A short LoRA continued-pretraining pass" on LLaDA2.1-mini, with no numbers in
  the abstract.

**Proposal.** Adding editing to a model that already denoises is closer to SCOPE than to the conversion, so plan on
10⁷–10⁸ tokens. Default to 100M (~3% of the conversion), with checkpoints at 10M / 30M / 100M checked on held-out
text:
- (i) T2T recovery accuracy on randomly corrupted text;
- (ii) M2T loss no worse than the original base's, i.e. no forgetting;
- (iii) the sweep-style fix / break rate of edits on the model's own drafts.

Stop early if (iii) plateaus. Going past ~300M tokens makes sense only if (iii) still improves.

### GPU memory (the user's question)

**Fixed costs.** 1.54B parameters (Qwen2.5-1.5B, 233M of them the tied embeddings), fp32 weights + bf16 autocast +
AdamW, as our Stage 2 runs.

**Activations.** Each corpus token passes through the model 4 times: the noisy and clean halves, and the two
complementary copies. So one 2,048-token row is 8,192 positions through 28 layers. That is roughly 3–5 MB of saved
activations per position without checkpointing (estimate; LoRA dropout keeps extra copies of layer inputs), and
~0.4 MB per position with per-layer activation checkpointing. The logits at supervised positions add ~2–4 GB per row.

| | Weights + grads + AdamW | Activations, one 2,048-token row | Total, 1 row per micro-step | With activation checkpointing |
|---|---|---|---|---|
| Full fine-tune | ~25 GB (6.2 + 6.2 + 12.3) | ~25–30 GB | **~55 GB** | **~30 GB** (1 row), ~40 GB (4 rows) |
| LoRA (r 8–64, 7 projections) | ~7 GB (frozen fp32 base + adapters) | ~30–40 GB | ~45 GB | **~10–15 GB** |

- **Batch size.** Fast-dLLM v2's 256-row batches come from gradient accumulation, so the batch does not change
  memory.
- **Fit.** Either variant fits the 96 GB card on its own. Neither should share it with a Stage 2 job: BIT's AR
  sentence stage peaked at 36 GB (DESIGN.md §2), and run 3's diffusion stage processes 4× the positions per trial.
- **Measure before trusting.** These are back-of-envelope figures. The first GPU smoke should log
  `torch.cuda.max_memory_allocated`.

### Wall time on our RTX PRO 6000 Blackwell

**Anchor.** Fast-dLLM v2's own conversion ran this objective at about 6M tokens per A100-hour (3.15B / ~512).

**The card is not pinned down.** Its dense BF16 peak is not settled in what we found (sources range ~250–500 TFLOPS,
against the A100's 312). Our flex-attention backward also runs with reduced tiles on this GPU (DESIGN.md §8). One
GPU has no ZeRO-3 communication, which helps.

**Assumption.** One card ≈ 1–1.5 A100s, i.e. ~6–9M tokens per hour for a full fine-tune. LoRA is ~1.3–1.5× faster (no
weight gradients for the base), and activation checkpointing costs ~1.3×.

| Step-1 tokens | Full fine-tune | Note |
|---|---|---|
| 10M | ~1–2 h | SCOPE scale |
| 100M | ~11–17 h | proposed default |
| 300M | ~1.5–2 days | |
| 1B | ~4.5–7 days | the "~1B" of Fast-dLLM v2's abstract |
| 3.15B | ~2–3 weeks | its actual 1.5B conversion |

A 15-minute GPU smoke would replace the assumed factor with a measured rate.

**Timeline.** The card is busy with the queue through about Saturday 2026-10-24.
- Two steps at 100M tokens: ~0.5–0.7 day for step 1, plus ~1.3 days for Stage 2, so results ~2 days after the queue
  clears.
- One step: ~1.3 days.
- The M2T-only control would add as much again.

### Step 1 script: `scripts/posttrain_edit.py` (written and CPU-tested, not run)

```
<bit env>/python scripts/posttrain_edit.py --out checkpoints/hf/Fast_dLLM_v2_1.5B-edit --exclude-bit \
    --tokens 100M --save-at 10M 30M [--grad-ckpt] [--lora 64] [--edit-rate 0.1]
```

**What it does:**
- **Data.** FineWeb-Edu `sample-10BT`, streamed, tokenised with the base snapshot's tokenizer, each document followed
  by `<|endoftext|>`, packed into 2,048-token rows with every token a target.
- **Leakage filter.** `--exclude-bit` drops documents containing any BIT train/val/test sentence of 4 or more words,
  word for word after stage2's normalisation. Shorter sentences ("thank you") are in every corpus and are skipped.
  The counts go to posttrain.json.
- **Held-out set.** The first 64 rows.
- **Optimisation.**
  - Each step is `text_block_diffusion_ce(model, ids, mask_id, edit_rate=0.1)`.
  - The loss is the mean over each micro-batch's supervised tokens, averaged over 64-row steps (131k tokens): about
    760 steps for 100M tokens.
  - AdamW at lr 2e-5 (Fast-dLLM v2's), 50 warmup steps, then constant, as SCOPE did, so that the 10M / 30M / 100M
    snapshots are comparable.
  - Clip 1.0; fp32 weights + bf16 autocast; seeded.
- **Variants.** `--lora R` (alpha 2R, no dropout, BIT's seven projections) is merged into every snapshot.
  `--grad-ckpt` checkpoints each layer: the vendored decoder has no checkpointing under transformers 4.48.
- **Held-out check.** At step 0 (the base), every 50 steps and at each snapshot:
  - (i) T2T CE and accuracy on corrupted tokens;
  - (ii) M2T CE and accuracy without corruption, against step 0, to catch forgetting.

  Check (iii), fix / break on the model's own drafts, is not in the script. It comes from decoding with the Stage 2
  model (sweep-style).
- **Outputs.**
  - `<out>/tokens_<N>/`: fp32 safetensors, about 6.2 GB each, kept in fp32 so that small updates survive. The base's
    tokenizer, added-tokens and code files are copied alongside, so the snapshot loads as the base does.
  - `<out>/log.jsonl` and `<out>/posttrain.json`: args, base, data counts, git commit, peak memory.
- **M2T-only control.** The same command with `--edit-rate 0`. The held-out T2T check still runs, at
  `--eval-edit-rate`.

**Not exercised on CPU.**
- FineWeb-Edu streaming under BIT's `datasets` 2.17.1. This needs network on the GPU host; `--text-file` can stand in.
- `--exclude-bit`, which needs BIT's data.
- The real model's memory and speed.

A 15-minute GPU smoke (e.g. `--tokens 2M --save-at --eval-every 5`) covers all three and measures tokens per hour.

**Registry entry for a snapshot,** for Stage 2. `llm_entry` needs only `local_dir`. Leave out `repo_id` and fetch other
entries by name, since `fetch_llms.py` without names expects one.

```yaml
fast-dllm-v2-1.5b-posttrained-edit:   # step 1 output; provenance in its posttrain.json
  local_dir: checkpoints/hf/Fast_dLLM_v2_1.5B-edit/tokens_100000000
  loader: diffusion
  sampler: {block_size: 32, small_block_size: 8, threshold: 0.9, align_sentence_block: true, edit_rate: 0.1,
            editing_threshold: 0.5, max_post_steps: 16, max_edits: 2}
```

**Stage 2 then needs nothing new:** `train_stage2.py --llm fast-dllm-v2-1.5b-posttrained-edit`, with run 3's other
arguments.

**Where to try the alternatives.** The open questions below (keep supervision, self-generated corruption) are cheapest
to try in step 1, which uses no brain data.

## Open questions

1. **Keep supervision.** Should the clean visible target tokens be supervised too (calibrated "keep or fix")? It is
   safer, but probably fewer edits. If the revision run shows edit storms (many edits that break words), switch to
   this. If it shows almost no edits, switch to question 2.
2. **Random vs self-generated corruption** (SCOPE / 2606.17175). Random tokens are easy to spot. The model's own
   errors are not.
3. **edit_rate 0.1** is untested, and so are the three sampler values. The sampler values can be tuned after training
   on BIT-val; the rate cannot.
4. **The layout is confounded.** The aligned layout changes the input too. If this run wins, (a) shows whether it won
   without revision. A layout-only training run would cost another ~30 h.
5. **Launch decision.** Run 3's revision-headroom numbers (fix vs break rates of one-round edits) are the cheapest
   evidence for whether revision has anything to fix. Read them before launching.
6. **Follow-ups for other sessions' files** (not edited here):
   - **Sweep.** `scripts/decoding_order_sweep.py`'s `traced_denoise` is a copy of run 3's sampler. On this run it
     would decode without edits, and `matches_run` would flag it. The sweep should use the model's own trace
     (`model.trace = []`) instead.
   - **Viewer.** `scripts/denoise_viewer.py`'s `replay` rejects passes with an empty `"m"`. It also does not apply
     `"e"` edits or `"r"` re-masks; with `"r"`, positions are listed again after they were committed.
   - **One-step probe.** Training on from a trained Stage 2 checkpoint needs `train_stage2.py` to load one (`--init`
     loads only the encoder and projector).
7. **Revising earlier blocks** (LLaDA2.1's "multiple block editing") is not implemented. It is irrelevant while every
   sentence fits in one block.

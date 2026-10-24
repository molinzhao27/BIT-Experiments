# Instruct + diffusion loss: a control for run 3 (prepared, next on the GPU)

**Status (2026-10-21).** The code and CPU tests are ready. Nothing has run on the GPU yet.
- **Next on the GPU** (user, 2026-10-21): it runs right after the current job (`B_ssl_s42`, ending ~13:00 UTC
  Oct 22).
- **Host:** the parent session installs it from a separate copy of the new-layout code, so the running queue's code is
  not touched. Smoke-test it there first (below).

**Question.** Run 3 (Fast-dLLM v2 1.5B, block diffusion) reached 25.2% test WER. Run 1 (Qwen2.5-1.5B-Instruct,
left to right, same recipe) reached 55.7%. The decoding-order sweep showed that decoding order is worth at most about
1 pp. So the gain comes from one of two things:
- (a) training with the block-diffusion objective;
- (b) NVIDIA's ~1B-token conversion of Instruct into a diffusion model.

This run starts from run 1's Instruct weights and changes only the objective and sampler to run 3's. It separates (a)
from (b).

**Design.** Everything is run 3's except the starting weights:
- the same model code (`BCI_DiffusionLLM`, vendored Fast-dLLM v2 architecture);
- the same loss, block size 32 and sampler (small block 8, threshold 0.9);
- arm B from `checkpoints/ssl_human/BEST_core`, the same projector, LoRA (r8/α32, seven projections), BIT prompt,
  contrastive term, 150 epochs, selection by greedy val WER, evaluation with null controls, and seed 42.

Entry `qwen2.5-1.5b-instruct-diffusion` in `configs/llms.yaml`: run 1's pinned snapshot (`Qwen/Qwen2.5-1.5B-Instruct`
@ `989aa798`, `checkpoints/hf/Qwen2.5-1.5B-Instruct`, already on the host), `loader: diffusion`, `mask_id: 151665`,
and run 3's `sampler`.

```
<bit env>/python scripts/train_stage2.py --name B_ssl_qwen25-1.5b-instruct-diffusion_s42 --arm B \
    --init checkpoints/ssl_human/BEST_core --seed 42 --llm qwen2.5-1.5b-instruct-diffusion
```

These are run 3's arguments (its run.json), with only `--name` and `--llm` changed. `train_stage2.py` needs no change.
Nothing needs downloading: the host has the snapshot from run 1. The code must be synced to the host first.

## 1. Instruct's weights in the Fast-dLLM v2 architecture

Fast-dLLM v2 is a Qwen2 with the same modules and parameter names: q/k/v biases, GQA with 2 KV heads, RMSNorm, SwiGLU
MLP, and tied embeddings. Both checkpoints hold the same 338 tensors with the same shapes, 3,087,467,144 bytes each,
with no `lm_head` tensor. Only the forward differs:
- in training, the [noisy ‖ clean] input, its block-diffusion mask, and RoPE applied to each half separately;
- in decoding, the block-causal mask and KV cache.

All of this lives in the vendored code that run 3 already uses.

**Loading.** `BCI_DiffusionLLM.from_pretrained` reads the snapshot's `config.json`:
- `model_type: qwen2` → `qwen2_as_fast_dllm_config` keeps every field, adds `bd_size: 32`, and loads the weights as
  they are, in fp32 as run 3. It refuses sliding-window attention, RoPE scaling and activations other than silu, none
  of which apply here.
- Otherwise (run 3) → the same call as before, unchanged.

**Config, Instruct converted vs Fast-dLLM v2 1.5B (pinned `config.json`s).** Equal:
- `vocab_size` 151936, `hidden_size` 1536, `intermediate_size` 8960, 28 layers, 12 heads, 2 KV heads;
- `rope_theta` 1e6, no RoPE scaling, `rms_norm_eps` 1e-6, `max_position_embeddings` 32768;
- all 28 layers full attention (`use_sliding_window: false`; Fast-dLLM's `sliding_window: null` is what the class
  derives from Instruct's unused 32768), `max_window_layers` 21;
- `tie_word_embeddings: true`, bos 151643 (`<|endoftext|>`), eos 151645 (`<|im_end|>`), `bd_size` 32 (added);
- `torch_dtype` bf16 on disk, loaded as fp32.

Different:
- `architectures` and `auto_map`: metadata, and the snapshot's own code is never run.
- `pad_token_id`: 151645 in Fast-dLLM v2, unset in Instruct. It only makes EOS's embedding row gradient-free, and the
  embeddings are frozen under LoRA anyway. Left as Instruct ships it.

**Tokenizer.** DESIGN.md §8 checked that Instruct's and Fast-dLLM v2's tokenizers give identical ids for BIT's prompt
and every sentence. Both have the same `vocab.json` blob, and both use EOS `<|im_end|>` and pad `<|endoftext|>`.
Fast-dLLM v2 adds only `|<MASK>|` = 151665, and lists it instead of Instruct's 13 chat/vision tokens as
`additional_special_tokens`. Its `merges.txt` only adds the `#version: 0.2` header line (14 bytes);
`tokenizer.json` is re-serialised with the mask added.

The control uses Instruct's tokenizer, as run 1 does. The mask never passes through a tokenizer: it is inserted by id,
and the sampler sets its logit to −∞. Instruct's tokenizer has no string for 151665 (`convert_ids_to_tokens` →
None), so a trace viewer should show masks by id.

## 2. The mask token

**Facts.** All read from the pinned safetensors by HTTP range request; scripts are in the session's scratchpad.
- **No new row is needed.** Qwen2.5 pads its vocabulary to 151,936 rows. The tokenizer uses 151,665 ids
  (151,643 BPE + 22 special), so row 151665 exists in Instruct and is never used. Fast-dLLM v2 put `|<MASK>|` at
  exactly that id, so `mask_id: 151665` in the entry is the same id as run 3's (from its `added_tokens.json`).
- **Instruct's row 151665 is the generic "untrained token" vector.** Rows 151664–151935 all have norm 0.425 and are
  nearly identical. A trained ordinary token has a mean norm of 1.05. This vector is at cos 0.90 with the mean of
  ordinary rows. Byte tokens that never occur in valid UTF-8 share it too, e.g. 0xC0, 0xC1 and 0xF5–0xFF: 14 of
  the first 512 ids.
- **NVIDIA started from that row and barely moved it.** Fast-dLLM v2's mask row is at cos 0.988 with Instruct's row
  151665 (norm 0.437). Their ~1B-token full fine-tune moved the EOS row far more (cos 0.886).
- **No BIT token shares the mask's direction except EOS.** I tokenized BIT's prompt and every val/test reference and
  decode of runs 1–3: 4,418 distinct ids. The highest cosines to the mask row were:
  - `<|im_end|>` (EOS, also the block padding): 0.882;
  - every other token: 0.54 or less (median 0.15).

  In Fast-dLLM v2, EOS vs mask is at 0.692: NVIDIA's training separated them by moving EOS.

**Decision: use Instruct's own row 151665, frozen.** Mask init = Instruct's existing row, not trainable.
- It is NVIDIA's own starting point. Their training left it nearly where it was (cos 0.988), so the conversion's effect
  is not in the mask row.
- Run 3's mask row was frozen too: LoRA r8 on the seven projections, `modules_to_save: []`. So the trainable set is
  exactly run 3's, and only the starting weights differ.
- Its norm is in range and its direction is distinct from every BIT token except EOS (0.88 cos). That is enough for a
  layer-0 LoRA to tell a mask from a word.

**Trainable parameters** (BIT's full Stage 2 model, real sizes, LLM on the meta device; asserted in the tests):

| | Run 3 | Control |
|---|---|---|
| LoRA (28 layers × 7 projections, r8) | 9,232,384 | 9,232,384 |
| NDT encoder | 9,882,368 | 9,882,368 |
| Projector | 591,360 | 591,360 |
| Contrastive head + temperature | 1,573,889 | 1,573,889 |
| **Total trainable** | **21,280,001** | **21,280,001** |
| Frozen LLM (tied embedding counted once) | 1,543,714,304 | 1,543,714,304 |

**Alternatives considered (not taken):**
- *Train only the mask row:* +1,536 parameters (+0.007%). peft 0.11.1 (BIT's env) has no `trainable_token_indices`,
  so it needs a separate parameter spliced into `block_diffusion_ce` and `_denoise`, which touches run 3's code paths.
  It would also make the control differ from run 3 in its trainable set.
- *`modules_to_save: [embed_tokens]`:* +233,373,696 parameters (11× run 3's total). Because the embeddings are tied,
  this also retrains the output head.
- *Re-initialise the row* (e.g. as the mean of the ordinary rows): this departs from NVIDIA's starting point, for no
  clear gain.

If the control lands far from run 3, "train the mask row" is the cheapest follow-up to rule out an input confound (see
Risks).

## 3. Block size and sampler

These are run 3's: `bd_size` 32, set by the conversion and checked against the sampler's `block_size` in `_configure`.
Training is the same: mask ratio U(1e-3, 1) per block, a complementary pair, token shift, CE summed, and targets
padded with EOS to their block. Decoding is the same too: small block 8, threshold 0.9, with the AR first token for
blocks after the first. The revision options are absent, so off.

## Verification (CPU, no GPU)

`tests/test_instruct_diffusion.py`, in BIT's env (torch 2.7.1, transformers 4.48.2, peft 0.11.1); the project .venv
skips it for lack of peft.
- **Run 3 unchanged.**
  - The `fast-dllm-v2-1.5b` and `qwen2.5-1.5b-instruct` entries equal those at `0b98fa9`.
  - `use_llm` gives run 3 the same config as `0b98fa9`'s `text_llm.py`.
  - A Fast-dLLM v2 snapshot loads exactly as through run 3's call.
- **Conversion.**
  - Instruct's pinned config converts to Fast-dLLM v2's except `architectures` / `auto_map` / `pad_token_id`.
  - A tiny random Qwen2 (all weights random, biases and norms too) keeps every tensor, with the head tied.
  - With one-token blocks, i.e. causal attention, the vendored model gives Qwen2's hidden states and logits exactly:
    max |diff| 0.0. With 32-token blocks the difference is 4.7, so the check is not vacuous.
- **Full size.** Trainable parameters are identical to run 3's (table above); the embeddings are frozen and tied.
- **The new entry end to end.** Registry → `use_llm` → BIT's own model build (NDT encoder, projector, contrastive
  head, LoRA) on a tiny converted Qwen2. Then:
  - the loss is finite;
  - every LoRA B matrix and the projector get gradients, while the embedding (mask row included) gets none;
  - greedy and nucleus decodes stop within 25 tokens and never emit the mask;
  - a mask id without an embedding row is refused.

```
tests/test_instruct_diffusion.py::test_registry_keeps_runs_1_and_3 PASSED
tests/test_instruct_diffusion.py::test_use_llm_gives_run3_what_run3s_code_gave_and_the_control_the_same_sampler PASSED
tests/test_instruct_diffusion.py::test_fast_dllm_snapshot_loads_as_before PASSED
tests/test_instruct_diffusion.py::test_instruct_config_converts_to_fast_dllm_v2s PASSED
tests/test_instruct_diffusion.py::test_converted_snapshot_keeps_every_weight_and_qwen2s_causal_forward PASSED
tests/test_instruct_diffusion.py::test_full_size_trainable_parameters_are_run3s PASSED
tests/test_instruct_diffusion.py::test_control_entry_builds_trains_and_decodes PASSED
7 passed in 2.61s
```

The whole suite in the same env: 71 passed, 1 skipped. `tests/test_diffusion_revision.py` passes, including its
bit-identity checks against run 3's code. `tests/test_bit_ssl.py` errors at import because this CPU env copy has no
matplotlib; that predates this change.

**Not tested on CPU:** the real 1.5B weights (no 3 GB download here), flex attention and memory. The shapes are run 3's,
so the compiled kernels and memory should be too.

## Expected GPU cost

- **Wall time.** Same shapes, sequence layout and parameter count as run 3, so about **32 h** for 150 epochs, i.e.
  ~12.4 min per epoch including evaluation.
  - Run 3 took 1,900 min in all, ~62 min per 5 epochs, peaking at 79.7 GB allocated on the 96 GB card. The control's
    tensors have the same shapes, so expect the same peak.
  - Validation decoding can be slower while the model is unsure: fewer tokens clear 0.9, so more sampler passes.
    Run 3's sweep gives 0.10 s per trial at 4.8 passes and 0.17 s at 8.8 passes (one token per pass). A 713-trial
    validation pass is about 1–6 min, so 30 of them add at most about 2.5 h.
- **If Instruct needs more epochs to learn denoising from scratch.**
  - Run 3 was at 75.6% val WER after 5 epochs and 30.1% after 20. It started from a model that already denoises.
  - Run 1's best checkpoint was at epoch 100, run 3's at 115.
  - **Keep BIT's 150 epochs** for the matched comparison: runs 1 and 3 both used 150, and BEST is chosen on val. If
    the control's val curve is still falling at 150, the budget is the limit, and a longer run (about 64 h for 300
    epochs) is a separate decision.
- **Optional pre-check, minutes, no training.** `text_block_diffusion_ce` on BIT's training sentences, text only, with
  each base model frozen. It shows how far Instruct starts from a denoiser compared with Fast-dLLM v2. It needs both
  snapshots, which the host has.

## Host smoke test, before the full run

The one piece not exercised on CPU is loading the real 1.5B Instruct weights through the conversion. Run 3 loads the
same 338 tensor names into the same class, so this is low risk, but check:
- **Load.** `--no-train --limit 8` with the launch arguments and a smoke `--name`.
  - The log has no "weights ... newly initialized" and no "model of type qwen2" warning.
  - `run.json` shows `diffusion_sampler.mask_id` 151665, and `llm_path` is the Instruct snapshot.
  - The 8 decodes are written.
- **Training.** `--epochs 1 --eval-every 1 --limit 64`. The step runs through flex attention, and
  `torch.cuda.max_memory_allocated` is in run 3's range. Scaled to 64 trials, the epoch time should match run 3's
  per-trial time.

## Risks

- **EOS and mask inputs are close** (cos 0.88 in Instruct vs 0.69 in Fast-dLLM v2), and the embeddings are frozen. The
  noisy block holds both masks and visible EOS padding. LoRA must separate them inside the network, where NVIDIA's
  full fine-tune separated them in the embedding. This is part of "the conversion", so the control should not
  remove it, but it may slow learning.
- **Instruct has never seen masks or within-block bidirectional attention.** At step 0 its predictions in the
  diffusion layout are those of an AR model reading mostly-masked context. LoRA r8 at BIT's learning rate may adapt
  more slowly than full fine-tuning did. The token shift helps: it keeps Instruct's next-token head in use.
- **Early selection noise.** Selection uses greedy val WER every 5 epochs. If early checkpoints decode badly, BEST
  simply comes later; nothing breaks.
- **One seed.** Run-to-run spread is not yet known (B seeds 43/44 are queued). A result between the two anchors needs
  more seeds before it is read.
- **Host layout.** Per the project notes, the GPU host keeps the old layout until the queue ends. Sync the repo before
  launching.

## How to read the result

**What the conversion adds** (user, 2026-10-21). NVIDIA's ~1B tokens add little language knowledge on top of
Qwen2.5's ~18T pretraining tokens. What they add is the denoising *skill*:
- block-bidirectional attention;
- a trained mask input;
- parallel multi-token prediction.

So the control asks whether Stage 2 alone can teach that skill, and whether it then helps.

Compare BIT-test greedy WER, and unseen-sentence WER as DESIGN.md §8 asks. Run 1: 55.7% (unseen 59.3%); run 3: 25.2%
(26.5%). All three runs select BEST on val the same way.
- **Near run 3 (≈ 25%):** the objective is what helps. BIT's ~8k sentences through LoRA r8 are enough to learn the
  skill, and NVIDIA's conversion adds little.
- **Bad (toward run 1, ≈ 56%): ambiguous.** Either:
  - (i) the objective does not matter here: trained in Stage 2, it generalises no better than left-to-right; or
  - (ii) ~8k sentences through LoRA r8, with a frozen, never-trained mask embedding (§2), cannot learn denoising from
    scratch.

  The signals below, and the follow-up conversion, separate the two.
- **In between:** a partial skill, or an epoch budget that is too short. Check the val curve before attributing it.

### Signals within the control run

**Reference: run 3, not run 1.** Run 3 has the same objective, data and contrastive term as the control, so its losses
are directly comparable to the control's. Run 1's losses are left-to-right CE and are not comparable.

| Signal | Where | Run 3's values |
|---|---|---|
| Training denoising loss: CE + contrastive, per supervised token, over the last 5 epochs | `train_avg_loss=` in the job's stdout log (printed by BIT's trainer at every eval; not in `log.jsonl`) | in run 3's job log on the host (not pulled here) |
| Validation denoising loss (same quantity on BIT-val, random masks) | `val_loss_tf` in `log.jsonl` | 1.133 (epoch 5), 0.721 (10), 0.655 (20, its minimum), then rising to 0.849 (150) while WER kept improving |
| Validation greedy WER | `val_free_greedy` in `log.jsonl` | 75.6% (5), 39.1% (10), 30.1% (20), 24.4% best (115) |
| Minutes per 5 epochs (longer = a less confident sampler, more passes) | `minutes` in `log.jsonl` | ~62 |

**Readings, by epoch ~50 or at the end:**

| Training loss vs run 3's | Val loss vs run 3's | Val WER | Reading |
|---|---|---|---|
| Stays well above at the same epoch (underfits) | high | poor | **(ii)**: the skill is not learned from BIT's sentences. Run the follow-up conversion. |
| Reaches run 3's level | stays well above | poor | The training sentences are fitted, but not as a general skill: either (i) or memorisation. The general-text check below separates them; the follow-up decides. |
| Near | near | poor | Not a skill deficit; suspect decoding. Re-decode BEST with small block 1 (`scripts/decoding_order_sweep.py`) to check the 0.9 threshold's commits. |
| Near | near | near run 3 | The objective helps (top case). |

**Eval-only check that separates (i) from memorisation** (a small script, not written; minutes). Measure masked-token
CE / accuracy, with fixed masks, on held-out general text: the M2T half of `posttrain_edit.py`'s `heldout_metrics`.
Run it for four models:
- Instruct as is;
- the control's LLM with its LoRA;
- Fast-dLLM v2 as is;
- run 3's LLM with its LoRA.

If the control stays near Instruct's value, it learned no general skill: (ii). If it approaches Fast-dLLM v2's, it
learned the skill and the skill did not help: (i).

### Follow-up for the bad case: our own small conversion, then Stage 2

**Step 1.** Convert Instruct ourselves with Fast-dLLM v2's objective on ~100M general-text tokens:
- data: FineWeb-Edu with BIT's sentences excluded;
- loss: `text_block_diffusion_ce` with no edit term (`--edit-rate 0`, plain mask-to-token, NVIDIA's objective);
- full fine-tune (`--lora 0`), so the mask row and the embeddings are trained, as NVIDIA's were;
- `scripts/posttrain_edit.py --llm qwen2.5-1.5b-instruct-diffusion --edit-rate 0 --lora 0 --exclude-bit
  --tokens 100M --save-at 10M 30M`.

`posttrain_edit.py` (owned by the revision-prep session; not changed here) needs one fix first. It reads the mask id
from the base's `added_tokens.json`, which Instruct's snapshot lacks, so it should take the entry's `mask_id`
(151665). Its direct `Fast_dLLM_QwenForCausalLM.from_pretrained` already loads a qwen2 snapshot, with `bd_size`
defaulting to 32 and a model-type warning; `BCI_DiffusionLLM.from_pretrained` would load it without the warning.

**Step 2.** Stage 2 exactly as here, on the step-1 snapshot. It needs a new `configs/llms.yaml` entry with
`mask_id: 151665`: the snapshot keeps Instruct's tokenizer, which has no `|<MASK>|`.

**Cost:** ~11–17 h for 100M tokens (docs/REVISION_RUN.md's unmeasured estimate; ~4–5 h to the 30M snapshot), plus
~32 h of Stage 2.

**Reading:**
- **Recovers ≈ run 3:** the skill, learned at text scale, is what helps, and 100M tokens are enough.
- **Stays bad:** NVIDIA's larger conversion (3.15B tokens per its appendix) matters, or the objective does not.

## Code changes

- `src/bit_experiments/diffusion_llm.py`:
  - `BD_SIZE` and `qwen2_as_fast_dllm_config` (new);
  - `from_pretrained` converts a qwen2 snapshot and otherwise makes the same call as run 3;
  - `_configure` refuses a mask id with no embedding row;
  - a docstring paragraph.
- `src/bit_experiments/text_llm.py`: `use_llm` takes `mask_id` from the entry when it has one, else from
  `added_tokens.json` as before.
- `configs/llms.yaml`: the new entry.
- `tests/test_instruct_diffusion.py` (new) and this note.

# Aero block diffusion: Aero-1-Audio's LM as a block-diffusion decoder (path 1 ready to launch)

**Status (2026-10-22).** **Path 1 is the priority** (parent session, on the control's early curve below). Its code
and CPU tests are done, and the launch command is below. Path 2 is only a costed fallback, at the end. Nothing has
run on the GPU and nothing is committed. Where path 1 goes in the queue is the user's call.

**Question.** Two effects each lowered arm B's test WER (one seed each):

| LLM | Test WER | Change |
|---|---|---|
| Qwen2.5-1.5B-Instruct (run 1) | 55.7% | |
| Aero-1-Audio (`B_ssl_s42`) | 34.2% | Instruct + audio post-training: −21 pp |
| Fast-dLLM v2 1.5B (run 3) | 25.2% | Instruct + NVIDIA's block-diffusion conversion: −31 pp |

Do they combine? Aero and Fast-dLLM v2 are both fine-tunes of the same Instruct checkpoint.

## The control so far, and the decision rule

The Instruct + diffusion control (`B_ssl_qwen25-1.5b-instruct-diffusion_s42`, `docs/INSTRUCT_DIFFUSION_CONTROL.md`)
is Instruct's weights with run 3's objective and sampler, without NVIDIA's conversion. Its early val greedy WER
(%; reported by the parent session, 2026-10-22) tracks run 3's, slightly ahead:

| Epoch | 5 | 10 | 15 | 20 |
|---|---|---|---|---|
| Control | 64.0 | 35.6 | 29.5 | 28.4 |
| Run 3 | 75.6 | 39.1 | 32.8 | 30.1 |

So plain Instruct learns denoising from BIT's 8k sentences, and the objective, not NVIDIA's conversion, looks like
what helps. That is the case for **path 1: Aero + the diffusion objective, directly in Stage 2.** The evidence is
preliminary: epoch 20 of 150, one seed, val only.

Keep reading the control to its end (~2026-10-23 22:00 UTC) with this rule, on BIT-test greedy WER, unseen-sentence
WER and its in-run losses:

| Control's final test WER | Route | GPU-hours |
|---|---|---|
| **Near run 3 (≲ 30%)**, as it is tracking now | **Path 1** | **~32 h** plain (matched); ~12–14 h with `fast_step` |
| In between (30–45%) | Path 1 if the control's losses tracked run 3's, else the fallback | as chosen |
| Bad (≳ 45%) | Fallback: path 2 (b) | ~18–22 h with `fast_step`, ~41–45 h plain |

These bands are for one seed. Arm B's seed spread is not known yet (`B_ssl_s43/s44` are queued).

**Path 1 completes a 2 × 2 table at seed 42.** Only the frozen LM changes across rows:

| | Left to right (BIT's objective) | Block diffusion in Stage 2 |
|---|---|---|
| Instruct | run 1: 55.7% | control: running (28.4% val at epoch 20) |
| Aero | `B_ssl_s42`: 34.2% | **path 1** |

Path 1 and the control start from bit-identical trainable weights at seed 42 (tested; see Verification). Run 1 and
`B_ssl_s42` do not, because BIT's Aero load draws extra random numbers (DESIGN.md §8).

---

## Path 1: Aero + the diffusion objective in Stage 2 (ready to launch)

**Design.** Everything is the control's, and so run 3's, except the starting LM weights:
- the same model code (`BCI_DiffusionLLM`, the vendored Fast-dLLM v2 architecture);
- the same loss, block size 32 and sampler (small block 8, threshold 0.9);
- arm B from `checkpoints/ssl_human/BEST_core`, with the same projector, LoRA (r8 / α32, seven projections), BIT
  prompt, contrastive term, 150 epochs, selection by greedy val WER, null controls, and seed 42;
- the plain training step, as the control uses.

**Entry** `aero-1-audio-diffusion` in `configs/llms.yaml`: `loader: diffusion`, `mask_id: 151665` and run 3's
`sampler`. It points at BIT's own pinned Aero snapshot (`lmms-lab/Aero-1-Audio` @ `2efba6e`,
`checkpoints/hf/Aero-1-Audio`, the same one as in `configs/bit_baseline.yaml`), so nothing needs downloading. Its
tokenizer is Aero's, as in `B_ssl_s42`.

### Install on the host

Use a separate code copy, as for the control; the running queue's checkout is not touched.
- **Files:** copy `src/bit_experiments/diffusion_llm.py`, `configs/llms.yaml` and `tests/test_aero_diffusion.py`.
  - `text_llm.py` and `fast_dllm_v2/` are the control's.
  - The fast variant also needs `src/bit_experiments/fast_block_diffusion.py`.
  - `diffusion_llm.py` also carries the revision and `fast_step` options. They are off by default, and the control's
    path is unchanged.
- **The control is still running from `/workspace/ctrl-instruct-diffusion`.** Its process has already imported its
  modules, but don't overwrite a running job's code. Either install into a fresh copy, or copy after the control
  ends.
- **The snapshot:** `<code copy>/checkpoints/hf/Aero-1-Audio` must resolve to BIT's snapshot. On the GPU host that is
  `/workspace/KunzDecoder/checkpoints/hf/Aero-1-Audio`, the path BIT's patched configs use.
- **Before the GPU:** `<bit env>/python -m pytest tests/test_aero_diffusion.py` (CPU, seconds) should give 10 passed.

### Smoke tests (under the GPU lock, minutes)

1. Real-weight load and decode:
   ```
   <bit env>/python scripts/train_stage2.py --name smoke_aero-diffusion_load --arm B \
       --init checkpoints/ssl_human/BEST_core --seed 42 --llm aero-1-audio-diffusion --no-train --limit 8
   ```
   Check that:
   - the log says "All the weights of Fast_dLLM_QwenForCausalLM were initialized from the model checkpoint", with no
     "newly initialized" and no "lm_head differs";
   - `results/stage2/smoke_aero-diffusion_load/run.json` has `diffusion_sampler.mask_id` 151665, and `llm_path` and
     `tokenizer_path` point at the Aero snapshot;
   - the 8 decodes are written.
2. One training epoch:
   ```
   <bit env>/python scripts/train_stage2.py --name smoke_aero-diffusion_e1 --arm B \
       --init checkpoints/ssl_human/BEST_core --seed 42 --llm aero-1-audio-diffusion --epochs 1 --eval-every 1 --limit 64
   ```
   Flex attention should run and `max_mem_gb` should be in run 3's range (79.7 GB at full size). Then delete
   `results/stage2/smoke_aero-diffusion_*` and `checkpoints/stage2/smoke_aero-diffusion_*`.

### Launch

These are run 3's and the control's arguments, with only `--name` and `--llm` changed:

```
<bit env>/python scripts/train_stage2.py --name B_ssl_aero-1-audio-diffusion_s42 --arm B \
    --init checkpoints/ssl_human/BEST_core --seed 42 --llm aero-1-audio-diffusion
```

Expect ~32 h and a ~80 GB peak, as run 3 and the control. Defaults in effect: 150 epochs, `--eval-every 5`,
`--label-fraction 1.0`, `--null-decodings greedy`.

**Optional fast variant**, not matched to the control: same masks, loss and gradients, but a different random stream
after step 1. It needs the `fast_step` GPU smoke first (`docs/DIFFUSION_SPEEDUP.md`); expect an unmeasured
~12–14 h.

```
<bit env>/python scripts/train_stage2.py --name B_ssl_aero-1-audio-diffusion-fast_s42 --arm B \
    --init checkpoints/ssl_human/BEST_core --seed 42 --llm aero-1-audio-diffusion-fast
```

### 1.1 Getting Aero's LM out of its snapshot

Aero-1-Audio is `AeroForConditionalGeneration`, a custom class (`model_type: "aero"`). It holds:
- a Qwen2.5-1.5B causal LM (`language_model`);
- a Whisper-large-v3-style audio encoder (`audio_tower`, 32 layers, d 1280, 128 mel bins);
- a single linear projector 1280 → 1536 (`audio_modal_projector.linear`).

BIT loads it through `Qwen2AudioForConditionalGeneration` and uses only the LM, with `project_to_audio: false`.

`BCI_DiffusionLLM.from_pretrained` gets a new `model_type == "aero"` branch, `aero_language_model`:
- reads the 339 `language_model.*` tensors from the safetensors and strips the prefix. They map one-to-one, by name
  and shape, onto the vendored model's state dict. This was checked against the real file's header
  (828 tensors).
- leaves out the 489 audio tensors (`audio_tower.*`, `audio_modal_projector.*`);
- converts Aero's `text_config` with the control's `qwen2_as_fast_dllm_config` (adds `bd_size: 32`; refuses sliding
  window, RoPE scaling and non-silu activations);
- builds the model through transformers' own `from_pretrained(None, config=…, state_dict=…)` in fp32, exactly as
  run 3 and the control are loaded. The log should say every checkpoint weight was used and nothing was newly
  initialised.

**The head.** Aero ships `lm_head` and `embed_tokens` as two tensors, but its `text_config` says
`tie_word_embeddings: true`. Every sampled row was bitwise equal: 1,400 rows across the vocabulary plus the 64 extra
rows. So BIT's tied load loses nothing, as DESIGN.md §8 already noted. The loader checks the full equality on load
and refuses the snapshot if the two ever differ, because a tied model can hold only one of them.

Run 3's and the control's loads are untouched: `model_type` `qwen2` and Fast-dLLM's own type take the same calls as
before.

### 1.2 Config, dimensions, RoPE, tokenizer

Aero's `text_config`, converted, equals the control's converted Instruct config **except `vocab_size`** (tested).

| | Instruct (control) | Aero's LM (path 1) | Fast-dLLM v2 (run 3) |
|---|---|---|---|
| hidden / intermediate / layers / heads / KV heads | 1536 / 8960 / 28 / 12 / 2 | same | same |
| RoPE θ, scaling, RMSNorm ε, max positions | 1e6, none, 1e-6, 32768 | same | same |
| attention | full in all 28 layers (sliding window off) | same | same |
| tied embeddings | yes | yes (text_config; the top-level `tie_word_embeddings: false` is the wrapper's and does not apply to the LM) | yes |
| bos / eos / pad | 151643 / 151645 / unset | same | 151643 / 151645 / 151645 |
| **vocab_size = embedding rows** | 151,936 | **152,000** | 151,936 |
| weights on disk | bf16 | bf16 | bf16 |

**The 64 extra rows** (151,936–151,999) come from Aero resizing its embedding. All 64 hold one identical vector:
norm 0.329, at cosine 0.983 with the mean ordinary row, which is what mean-initialisation gives. No token uses them.
They add 64 logit columns, exactly as in `B_ssl_s42`, and 98,304 frozen parameters.

**Tokenizer.** Aero's `vocab.json` is byte-identical to Instruct's, and its merges are the same 151,387 pairs.
- Its `merges.txt` adds a `#version: 0.2` header.
- Its `tokenizer.json` writes `ignore_merges: false`, which is the default.
- DESIGN.md §8 already checked that Aero's, Instruct's and Fast-dLLM v2's tokenizers give identical ids for BIT's
  prompt and for every sentence.
- Aero adds three special tokens after Qwen's 22: `<image>` = 151665, `<video>` = 151666, `<|AUDIO|>` = 151667.

### 1.3 The mask token

**Facts** (read from the three pinned safetensors by HTTP range request):

| | Instruct | Aero | Fast-dLLM v2 |
|---|---|---|---|
| row 151665 | the "untrained" vector (norm 0.425), shared by rows 151664–151935 | the same kind of vector (norm 0.431), **bitwise identical** across rows 151664–151935, including `<image>`, `<video>` and `<|AUDIO|>` | NVIDIA's trained `|<MASK>|` (norm 0.437) |
| cos(row 151665, Instruct's row 151665) | 1 | 0.995 | 0.988 |
| cos(row 151665, its EOS row) | **0.882** | **0.740** | **0.692** |
| highest other BIT token (prompt, references and all decodes of runs 1–3 and `B_ssl_s42`) | ≤ 0.54 | 0.548 (a decode-only token), ≤ 0.44 for reference tokens | |

- **Aero never trained these rows as inputs.** `<|AUDIO|>`'s row is always replaced by audio features, and Aero's
  data has no image or video. They moved only as every unused row did, through the tied output softmax.
- **Aero's EOS already moved away from the mask direction** (0.740), much as NVIDIA's conversion moved it (0.692).
  Instruct's EOS did not (0.882). This makes the control's main input risk ("EOS and mask inputs are close") smaller
  here.

**Decision: `mask_id: 151665`, Aero's own row, frozen** (the control's rule).
- It is the same id as run 3 and the control, so traces, the viewer and the sweep keep "mask = 151665".
- Its vector is the same as any unused row's. Choosing 151668 (no token at all) would give the same model.
- In Aero's tokenizer, the literal text `<image>` maps to 151665. BIT's text never contains it, the mask is inserted
  by id, and the sampler sets its logit to −∞.

**Trainable parameters** (tested at full size, with the LLM on the meta device): the same set as run 3's and the
control's, tensor by tensor. LoRA is 9,232,384 (derived from the LLM and LoRA configs). The total depends on BIT's
data config, which sizes the neural read-in: 21,280,001 with the repo's config, 22,657,281 with the GPU host's patched one
(run 3's and the control's job logs). Embeddings are frozen and tied. Only the frozen LLM grows, by 98,304, to
1,543,812,608.

**How far Aero is from Instruct.** Relative weight change on sampled layers (0 / 14 / 27, attention, norms, one MLP)
is 2.0% for Aero and 5.2% for Fast-dLLM v2. The two changes are almost orthogonal: cosine ~0.01 (0.3 for layer 0's
q/k). Path 1 starts 2% away from the control's start, with every other weight identical.

### 1.4 Verification (CPU)

`tests/test_aero_diffusion.py` has 10 tests, run in BIT's env copy (torch 2.7.1, transformers 4.48.2, peft 0.11.1);
it is skipped without peft.
- **Registry.** Both entries are the control's with BIT's pinned Aero snapshot. The fast entry adds only
  `fast_step: true`. Earlier entries are untouched.
- **Extraction against BIT's own load.** A tiny snapshot is written in Aero's real layout: `model_type: "aero"`, bf16,
  `language_model.*` plus a separate equal `lm_head`, `audio_tower.*` and `audio_modal_projector.*`, with every
  weight random.
  - It loads into the vendored model with exactly the state dict that `Qwen2AudioForConditionalGeneration.from_pretrained`
    gives, which is BIT's own call. The head is tied in both.
  - With one-token blocks (causal) its hidden states and logits equal BIT's LM's (1e-5). With 32-token blocks they
    differ, so the check is not vacuous.
  - A snapshot with a differing head, or one that is not Aero's, is refused.
  - **Planted bug:** a wrong key prefix fails both load tests.
- **Config.** The real `text_config` converts to the control's config except `vocab_size`.
- **Full size.** The trainable set equals run 3's and the control's, and the LoRA count is derived from the configs.
  No total is hard-coded, because the host's data config changes it.
- **End to end.** registry → `use_llm` → BIT's model build (NDT encoder, projector, contrastive head, LoRA) on the
  tiny Aero snapshot.
  - The loss is finite. Every LoRA B matrix and the projector get gradients; the embedding gets none, and the mask
    row is unchanged.
  - Greedy and nucleus decodes stop within 25 tokens and never emit the mask.
  - A mask id with no embedding row is refused.
- **`fast_step`.** The fast entry gives the plain entry's loss and every gradient, in float64 with dropout off
  (worst relative difference ~1e-15). In fp32 the NDT encoder's query/key gradients differ by up to ~3e-3 of their
  max, from summation order alone.
- **Same start as the control.** At one seed, every trainable tensor (encoder, projector, head, LoRA) is bitwise
  equal between the two builds; only the frozen LM differs.
- **Path 2 tools** (below): `posttrain_edit.py` on an Aero base, and the merge.

**Whole suite** in the same env, without `test_bit_ssl.py` (no matplotlib in the env copy): **146 passed, 1 skipped**.
That includes the run 3 identity tests, the control's tests, and the revision and `fast_step` tests.

**Not tested on CPU:** the real 1.5B weights (a 4.8 GB download, not done here), flex attention and memory. The
shapes are run 3's except the 64-row larger head.

### 1.5 Cost and risks

**Cost.**
- **Plain:** the same shapes as run 3 and the control, plus 0.04% more head. Expect about **32 h** for 150 epochs
  (~12.4 min per epoch including evaluation) and a peak near run 3's 79.7 GB.
- **Fast:** estimated at ~12–14 h, but unmeasured.
- Early validation decodes can be slower while the sampler is unsure, at most ~2.5 h over the run. The control's
  early epochs show no sign of this.

**Risks.**
- **The control's caveats.** The mask row is frozen and never trained, the LM has never seen masks or within-block
  bidirectional attention, and LoRA r8 adapts slowly. The control's early curve says Instruct overcame all three.
  Aero's starting point is closer to Fast-dLLM v2's on the one input risk, EOS vs mask: cos 0.740 vs Instruct's
  0.882.
- **Aero's skill may not survive the layout change.** It was learned with causal attention over the audio prefix. In
  the diffusion layout the neural prefix sits in clean blocks, which see earlier blocks causally and their own block
  bidirectionally. The LoRA has to bridge that. Path 1 measures exactly this: does Aero's −21 pp survive under the
  diffusion objective?
- **One seed.** Compare against `B_ssl_s43/s44` once they exist.

---

## Fallback: path 2, a conversion step before Stage 2 (only if the control ends bad)

Not prototyped further; costed only. The idea: keep a model that already denoises (Fast-dLLM v2, NVIDIA's ~3.15B
tokens) and add Aero's skill of transcribing a span of non-token embeddings. Aero learned that skill from ~50k h of
public audio (~5B tokens, one day on 16 H100s; LibriSpeech test-clean 1.49% WER, per
[lmms-lab's post](https://www.lmms-lab.com/posts/aero_audio/)).

The time estimates are anchored on run 3's measured step: ~18k decoder tokens/s with LoRA, ~12k/s estimated for a
full fine-tune. They are unmeasured otherwise.

| Option | Data (licence) | Before Stage 2 | Total with Stage 2 (`fast_step` / plain) | Audio skill | New code |
|---|---|---|---|---|---|
| (a) text-only conversion of Aero's LM (`posttrain_edit.py --llm aero-1-audio-diffusion --edit-rate 0 --lora 0 --exclude-bit --tokens 100M`) | FineWeb-Edu, streamed (ODC-By) | ~9–17 h, ~55 GB (30 with `--grad-ckpt`) | ~21–31 h / ~41–49 h | at risk, unmeasured; re-learns denoising from ~3% of NVIDIA's tokens | none (script fixed, below) |
| **(b) Aero's audio tower + projector feeding Fast-dLLM v2, ASR with the diffusion loss** | LibriSpeech 960 h (CC BY 4.0, ~60 GB) | ~1 h gates + ~5–7 h per epoch with `fast_step` (~8–12 h plain), ~40–65 GB | **~18–22 h** / ~41–45 h | trained, and measured by dev-clean WER | audio front end + loader + loop, ~300 lines |
| (c) full Aero on audio + text | both | ~14–24 h | ~26–38 h / ~49–61 h | kept; denoising re-learned from ~100M tokens | (a) + (b) + a mixer |
| (d) task-vector merge, W_FastdLLM + α (W_Aero − W_Instruct) (`scripts/merge_aero_delta.py`, CPU minutes) | none | 0 (+ ~1 h gates) | ~13–15 h / ~33 h | unknown until probed | done |

**If needed, use (b), with (d) as its free first step.**
1. Merge, then check the merge's text denoising against Fast-dLLM v2's:
   `posttrain_edit.py --base <snapshot> --tokens 0 --save-at --edit-rate 0` logs the held-out M2T metrics at step 0
   and saves nothing.
2. Check zero-shot LibriSpeech dev-clean WER through Aero's tower and projector. This needs (b)'s front end.
3. Then run Stage 2 on the merge if it passes, else on (b)'s audio-post-trained model.

Why (b) and (d) over (a) and (c): a bad control would mean the denoising skill is the costly part, and only these two
keep NVIDIA's version of it.

## Code changes (all additive; nothing committed)

- `src/bit_experiments/diffusion_llm.py`:
  - `AERO_LM_PREFIX` and `aero_language_model` (new);
  - a `model_type == "aero"` branch in `BCI_DiffusionLLM.from_pretrained`;
  - a docstring paragraph.

  Run 3's, the control's, the revision options' and `fast_step`'s paths are unchanged.
- `configs/llms.yaml`: the entries `aero-1-audio-diffusion` (plain, matched) and `aero-1-audio-diffusion-fast`,
  appended.
- `scripts/posttrain_edit.py`, owned by the revision-prep session: the fix the control doc asked for.
  - The mask is `--mask-id`, else the entry's `mask_id`, else `added_tokens.json` as before.
  - The base loads via `BCI_DiffusionLLM.from_pretrained`, which is the same call for a Fast-dLLM v2 snapshot and
    also converts Instruct and Aero bases.
  - Its own tests still pass. This unblocks (a) and the control's own follow-up.
- `scripts/merge_aero_delta.py` (new): option (d).
- `tests/test_aero_diffusion.py` (new), and this note.

Not touched: `docs/DESIGN.md`, `CLAUDE.md`, `scripts/decoding_order_sweep.py`, `scripts/denoise_viewer.py`,
`text_llm.py`, `fast_block_diffusion.py`, `modeling.py`, the GPU host, `data/raw`.

---

## Arm A + Aero block diffusion (queued 2026-10-24, right after Aero arm B)

**Why.** It completes a 2 × 2 at seed 42, with Aero's LM in every cell: phoneme stage yes/no × left-to-right vs
block-diffusion objective. The question is whether diffusion helps arm A as much as it helps arm B.

| | Left to right | Block diffusion |
|---|---|---|
| Arm B (direct) | `B_ssl_s42`: 34.2% test | `B_ssl_aero-1-audio-diffusion_s42` (running) |
| Arm A (phoneme stage) | `A_ssl_s42` (queued) | **`A_ssl_aero-1-audio-diffusion_s42`** |

**Run.** Path 1's command with only `--name`, `--arm` and `--init` changed. It uses the plain step: `fast_step` stays
off until its train-mode GPU check passes.

```
<bit env>/python scripts/train_stage2.py --name A_ssl_aero-1-audio-diffusion_s42 --arm A \
    --init checkpoints/bit/finetune-ndt-phoneme-brandman_2024_text-feat_all/BEST --seed 42 --llm aero-1-audio-diffusion
```

**Init.** It is the same one `A_ssl_s42` uses: arm A's CTC phoneme encoder from 4c.
- It was fine-tuned from the SSL core (`checkpoints/ssl_human/BEST_core_ft`, masker off) with TX + SBP for 800
  epochs. BEST is epoch 709, with PER val 9.09% / test 9.25%.
- "finetune" in the path is BIT's training mode for a run with an init. BIT's own from-scratch phoneme stage (4a) is
  in a different directory.
- Unlike arm B's TX-only SSL core, this encoder includes its 512-channel read-in. Stage 2 loads all 127 tensors
  (`read_in_loaded: true`). The projector, LoRA and heads start fresh, as in arm B.

**Dry run (CPU, GPU hidden, the host's code copy).** It ran `train_stage2.py --no-train --limit 8` with this run's
arguments and stopped at the first decode.
- The model built, Aero's LM loaded, and the init passed `stage2.load_init`: 127 loaded; nothing skipped, missing or
  unexpected.
- There are 22,657,281 trainable parameters, the same as arm B's diffusion run.
- Against `B_ssl_aero-1-audio-diffusion_s42`'s `run.json`, 15 of 152 paths differ:
  - name, arm and init;
  - the dry-run flags;
  - the init report and trial counts.

  The whole resolved config is identical apart from the run name.

**Launcher.** Local, in `ops/arm-a-diffusion/`: a driver that runs under the shared GPU lock as one of the main
queue's jobs. It runs the training, with a guard that checks the init load in `run.json` before the first step. Then
it runs the eval-only ablations (`docs/EVAL_ABLATIONS.md`) on the finished run.

**Cost.** About 31 h: 11.9 min per epoch for 150 epochs plus final evaluation (~30 h), then ~0.5 h of ablations. Peak
memory is ~80 GB.

**Order.** The user first chose "after our queue". On 2026-10-24 they asked to finish the 2 × 2 first ("move it up in
the queue, before the contrastive"). The main queue now runs, after Aero arm B's diffusion run:
1. `A_ssl_s42`;
2. this run;
3. `C_ssl_l14_b1024_s42`;
4. `B_ssl_s43`, then `B_ssl_s44`;
5. Instruct without a prompt.

Estimates (UTC):
- this run starts ~Sun Oct 25 15:00–16:30 and ends ~Mon Oct 26 21:30–23:00;
- the queue ends ~Wed Oct 28 08:00–10:00, about 31 h later than without this run.

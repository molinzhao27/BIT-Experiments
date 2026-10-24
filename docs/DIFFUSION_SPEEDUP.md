# Faster block-diffusion training step (`fast_step`)

Status (2026-10-21): the code and its CPU tests are done, but it has not run on a GPU. It is off by default. Run 3's
step, and the Instruct + diffusion control (which must stay comparable with run 3), are untouched. This is
DESIGN.md §8 "Training speed for later diffusion runs", item 1, plus part of item 2.

## What changed

`block_diffusion_ce` (run 3) runs `[noisy ‖ clean]` at the batch length L for each trial and complementary copy:
2B = 16 rows of 2L tokens. Its attention lets noisy block k see itself and clean blocks < k, and clean block k see
clean blocks ≤ k. The noisy output at position p is decoded only when p + 1 is supervised, and it depends only on
noisy block(p) and the clean blocks before it. So each row needs:

- **noisy blocks** holding a position whose next position is a target. These run from the block of the last prefix
  token (the "tail" block) to the last target block. When the prefix fills whole blocks (tail 0, 3.4% of T15
  trials), the first of them is the pure-prefix block before the sentence, kept so that its last token can predict
  the first word;
- **clean blocks** from 0 up to the last needed noisy block, exclusive.

Everything else is skipped: the noisy copy of the rest of the prefix, the clean copy of the last target block, and
the batch's padding blocks in both halves. `src/bit_experiments/fast_block_diffusion.py` packs the needed blocks of
all rows into one sequence, padded to a multiple of `pack_quantum`. Each token keeps its original position for RoPE.
Token q may attend token k iff both are in the same row and `block_diff_mask` holds between their original
positions. On CUDA, this mask is a flex attention `mask_mod` run through the packed step's own compiled kernel, with
tiles that fit 101 KB of shared memory; on CPU it is a dense mask through SDPA, as in the vendored model. A stand-in decoder (`PackedDecoder`)
scatters the outputs back to their `[2B, 2L]` slots. `block_diffusion_ce` itself then runs unchanged: its mask and
T2T draws, token shift, selection and summed CE. Which blocks are needed depends only on the target labels, never
on the masks, so the shapes do not depend on the random draws. A device-side assert (no sync) checks that every
decoded output was computed.

**Option `share_clean`** (off by default): the two complementary copies of a trial read one clean copy of its context
instead of two. Without dropout it is exact too (tested). With LoRA dropout, both copies' predictions then share one
dropout draw on the context. The expected loss and gradient are unchanged, but the noise is correlated. That is a
small change to the recipe's noise, so it needs the user's OK before use.

**Random stream.** The masks come from the same calls in the same order, so a seed gives the same masks. LoRA
dropout draws a mask only for the token copies that are computed. So (i) those copies get different draws than in
run 3, with the same distribution, and (ii) the generator's state after the step differs, which changes every later
mask. A `fast_step` run is the same recipe, but not a bit-for-bit replay of a plain-step run with the same seed.
That is why the Instruct control keeps the plain step.

## Equivalence evidence (CPU, `tests/test_fast_block_diffusion.py`, 68 tests)

The model is a tiny random Fast-dLLM v2 (2 layers, 64-d, block 32) in BIT's LoRA (r 8, α 32, seven projections,
nonzero B). The tests run in BIT's env; the local CPU copy is described in `docs/REVISION_RUN.md` "Verification".

- **Plan.** The packed tokens are exactly the dependency closure of the decoded outputs under `block_diff_mask`,
  found independently by iterating the mask to a fixpoint. The packed mask is `block_diff_mask` restricted to those
  tokens, and pads see only pads. The flex block mask built from the same `mask_mod` marks exactly the tiles that
  hold an allowed pair.
- **Step, dropout off.** The tests check the same masks, labels, `n` and `corrupted`, the same summed CE and logits,
  and the same gradients of every LoRA parameter and of the input embeddings. The input-embedding gradients carry
  the encoder's and projector's. This holds over 3 seeds × 4 layouts: tails 0, 1, 5, 17 and 31; prefixes inside
  block 0; one-token targets; targets spanning two blocks; T15-like prefixes of 96–252; label holes. It also holds
  for dense and flex attention, `share_clean` on and off, and T2T (`edit_rate` 0.3) on.
  Worst case over 72 configurations: CE relative 2e-7, logits 6e-6 absolute, gradients 3e-6 relative to each
  tensor's max. Under bf16 autocast: CE 8e-6 relative, gradient cosine ≥ 0.999997.
- **Dropout on.** A test-only dropout keyed on the token copy (row, half, position) gives each copy the same mask in
  both steps. With it, the steps agree to the same tolerance with p = 0.2. So their only dropout difference is which
  random draw each copy gets. For `share_clean`, they agree once the plain step gives a trial's two clean copies one
  mask. With real dropout, the paired loss difference over 200 seeds has mean 0 (|z| < 4). With dropout off, the
  generator's state after the step is identical; with it on, it differs, as stated above.
- **Hook-up.** `BCI_DiffusionLLM.forward` gives the same loss, examples, logits, targets and gradients with
  `fast_step` on and off, including the neural tokens' gradient, `share_clean`, `pack_quantum`, and the revision
  options (aligned block, T2T). Bad option values are refused. Only the new registry entry turns the step on.
- **The tests catch real bugs.** Three planted bugs each fail 44–50 of the 52 plan, mask and step tests: forgetting
  the tail-0 block, letting noisy tokens see their own clean block, and wrong RoPE positions.
- **Recompiles (Dynamo, CPU `aot_eager`).** New plans of the same length reuse the compiled flex graph. The first
  new length recompiles once into a dynamic-shape graph, and later lengths reuse it. Inductor can't compile flex on
  macOS CPU, so the CUDA kernel's dynamic-length build is left to the GPU smoke. Run 3 relies on the same path.

The full suite, run in BIT's env copy without `test_bit_ssl.py` (which needs matplotlib), gives 136 passed and 1
skipped.

## Expected speed-up

The layout is run 3's real one: 1,426 val + test trials from run 3's decode traces, with prefixes of 85–502 tokens
(mean 210). Targets including EOS are 3–18 tokens (mean 8.0), tokenized with the local Qwen3-VL tokenizer as a
stand-in for Qwen2.5's. The estimate averages 300 random batches of 8, BIT's batch size. Train prefixes should be
similar (same task and prompt), but this was not checked.

| per step (mean)              | plain (run 3) | `fast_step` | + `share_clean` |
|------------------------------|---------------|-------------|-----------------|
| tokens through the decoder   | 12,384        | 4,111 (0.34×) | 2,512 (0.21×) |
| flex attention tiles (128²)  | 1×            | 0.30×       | 0.21×           |
| decoder FLOPs, fwd+bwd, incl. lm_head | ~100 TFLOP | ~33 (0.34×, range 0.22–0.50) | ~20 (0.21×) |

Where the plain step's tokens go: 39% are batch padding blocks in both halves (L is the batch maximum rounded up to
128: 384 in 76% of batches), 25% the noisy prefix copy, and 4% the clean copy of the last target block. Only 31% are
needed. The packed length's own padding (`pack_quantum` 512) adds 5.7%.

**Wall time (rough).** Run 3 took ~60–63 min per 5 epochs including eval, and the AR Instruct run ~15.5. If time per
step scales with decoder FLOPs (fit through both runs: ~0.03 s fixed + ~6.5 ms per TFLOP), training goes from ~11.5
to **~4–5 min/epoch** with `fast_step`, or ~3 with `share_clean`. Eval decodes are unchanged. So 150 epochs take
roughly 12–14 h instead of 31.7 h (~9–10 h with `share_clean`). Peak memory (79.7 GB) should fall roughly with the
token count. All of this needs the GPU smoke to confirm. The packed attention has relatively more partial tiles, and
the CPU-side costs per step don't shrink, except that building the mask gets cheaper. Run 3 materializes
`[16, 12, 768, 768]` for its block mask; the packed step materializes `[1, 1, T, T]`, with T ≈ 4k. On CPU that is
103 ms vs 41 ms. Building the plan takes 0.12 ms.

## How to enable

Add `fast_step: true` to the entry's `sampler` in `configs/llms.yaml`. For example, the new entry
`fast-dllm-v2-1.5b-fast` is run 3's entry plus `fast_step: true`. Optional keys (each needs `fast_step`):

- `share_clean: true`: see the caveat above;
- `pack_quantum: N`: a multiple of 128, default 512. In 300 batches, quanta of 128 / 512 / 1024 gave 19 / 6 / 4
  distinct packed lengths.

A future Aero → block-diffusion entry or revision-trained entry gets the same key. The step is used in training and
in the teacher-forced validation loss. Decoding is unaffected.

**GPU smoke (before a long run; the user's call, not done):**

1. `train_stage2.py --name smoke_fast --arm B --init checkpoints/ssl_human/BEST_core --seed 42 --llm fast-dllm-v2-1.5b-fast --epochs 1 --eval-every 1`
   (a full epoch, so that most packed lengths occur). Run it with `TORCH_LOGS=recompiles`. Expect one or two
   flex recompiles and no `cache_size_limit` warning. If that warning appears, raise `pack_quantum` to 1024: past
   the limit, flex falls back to its unfused, memory-hungry path.
2. Compare minutes per epoch and `max_mem_gb` with run 3, and train loss over the first epoch with run 3's log. The
   loss should match in distribution; it can't match exactly, because of the random stream.
3. Optional: rerun run 3 with this entry to show the WER matches within seed noise.

## Optional items (DESIGN.md §8)

- **Fewer small ops in mask building:** the packed step builds its block mask once per step over `[1, 1, T, T]`
  instead of `[2B, 12, 2L, 2L]` (above). Done.
- **Finer length bucket:** the packed step skips padding blocks, so the 128 bucket no longer costs compute. What
  remains is `pack_quantum` (configurable). The bucket is subsumed, so nothing more is needed.
- **`torch.compile` / CUDA graphs of the whole step:** not done. Can't be tested on CPU, since Inductor can't compile
  flex here. The step also has data-dependent shapes: the packed length per batch, and the supervised selection
  `hidden[supervised]` that `block_diffusion_ce` uses. CUDA graphs would need one capture per shape plus a
  fixed-size selection. Worth trying only if the smoke shows the packed step to be launch-bound.
- **Host syncs in `BCI_DiffusionLLM.forward`** (about 3 per trial in its per-trial loop, run 3's code): not
  changed, since run 3's path stays as it is. They come before the decoder runs and are minor.
- **Flex kernel options:** the packed step has its own forward tiles, sized for this GPU's shared memory (see "Kernel tiles" below). It keeps the vendored backward tiles. They have not been tuned for speed.

## Kernel tiles (fix after the first GPU smoke, 2026-10-23)

The first smoke (`smoke_aero-diffusion-fast_e1`, the GPU host) crashed on batch 1, in the first forward call of the
compiled flex attention: `OutOfResources: shared memory, Required: 107520, Hardware limit: 101376`. The plain step
compiles the same function (the vendored `fused_flex_attention`) without trouble. The arithmetic below is for bf16
and head_dim 128, and shows why.

- **Torch 2.7's forward default:** torch treats capability ≥ 9.0 (this GPU's 12.0 included) as an H100, so it runs
  the forward at `BLOCK_M 128, BLOCK_N 64, 8 warps, 3 stages`. The pipeline holds 3 stages of a K and a V tile,
  64 × 128 × 2 bytes each: 3 × 32,768 = **98,304 bytes**. That fits under 101,376, which is why the plain step
  runs: its mask is index arithmetic.
- **The packed mask's extra:** the packed mask reads four tables (`noisy`, `blk`, `ctx`, `grp`). Converting its
  128 × 64 boolean output into the score layout takes 128 × (64 + 8 padding) = **9,216 bytes** more.
  98,304 + 9,216 = **107,520**, exactly the error.
- **The fix:** the packed step now calls its own compiled kernel, `packed_flex_attention`, with
  `PACKED_FLEX_KERNEL_OPTIONS`. The vendored `fused_flex_attention` and `FLEX_KERNEL_OPTIONS` (plain step, run 3,
  the control) are untouched.
  - **Forward:** `BLOCK_M 64, BLOCK_N 64, 4 warps, 2 stages`. That is 2 × 32,768 + 64 × 72 = **70,144 bytes**,
    leaving 31,232 spare. The worst case adds a resident Q tile (64 × 128 × 2 = 16,384) and an fp32 mask scratch
    (64 × 68 × 4 = 17,408), which comes to 99,328 and still fits.
  - **Backward:** explicit, and the same as the vendored tiles (`BLOCK_M1 32, BLOCK_N1 64, BLOCK_M2 64,
    BLOCK_N2 32, 4 warps, 2 stages`), which ran for run 3. Each loop pipelines 2 stages of two 32 × 128 tiles
    (2 × 16,384 = 32,768) next to one resident 64 × 128 tile (16,384), so about 49 KB before scratch. The packed
    mask adds the conversion scratch of a 64 × 32 tile: 64 × 40 = 2,560 bytes, or 64 × 36 × 4 = 9,216 at worst.
    That totals about 52–59 KB, well under the limit.
  - **Divisibility:** every tile (64, 32) divides the block mask's 128-token tile, as flex requires.
- **Not measured.** This is a model of Triton's allocation, checked only against the two observed numbers (the
  plain step fits; the packed default needs 107,520). Inductor can't build flex on the local CPU, so the restart's
  smoke is the GPU check, with the plain step as the fallback. The smaller forward tile costs some attention speed,
  since K and V are read once per 64 queries instead of 128. Attention is a minor part of the step.
- **Tests:** the packed kernel passes these options. On the forced CUDA branch, the step calls it once per layer,
  never the vendored kernel, and still matches the plain step. The arithmetic above is a test.

## Host waits (`sync_free`, 2026-10-23): code and CPU tests only, not yet run on the GPU

**Background:** the fast run (`B_ssl_aero-1-audio-diffusion-fast_s42`) trains at 3.6–3.9 it/s: 4:38 per epoch,
against the plain step's 11:50. But the GPU is busy only ~65% of the time: 30% of 0.1-s samples are under 50%, and
it draws 416 of 600 W. The plain step ran at ~100%. Neither training thread is maxed out (main 56%, autograd 40%),
so the step stalls where the CPU waits for the GPU, which then idles while the CPU prepares the next work.

**Audit: where the fast step waits** (per step of 8 trials):

| where | waits | what follows the wait |
|---|---|---|
| `BCI_DiffusionLLM.forward`, per trial: `int(attention_mask.sum())`, `int((~is_target).sum())`, `bool(.all())`, `int(input_split)`, boolean indexing of the neural tokens | ~40 | Python between waits, with the GPU idle |
| `torch.tensor([eos] / [mask_id], device=…)`: blocking host-to-GPU copies, each followed by a stream sync | 2 (+8 with the aligned block) | the same |
| `pack_plan`: `labels.cpu()` | 1 | plan building, `create_block_mask` |
| `create_block_mask`: evaluates the mask at T² ≈ 17M positions by vmap, dozens of small kernels | 0, but CPU-heavy (41 ms on CPU tensors) | the GPU runs only tiny kernels |
| `hidden[supervised]`, `shifted[supervised]`: nonzero, after the decoder | 2 | the CPU cannot queue the head, loss or backward until the forward has run, so the backward starts with an empty queue (the autograd thread launches it while the GPU keeps up) |
| BIT's trainer: two `.item()` per step (loss log); accelerate's blocking batch copies to the GPU (one per tensor) | ~2 + ~9 | the next step's data loading (main thread, no workers), batch copy and encoder launches, with the GPU idle |
| BIT's encoder (`prepare_embeds`, vendored) | unknown | the profiling script lists them |

**`sync_free: true`** (needs `fast_step`; module `src/bit_experiments/sync_free_step.py`; entry
`aero-1-audio-diffusion-fast-syncfree`). The same step, waiting once, right after the encoder: one copy to the CPU of
the batch's small integer tensors and the encoder's neural-token mask, which exists only on the GPU. Everything that
depends on lengths is computed on the CPU from them, about 2 ms for a T15 batch, and sent in non-blocking copies from
pinned memory:

- **Clean sequences:** one gather from `[text embeddings ‖ neural tokens ‖ EOS ‖ filler]`, the same rows forward
  concatenates. The labels are built on the CPU.
- **Pack plan:** from the CPU labels. No `labels.cpu()` is needed.
- **Flex block mask:** built at block level. Each packed 32-token block has one mask value per pair of blocks, so the
  128-token tiles follow from a (T/32)² table, followed by torch's own `_create_sparse_block_from_block_mask`. Its
  fields are identical to `create_block_mask`'s (tested).
- **Selection:** each target is masked in exactly one copy, so the number of decoded outputs is known (the number of
  targets). Which copy masks each target is read on the GPU, and a GPU sort of their keys gives the boolean
  selection's order, so the head, loss and backward are queued behind the forward.
- **Mask and EOS rows:** `embed(torch.full(...))` instead of `embed(torch.tensor(..., device=))`.
- **T2T (`edit_rate` > 0):** the number of decoded outputs is random, so it keeps the boolean selection (one more wait
  after the decoder).

**Same maths, bit for bit:** the same random draws in the same order (masks and LoRA dropout), so also the same
generator state after the step. `tests/test_sync_free_step.py` (86 tests) compares it with `fast_step` using
`torch.equal`, with real LoRA dropout (p 0.2) on, over 3 seeds × 4 layouts. It covers dense and flex attention,
`share_clean`, `pack_quantum`, the aligned block, T2T, and full `BCI_DiffusionLLM.forward` calls (loss, examples,
logits, targets, every LoRA gradient, the neural tokens' gradient). Three planted bugs fail 10–56 tests each: no sort,
the wrong copy, no full tiles. Not a single random draw changes, so a `sync_free` run replays a `fast_step` run with
the same seed, up to the GPU's own nondeterminism.

**Expected effect** (estimates; the profiling script measures them). Of the ~93 ms of GPU idle per 267-ms step:

- **Per-trial waits and blocking copies:** a few ms.
- **`create_block_mask` with an idle GPU:** ~5–10 ms. It becomes ~2 ms of CPU work.
- **The wait after the decoder:** ~10–30 ms. Without it, the CPU queues the backward while the GPU still runs the
  forward, a head start of roughly forward GPU time minus forward CPU time, which covers the backward's launch gaps.

In all, perhaps 15–45 ms, so ~0.23–0.25 s per step (5–17% faster). The rest of the idle time is the next point.

**Left as is (proposals, not done):**

- **Trainer `.item()` and data loading:** BIT's trainer reads the loss with `.item()` at every step's end. The GPU
  queue then empties, and the next step's data loading (main thread, `num_workers` 0), batch copy and forward
  prologue all run with the GPU idle. This is likely the largest remaining part; the script's `fetch` and `item`
  phases measure it. Two fixes, in `train_stage2.py`'s `Stage2Trainer` rather than BIT's code:
  - DataLoader workers (2, persistent). BIT's dataset and collate draw no random numbers, and the batch sampler
    draws from its own seeded generator in the main process, so the random stream doesn't change;
  - reading the two logged values a step (or N steps) late. The values are the same, and the CPU can then prepare
    step k+1 while the GPU finishes step k.
  Together they could close most of the remaining gap to the plain step's ~100%. Both override BIT's training loop,
  so they are worth doing only once the profile shows how much they recover.
- **Recompiles:** none in steady state (the smoke had 2 recompile lines, then the shapes are dynamic), so bucketing
  more would buy nothing.
- **CUDA graphs** (`mode="reduce-overhead"`): they would need static shapes (packed length, batch length and
  decoded-output count padded to buckets) and graph-safe ops through peft and autograd. That's a large change, worth
  it only if, once the waits are gone, the profile still shows the GPU idling inside the forward or backward
  (launch-bound).

**Measure:** run `scripts/profile_fast_step.py --llm aero-1-audio-diffusion-fast` (about 4–5 min, through the GPU
lock, from a code copy with these files). In one process, on one model, it alternates windows with and without the
waits. It reports s/step, it/s, the phases (fetch, forward, backward, optimizer, item wait), nvidia-smi utilization
and power, the host waits per step by file:line (torch's CUDA sync debug mode), and profiled GPU-busy %. It writes
`results/profile/<name>.json` and saves no checkpoint.

## Training failure (2026-10-24): fix in place, gated at launch

**What happened:** the fast-step Aero run trained a much worse model than the plain step from the same init, seed
and config: val WER 108.6% vs 42.0% at epoch 5, val loss 1.665 vs 0.682, train loss 1.461 vs 1.088. Yet the
one-checkpoint check (`scripts/gpu_fast_step_check.py`) passed on CUDA. That check ran in eval mode with the same
masks, on 4 batches covering only packed lengths 4096 and 3584.

**Ruled out by reading the code:**
- mask and ratio sampling, complementary masks, and labels. The packed step calls `block_diffusion_ce` itself and
  replaces only the decoder;
- supervised positions dropped. An async assert would have stopped the run;
- loss normalisation and the contrastive weighting (the same `forward`);
- dropout (LoRA only; one draw per computed token copy), encoder noise, gradient checkpointing (none), and
  `diffusion_layout`;
- torch 2.7.1's flex templates on the non-divisible path, read for a template bug: none found.

**Leading cause:** the packed step's flex `mask_mod` captures tables whose length is the packed length T, and T
changes from batch to batch. With torch's default (automatic dynamic shapes), the second T recompiles
`packed_flex_attention` with dynamic shapes, and every later T runs on that graph. The failed run logged 2
recompile lines, and on CPU the default gives exactly 2 graphs for 3 lengths. The eval check's grad graph ran only
at its first two lengths. The plain step's `mask_mod` captures an int, so it never builds this graph.

**Fix:** `src/bit_experiments/fast_block_diffusion.py` sets `PACKED_FLEX_DYNAMIC = False`. Each packed length gets
its own static graph, the same kind of compile the eval check validated, and Dynamo's recompile limit is raised to
64. It costs a ~1 min compile per new length (about 6–10 per run) and changes no maths. CPU test:
`test_packed_kernel_compiles_statically_per_packed_length`.

**Launch gate:** `scripts/fast_step_gate.py`, ~15–20 min, peak ~80 GB.
- **Exit code:** 0 = the fixed fast step is validated, 3 = not validated (use plain), anything else = error (use
  plain).
- **The old step:** the 10-23 host module, byte-identical, is kept as
  `scripts/fast_gate/fast_block_diffusion_79bbfef.py` (md5-checked at load). A CPU test shows it computes exactly
  the fixed module's maths; only the compile mode differs.
- **Phases, in a fresh process so the compiles happen in a training run's order:**
  - 0, sanity: the dense-SDPA packed step equals plain on one batch, which makes it the reference.
  - 1, kernels: on ≥5 real packed lengths, the old and the fixed step against dense (loss, logits, gradients per
    group). The old step is compiled static at the first length, dynamic from the second.
  - 2, train mode: 160 steps each of plain, old and fixed (BIT's training step, dropout and noise on, the same
    batches), checking per-step losses, supervised counts and probe losses on fixed val batches.
- **rc 0** iff the sanity check passes, the old step's bug shows (it fails a phase 1 length, or trains worse than
  plain in phase 2), and the fixed step passes every phase 1 length and trains no worse than plain. If the old step's
  bug does not show, the gate cannot vouch for the fix and returns 3.
- **CPU tests:** `tests/test_fast_step_gate.py`. They cover the decision table, the old module's identity and maths,
  and three end-to-end runs on a tiny model: a broken old kernel gives 0, correct kernels give 3, a broken fixed
  kernel gives 3.

## Files

- `src/bit_experiments/fast_block_diffusion.py` (new): `pack_plan`, `packed_mask`, `PackedDecoder`,
  `fast_block_diffusion_ce`, `step_cost`.
- `src/bit_experiments/diffusion_llm.py`: `FAST_STEP_DEFAULTS`, option checks in `_configure`, the dispatch in
  `denoising_ce`, and a docstring paragraph. These are additive; with the flag off, the code path is unchanged.
- `configs/llms.yaml`: the new entry `fast-dllm-v2-1.5b-fast`, appended.
- `tests/test_fast_block_diffusion.py` (new).
- `src/bit_experiments/sync_free_step.py`, `tests/test_sync_free_step.py`, `scripts/profile_fast_step.py` (new),
  plus the `sync_free` option (`diffusion_llm.py`: `FAST_STEP_DEFAULTS`, its check, and a dispatch at the top
  of `forward`) and the entry `aero-1-audio-diffusion-fast-syncfree` (`configs/llms.yaml`); `packed_forward`
  and `PackedDecoder` take an optional prebuilt block mask.
- `scripts/fast_step_gate.py`, `scripts/fast_gate/fast_block_diffusion_79bbfef.py` (the old module, byte-identical),
  `tests/test_fast_step_gate.py` (new); `PACKED_FLEX_DYNAMIC` in `fast_block_diffusion.py`.

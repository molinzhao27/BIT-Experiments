# Retraining ablations: does the decoder use anything beyond phonemes? (prepared, not launched)

**Status (2026-10-24).** The erasure pair now runs with both decoders, AR and block diffusion (approved; §9). The
code, configs and CPU tests are ready (`tests/test_neural_ablations.py`, 21 passed in
BIT's environment). Nothing has run on the GPU, and nothing is committed. Every flag is off by default: without
`--ablation`, `train_stage2.py` runs exactly as before.

**Tier.** 2 (DESIGN.md §0): it explains the result, and is never in the headline table.

**Question.** The phoneme-free decoders make strongly sound-alike substitutions (`results/stage2/phonetic_errors.md`:
Fast-dLLM v2 23.6% vs 2.7% chance; Aero arm B 16.1%). The shuffled-input control removes them. Is phonetic
information all that reaches the LLM and gets used, or does anything beyond it matter? Two Stage 2 runs change
only what reaches the projector and LLM. Everything else is arm B's / arm A's Aero recipe: Aero-1-Audio, BIT's
prompt, LoRA r8 / α32, CE + BIT's contrastive term, 150 epochs, selection by greedy val WER, test with null controls.

| Run (proposed name) | `--init` | `--ablation` | What reaches the projector | Compare with |
|---|---|---|---|---|
| (a) `A_ssl_phonbottleneck_s42` | arm A's CTC model | `phoneme-bottleneck` | arm A's 41 per-frame log-posteriors (frozen model) | `A_ssl_s42` (queued, ~Sun 05:00 UTC) |
| (a′) `A_ssl_frozen_s42`, conditional | arm A's CTC model | `frozen-A` | arm A's frozen 384-d encoder output | (a) and `A_ssl_s42` |
| (b0) `B_ssl_frozen_s42`, control | `B_ssl_s42`'s encoder | `frozen-B` | `B_ssl_s42`'s frozen 384-d encoder output | (b) |
| (b) `B_ssl_phonerased_s42` | `B_ssl_s42`'s encoder | `phoneme-erasure` | the same output after a fixed LEACE phoneme eraser | (b0) |

**Free, decode time only, no retraining** (`--eval-only`, ~15 min each):
- `B_ssl_s42`'s BEST with `frozen-B`: must reproduce its summary.json exactly. This is the machinery check.
- `B_ssl_s42`'s BEST with `phoneme-erasure`: how much the trained decoder relies on the erased directions.

**Cost.** About 27 GPU hours for (a), (b0) and (b) (B_ssl_s42 took 8.9 h). Prep and checks add about 1 h. The
conditional (a′) adds 9 h. Details are in §6.

## 1. (a) Phoneme bottleneck

**Design.**
- The encoder is arm A's CTC phoneme model (`checkpoints/bit/finetune-ndt-phoneme-brandman_2024_text-feat_all/BEST`;
  test PER 9.25%). It is loaded whole: its TX+SBP read-in, transformer core and CTC read-out (`decoder.bin`,
  Linear 384 → 41, then log-softmax). It is frozen.
- Its per-frame output over BIT's 41 classes is the only neural input to the projector:
  - the classes are blank, 39 phonemes and SIL;
  - one frame = one 100 ms patch token, the same time base as arm A's Stage 2;
  - the output is floored at log 1e-8 and standardized per class with fixed train-split constants.
- The projector is BIT's. Only its first Linear is replaced, by a fresh one with 41 inputs: on the host's MLP
  (`configs/bit_baseline.yaml` patches `inter_size: 1024`) that is 41 → 1024, then the ReLU and the 1024 → 1536
  Linear exactly as the model built them; 1,617,408 parameters instead of 1,968,640. (The repo's unpatched BIT
  config has a single Linear; then that Linear is the one replaced. Both shapes are tested.) Everything after it is
  BIT's.

**Log-posteriors, not posteriors.** Posteriors and log-posteriors carry the same information, since one is a
bijection of the other. The choice is about what a ReLU MLP can use.
- CTC posteriors are extremely peaked: blank or one phoneme takes nearly all the mass on almost every frame. The
  competing phonemes ("B or P?") sit at 1e-3 to 1e-6, a range that is invisible in linear probability space at
  initialization scale. In log space the same ranking is spread over about 15 units.
- Log-posteriors are what arm A's CTC loss and any phoneme decoder (beam search, lattice) consume.
- They are the standard choice for posterior features in ASR (tandem features, Hermansky, Ellis & Sharma 2000, take
  the log of the MLP posteriors because the raw posteriors are too skewed to model).
- **Floor at 1e-8.** It bounds the input. Below it nothing usable remains: in bf16 the logits are resolved to about
  0.06–0.13 at their magnitudes.
- **Standardization.** The per-class mean and SD are fixed constants from `prepare_ablations.py bottleneck` over all
  train frames. The affine map is invertible and the projector's first Linear could absorb it, so the function class
  is unchanged. It only puts the 41 inputs on the O(1) scale of a LayerNormed encoder output, which keeps AdamW at
  lr 5e-5 in the same regime as BIT's runs.
- **All 41 classes, blank and SIL included.** They are the model's whole output distribution: blank carries timing
  between phonemes, SIL carries word boundaries. Dropping blank would leave an unnormalized vector.
- **Variants, configured but not proposed.** `feature: posterior`, and `feature: one_hot` (argmax class only). One-hot
  is a stricter bottleneck without confidence; it would answer a different question (is the phoneme string enough?).

**Frozen, in eval mode.** A trainable encoder could turn the 41 "posteriors" into an arbitrary 41-d code for the
LLM, and the bottleneck would no longer be phonetic. So the whole phoneme model is frozen, and it runs in eval mode
even while the LLM trains: no dropout, no gradient. Its features are therefore exactly the posteriors that give its
9.25% PER. `prepare_ablations.py labels` re-measures that PER through Stage 2's input pipeline (smoothing, padding fix)
and stops (exit 3) if test PER is more than 1.5 pp from 9.25%.

**Reading.** A difference counts only if the paired trial-bootstrap 95% CI of the test WER difference excludes 0. It
must also exceed arm B's seed spread once `B_ssl_s43` / `B_ssl_s44` exist (DESIGN.md §2: differences smaller than
seed variability are not claimed).
- **(a) ≈ `A_ssl_s42`.** Arm A's decoder uses nothing beyond what the phoneme posteriors carry: phoneme identity,
  confidence and timing.
- **(a) better than `A_ssl_s42`.** The same conclusion, and the bottleneck also regularizes on 8k trials.
- **(a) clearly worse than `A_ssl_s42`.** Two differences are confounded: the bottleneck and the frozen encoder. Run
  (a′), arm A's encoder frozen with its full 384-d output:
  - (a′) ≈ `A_ssl_s42`: the rich 384-d features carry information beyond the posteriors, and the decoder uses it.
  - (a′) ≈ (a): the gap comes from fine-tuning the encoder end to end, not from the bottleneck. Then the bottleneck is
    not shown to lose anything.
- **Also against `B_ssl_s42` (34.2%).** If phonemes alone match or beat the phoneme-free arm B, all of arm B's
  performance is within reach of a purely phonetic channel. That agrees with the error analysis, but it is not
  proof that B uses only phonetics.

## 2. (b) Phoneme erasure

**Which encoder.** The brief says "the arm B SSL encoder's outputs", but those do not exist on T15 TX+SBP before
Stage 2.
- `checkpoints/ssl_human/BEST_core` is the transformer core only. The SSL corpus was TX-only, so the T15 read-in
  restarts at Stage 2 (DESIGN.md §7.2; B_ssl_s42's run.json has `read_in_loaded: false`).
- So the frozen encoder is **`B_ssl_s42`'s trained Stage 2 encoder**: the SSL core plus the TX+SBP read-in that
  Stage 2 trained, all of it phoneme-free. It is exported by `prepare_ablations.py export-encoder` to
  `checkpoints/ablations/B_ssl_s42-encoder/` and checked against the run's slim checkpoint.
- Its features are known to support 34.2% test WER, so whatever the eraser removes, it removes from a working code.
- *Not chosen:* the full SSL model with its own TX-only T15 read-in (`checkpoints/ssl_human/BEST`). Its features
  exist, but Stage 2 would then need TX-only input, unlike every other run. And a frozen SSL encoder is probably a
  weak decoder by itself, so its control would decode badly and the erasure effect would be hard to read. Its probe
  is in docs/PROBES_FOILS.md.

**LEACE, not INLP.**
- LEACE (Belrose et al., NeurIPS 2023) is closed form: r(x) = x − W⁺PW(x − μ), with W the (pseudo-)whitening and P the
  projection onto colspace(W Cov(x, z)).
  - It makes Cov(r(X), Z) = 0 exactly. Every class-conditional mean then equals the overall mean, so no linear
    classifier beats a constant predictor, for any convex loss, on the fitting data.
  - It does so with the least mean-squared change to X.
  - It is deterministic and needs one pass of sums.
- INLP (Ravfogel et al., ACL 2020) projects out the weight vectors of linear classifiers trained one after another.
  - It has no guarantee that all linear information is gone, and it usually removes more directions than needed.
  - It depends on how the classifiers were trained, so it is stochastic and needs tuning.
- Tests: zero cross-covariance (< 1e-10); an idempotent eraser; no more change than the orthogonal projection out
  of span(Cov(x, z)); a held-out linear probe at the majority class while an MLP still reads a planted XOR code. Pseudo-inverses handle the singular covariance
  of a LayerNorm output, which lies in a hyperplane.

**Labels (train split only for the fit).**
- Frame phonemes come from CTC forced alignment (Viterbi) with arm A's model. The label sequences are BIT's own:
  g2p_en, i.e. the CMU dictionary plus G2P, stress removed, SIL after each word. These are the sequences arm A was
  trained on.
- Each label owns its run of frames on the Viterbi path. The blank frames between two labels are split at the
  midpoint. Up to 2 blank frames before the first / after the last label count; frames further out are unlabelled.
- Concept: 40 classes (39 phonemes + SIL; blank is never a segment label), one-hot, so the erased rank is ≤ 39 of 384.
- `offsets: [-1, 0, 1]` would also erase the previous and next label (rank ≤ 117). It is configured, not proposed.
- The labels are used for nothing else (DESIGN.md §5).

**Frozen encoder (decided), and the trade-off.**
- *A trainable encoder with a fixed eraser is invalid.* LEACE removes fixed directions. Training would re-encode
  phoneme identity in the directions that survive, and the guarantee would hold only at step 0.
- *Trainable, with the eraser re-fit periodically (e.g. every 5 epochs).*
  - The erasure holds only at re-fit moments; in between the encoder drifts back.
  - The LLM sees a moving, non-stationary input.
  - Each re-fit costs a full pass over the train frames.
  - The final checkpoint's erasure must be re-fit again and then no longer matches what the LLM was trained on.
- *Adversarial invariance* (gradient reversal against a phoneme classifier) gives no guarantee, and is an unstable
  game.
- *Frozen (chosen).* The erasure holds exactly during all of training and evaluation, on the features it was fit
  on. It costs encoder adaptation, which is why (b0), the same frozen encoder without the eraser, is required. The
  frozen encoder runs in eval mode, so the LLM sees deterministic features, exactly what the eraser was fit on and
  what evaluation sees. B_ssl_s42 trained with encoder dropout; (b0) and (b) both lose it equally.

**Erasure check** (`prepare_ablations.py erasure`, written to `eraser_check.json`; exit 4 if it fails):
- a linear probe trained on train frames (seeded subsample, ≤ 200k) and scored on held-out val frames, before and
  after erasure, with its L2 chosen by val log-loss. **Pass:** after erasure, val accuracy ≤ majority class + 2 pp
  and information ≤ 0.02 bits/frame;
- an MLP probe (one hidden layer of 256) before and after, which measures the non-linear leftover;
- linear probes for the previous and the next label, after erasure;
- the in-sample cross-covariance, before vs after;
- the share of feature variance removed (val).

**What LEACE does not remove, and so what (b) can and cannot show.**
- It removes linearly decodable phoneme identity frame by frame. Class-conditional covariances survive, and so can
  phoneme information that only a non-linear reader recovers. BIT's projector is an MLP and the LLM is very
  non-linear.
- If the MLP probe after erasure is well above chance, "decoding left after erasure" can still be phonetic, only
  non-linearly coded.
- Neighbouring labels and the temporal pattern across frames may still be readable; the neighbour probes quantify
  the first.

**Reading** (test WER; the same counting rule as in §1):
- **(b) ≈ (b0):** removing linear phoneme identity costs nothing.
  - MLP probe near chance after erasure: the decoder uses non-phonetic information, or phoneme information in a form
    no frame-level probe reads.
  - MLP probe well above chance: the decoder reads phonemes non-linearly, and the result says nothing beyond that.
- **(b) near the null controls (100–126%):** the decoder depends on linearly decodable phoneme identity, and what is
  left is not usable.
- **In between:** report the fraction of the gap kept, (null − (b)) / (null − (b0)), with its bootstrap CI.
- **Decode-time erasure** (eval-only on `B_ssl_s42`): how much the trained decoder relies on those directions, with
  no chance to adapt. A large jump there plus a recovery in (b) means the LLM can re-learn from what is left.

## 3. Mechanics

- **Code.**
  - `src/bit_experiments/neural_ablations.py`: the modules, LEACE, `apply` and `verify`.
  - `src/bit_experiments/phoneme_frames.py`: forced alignment, label cache, batched frozen features.
  - `scripts/prepare_ablations.py`: the artifacts.
  - `configs/ablations.yaml`: the entries.
- **`train_stage2.py` (additive, all under `if args.ablation`; a test checks this):**
  - `--ablation NAME`;
  - in `set_model`, after `load_init` and before BIT's Trainer builds the optimizer, `neural_ablations.apply`, so
    the optimizer sees the re-sized projector and no encoder gradients;
  - after BEST is loaded, `neural_ablations.verify`: the encoder is still frozen and is the one the artifacts were
    fit on (and, after training, the one training started from), written to run.json;
  - `"ablation"` added to each decode's meta.json.
- **Hook.** The model instance's `prepare_neural_embeds` is replaced, as `stage2.fix_padding` replaces
  `merge_embeds`. BCI_AudioLLM (Aero), BCI_TextLLM and BCI_DiffusionLLM all reach the encoder through it.
  - The frozen encoder runs under no_grad and in eval mode, under the ambient bf16 autocast, as when the artifacts
    were fit.
  - Read-out, standardization and eraser run in float32.
  - With the padding fix, a trial's features in a training batch equal its features alone (tested).
- **Artifacts and hashes.**
  - Read-out weights, statistics and eraser are non-persistent buffers, loaded from their files. The slim checkpoint
    format is unchanged, and any checkpoint can be evaluated with an ablation inserted.
  - stats.pt and eraser.pt record the digest of the encoder they were computed on. `apply` refuses a different one,
    a `--init` other than the entry's, and an encoder not fully loaded from `--init` (a frozen random read-in would be
    meaningless).
- **RNG.** The bottleneck's new first Linear is built under `torch.random.fork_rng` seeded with `--seed`. The rest of the
  initialization (LoRA, contrastive head) is the seed's usual one, and the global random stream continues as without
  the ablation (tested). The training stream still differs from B_ssl_s42's, because a frozen eval-mode encoder draws
  no dropout.
- **Trainable parameters** (host config, MLP projector): the NDT encoder's 8,394,752 parameters (1,484,544 of them
  the read-ins) are frozen in every kind. The bottleneck's projector has 351,232 fewer. BIT's unused NDT
  `decoder` (1,487,616) stays as in every Stage 2 run.

## 4. Commands (in order; BIT's env; from a fresh code copy on the host)

**Host copy.** Install into a fresh new-layout copy (e.g. `/workspace/ablations`), as the Aero and control runs do;
never overwrite a running job's code.
- **Files.** All additive:
  - `src/bit_experiments/{neural_ablations,phoneme_frames,probes,foils}.py`;
  - `scripts/{train_stage2,prepare_ablations,run_probes_foils,build_foils}.py`;
  - `configs/{ablations,probes,foils}.yaml`;
  - `tests/test_{neural_ablations,probes_foils}.py`.
- **Links:** `checkpoints/{bit,ssl_human,stage1,stage2,hf}` and `data/{raw,processed}`, pointing to the main tree;
  `checkpoints/ablations/` is new.
- **Before the GPU:** `<bit env>/python -m pytest tests/test_neural_ablations.py tests/test_probes_foils.py`
  (CPU, ~1 min) → 37 passed (+ `tests/test_ablation_report.py` 2, `ops/eraser-pair/test_launch.py` 19). The host's env
  has no pytest: run them on the Mac in a copy of BIT's env.

```bash
# 1. Artifacts (GPU, ~20 min; exit 3 = arm A's PER not reproduced, exit 4 = erasure check failed)
<bit env>/python scripts/prepare_ablations.py all --device cuda
#    = labels -> bottleneck -> export-encoder --run B_ssl_s42 -> erasure; read
#      data/processed/phoneme_frames/<sha>.json (PER, labelled share) and checkpoints/ablations/phoneme-erasure/eraser_check.json

# 2. Decode-time checks (~15 min each). The first must reproduce results/stage2/B_ssl_s42/summary.json exactly.
<bit env>/python scripts/train_stage2.py --name B_ssl_s42_frozenB-evalonly --arm B --seed 42 \
    --eval-only checkpoints/stage2/B_ssl_s42/BEST --ablation frozen-B
<bit env>/python scripts/train_stage2.py --name B_ssl_s42_erased-evalonly --arm B --seed 42 \
    --eval-only checkpoints/stage2/B_ssl_s42/BEST --ablation phoneme-erasure

# 3. Smoke (minutes): builds, loads, evaluates at step 0
<bit env>/python scripts/train_stage2.py --name smoke_bottleneck --arm A --seed 42 --no-train --limit 8 \
    --init checkpoints/bit/finetune-ndt-phoneme-brandman_2024_text-feat_all/BEST --ablation phoneme-bottleneck

# 4. The runs (~8.9 h each)
<bit env>/python scripts/train_stage2.py --name A_ssl_phonbottleneck_s42 --arm A --seed 42 \
    --init checkpoints/bit/finetune-ndt-phoneme-brandman_2024_text-feat_all/BEST --ablation phoneme-bottleneck
<bit env>/python scripts/train_stage2.py --name B_ssl_frozen_s42 --arm B --seed 42 \
    --init checkpoints/ablations/B_ssl_s42-encoder --ablation frozen-B
<bit env>/python scripts/train_stage2.py --name B_ssl_phonerased_s42 --arm B --seed 42 \
    --init checkpoints/ablations/B_ssl_s42-encoder --ablation phoneme-erasure
# conditional on (a) being clearly worse than A_ssl_s42:
<bit env>/python scripts/train_stage2.py --name A_ssl_frozen_s42 --arm A --seed 42 \
    --init checkpoints/bit/finetune-ndt-phoneme-brandman_2024_text-feat_all/BEST --ablation frozen-A
```

**Under the GPU lock:**

```bash
<GPU-lock wrapper> stage2-<name> scripts/train_stage2.py ...
```

The wrapper `cd`s to /workspace/KunzDecoder. For a separate copy, use that copy's own wrapper or launcher, as the
Aero / control runs do.

**Smoke after the run (smoke 1).** `run.json`'s `init_report.ablation.projector_resized` shows the first Linear
re-sized to 41 inputs, with `kept: [ReLU, Linear]` on the host (bottleneck), and `eval_checkpoint.ablation` has `frozen: true` and every `*_matches: true`.

## 5. Tests (`tests/test_neural_ablations.py`, CPU, BIT's env)

- **Alignment:**
  - the Viterbi path equals the best of all valid CTC paths (brute force) and collapses to the labels;
  - a too-short trial is refused;
  - segment labels follow the midpoint and margin rule;
  - concept offsets;
  - BIT's PER collapse vs standard CTC's.
- **Label cache:** round trip and tamper check.
- **LEACE:** see §2; streaming moments equal batch moments.
- **Read-out:** loads from decoder.bin; floor, standardization, one-hot; wrong vocabulary refused.
- **Both projector shapes** for every test that touches the projector: the host's MLP (`inter_size: 1024`) and
  the repo's single Linear. On the MLP only the first Linear is new; the ReLU and second Linear are the same
  objects with the same weights (a mutation check that rebuilds the whole projector fails).
- **Bottleneck end to end** (tiny Qwen2 in BCI_TextLLM, artifacts from `prepare_ablations.py`'s own functions):
  - the neural tokens equal projector(standardized floored log-posteriors);
  - the frozen part is deterministic in train mode;
  - a batch row equals its lone trial;
  - gradients reach the projector and LoRA and never the encoder;
  - the global RNG is unchanged;
  - slim round trip with the ablation re-applied;
  - stats from another encoder are refused;
  - `frozen-A` leaves the features and projector unchanged.
- **Erasure end to end:**
  - export, labels (`make_labels`) and LEACE (`make_eraser`) on synthetic trials;
  - neural tokens = projector(A·h + b);
  - eval-only on the "trained" checkpoint passes `verify`, and the wrong checkpoint is refused.
- **Bottleneck in BCI_DiffusionLLM:** loss, backward and greedy decode.
- **train_stage2.py:** every `neural_ablations` call is under `if args.ablation`.
- **Mutation checks** (run once, not in the suite): the eval-mode and RNG tests fail when those mechanisms are
  removed.

## 6. GPU cost

| Step | GPU time | Basis |
|---|---|---|
| `prepare_ablations.py all` | ~20 min | arm A + B encoders over ~9.5k trials (7M-param NDT), Viterbi and g2p on CPU, 6 probe fits |
| 2 decode-time checks | ~30 min | B_ssl_s42's final evaluation took 14.6 min |
| smoke | ~5 min | |
| (a), (b0), (b) | ~3 × 8.9 h | B_ssl_s42: 531 min in all (17.2 min per 5 epochs incl. val decode); the frozen encoder saves only its backward (<1% of the FLOPs) |
| (a′), conditional | ~8.9 h | |
| **Total** | **~28 h (+9 h)** | |

**When.** Everything needed exists now: arm A's phoneme model, B_ssl_s42's BEST, the T15 data. Reading (a) needs
`A_ssl_s42` (~Sun 2026-10-25 05:00 UTC). The queue ends ~Mon 2026-10-26 08:00 UTC, so the natural slot is after it.
The order is the user's call.

## 7. Diffusion versions later?

*Superseded for (b) on 2026-10-24: the user approved the erasure pair with both decoders (§9). What follows is the
2026-10-22 reasoning, kept as the record; (a) stays AR only.*

Both ablations work unchanged with a diffusion `--llm`: the hook sits before any decoder, tested in BCI_DiffusionLLM
and with `aero-1-audio-diffusion` (§9). A full diffusion version is **not worth doing now**.
- **Cost.** Each needs a diffusion reference that does not exist yet:
  - for (a), an arm A diffusion Stage 2;
  - for (b), run 3's encoder frozen, with and without the eraser.

  That is 4 runs at ~12–14 h each with `fast_step` (unmeasured, docs/DIFFUSION_SPEEDUP.md), about 50 h.
- **Value.** It would extend the claim to the best decoder (25.2%), whose errors are the most sound-like. That
  matters only if the Aero results are clear-cut and the claim is meant to cover diffusion.
- **The cheap step first:** decode-time erasure on run 3 (entry `phoneme-erasure-run3`, ~30 min):

  ```bash
  <bit env>/python scripts/prepare_ablations.py export-encoder --run B_ssl_fast-dllm-v2-1.5b_s42 --erasure phoneme-erasure-run3
  <bit env>/python scripts/prepare_ablations.py erasure --erasure phoneme-erasure-run3 --device cuda
  <bit env>/python scripts/train_stage2.py --name B_ssl_fast-dllm-v2-1.5b_s42_erased-evalonly --arm B --seed 42 \
      --eval-only checkpoints/stage2/B_ssl_fast-dllm-v2-1.5b_s42/BEST --llm fast-dllm-v2-1.5b --ablation phoneme-erasure-run3
  ```

  If run 3 collapses under decode-time erasure much as `B_ssl_s42` does, the Aero runs probably speak for both.

## 8. Not done / open

- The erasure pair with both decoders is prepared, not launched (§9).
- Nothing has run on real data. The PER re-measurement and the erasure check are the first real-data gates.
- The seed spread for the counting rule needs `B_ssl_s43` / `B_ssl_s44` (queued last).
- `scripts/eval_ablations.py` (the other session) rebuilds runs from run.json without the `ablation` argument. To
  apply its input ablations to (a) / (b) runs, it would need `neural_ablations.apply` before loading BEST, as
  `run_probes_foils.py` does.

## 9. The erasure pair with both decoders (approved 2026-10-24; prepared, not launched)

**The decision.** The user chose "eraser pair": the phoneme-eraser pair (frozen B encoder vs LEACE-erased) with both
the AR and the diffusion decoder, the decode-only checks first, after arm C + diffusion.

**The question.** The motivation is decoding data with no phonetic signal. Which decoder, AR or block diffusion,
follows the information left after erasure, and which falls back on fluent, ungrounded LM output?

**Design** (the parent's proposal; I agree with it, and the reasons are below):
- **One frozen encoder for all four runs:** `B_ssl_s42`'s Stage 2 encoder, as in §2, with the same fixed eraser.
  Both decoders see identical input information; only the decoder objective differs.
  - The encoder was co-trained with Aero AR, so its features are AR-shaped. That is why each erased run is compared
    with its **own** frozen control, (b0) for AR and (b0-diff) for diffusion, never with the other decoder.
  - The alternative, each decoder on its own co-trained encoder (`B_ssl_aero-1-audio-diffusion_s42`'s for
    diffusion), would erase different information from different features. The decoder comparison would then be
    confounded.
- **Recipe:** 150 epochs, selection by greedy val WER, test with null controls, seed 42, `--init
  checkpoints/ablations/B_ssl_s42-encoder`.
- **Diffusion step** (decided by the launcher; the run name is the same either way, as C's launcher does; the step is
  recorded in `launcher.json` and `run.json`):
  - (b0-diff) runs on the fast step (`aero-1-audio-diffusion-fast`) only if all of these hold, else plain
    (`aero-1-audio-diffusion`):
    - C's train-mode gate validated it (`/workspace/c-diffusion/results/fast_step_gate.json`);
    - C finished on it with rc 0, i.e. past C's epoch-5 backstop;
    - this copy's `fast_block_diffusion.py` is the gated file (md5).
  - On the fast step, (b0-diff) carries C's epoch-5 guard: val WER above 70%, or any failure → archive to
    `aborted-<UTC>/` and restart plain under the same name.
  - (b-diff) always uses (b0-diff)'s final step, so the pair is matched. It has no epoch-5 WER guard, because an
    erased run may legitimately be that bad; a failed fast attempt still restarts plain and is flagged `unmatched`.
  - CLAUDE.md, 2026-10-24: the unfixed fast step trains a worse model.

| Run | `--ablation` | `--llm` | GPU |
|---|---|---|---|
| (b0) `B_ssl_frozen_s42` | `frozen-B` | Aero (default) | ~8.9 h |
| (b) `B_ssl_phonerased_s42` | `phoneme-erasure` | Aero | ~8.9 h |
| (b0-diff) `B_ssl_aero-1-audio-diffusion_frozen_s42` | `frozen-B` | `aero-1-audio-diffusion` | ~30 h plain, ~11 h fast |
| (b-diff) `B_ssl_aero-1-audio-diffusion_phonerased_s42` | `phoneme-erasure` | `aero-1-audio-diffusion` | ~30 h plain, ~11 h fast |

**First, ~1.5 h** (`ops/eraser-pair/launch.py prep`, one job):
- **Artifacts:** `prepare_ablations.py all`, with exit 3 / 4 as gates (§4). Plus the export and eraser of
  `B_ssl_aero-1-audio-diffusion_s42`'s own encoder (entry `phoneme-erasure-aero-diffusion`, for its decode-time check
  only).
- **Decode-time checks** (`--eval-only`, the run's own seed):
  - `B_ssl_s42` + `frozen-B` must reproduce B_ssl_s42's greedy decodes. **Gate for all runs.**
  - `B_ssl_s42` + `phoneme-erasure`.
  - `B_ssl_aero-1-audio-diffusion_s42` + `frozen-evalonly` must reproduce that run's decodes. **Gate for the
    diffusion runs.** It is a check I added (~12 min): it tests the hook in the diffusion decode path on the GPU, as
    the first check tests the AR path.
  - `B_ssl_aero-1-audio-diffusion_s42` + `phoneme-erasure-aero-diffusion`: the cheap first look at the decoder
    comparison, though on different encoders.
- **"Reproduce"** means the greedy val and test WER within 0.5 pp, with every differing trial reported. Bit-exact
  equality is expected; the tolerance only absorbs GPU nondeterminism, while a broken hook changes far more.

**The readout** (`scripts/ablation_report.py`, CPU, from saved decodes; paired trial bootstrap over BIT-test; written
as `results/stage2/eraser_pair{,_decode_time}.md`):
- **Per run:** WER on the real input and under the nulls; the **grounding gap** G = WER(shuffled input) −
  WER(real), how much the output depends on the trial's own neural data; the share of derailed sentences (sentence
  WER ≥ 1). For scale, from saved decodes:

  | Run | G |
  |---|---|
  | B_ssl_s42 | 81.3 pp |
  | Instruct + diffusion control | 91.4 pp |
  | run 3 | 89.5 pp |
  | Instruct AR | 58.0 pp |

- **Per pair:**
  - the cost of erasure, WER(erased) − WER(frozen control);
  - the **grounding kept**, G(erased) / G(control): 1 = still as tied to the input; 0 = no more tied to its own trial
    than to another one's, which is fluent, ungrounded output.
- **AR vs diffusion:** the difference-in-differences of both, on the same 713 trials with the same resamples. A
  decoder that "follows the information left" pays less and keeps more grounding.
- **Error types:** run `phonetic_error_analysis.py` / `ablation_error_types.py` on the four runs' decodes. Erasure
  should cut the sound-alike excess; a decoder that falls back on the LM shows unrelated fluent words instead.
- **The probes** in `eraser_check.json`, copied into the launcher's state file, say what "information left" means:
  - **linear after erasure ≈ majority:** the guarantee held;
  - **MLP after erasure:** phonetic information left in non-linear form;
  - **neighbour-label probes:** the previous / next phoneme still readable.

  If the MLP probe stays well above chance, a decoder that keeps working may be reading phonemes non-linearly
  (§2), not something non-phonetic.
- **Optional, not in the chain:** `run_probes_foils.py` on the four runs (~2 h). Its forced-choice neural evidence
  measures grounding sentence by sentence.

**Host.**
- **Code copy:** `/workspace/eraser-diffusion`, its own: `/workspace/c-diffusion`'s tree (the 2 × 2's code + the
  gated fast-step fix, same symlinks into the main tree), plus the ablation files and `train_stage2.py` with
  `--ablation`.
  - c-diffusion itself must not change before C has finished, at the arm C diffusion session's request.
  - The arm C diffusion session builds the copy.
- **Chain:** the six lines are in that session's after-others chain, installed on the host 2026-10-24 18:04 UTC
  (its host-side job list), right after `C_ssl_aero-1-audio-diffusion_s42`:
  prep → b0 → b → b0-diff → b-diff → report.
- **Gates:** a failing line does not stop that chain, so each step checks its own prerequisites. A `run` refuses
  (exit 2) unless prep's gates passed, and (b-diff) also until (b0-diff) has finished. A run.json guard stops a run
  with the wrong arguments, ablation or decoder step.
- **Read only:** nothing is written to `/workspace/aero-diffusion` or `/workspace/c-diffusion`.
- **Install, lines, outputs:** `ops/eraser-pair/README.md`.

**ETA.** T0 = the end of `C_ssl_aero-1-audio-diffusion_s42`.

| Step | When |
|---|---|
| prep | T0 + 1.5 h |
| AR pair | T0 + ~19 h |
| diffusion pair, plain | T0 + ~79 h (~3.3 days) |
| diffusion pair, fast (once validated) | T0 + ~41 h |

If arm C diffusion starts ~Thu 10-29 and takes ~31 h on the plain step, T0 is ~Fri 10-30. The AR answer would then
come ~Sat 10-31, and the diffusion answer ~Mon 11-02 (plain).

**Tests** (CPU, BIT's env):
- **`tests/test_neural_ablations.py::test_eraser_pair_with_aero_block_diffusion`**, Linear and MLP projector, on the
  real `aero-1-audio-diffusion` entry (Aero's LM read out of a tiny Aero-layout snapshot):
  - B's frozen encoder with `frozen-B` and `phoneme-erasure`: neural tokens = projector(h) / projector(A·h + b);
  - loss and backward reach the projector and LoRA, never the encoder, and the mask row stays frozen;
  - greedy and nucleus decode;
  - slim round trip with the ablation re-applied;
  - on a diffusion run's own BEST from another copy (`export-encoder --best`), `frozen-evalonly` reproduces its greedy
    and nucleus decodes exactly; its own eraser verifies; B's eraser on it is refused;
  - `frozen-evalonly` with an `--init` is refused.
- **`tests/test_ablation_report.py`:** planted decodes give the WERs, gaps, costs, grounding kept and the
  difference-in-differences with CIs; runs decoded on different trials are refused.
- **`ops/eraser-pair/test_launch.py`:** every gate, refusal and command, offline. 19 tests, incl. a dry run writing no state and a prep that dies part way leaving the runs refusing. The run steps drive a
  real subprocess (a fake `train_stage2.py`), covering:
  - the step decision in each case: no gate record, gate failed, C on plain, C failed, another file;
  - a matched fast pair;
  - the epoch-5 guard archiving and restarting plain;
  - an unmatched fallback;
  - the run.json guard.

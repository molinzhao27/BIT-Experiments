# Design

## 0. Aim and scope guard (read this first)

**Aim: phoneme-free brain decoding.** Decode attempted-speech sentences from intracortical activity (T15,
Brain-to-Text '25) end-to-end through an LLM, with **no phoneme labels anywhere in the pipeline**. Measure how close
that gets to the phoneme-first recipe, and explain the gap.

**Headline result (tier 1):** Arm C (SSL → contrastive alignment → LLM) vs Arm A (SSL → phoneme CTC → LLM) vs Arm B
(SSL → LLM). Free-generation WER on BIT-test, also split by sentences seen / unseen in training, with null controls,
3 seeds.

**Tier 2, explains the result:**
- Arm E: a non-contrastive alignment alternative.
- Arm D: the hybrid upper reference.
- B-long: budget-matched control for B.
- C at small batch: is batch size what makes C work?
- Random-init A and C: what does pretraining buy?
- Probes after each stage: phonetic vs semantic content.
- LLM swap (§8): does the decoder LLM matter (Aero's audio post-training first, then a larger text LLM)? Arm B
  only, never in the headline table.

**Tier 3, generality:**
- Label-fraction curves.
- Kunz transfer (4f, only on request).

**Out of scope, and what each would cost us:**
- Chasing state of the art. No n-gram/LLM rescoring, ensembling or phoneme-based correction in any headline number.
- Non-speech pretraining data (handwriting, counting, sequence recall, monkeys), unless tier-1 results show
  pretraining is the bottleneck.
- New architectures beyond BIT's encoder, projector and LLM.
- Other benchmarks.
- Phases 1–3-style analyses inside Phase 4.

Every new piece of work should say which tier it serves; if none, it waits.

**Central question:** can contrastive (or other non-phonetic) alignment *replace* phoneme supervision in an
end-to-end neural → LLM speech decoder, and if not, why? BIT (arXiv:2511.21740v5, App. C) and Feng et al. 2024
report that SSL → sentence-level LLM training without a phoneme stage does not reach low WER.

## 1. What the literature says (read 2026-10-18; full texts)

| Paper | Setting | Relevant findings | Consequence for us |
|---|---|---|---|
| **BIT**, Zhang et al., arXiv 2511.21740v5 | T12/T15, NDT + Aero-1-Audio 1.5B, LoRA | Phoneme CTC → LLM (CE + contrastive, batch 8–16, mean-pooled input-embedding text target, τ init 0.1 / max 100). No phoneme stage → no low WER (App. C). T15 holdout 11.06% end-to-end | Arm A = BIT recipe; Arm B reproduces App. C |
| **MoDAl**, Chen & Chin, arXiv 2605.00025v3 (ICMI 2026) | T12 (B2T '24), BiGRU encoders, Aero-1-Audio 1.5B + QLoRA | **Full** per-frame sequence of the CTC-trained 6v encoder goes to the LLM. Pooled vectors only for the extra area-44 encoders; full sequences from those "destabilize training". Symmetric InfoNCE, in-batch negatives, **batch 128 (batch 64: +2.3 WER)**, learnable τ (0.1), λ warmed up from 0 to 1. CE only 23.9 → CE + contrastive 22.0% test WER. **No phoneme-free ablation**; text target not specified; model selection on the reported split | Amendment 1 (full sequence to LLM, pool only in the contrastive head); Arm D ≈ MoDAl without decorrelation; batch size matters |
| **BAT-CLIP**, Kim et al., arXiv 2609.31180v1 | iEEG Podcast, encoder from DIVER-1 (SSL, ~5.3k h) | CLIP alignment from **scratch "largely fails"**; it helps from an SSL foundation model. Frozen anchors. Pooled CLIP alignment **hurts onset/transient information** | Arm C from SSL is the main arm; random-init C is the control; pool only inside the contrastive head |
| **The Semantic Bottleneck**, Landau et al., arXiv 2609.10296v2 | MEG (LibriBrain, 62 h), sentence → ADA embedding → inversion | **Noise controls** expose text/corpus priors; WER can exceed 1 (over-generation); semantic metrics disagree with each other. Loss = SigLIP + **global cosine** (largest effect) + VICReg | Null controls are mandatory; WER is primary; Arm E (cosine/MSE to the fixed target) is a real alternative |
| **Züfle & Niehues**, arXiv 2412.15712v2 | Speech LLM, HuBERT + Q-Former + Llama-3.1-8B, frozen LLM | Contrastive pretraining of the projector to the LLM's space. **Targets from deeper / all layers ≫ embedding layer**. Token-level (Wasserstein) ≥ pooled cosine. With **10% of task data** it beats ASR pretraining | Text target = a mid/deep layer (configurable), input-embedding pool as the BIT-matching control; label-fraction curves |

None of these reports a phoneme-free model that matches phoneme-first on intracortical benchmarks, so the
question is open.

## 2. Fixed across all arms

- **Data:** Brain-to-Text '25 (T15): train 8,072 / BIT-val 713 / BIT-test 713 (BIT's seed-42 halves of the official
  val split). The official test set (1,450, no labels) is never used. B2T '24 (T12) is used for pretraining only
  until the user decides otherwise.
- **Encoder:** BIT NDT (~7M core), initialized from the 4b SSL checkpoint (or random, for the random-init controls).
- **LLM:** Aero-1-Audio (Qwen2.5-1.5B LM), pinned snapshot `2efba6e`. Frozen base + LoRA r8 / α32 / dropout 0.2
  on q, k, v, o, gate, up, down. Only the §8 side question varies it.
- **LLM input (amendment 1):** the **full time-resolved encoder sequence** → BIT's MLP projector → LLM embedding
  space, then the prompt "decode the above neural activity into an English sentence:", then the target. There is
  **no K-token resampler** anywhere in the LLM path. Pooling happens **only inside contrastive / alignment heads**.
- **Sentence-stage loss:** next-token CE on the target only, + BIT's contrastive term (same weight in every arm).
- **Decoding / reporting (amendment 6):** free generation only, no reference-token prefixes. Settings: greedy,
  and nucleus p=0.9 / T=0.7 / ≤25 new tokens (BIT's). **WER is primary**; CER, BERTScore F1 and content-word errors
  are secondary. No LM rescoring, ensembling or phoneme-based correction in any headline number (Arm C
  especially).
- **Model selection:** one criterion for every arm's sentence stage: free-generation WER on BIT-val (BIT itself
  selects by teacher-forced WER; reported separately for the 4a reproduction). Final numbers on **BIT-test**
  (never used for any choice). Phoneme labels are never used to select models in arms B, C, E (amendment 5).
- **Seeds:** 3 per configuration; differences smaller than seed variability are not claimed.
- **Padding in Stage 2 (deviation from BIT, user-approved 2026-10-20; `bit_experiments.stage2.fix_padding`, every
  Stage 2 run, all arms and LLMs).**
  - **Problem.** In BIT a trial's training output depends on its batch:
    - The encoder smooths the padded batch, so the padding value (−100) leaks into the last ~6 bins of each
      shorter trial (|x| up to ~40, vs ~1–2 alone).
    - The merged LLM input keeps the neural padding between a trial's neural tokens and its text. This shifts
      positions, and with no prompt after the neural tokens, the first word is predicted from a padding slot.
    - Evaluation runs one trial at a time, without padding. Measured on real trials (CPU, before training), a
      shorter trial's logits in a batch differed from its lone logits by 2.5–4.8 (BIT's prompt) and 8–11 (no
      prompt).
  - **Fix.**
    - Smoothing treats padding as the zero edge a lone trial gets.
    - Each trial's real tokens are contiguous, with its padding after them.
    - Batch and lone logits then agree to ≤ 0.03. A lone trial is unchanged, so evaluation is exactly BIT's.
  - **Not changed:**
    - the SSL pretraining (4b) and the phoneme stages (4a, arm A), which ran with BIT's behavior
      (`docs/SSL_PLAN.md` §1.3);
    - BIT's contrastive pooling: its means include padding, so a short trial's summary vectors are mostly
      padding. This affects only that loss term, equally in every arm, and decoding never uses it. Kept as BIT
      (user, 2026-10-20: a faithful reproduction).

## 3. Arms

| Arm | Intermediate stage (after SSL) | Then |
|---|---|---|
| **A** phoneme-first (BIT) | CTC phoneme fine-tuning | sentence-level LLM training |
| **B** direct (BIT App. C) | none | sentence-level LLM training |
| **B-long** budget-matched | none | sentence-level LLM training for (A's intermediate + sentence) total updates |
| **C** contrastive-first (ours) | standalone contrastive alignment to the fixed text target (large batch, GradCache/memory bank) | sentence-level LLM training |
| **D** hybrid (BIT/MoDAl-style) | CTC + contrastive jointly | sentence-level LLM training |
| **E** non-contrastive alignment | cosine / MSE regression to the same fixed target | sentence-level LLM training |
| A, C from random init | as A, C, without SSL | — |
| C small-batch ablation | C at BIT's batch (8 for T15) | — |

### Contrastive / alignment head (C, D, E; amendment 2)
- **Neural side:** encoder sequence → projector → **masked mean pool** → linear → L2 norm.
  **Text side:** cached target → linear → L2 norm.
- **Loss:** symmetric InfoNCE with learnable τ (init 0.1, max logit scale 100, as in BIT).
  **Negatives:** GradCache (exact large-batch gradients through the encoder; a 1024-pair batch splits into chunks)
  as the primary method. Alternative: a MoCo-style memory bank of past neural keys, with momentum encoder.
  Batch sweep: 64, 256, 1024, 4096 (GradCache).
- **False negatives:** pairs with the same normalized sentence are masked out of the denominator; the B2T '25
  train set has 8,072 trials but 7,307 distinct sentences. Every step logs **distinct sentences per InfoNCE
  denominator**.
- **Fixed target (cached once, hashed):** teacher = the Aero-1-Audio LM (the same LLM we decode with), run
  frozen over each cued sentence (normalized text, no prompt).
  - (i) **input-embedding mean pool**: BIT's aligner target;
  - (ii) **middle hidden layer** (layer 14 of 28) mean pool: the default for C, following Züfle & Niehues;
  - (iii) optional: the mean of every 5th layer.
  - Each is L2-normalized, cached to `data/processed/text_targets/<sha>.npz`, and a manifest records model,
    revision, layer, pooling, normalization and tokenizer, so the target is bit-for-bit fixed across runs.
- **Stopping:** C stops on the plateau of val top-1/top-5 sentence retrieval among BIT-val sentences.

## 4. Label-fraction curves (amendment 4)
Arms A, B, C at 5 / 10 / 25 / 50 / 100% of training trials (nested subsets, sampled by session-stratified
sentence), ≥3 seeds. Each point reports pairs, distinct sentences and hours of neural data. SSL is unaffected:
it is label-free, and its corpus never includes BIT-val/test.

## 5. Probes (amendment 5)
Frozen encoders (and projector outputs) are probed after **SSL**, after the **intermediate objective** (CTC /
contrastive / regression), and after **LLM fine-tuning**, for arms A–E:
- **Frame-level phoneme identity:** linear probe on CMU-dict phonemes aligned by CTC forced alignment from
  Arm A's model. Probe only; never used for training or selection.
- **RSA:** BIT App. L.1. 10-segment mean pooling, 1 − cosine RDMs, Pearson on upper triangles, against the
  cached LLM sentence embeddings.
- **Diagnostic sentence contrasts:** forced choice between the true sentence of a held-out trial and a
  constructed foil, scored (a) by contrastive similarity and (b) by the LLM's conditional log-likelihood given
  the neural tokens. The neural data are fixed, so contrasts are built on the text side. Foil types:
  - paraphrase (should *not* be rejected by a semantic code);
  - phonetically similar / different meaning (CMU-dict neighbours);
  - word-order swap;
  - negation insertion;
  - number / name substitution.

  The pattern of forced-choice accuracies separates phonetic, lexical and semantic coding.

## 6. Compute (to be finalised with the user after 4a)
On this GPU, measured with BIT smoke runs (while sharing the GPU, so pessimistic):
- **Phoneme stage:** ~1.3 steps/s at batch 32. 800 epochs ≈ 20–40 h.
- **Sentence stage:** ~1.85 steps/s at batch 8. 150 epochs ≈ 25 h, peak 36 GB.

**At BIT's schedule the amended plan is far beyond one shared GPU:**
- 8 configs × 3 seeds × ~25 h sentence stage ≈ 600 h;
- label-fraction curves ≈ 3 arms × 3 seeds × 47 h ≈ 425 h.

**Proposal:** set a fixed sentence-stage step budget from 4a's validation-WER learning curve, typically much
shorter than 150 epochs once converged, and use it for every arm. Run 1 seed of all arms first (the "first full
run" checkpoint), then add seeds and label-fraction points in order of informativeness.

## 7. SSL corpus decisions (approved 2026-10-19; `configs/ssl_human.yaml`)

**Corpus = human speech-sentence recordings only (attempted / imagined), minus every evaluation trial.**

| Group | Data | Approx. hours |
|---|---|---|
| T15 | B2T '25 train + Kunz imagined sentences (non-eval) | 42.3 |
| T12 | B2T '24 train + Kunz sentences | 16.4 |
| T16 | Kunz sentences | 1.4 |

That is about 60 h in total, vs BIT's ~98 h, which also included handwriting.

1. **Exposure.**
   - Official train splits only.
   - BIT-val is used only for checkpoint selection, and label-free.
   - BIT-test and both competition holdouts are never read.
   - Kunz isolated/interleaved (Phase 1–3 evaluation) and all `*_eval.mat` sentence files (closed-loop evaluation
     sessions, held out for any later imagined-sentence decoding) are excluded.
   - Kunz blocks matching any B2T '25 split are dropped (T15 2024.03.03 attempted = B2T '25 train blocks).

   BIT pretrains on validation and holdout neural data too. Its App. P shows excluding fine-tuning data makes no
   substantial difference (T15 end-to-end 11.06% vs 11.00%), so being strict costs nothing measurable.
2. **Features: threshold crossings only,** exactly as BIT ("Since SBP is unavailable in some pretraining datasets,
   we use only thresholded spikes"). Decoding uses TX + SBP. The T15 read-in therefore restarts at fine-tuning
   (1280 → 2560 inputs) and the transformer core transfers, identically in every arm.
3. **Read-ins.** Kunz T15 shares B2T '25's read-in: duplicated trials match channel for channel. T12 (Kunz vs
   B2T '24) and T16 are kept separate because their channel identity is unverified.
4. **Normalization.**
   - B2T '25 is used exactly as released. Measured: already rolling-z-scored and clipped at |z| = 10.0; BIT does not
     re-normalize it.
   - Everything else uses BIT's B2T '24 loader rule: z-score per block (own bins), then per day (train-split stats),
     with a 0.01 SD floor.
   - Clip at ±10 everywhere, the B2T '25 convention. Measured: BIT's B2T '24 loader gives |z| up to 115, at
     8 × 10⁻⁶ of values; near-silent Kunz channels reach 136.
5. **Checkpoint.** Masked-bin R² with fixed eval masks on the T12 and T15 BIT-val halves, equally weighted.
6. **Deferred,** not aim-critical:
   - Counting, keyword (T12 256-ch incl. area 44) and sequence-recall archives (no block info).
   - The handwriting datasets (downloaded and verified, `data/raw/{willett_2021_handwriting,fan_2023_corp,falcon_h2}`).
7. **New evaluation split.** 81/713 BIT-val and 81/713 BIT-test trials repeat a training sentence (11.4%), and 13
   sentences occur in both halves. Every arm reports WER on seen and unseen sentences separately, because
   sentence-level contrastive alignment is retrieval-like.

## 8. LLM-swap side question (approved 2026-10-19; `configs/llms.yaml`)

**Question.** Does the decoder LLM change the result? BIT's neural tokens never pass through Aero-1-Audio's audio
tower (`project_to_audio: false`); they enter the LM's text-embedding space directly. So the first question is
whether Aero's audio post-training matters at all. Only then: does a stronger language prior help?

**Tier 2.** The phoneme-free arms lean on the LM prior more than arm A does, so the answer bears on the gap. It is
never mixed into the headline table, and Aero stays the LLM for every other run.

**Fixed.**
- **Arm:** arm B from the SSL encoder (`checkpoints/ssl_human/BEST_core`), because its intermediate stage involves
  no LLM. Arm C's Stage 1 targets are Aero's LM states, so swapping the LLM there would mean redoing Stage 1.
- **Recipe:** everything else as §2:
  - the Stage 2 recipe;
  - LoRA r8 / α32 on the same seven projections;
  - fp32 weights + bf16 autocast;
  - CE + BIT's contrastive term;
  - selection by free-generation WER on BIT-val, with BIT-test reported.
- **Reference:** arm B with Aero, seed 42 (`B_ssl_s42`).
- **Generation:** every LLM's shipped generation defaults are replaced by neutral ones (bos/eos only, as Aero
  ships). Qwen2.5-Instruct ships repetition_penalty 1.1 / top_k 20 / top_p 0.8, which would otherwise change greedy
  and nucleus decoding.
- **Comparison:** unseen-sentence WER. One seed per swap first; a difference counts only if it is larger than arm
  B's seed spread. So arm B needs its other seeds (§2) before runs 1–2 can be judged: `B_ssl_s43` and `B_ssl_s44`
  (same flags as `B_ssl_s42`) are run last. The same seed does not give
  the same start: BIT builds the projector, read-in, contrastive head and LoRA after loading the LLM, and Aero's
  load draws extra random numbers (its unloaded audio projector, LoRA on its audio tower). At seed 42 every one of
  those tensors differs between the Aero and the Instruct builds, so a single-seed swap difference includes
  seed-to-seed noise.
- **Code:** `src/bit_experiments/text_llm.py` (BIT's `BCI_AudioLLM` with an `AutoModelForCausalLM` decoder), selected
  with `train_stage2.py --llm <name>`; prompt with `--prompt` ("#" marks the neural tokens).

The swaps can only be compared once `B_ssl_s42` is done.

**Runs**, in this order:

| Run | LLM | Prompt | Change vs the reference |
|---|---|---|---|
| 1 `B_ssl_qwen25-1.5b-instruct_s42` | Qwen2.5-1.5B-Instruct | BIT's | Audio post-training removed: Aero's own base model (same tokenizer, 1536-d, 28 layers) |
| 2 `B_ssl_qwen25-1.5b-noprompt_s42` | Qwen2.5-1.5B (base) | None: neural tokens, then the sentence | Instruction tuning and the prompt removed |
| 4 `B_ssl_qwen25-1.5b-instruct-noprompt_s42` | Qwen2.5-1.5B-Instruct | None | The prompt removed only (vs run 1) |

- Run 2 differed from run 1 (seed 42, BIT-test free-generation WER 43.8% vs 55.7%). So run 4 removes only the prompt
  from Instruct, to tell the two changes apart (user, 2026-10-20; queued last, after the arms and seeds). Base +
  BIT's prompt would complete the 2 × 2 table; it is not queued.
- With no prompt, BIT's contrastive text side (the mean of the input's text embeddings) loses the constant prompt
  tokens, and the prompt-only null control does not exist.

**Checks (2026-10-20, CPU).**
- **Same model code path.** `BCI_TextLLM` given Aero's own LM weights reproduces BIT's `BCI_AudioLLM` exactly, with
  the same encoder, projector and nonzero LoRA weights. Logits differ by 0.0 in fp32 and in bf16 autocast, and
  greedy and nucleus decodes are identical. So run 1 vs the reference changes only the LM weights (and the init
  noise above). Aero's LM differs from Qwen2.5-1.5B-Instruct's by ≈3% (relative weight norm), more than Instruct
  differs from base (≈1%). Aero's separate `lm_head` equals its embeddings, so BIT's tied load loses nothing.
- **Tokenizers.** Aero's, Instruct's, base's and Fast-dLLM v2's tokenizers give identical ids for BIT's prompt
  and every sentence (train / val / test). EOS is `<|im_end|>`, except `<|endoftext|>` for base. Each run trains
  and stops on its own tokenizer's EOS.
- **25-new-token cap.** It never binds on BIT-val / BIT-test: the longest sentence is 18 tokens with EOS (one
  train sentence is 26).

**Next, conditional on run 1.**
- **If Instruct ≈ Aero** (audio training not needed), try a larger, newer text LLM:
  - Qwen3.5-9B first: the same model class as Qwen3.8-27B, so it de-risks that code path;
  - Qwen3.8-27B only after that.

  Both need:
  - a second environment (transformers ≥ 4.57 / 5.8; BIT's env pins 4.48.2 / peft 0.11.1), with the reference
    re-run in it;
  - LoRA targets for their linear-attention layers;
  - bf16 weights (27B in fp32 does not fit in 96 GB);
  - the contrastive head resized (`projector.embed_size` = hidden / 2).

  Rough cost at BIT's 150 epochs: ≈ 6× and 18× a 1.5B run per seed.
- **If Aero clearly beats Instruct,** larger audio LLMs are the route instead.

### Run 3: block-diffusion decoder (user request, 2026-10-19)

**Question.** With the same base weights, does decoding the sentence by block diffusion beat left-to-right
decoding? The decoder sees the whole trial's neural tokens before it writes anything, so it could fill in the
confident words first and settle the uncertain ones from both sides.

**Scope.** This is a new decoding method. §0 lists those as out of scope, so this is a user-requested exception: it
sits next to the LLM swap, in no tier, and is never in the headline table.

**Model.** Fast-dLLM v2 1.5B (`Efficient-Large-Model/Fast_dLLM_v2_1.5B` @ `25093b6f`, Apache-2.0). It is a block
diffusion LM converted from Qwen2.5-1.5B-Instruct with about 1B tokens of extra training: same 1536-d, 28 layers,
tokenizer and LoRA module names. Run 1 vs run 3 therefore swaps the AR model for its block-diffusion conversion:
the converted weights (those ~1B tokens), the training objective and the sampler change together.

**Matched to run 1:** arm B from the SSL encoder, the same projector, LoRA (r8 / α32, seven projections), BIT
prompt, contrastive term, epochs, selection (free-generation WER on BIT-val) and evaluation, including null
controls.

**New, not in BIT:**
1. **Loss:** Fast-dLLM v2's own objective, applied to embeddings:
   - the sentence tokens are masked at a random ratio per 32-token block, with a complementary-mask copy;
   - masked tokens are predicted with its token shift (position j from position j − 1);
   - its block-diffusion attention: bidirectional within a block, causal across blocks.

   The neural tokens + prompt are a clean prefix that is never masked. The CE is **summed** over the masked tokens
   (each target token once per complementary pair), as BIT sums its CE in run 1, so a sentence token has the same
   weight next to the contrastive term. Fast-dLLM v2 averages; a mean would halve that weight here, since about half
   the supervised tokens are block padding.
2. **Targets:** sentence + EOS, the rest of its block padded with EOS (as in LLaDA's instruction tuning). Decoding
   cuts at the first EOS and at 25 new tokens, as for the AR runs. Fast-dLLM v2's own SFT script instead packs
   examples back to back (LMFlow `group_texts`, no padding). Here each trial is its own sequence, because each
   carries its own neural prefix.
3. **Sampler:** Fast-dLLM v2's sub-block confidence decoding (small block 8, threshold 0.9, the model card's
   defaults) for selection and the main numbers. Other settings are tuned on BIT-val only, by evaluating BEST
   again.
   - **Decoding-order sweep** (user, 2026-10-20; eval only, `scripts/decoding_order_sweep.py`). Run 3's BEST,
     greedy, threshold 0.9, small block 8 / 1 / 4 / 32:
     - 1 = strictly left to right;
     - 4 / 8 = chunked;
     - 32 = the whole block in confidence order.

     The weights are the same, so this separates the decoding order from the conversion confound above. The run's own
     setting (8) must reproduce its summary.json. Training masks random ratios per block, so every order is in
     distribution. BIT-test is reported, but the order is chosen on BIT-val. The sweep runs under the GPU lock
     straight after run 3; it is a separate job, outside the queue script. Every decode also writes a per-pass trace
     (`traces/*.json.gz`: each masked position's proposal, its probability and whether it was committed), for
     visualising the denoising (user, 2026-10-21). The traced sampler is a line-for-line copy plus recording, so
     reproducing the run's summary.json also checks it.
   - **Revision-headroom check** (user, 2026-10-21; same job, after the sweep, chunk size 8, val and test). Each
     decoded token is hidden on its own and re-predicted with the rest of its block visible, which is what a
     within-block editing sampler would see. Reported: how often the prediction changes, and whether each single
     change fixes or breaks the sentence (word errors vs the reference); corpus WER after one round of edits above
     0.5 / 0.7 / 0.9; and an oracle that applies only the fixing edits. Run 3 was never trained to revise, so this
     estimates the headroom before a revision-trained run (about 30 GPU hours) is considered.
   - **Re-masking sampler, decode-only** (user, 2026-10-21; same job, last, chunk size 8, val and test; rule of
     `docs/REVISION_RUN.md` §4). When a block is done, the tokens committed below 0.9 (the "best guess" commits) are
     masked again and re-filled with the rest of the block visible, up to 3 rounds. No training. Compared sentence by
     sentence with the plain chunk-8 decode; BIT-val decides whether it is used, BIT-test is reported.
   - **Training speed for later diffusion runs** (user, 2026-10-22: "we need to make training faster"; required
     before an Aero → block-diffusion conversion or a revision-trained run, not for the Instruct + diffusion control,
     which must stay bit-comparable with run 3). Run 3 took ~12 min/epoch vs ~3 for the AR runs (31.7 h in all).
     Each step runs 2B sequences ([noisy ‖ clean] × two complementary masks) of twice the length, so about 4× the
     tokens, through flex attention with tiles shrunk to fit 101 KB of shared memory. The main thread was CPU-bound.
     In order of expected gain, all keeping the recipe's maths (masks, loss, batch, epochs):
     1. **Skip the noisy copy of the prefix.** Noisy blocks attend only to themselves and to *clean* earlier blocks,
        so the noisy duplicate of the prompt and neural tokens (~150–260 of each sequence's tokens) is computed and
        never used, except for the block the prefix shares with the sentence. Encode the prefix once, as clean
        context (a cache or a shortened noisy half). Expect a third to half less compute. Needs a test that the
        loss and gradients match the current step to bf16 tolerance.
     2. **Cut launch overhead:** CUDA graphs / `torch.compile` of the whole step, fewer small ops in the mask and
        label building. The CPU-contention episode (11:27 → 15:00 per epoch) showed the step is launch-bound.
     3. **Less padding:** a finer length bucket than 128, if the number of compiled shapes stays manageable.
     4. **Attention kernel:** re-tune flex attention's kernel options, or use a block-sparse kernel that fits the
        GPU's shared memory with larger tiles.
     Not allowed: shorter epochs, smaller batches (BIT's contrastive term uses in-batch negatives) or length-sorted
     batches, since they change the recipe.

**Code.**
- **Vendored decoder:** a copy of the model's `modeling.py` / `configuration.py` (`src/bit_experiments/fast_dllm_v2/`),
  shimmed so it runs in BIT's env (transformers 4.48.2, the same env as runs 1–2). The shims replace 4.52+
  decorators / base classes and expand the inference mask to 4-D. Flex attention's backward kernel uses smaller tiles
  (32/64, 2 stages). Torch 2.7 picks its H100 tiles for this GPU (capability 12.0), which need 132 KB of shared
  memory per block against the GPU's 101 KB. The first GPU timing smoke failed on exactly that (2026-10-20).
- **Model:** `src/bit_experiments/diffusion_llm.py` holds the embedding-level training step and sampler. The released
  code takes only `input_ids` in both places.
- **Checks (CPU, fp32).**
  - The training step reproduces the released forward's loss on the same masks (difference 5e-7).
  - The sampler is token-identical to the released `generate` (greedy, small block 8, thresholds 0.9 and 1.0) on
    chat prompts of 23, 30, 41 and 64 tokens.
  - At exactly 32 prompt tokens, the released `generate` ignores the prompt: it caches only prompts longer than one
    block. Ours handles it.

**Cost:** each step
sees about 4× the tokens of an AR step (noisy + clean halves, and the complementary copy), so expect several times run
1's wall time. This will be measured in the GPU smoke test.

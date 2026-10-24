# Probes, RSA and foil contrasts (DESIGN.md §5; prepared, not run)

**Status (2026-10-22).**
- The code, configs and CPU tests are ready (`tests/test_probes_foils.py`: 16 passed in BIT's environment; the 8 that
  need no model also pass in the project venv).
- The foil set is built and ready for review: `data/processed/foils/4ebac2f19f10/`, with `review.md` in it, 4,533
  foils.
- Nothing has run on real neural data. Nothing is committed.

**Tier.** 2 (DESIGN.md §0, "Probes after each stage: phonetic vs semantic content"). The probes and foils never train
or select a model.

**Question.** What do the frozen representations carry, stage by stage and arm by arm: phoneme identity, sentence
meaning, or both? And when a decoder weighs two candidate sentences given the same neural data, what kind of
difference does the neural evidence pick up? The error analysis (`results/stage2/phonetic_errors.md`) looked only at
the outputs. This looks inside, and at forced choices.

**One host job:**

```bash
<bit env>/python scripts/run_probes_foils.py --runs <run> ...
```

## 1. What is probed

Every run of the list is turned into frozen representations. All of them are eval mode, Stage 2's input pipeline
(smoothing 2, the padding fix: a trial's features in a batch equal its features alone) and bf16 autocast on CUDA, at
the encoder's frame rate (100 ms patches).

| Run name | Source | Representations | Stage |
|---|---|---|---|
| `input` | Stage 2's smoothed TX+SBP bins, patched (2,560-d) | `input` | baseline: what a linear reader gets from the array directly |
| `random` | Stage 2's NDT at its seeded random init | `encoder` | baseline: the architecture alone |
| `ssl` | `checkpoints/ssl_human/BEST`, with its own TX-only T15 read-in (TX features only) | `encoder` | after SSL |
| `A_ctc` | arm A's CTC model | `encoder`, `posterior` (41 log-posteriors) | after arm A's intermediate stage |
| `C_stage1` | `checkpoints/stage1/C_l14_b1024_s42/best` | `encoder`, `projector` | after arm C's intermediate stage |
| any Stage 2 run | `checkpoints/stage2/<run>/BEST` (or the checkpoint an eval-only run used) | `encoder`, `projector`, and `ablated` for an ablation run | after LLM fine-tuning |

- Arm B has no intermediate stage: its "after SSL" is `ssl`, its "after Stage 2" is `B_ssl_s42`.
- An ablation run (docs/RETRAINING_ABLATIONS.md) is rebuilt with its ablation module between encoder and projector.
- Stage 2 runs available now on the host:
  - `B_ssl_s42`;
  - `B_rand_bit4a` (BIT's own sentence model: its encoder never saw SSL);
  - `B_ssl_qwen25-1.5b-instruct_s42`, `B_ssl_qwen25-1.5b-noprompt_s42`;
  - `B_ssl_fast-dllm-v2-1.5b_s42` (run 3).
- Later: `C_ssl_l14_b1024_s42` (~Sat 20:00 UTC), `A_ssl_s42` (~Sun 05:00), the Instruct and Aero diffusion runs, the
  B seeds, and the ablation runs.

## 2. Frame-level phoneme probe

- **Labels.** Each frame's phoneme from CTC forced alignment with arm A's model, the same labels as the eraser
  (docs/RETRAINING_ABLATIONS.md §2, `bit_experiments.phoneme_frames`):
  - BIT's g2p_en sequences, 40 classes (39 phonemes + SIL);
  - each label owns its Viterbi run;
  - blank gaps are split at the midpoint, with a 2-frame margin;
  - `prepare_ablations.py labels` computes them for all three splits, once.

  They are probe targets only, never used to train or select a decoder (DESIGN.md §5). Arm A was trained on the
  train split's phoneme sequences, so its train alignments are in-sample. Val and test alignments use the true
  transcript and arm A's posteriors, which is forced alignment, not decoding.
- **Probe.**
  - Multinomial logistic regression on standardized features, trained on a seeded subsample of ≤ 200k labelled
    train frames (the same frames for every run).
  - Full-batch L-BFGS, with L2 chosen by val log-loss from {1e-5, 1e-4, 1e-3, 1e-2}. Selecting on val log-loss keeps
    the probabilities calibrated, so the information estimate means something.
  - Reported on test:
    - accuracy and balanced accuracy;
    - **information** = prior log-loss − probe log-loss, in bits per frame, with the prior from the train class
      frequencies;
    - the majority-class rate;
    - cluster-bootstrap 95% CIs over trials (frames within a trial are not independent).
- **Reading.**
  - The comparison is across stages and arms against two baselines: `input` (motor cortex is phoneme-decodable
    linearly, so this is not chance) and `random`.
  - "Information beyond the input" is meaningless for a deterministic function of it. The question is how
    accessible the phoneme code is, linearly, at each stage.
  - A Stage 2 encoder that holds as much linear phoneme information as `A_ctc`, with no phoneme labels ever (arm B),
    is direct evidence that phoneme-free training rebuilds a phonetic code.
  - The MLP probe (`probes.fit_mlp_probe`) is available for a non-linear read-out. It is used in the erasure check,
    and not by default here (cost).

## 3. RSA (BIT App. L.1) and partial RSA

- **Neural vectors.** Each trial's representation sequence is cut into 10 contiguous segments, and the segment means
  are concatenated. BIT App. L.1 (read 2026-10-22): "each sequence is divided into 10 segments and the segment
  averages are concatenated".
- **Model vectors.** The cached LLM sentence embeddings, `data/processed/text_targets`: Aero's LM over the sentence
  alone, mean-pooled, L2-normalized.
  - `emb`, the input-embedding pool: BIT's aligner target, and BIT's own "mean pooling across tokens";
  - `layer14`: arm C's target.
  - `layer28` and others can be added in `configs/probes.yaml`.
- **RDMs and test.** RDMs are 1 − cosine, compared by the Pearson r of their upper triangles.
- **Trials.** BIT-test. Both BIT's T15 selection (120–200 patch tokens; "mean ≈160") and all 713 trials.
- **Partial RSA.** Covariate RDMs are regressed out of both RDMs, by least squares on the upper triangles, and the
  residuals are correlated.
  - `phonetic`: the normalized phoneme edit distance between the two sentences' label sequences;
  - `phonetic_length`: phonetic, plus |Δ phoneme count| and |Δ duration|, both relative.
- **Statistics.** A one-sided Mantel permutation test (1000 permutations of the neural RDM's rows and columns
  together). A trial-bootstrap 95% CI (100 resamples, self-pairs of a resampled trial dropped; Nili et al. 2014).
  Also reported: r(neural, phonetic RDM) and r(target, phonetic RDM), the size of the confound.
- **Reading.**
  - The plain r alone cannot separate meaning from sound: sentences that sound alike also tend to share words and
    hence embeddings.
  - The partial r (phonetic regressed out) is the test for sentence-level structure beyond phonetics: positive with
    a small Mantel p means the neural geometry follows the LLM's meaning geometry beyond what phoneme overlap
    explains.
  - Edit distance is a linear stand-in for phonetic similarity. A residual can still be phonetic in some other
    form, so a positive partial r is suggestive, not proof.

## 4. Foil forced choice

### 4.1 The foil set (text side, deterministic, offline, reviewed)

- **Build.** `scripts/build_foils.py`, run on the Mac in the project venv (CMU dictionary: `cmudict` 1.1.3;
  frequencies: `wordfreq` 3.1.1). The tables are in `configs/foils.yaml` (curated; the manifest records their sha256).
  - Each foil has its own generator, seeded by (seed, split, trial, type, rule), so adding trials or rules changes
    no other foil.
  - Sentences: the references of `results/stage2/B_ssl_s42/decodes/{val,test}_greedy_real.csv`. The host job checks
    each one against BIT's loader before scoring.
  - No external API, no LLM-generated text.
- **Review.** `review.md` holds 12 seeded examples per variant. I reviewed them and fixed the rules that misfired:
  - compound numbers ("two thousand" → "two sixty");
  - adverbial "first";
  - first names that are common words (rose, mark, bill);
  - "new england" → "new japan";
  - "help out" → "assist out";
  - "they'll not";
  - "you've really strong views".

  The user should still skim `review.md`, especially the synonym rows. If a rule is bad, edit the tables and
  rebuild; the CSV is hash-checked and is never edited by hand.

| Type / variant | Rule | Test foils | Median phoneme distance (test) | A semantic code would | A phonetic code would |
|---|---|---|---|---|---|
| paraphrase / synonym | one content word → curated near-synonym (big → large, kids → children) | 248 | 0.19 | accept (~50%) | reject |
| paraphrase / contraction | "do not" ↔ "don't", "it is" ↔ "it's" ("it's been" → "it has been") | 215 | 0.10 | accept | partly reject |
| sound_alike / neighbour | one content word → the most frequent real word at phoneme edit distance 1; no homophone, no inflection, Zipf ≥ 3 | 680 | 0.05 | reject | barely reject |
| order_swap / adjacent | two adjacent different words swapped | 713 | 0.21 | partly reject | reject (sequence) |
| negation / remove, insert | first negation removed ("don't" → "do"), else "not" after the first copula / modal | 88 + 281 | 0.11 | reject | reject (the extra / missing "not") |
| number_name / number:\*, name:\* | number word or day / month / place / first name → another of its category, same syllable count where possible; no compounds, "the first …" only, not after "new" | 45 (val + test: 87) | 0.10–0.27 | reject | reject when distinct |

Every test trial has at least one foil (3.2 per trial). number_name is small, so it is also reported on val + test
pooled.

### 4.2 Scores and conditions

| Scorer | Runs | Score of a sentence s given the trial's neural data |
|---|---|---|
| `loglik` | AR Stage 2 runs (Aero, Qwen) | log p(s + EOS │ prompt, neural tokens), teacher forced: BIT's own forward with the padding fix, exactly the training CE (tested against explicit teacher forcing) |
| `denoise` | block-diffusion runs | minus the training loss of `block_diffusion_ce` for s, averaged over 8 complementary mask pairs (below) |
| `aligner` | every Stage 2 run | cosine of BIT's in-loop contrastive head (ModalityAligner): mean projector output vs the mean text embedding of [prompt, s, EOS] (tested equal to BIT's module) |
| `stage1` | `C_stage1` | cosine of arm C's Stage 1 head between the trial's neural embedding and s's fixed layer-14 target |

**The diffusion score, consistent with the training loss.**
- **The step.** The decoder was trained with Fast-dLLM v2's block-diffusion step:
  - a mask ratio t ~ U(ε, 1) per 32-token block; each target token masked with probability t;
  - a complementary copy masking the other tokens;
  - the CE summed over the masked tokens of both copies (sentence, EOS and the block's EOS padding, each exactly
    once per pair).
- **The score.** `denoise` is the expected value of that objective, estimated by Monte Carlo. The prefix and targets
  are built by the model's own forward (captured just before its loss), and the per-row CE equals
  `block_diffusion_ce`'s with the same masks (tested).
- **Common random numbers.** The masks are drawn once per (trial, draw) and shared by every sentence of the trial and
  every condition. The same sentence in any row gets the same score (tested), which cancels most of the Monte Carlo
  noise in a margin. The per-sentence SE is saved (`se_true`, `se_foil`).
- **Not the 1/t-weighted ELBO.** With complementary masks the complement's weight 1/(1 − t) is unbounded, so its
  variance is too. The forced choice needs a consistent score, not a calibrated likelihood.
- **Not left-to-right unmasking.** It is a single order, 25 passes per sentence. A possible extension.

**Conditions:**
- `real`;
- `null_shuffled`: another trial's neural data, from `stage2.derangement` with the run's seed, i.e. exactly the
  pairing of the run's own null decodes;
- `null_prompt_only`: no neural tokens, the decoder's text prior. Skipped for the no-prompt run and for the cosine
  scorers.

### 4.3 Analysis (`foils.json`, per scorer; test and val + test)

- **Forced-choice accuracy** per type and per variant, in each condition: true sentence preferred, ties counted 1/2.
  Also the mean margin, score(true) − score(foil).
- **Neural evidence.** The real margin minus the null margin, and the real accuracy minus the null accuracy, with
  paired trial-bootstrap 95% CIs (2000 resamples). `null_shuffled` is the primary null, because it stays in the
  decoder's input distribution.
- **Type effects at matched phonetic distance.** An OLS of the per-foil evidence (real − shuffled margin) on the
  foil's phoneme distance plus one dummy per type, with sound_alike as the reference, and trial-bootstrap CIs.
  - A purely phonetic code: the evidence depends on distance only, and the type effects are ≈ 0.
  - A semantic code: paraphrase is clearly negative (less evidence than its distance predicts); negation and
    number_name are positive.

  This is the single most direct test of meaning beyond sound in the decoder's use of the neural data.
- **What the pattern means:**
  - The LM prior alone rejects order swaps (they are ungrammatical), whatever the neural data; hence the nulls.
  - A decoder that rejects synonyms as strongly as their phonetic distance predicts, and sound-alikes only weakly,
    is reading phonetics.
  - A decoder that accepts synonyms and rejects sound-alikes is reading meaning.
  - Arm C's `stage1` score is a pure sentence-embedding match, so its pattern is the semantic reference point.

## 5. Commands

**Once, on the Mac** (already done; repeat only after editing `configs/foils.yaml`):

```bash
.venv/bin/python scripts/build_foils.py
```

Then copy `data/processed/foils/` to the host copy's `data/processed/foils/` (e.g. `scripts/remote/sync.sh data`).

**On the host** (BIT's env, a fresh new-layout code copy as in docs/RETRAINING_ABLATIONS.md §4). It needs the label
cache from `prepare_ablations.py labels` (~5 min GPU, shared with the ablations) and the existing
`data/processed/text_targets`.

```bash
<bit env>/python -m pytest tests/test_probes_foils.py          # CPU, ~30 s: 16 passed
<bit env>/python scripts/prepare_ablations.py labels --device cuda
# smoke (minutes): every part on 16 trials per split
<bit env>/python scripts/run_probes_foils.py --runs A_ctc B_ssl_s42 --limit 16 --out results/probes_smoke
# the runs available now
<bit env>/python scripts/run_probes_foils.py --runs input random ssl A_ctc C_stage1 B_ssl_s42 B_rand_bit4a \
    B_ssl_qwen25-1.5b-instruct_s42 B_ssl_qwen25-1.5b-noprompt_s42 B_ssl_fast-dllm-v2-1.5b_s42
# later, as they finish (results/probes/SUMMARY.md is rewritten after every job)
<bit env>/python scripts/run_probes_foils.py --runs C_ssl_l14_b1024_s42 A_ssl_s42
<bit env>/python scripts/run_probes_foils.py --summarize
```

**Under the GPU lock:**

```bash
<GPU-lock wrapper> probes-foils scripts/run_probes_foils.py --runs ...
```

The same caveat about the wrapper's working directory as in RETRAINING_ABLATIONS.md §4 applies.

**Outputs.** In `results/probes/<run>/`:
- `probe.json` (per representation: test and val scores, the L2 grid);
- `rsa.json` (per representation and trial selection: phonetic, then each target plain and partial);
- `foil_scores.csv` (one row per foil × condition × scorer);
- `foils.json`;
- `timing.json` (wall minutes per part).

Plus `results/probes/SUMMARY.md` across runs.

## 6. GPU minutes

Measured on the Mac CPU:
- RSA: 3.6 s per plain and 7.2 s per partial comparison at n = 713 (1000 permutations + 100 bootstraps);
- phoneme RDM: 0.7 s;
- an L-BFGS probe pass at 200k × 1536: 2.2 s per 100 iterations.

The rest are estimates from FLOPs and from run 3's training throughput. Each job holds the lock for all of its
parts.

| Part | Per | Estimate |
|---|---|---|
| Data load (hdf5) | job | ~3 min |
| Frozen features, all splits | source | ~1–2 min (7M-param NDT over 9.5k trials) |
| Phoneme probe | representation | ~0.5–1 min |
| RSA (2 targets × plain / 2 partial + phonetic, 2 trial selections) | representation | ~1.3 min (CPU) |
| Foils `loglik` + `aligner`, val + test | AR run | ~8 min: ~18k sentence scorings in ~4.3k forwards of ≤ 8 rows × ~200 tokens, plus model load |
| Foils `denoise` + `aligner`, test | diffusion run | ~40 min: ~3k sentences × 3 conditions × 8 pairs × 1,024 tokens ≈ 73M tokens, vs run 3's ~2M forward-equivalent tokens / min in training |
| Foils `stage1` | C_stage1 | ~3 min (layer-14 targets for ~6k sentences through Aero's LM) |

**Per job:**
- **Runs available now** (5 sources + 5 Stage 2 runs; 4 AR, 1 diffusion; 17 representations): **~2.2 GPU hours**.
- **Each later AR run:** ~12 min. **Each later diffusion run:** ~45 min. `C_ssl_l14_b1024_s42` + `A_ssl_s42`:
  ~25 min.
- **Cheaper diffusion option:** `diffusion_draws: 4` halves the diffusion part, at a larger Monte Carlo SE (the
  margins share masks, so the SE of a difference is much smaller than either score's).

## 7. Tests (`tests/test_probes_foils.py`)

- **Probes:**
  - planted linear information found; none in noise (≈ 0 bits, majority accuracy);
  - an XOR code read by the MLP, not by the linear probe.
- **RSA:**
  - BIT's pooling;
  - r = 1 for equal RDMs with a small Mantel p;
  - a shared phonetic factor gives a positive plain r, a partial r ≈ 0 and p > 0.01;
  - CIs bracket r.
- **Foil rules** (the real tables, a fake 31-word lexicon): every rule and every exclusion in §4.1; determinism and
  trial independence; YAML booleans refused.
- **Scorers** (tiny Qwen2, word-level tokenizer):
  - `loglik` = explicit teacher forcing, batch = lone, prompt-only = the text prior;
  - `aligner` = BIT's ModalityAligner;
  - `denoise` per-row CE sums to `block_diffusion_ce`'s; deterministic per seed; common random numbers;
    batch = lone.
- **Analysis:** forced choice, CIs, ties, and type effects recovering planted coefficients.
- **The launcher end to end** on a synthetic world:
  - all five source kinds;
  - an AR, a diffusion and a bottleneck Stage 2 run;
  - an eval-only checkpoint path from another code copy;
  - artifacts from `prepare_ablations.py` and `build_foils.py`;
  - every part and the summary.

## 8. Not done / open

- Nothing has run on real data. The label cache (`prepare_ablations.py labels`) comes first. Its PER check confirms
  arm A's model works through Stage 2's pipeline.
- Probes of the LLM's own hidden states at the neural-token positions are not included (they need the LLM forward
  per trial). That is the next step if the projector-level results leave the question open.
- The paraphrase foils are single-word synonyms and contractions, not free rewrites. A reviewed set of full
  paraphrases could be added as a CSV of the same columns (variant `curated`) if the synonym results are ambiguous.
- `wordfreq` and `cmudict` are installed only in the Mac's project venv, so the foil set is built there and copied.

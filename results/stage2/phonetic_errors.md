# Sound-alike or language-driven? The word errors of the phoneme-free decoders

2026-10-22. BIT-test, greedy decodes, one seed per decoder. CPU only.

```bash
.venv/bin/python scripts/phonetic_error_analysis.py
```

The script writes `results/stage2/phonetic_errors/`: `summary.json`, `tables.md` (every table, all slices and contrasts), and one CSV of pairs per run and condition. Its docstring has the full method. I installed three packages into `.venv`: `pronouncing` 0.3.0 with `cmudict` 1.1.3, and `wordfreq` 3.1.1. distilbert-base-uncased was already in the local HF cache.

**Question.** Is the model "still learning phonemes, just inside the model"? This note asks a narrower, cheaper question. When a decoder gets a word wrong, does the wrong word sound like the right one, or does it fit the sentence and its meaning?

**Caveat.** This tests the error pattern of the outputs only, not internal representations. A continuous articulatory code would give the same pattern as discrete phoneme units. The frozen-encoder phoneme probe and the foil contrasts (`docs/DESIGN.md` §5) are the stronger tests.

## Method in brief

- **Alignment.** Words are normalized as the project's WER does (`textmetrics.normalize`). They're aligned by word-level Levenshtein with unit costs, and the edit count matches `word_errors` in every trial. The recomputed WERs match each run's `summary.json`. The costs ignore what the words are, so the alignment can't favour sound-alike pairs. Each substitution gives a pair (reference word → decoded word).
- **Phonetic measures** (CMU dictionary, stress removed, best match over listed pronunciations):
  - **phoneme similarity** = 1 − phoneme edits / longer length;
  - a version with articulatory-feature costs;
  - first and last phoneme match;
  - **sound-alike** = at most half the phonemes changed;
  - **homophone** = identical pronunciations.

  Pairs with a word missing from the dictionary are dropped: 0.2–2.5% of substitutions.
- **Language-side measures:**
  - **meaning**: cosine of distilbert word embeddings, each word embedded alone. **Related** = cosine ≥ 0.4; for scale, cat/dog is 0.56, car/truck 0.39 and cat/hat 0.16.
  - **context fit**: distilbert masked-LM log-probability of the word in the true word's slot of the reference sentence. It covers single-piece words, 95–97% of pairs.
  - **frequency**: wordfreq Zipf.
- **Chance level (null).** The decoded word is replaced by a random substitute drawn from the same run's substitutions, with the same number of phonemes. **Excess** = observed − chance. CIs are 95% cluster bootstraps over trials, with the same resampled trials for every run, so run contrasts are paired. p is a one-sided Monte Carlo permutation (5000 draws).
- **Spelling confound.** The LLMs read and write spelling-based BPE pieces, so a second null matches each random substitute's spelling similarity to the reference word. Its excess is sound similarity beyond spelling.
- **Negative control.** The same analysis runs on each run's `null_shuffled` decodes, which use another trial's neural data. Any excess there comes from alignment or sentence position, not from the neural signal.

## Results

### 1. Every decoder makes sound-alike errors far above chance, and the neural signal causes them

| Decoder | test WER | pairs | sound-alike: obs / chance | phoneme-sim. excess [95% CI] | beyond spelling | homophones | related in meaning: obs / chance | meaning excess | context fit excess (nats) | fits context better than the true word: obs / chance |
|---|---|---|---|---|---|---|---|---|---|---|
| Fast-dLLM v2 (diffusion) | 25.2% | 1093 | **23.6% / 2.7%** | **+0.192** [+0.178, +0.206] | +0.107 [+0.097, +0.117] | 20 (1.8%) | 9.7% / 2.9% | +0.084 [+0.075, +0.094] | +1.90 [+1.71, +2.10] | 16% / 8% |
| Aero | 34.2% | 1371 | 16.1% / 2.7% | +0.118 [+0.105, +0.131] | +0.063 [+0.054, +0.073] | 7 (0.5%) | 8.4% / 3.5% | +0.062 [+0.052, +0.072] | +1.82 [+1.60, +2.05] | 15% / 8% |
| Aero, random-init encoder (4a) | 33.6% | 1368 | 16.2% / 2.7% | +0.124 [+0.111, +0.137] | +0.068 [+0.059, +0.077] | 11 (0.8%) | 9.1% / 4.0% | +0.063 [+0.054, +0.072] | +1.83 [+1.63, +2.03] | 16% / 8% |
| Qwen base, no prompt | 43.8% | 1631 | 10.4% / 2.5% | +0.074 [+0.065, +0.084] | +0.043 [+0.036, +0.050] | 6 (0.4%) | 9.3% / 3.8% | +0.052 [+0.045, +0.061] | +1.57 [+1.39, +1.75] | 13% / 8% |
| Qwen Instruct | 55.7% | 2096 | 6.9% / 2.5% | +0.048 [+0.040, +0.056] | +0.027 [+0.021, +0.033] | 2 (0.1%) | 7.7% / 4.8% | +0.038 [+0.032, +0.044] | +1.20 [+1.05, +1.34] | 10% / 6% |

- All p < 0.001.
- Raw phoneme similarity for Fast-dLLM v2 is 0.288, against 0.096 by chance.
- Homophones include their→there, clothes→close (×3), threw→through, hear→here, sea→see and four→for. A text LM rarely produces these swaps by itself in context.
- **Shuffled-input control, all five runs:**
  - Phoneme-similarity excess is −0.002 to +0.004, every CI includes 0.
  - Sound-alikes are 3.0–3.5% against 3.2–3.4% by chance.
  - The alignment and sentence position alone create no sound-alike effect.
  - They do create a small positional effect on the language side: meaning +0.006 to +0.010, context fit +0.13 to +0.20 nats.
  - Subtracting the control leaves the language-side effects almost whole. For Fast-dLLM v2, real − control is meaning +0.074 [+0.064, +0.085] and context +1.72 nats [+1.50, +1.93].
- **The phonetic effect survives every check:**

  | Fast-dLLM v2 slice | Phoneme-sim. excess |
  |---|---|
  | Content words only | +0.184 [+0.169, +0.199] |
  | Pairs sharing a stem dropped (use/used) | +0.184 [+0.171, +0.198] |
  | Spelling-matched null (all pairs) | +0.107 |

  The other runs show the same pattern (`tables.md`).
- **Val replicates it** (the split used for checkpoint selection): the same ordering, with Fast-dLLM v2 at +0.188, Aero +0.127, base +0.089, Instruct +0.048 and Aero random-init +0.123.

### 2. The errors are also language-driven, not only sound-driven

Substitutes are related in meaning about 3× more often than chance. They fit the true sentence about 1.9 nats better than a random substitute, and twice as often fit it better than the true word does.

**The meaning effect is not a by-product of sound.** Among pairs that do **not** sound alike, the meaning excess is still +0.074 [+0.064, +0.084] for Fast-dLLM v2 and +0.030 to +0.050 for the others. The context-fit excess there is +1.85 nats.

**Word frequency is not the driver.**
- Rare words are the ones that get replaced: the median Zipf of substituted reference words is 4.96, against 6.38 for correctly decoded ones (Fast-dLLM v2; the others are similar).
- But the chosen substitute is no commoner than a random substitute would be: 68% vs 64% commoner than the reference word for Fast-dLLM v2, and 53–60% vs 53–58% for the others.

Sound-alike (phoneme sim. ≥ 0.5) × related in meaning (cosine ≥ 0.4), share of pairs:

| Decoder | both | sound only | meaning only | neither |
|---|---|---|---|---|
| Fast-dLLM v2 | 4.0% | 19.6% | 5.7% | 70.7% |
| Aero | 3.1% | 13.0% | 5.3% | 78.6% |
| Aero, random-init encoder | 3.8% | 12.4% | 5.3% | 78.5% |
| Qwen base | 2.8% | 7.6% | 6.6% | 83.1% |
| Qwen Instruct | 1.9% | 5.0% | 5.9% | 87.3% |
| *shuffled control (range)* | *0.9–1.2%* | *1.8–2.3%* | *5.6–6.2%* | *90.6–91.2%* |

These thresholds are crude. "Neither" still carries phonetic signal: among pairs that don't sound alike, the phoneme-similarity excess is +0.089 for Fast-dLLM v2.

### 3. Better decoders make errors that are more sound-like; the context fit hardly changes between them

Paired contrasts in excess (a − b), 95% CI. "Mostly-right trials" means each run's own trials with sentence WER ≤ 0.5.

| a − b | phoneme sim., all | phoneme sim., mostly-right trials | beyond spelling | meaning | context fit (nats) |
|---|---|---|---|---|---|
| Fast-dLLM v2 − Aero | **+0.074** [+0.058, +0.092] | +0.031 [+0.006, +0.056] | +0.044 [+0.032, +0.057] | +0.022 [+0.010, +0.034] | +0.08 [−0.17, +0.34] |
| Fast-dLLM v2 − Qwen base | +0.118 [+0.102, +0.134] | +0.098 [+0.074, +0.123] | +0.064 [+0.053, +0.076] | +0.032 [+0.021, +0.043] | +0.33 [+0.08, +0.59] |
| Aero − Qwen base | +0.043 [+0.029, +0.057] | +0.067 [+0.043, +0.093] | +0.020 [+0.010, +0.031] | +0.010 [−0.001, +0.021] | +0.25 [+0.01, +0.49] |
| Aero − Qwen Instruct | +0.070 [+0.056, +0.084] | +0.105 [+0.078, +0.133] | +0.037 [+0.026, +0.047] | +0.024 [+0.014, +0.035] | +0.63 [+0.39, +0.87] |
| Qwen base − Qwen Instruct | +0.027 [+0.015, +0.038] | +0.038 [+0.012, +0.062] | +0.017 [+0.008, +0.025] | +0.015 [+0.006, +0.024] | +0.38 [+0.18, +0.57] |
| Aero − Aero random-init | −0.006 [−0.021, +0.009] | −0.000 [−0.026, +0.027] | −0.005 [−0.016, +0.006] | −0.001 [−0.013, +0.011] | −0.01 [−0.25, +0.24] |

On val, Fast-dLLM v2 − Aero is +0.062 [+0.043, +0.080] for phoneme similarity.

Sound-alike errors in excess of chance, per 100 reference words: Fast-dLLM v2 4.8, Aero 3.9, Aero random-init 3.9, base 2.7, Instruct 1.9.

The best decoder makes the *most* sound-alike errors in absolute terms while making the fewest errors overall. What separates the decoders is the unrelated words. Qwen Instruct's extra errors are mostly words that neither sound alike nor share meaning (87%).

### 4. Slices

- **Seen vs unseen sentences:** no reliable difference.
  - Only 81 seen trials, 55–108 pairs per run.
  - Seen − unseen phoneme-similarity excess is −0.030 for Fast-dLLM v2, −0.039 for Aero and −0.006 for Instruct, all with CIs spanning 0.
  - Qwen base is the exception: −0.044 [−0.077, −0.012].
- **Mostly-right vs mostly-wrong trials:**
  - Errors in nearly right sentences are more sound-like (+0.026 to +0.091) and fit the context better (+1.2 to +2.4 nats).
  - Even in mostly-wrong (hallucinated) sentences, the aligned substitutes still sound alike above chance: Fast-dLLM v2 +0.164 [+0.140, +0.187], Instruct +0.043.
  - The decoders keep a sound skeleton while rewriting the sentence. For test trial 695, "and hurricane jose had not turned north yet", the decoders wrote:
    - Fast-dLLM v2: "and android orbit had not turned north yet"
    - Aero: "and uranium oxide and not uranium plutonium"
    - Qwen base: "an american auto company had not built cars yet"
    - Qwen Instruct: "an american grocery store now looks like a super mall"

  The brief's examples are val trials from Fast-dLLM v2: "used to hassle somebody" → "use to hold somebody", and "grocery store" → "concert store".
- **Non-words:** 0.47% of Fast-dLLM v2's output words aren't words (22 of 4730), against ≤ 0.1% for the autoregressive decoders. They're sub-word mash-ups that often keep the target's onset: occasionally → "enasionally", supported → "sumendant", require → "recar". These pairs have no CMU pronunciation and are dropped, which slightly *understates* the diffusion model's form-based errors.

### 20 most common substitutions (test)

Most pairs occur once; the top ones are mostly function words.

| # | Fast-dLLM v2 | Aero | Qwen base, no prompt | Qwen Instruct | Aero, random-init encoder |
|---|---|---|---|---|---|
| 1 | has → as (4) | the → a (3) | in → the (2) | of → in (3) | in → and (3) |
| 2 | the → they (3) | to → the (3) | the → it's (2) | as → a (3) | for → your (3) |
| 3 | their → there (3) | he's → it's (3) | these → those (2) | in → and (3) | in → it (3) |
| 4 | such → just (3) | out → up (3) | on → and (2) | of → the (3) | to → do (3) |
| 5 | clothes → close (3) | the → that (2) | the → a (2) | a → and (3) | well → will (2) |
| 6 | in → and (3) | it → in (2) | a → to (2) | in → it (3) | has → as (2) |
| 7 | and → a (3) | the → do (2) | not → but (2) | in → of (3) | a → he (2) |
| 8 | in → it (3) | of → is (2) | to → the (2) | it → it's (3) | an → a (2) |
| 9 | anyway → anymore (2) | up → of (2) | to → a (2) | of → over (3) | in → is (2) |
| 10 | is → in (2) | such → just (2) | that → there (2) | in → the (2) | as → you (2) |
| 11 | it → he (2) | look → like (2) | the → go (2) | on → of (2) | how → i (2) |
| 12 | oh → who (2) | him → of (2) | of → do (2) | in → with (2) | are → a (2) |
| 13 | got → not (2) | year → here (2) | as → have (2) | in → a (2) | as → for (2) |
| 14 | its → it's (2) | a → he (2) | i → you (2) | the → little (2) | year → here (2) |
| 15 | has → is (2) | and → a (2) | for → of (2) | work → week (2) | the → they (2) |
| 16 | bay → pay (2) | of → the (2) | more → for (2) | in → to (2) | education → experience (2) |
| 17 | once → what's (2) | had → and (2) | money → morning (2) | for → of (2) | building → buying (2) |
| 18 | look → like (2) | new → you (2) | his → is (2) | up → it (2) | its → it's (2) |
| 19 | half → have (2) | they're → there (2) | run → one (2) | her → the (2) | high → i (2) |
| 20 | what's → what (2) | as → has (2) | have → i (2) | of → her (2) | and → at (2) |

## Conclusions

1. **The errors carry a clear sound signature.** About a quarter of the diffusion model's substitutions are sound-alikes, against 3% by chance. It produces homophones (their→there, four→for, sea→see) and keeps first and last phonemes. The effect beyond spelling is about half as large as before the spelling match, and it is still well above chance. Shuffled neural input removes it entirely. So the effect comes from the neural signal, not from the alignment, spelling or the LLM alone.

   This fits an encoder that passes on phonetic or articulatory information, which is expected for speech-motor cortex. It doesn't show that the encoder has phoneme *units*.
2. **The LLM's prior shapes the errors too.** Substitutes fit the true sentence context and are related in meaning above chance, even when they don't sound alike. But the chosen word is no commoner than chance. Most errors are a sound-shaped guess that the language model makes plausible in context.
3. **The better decoders have more sound-like errors, not more fluent ones.** The ranking matches WER, Fast-dLLM v2 > Aero ≈ Aero random-init > Qwen base > Qwen Instruct, for both the phonetic and the language-side measures. Fast-dLLM v2's errors are more sound-like than Aero's, by +0.074 overall and +0.031 on mostly-right trials, but they fit the context no better (+0.08 nats, n.s.). One reading: the diffusion model's gain comes from using more of the neural (articulatory) signal, not from leaning harder on the language prior. Instruct's extra errors are mostly unrelated words.
4. **SSL vs random encoder init (Aero):** identical error profiles on every measure.
5. **Seen vs unseen sentences:** no reliable difference, with too few seen trials to say more.

## Limitations

- Error pattern only. See the caveat at the top: §5's frozen-encoder phoneme probe and phonetic-neighbour foils are the real test of "phonemes inside the model".
- One seed per decoder. The runs differ in WER, so contrasts between runs mix error *type* with error *rate*. The mostly-right slice only partly separates them.
- **Measurement choices:**
  - The meaning measure is crude: decontextualized distilbert embeddings, and function-word pairs (the/a) score high.
  - The context measure uses a different LM from the decoders.
  - The 0.5 and 0.4 thresholds are arbitrary; the continuous excesses don't depend on them.
- **Pair independence.** The permutation p treats pairs as independent; the trial-cluster bootstrap CIs are the main uncertainty measure.
- **Instruct + diffusion control:** `B_ssl_qwen25-1.5b-instruct-diffusion_s42` is still training on the GPU host and wasn't analysed. Once its `decodes/test_greedy_real.csv` is pulled to `results/stage2/`, re-running the script includes it automatically.

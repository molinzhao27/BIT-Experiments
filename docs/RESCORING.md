# N-best rescoring (side test)

**Side test, outside the tiers, never a headline number.** DESIGN.md §0 keeps n-gram / LLM rescoring out of every
headline result. This asks a narrower question about the decoders we already trained: how much of their error is in
the choice among sentences they already propose, and does a text-only LM pick better than they do? The headline
numbers stay free-generation greedy decodes.

Code: `scripts/nbest_decode.py` (candidates), `scripts/rescore_nbest.py` (OPT scores, selection, metrics),
`src/bit_experiments/rescoring.py`. Config: `configs/rescoring.yaml`. Tests: `tests/test_rescoring.py`.

## Pipeline

1. **Candidates** (`nbest_decode.py`, a GPU with BIT's environment). The run is rebuilt eval-only from its `run.json`
   as `train_stage2.py --eval-only` builds it (BIT's sentence config, the run's LLM entry and prompt,
   `stage2.fix_padding`, BEST via `stage2.load_model_state`), without BIT's Trainer. This repository's copy of BIT's
   YAMLs is unpatched, so the patched projector (`inter_size` 1024) is taken from `run.json`. The model, method and
   data sections must then equal `run.json`, paths aside. Per trial of BIT-val and BIT-test, N = 12 candidates:
   - the greedy decode, exactly as `stage2.generate` produces it (checked: the first 20 trials equal its output);
   - K = 11 nucleus samples at BIT's settings (p 0.9, T 0.7, ≤ 25 new tokens), seeded once per trial with
     `seed * 100003 + trial`. AR decoders draw them in one `generate` call (`num_return_sequences`). A diffusion
     decoder runs its plain sampler 11 times, plus 11 more per extra sub-block size in `diffusion_orders` (default
     none).

   Duplicates are kept. Per candidate it records the decoder's own score: summed and mean log-probability of its
   tokens, up to and including EOS. For AR decoders this comes from the raw logits of the generate call. For
   diffusion decoders it is the log-probability each token had in the pass that committed it
   (`BCI_DiffusionLLM.commit_logprobs`, off by default; it changes no decode, `tests/test_rescoring.py`). BIT's Aero
   path carries Qwen2-Audio's audio tower, which BIT never runs (`project_to_audio: false`). It is replaced by an
   empty stand-in, so the model fits a 12 GB card; numerics are unchanged.
2. **OPT-6.7B** (`rescore_nbest.py opt`, fp16; on the Mac only while its GPU is free). Each unique candidate text is
   scored after BIT's normalization (`textmetrics.normalize`, lowercase, no punctuation), as Card et al. 2024 rescore
   with OPT: `</s>` (OPT's BOS) first, every later token scored, no end token. The log-prob of `</s>` after the
   sentence is cached too (`opt.eos`). Scores are cached per text and shared by all runs.
3. **Selection** (`rescore_nbest.py report`). Candidates that normalize alike are one candidate with a count. Rules:
   - `greedy`: the baseline;
   - `opt`: the highest OPT log-likelihood;
   - `dec_opt`: decoder log-prob + α · OPT log-prob + β · words.

   α and β are grid-searched on **BIT-val only** (corpus WER), then applied unchanged to BIT-test. Ties go to the
   greedy decode, then to the most-sampled candidate.
4. **Metrics on BIT-test:**
   - corpus WER (BIT's normalization), with a paired trial bootstrap vs greedy (10,000 resamples, seed 0);
   - oracle WER (the best candidate per trial = the headroom);
   - seen / unseen sentences (the flags come from the run's own decodes);
   - BERTScore F1 of the chosen sentences (`scripts/evaluate_decodes.py` in the project `.venv`, `configs/eval.yaml`).

Written but held (off in `configs/rescoring.yaml`, untested beyond the unit tests): the rules `decoder`, `opt_norm`,
`mbr`, `contrastive`, `dec_contrastive` and `all`, and selective-prediction coverage. Not built: shuffled-input null
candidates, the contrastive-head score, other text LMs, cross-decoder scoring and a diffusion pseudo-likelihood.

## Commands

The candidates are generated on the spare PC's RTX 4070 (CUDA, BIT's versions: torch 2.7.1+cu128, transformers
4.48.2, peft 0.11.1), under that machine's shared GPU lock. OPT runs on the Mac (MPS). The PC copy holds `src`,
`scripts`, `configs`, `external/bit` and `pyproject.toml`, plus each run's `results/stage2/<run>/{run.json,
summary.json, decodes}`. The models and the T15 inputs are streamed from the GPU host through the Mac and are never
stored on it: the Aero-1-Audio snapshot, each run's BEST, and the 41 `data_val.hdf5` files that BIT-val and BIT-test
are drawn from.

```bash
# on the PC (paths are arguments; <root> = the PC's working directory)
<root>/env/bin/python scripts/nbest_decode.py --run RUN --gate \
    --data-root <root>/data/raw --hf-root <root>/models/hf --ckpt-root <root>/models/stage2   # reproduction check
<root>/env/bin/python scripts/nbest_decode.py --run RUN --data-root ... --hf-root ... --ckpt-root ...
# back on the Mac, after copying results/rescoring/RUN/ back
~/envs/bit-mac/bin/python scripts/rescore_nbest.py opt --run RUN      # BIT's versions, MPS, fp16
.venv/bin/python scripts/rescore_nbest.py report --run RUN            # tuning, metrics, BERTScore, tables
```

## Reproduction check (B_ssl_s42, RTX 4070)

| | exact-string match with the GPU host's greedy BIT-test decodes | BIT-test WER | Δ vs the host's 34.23% |
|---|---|---|---|
| bf16 autocast (as the host), 713 trials | 95.65% | 34.34% | +0.10 pp |
| fp32, first 150 trials | 96.0% (bf16: 95.3% on the same trials) | — | — |

The decodes that differ diverge at near-tied tokens (both precisions differ from the host about equally; fp32 vs bf16
on the PC: 94%), which points to the two GPUs' kernels (Ada vs Blackwell), not to the pipeline: the candidate path's
greedy decode equals `stage2.generate`'s, and the WER moves by 0.1 pp. Rescoring compares rules on the same PC
candidates, so its greedy baseline is the PC's own greedy decode.

## Runtime per cell

| step | where | B_ssl_s42 (AR, Aero) |
|---|---|---|
| reproduction check, 713 greedy BIT-test decodes | RTX 4070 | 3 min (0.25 s/trial) |
| candidates, 2 × 713 trials × 12 | RTX 4070 | 18.3 min (0.064 s/candidate, 0.77 s/trial) |
| OPT-6.7B, unique texts (4,803 for B_ssl_s42) | Mac, MPS | pending (the Mac GPU was busy) |
| report (tuning, bootstrap, BERTScore on CPU) | Mac | ~1 min per rule file |

A block-diffusion run's candidates are slower: its sampler runs once per sample, and a trial's 12 samples are not
batched. The structural check (Aero block diffusion before its BEST existed, untrained heads, 5 trials) ran at
1.0 s/candidate, nearly all of them at the 25-token cap, which bounds a cell at ~4.8 h. A trained model ends at EOS
and commits several tokens per pass, so it should take a fraction of that (to be measured on the Aero diffusion
BEST).

## B_ssl_s42 so far (before OPT)

12 candidates per trial, but only 3.42 unique on BIT-test (3.33 on BIT-val): at T 0.7 most samples repeat the greedy
decode. Greedy (the PC's) BIT-test WER is 34.34%, and the oracle over the 12 candidates is 26.29% (seen 17.07%,
unseen 27.46%; BIT-val 23.70%). That 8 pp is the most any rule could recover. BERTScore F1 of the greedy decodes is
0.635. `results/rescoring/B_ssl_s42/summary.md` fills in the OPT rules once `rescore_nbest.py opt` has run.

## Caveats

- One seed per run, and a side test: a gain here is not a decoder result.
- Every rule picks among the decoder's own candidates, so none can beat the oracle. A rule that changes few trials
  cannot move WER much, whatever the LM.
- OPT sees lowercase text without punctuation, as BIT's references are, and sums log-probs without an end token. That
  favours short sentences, which the tuned per-word β offsets.
- AR decoder scores come from the sampled path's tokens. A sentence reached through two tokenizations keeps its best
  path's score.

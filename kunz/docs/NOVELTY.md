# Novelty assessment (2026-10-18)

A literature check by a research subagent (web search plus reading the sources), for Phases 2–3.
I spot-checked the key citation myself: de Borman et al. 2026, *J. Neural Eng.* 23:016016, which
does report that ECoG "patterns learned from performed and perceived speech could generalize to
imagined speech". The other citations come from the subagent's reading and are listed so you can
check them.

**Bottom line.** We found no paper or preprint that trains a word decoder on **listening or silent
reading** and tests it on **inner speech** in **human precentral intracortical** recordings, on this
dataset or any other. But:
- close analogues exist in ECoG and in other regions;
- Kunz et al. already report the shared representation (by correlation) that Phase 2 tests with
  decoders;
- attempted→inner transfer is already in the paper.

Phase 2 is **partially novel**; Phase 3 is a **standard benchmark**.

| Analysis | Novelty | Confidence | Closest prior work |
|---|---|---|---|
| Listening → inner speech transfer | Partially novel | Medium | Kunz 2025 Fig 2A/B (shared code by correlation); de Borman 2026 (ECoG); Wandelt 2024 (SMG); Xu 2024 (precentral Neuropixels, listening↔speaking sentences) |
| Reading → inner speech transfer | Partially novel | Medium | Wandelt 2024 (written cue ↔ internal speech, same trial); Kunz 2025 Fig 2 |
| Reverse directions / full 7×7 matrix | Partially novel. **Attempted → inner is not novel** | Medium | Kunz 2025 Fig 3E (sentence RNN, attempted → inner), Fig S2; Wandelt 2024; Xu 2024 |
| Strict normalization + interleaved drift controls | The techniques are standard; applying them to cross-behavior transfer is new | High | Kunz 2025 STAR Methods, Fig 6 (14-class GNB on interleaved attempted + inner; listening trials unused) |
| Phase 3 decoder comparison | Not novel (incremental benchmark on new data) | Medium-high | Wilson 2020; Livezey 2019; Glaser 2020; Kunz 2025 ("more advanced methods could likely improve classification accuracy") |
| Project overall | Partially novel | Medium | All of the above |

## Evidence

**What Kunz et al. 2025 already did** (PMC12360486; github.com/nptl-stanford/inner_speech)
- **Shared code across behaviors:** Fig 2A says "words are encoded similarly across behaviors—including
  listening and silent reading". Fig 2B gives cross-validated correlations between behaviors.
- **Decoder transfer they did run:** attempted → inner with a sentence RNN (Fig 3E, above chance for
  all participants); attempted ↔ verbal memory (Fig S2/S3F); inner → counting and prompted thought
  (Fig 5/S5).
- **What they did not run:** a decoder trained on listening, reading or mimed speech, or any
  cross-behavior decoding on the 7-word isolated set.
- **Motor-intent dimension:** the normalized attempted − inner centroid difference.
- **Their own caveat:** isolated blocks "did not allow us to assess differences… between behaviors"
  because of "spurious neural nonstationarity".

**Closest prior cross-condition work**
- **de Borman et al. 2026,** *J. Neural Eng.*, doi:10.1088/1741-2552/ae3a1b. ECoG, LDA, 5 words. Models
  transferred from performed or perceived speech to imagined speech; reading was not analyzed. Limited
  sensorimotor coverage.
- **Wandelt et al. 2024,** *Nat. Hum. Behav.*, PMC11199147. Intracortical recordings in supramarginal
  gyrus. An internal-speech model generalized to vocalized speech and written cue words, but less to the
  auditory cue phase. Cue and speech fall in the same trial, so this is a weaker test than separate
  blocks.
- **Xu et al. 2024,** bioRxiv doi:10.1101/2024.12.27.630563. Precentral gyrus, Neuropixels. "Decoders
  fitted during listening can both decode sentence identity… during speaking, and vice versa". It also
  uses train-time × test-time maps, so that diagnostic is not new.
- **Martin et al. 2014,** doi:10.3389/fneng.2014.00014: decoding covert speech from models trained on
  overt speech.
- **Cheung et al. 2016,** eLife doi:10.7554/eLife.12577, a counterpoint: during listening, motor cortex
  is organized along acoustic rather than articulatory features.
- **Schippers et al. 2025,** doi:10.1186/s12984-025-01689-7: production-trained ECoG decoders give false
  positives during perception.

**Reuse of this Dryad dataset.** None of these does listening/reading → inner transfer.
- **Papers and preprints:**
  - Zhang et al. 2025, arXiv:2511.21740: attempted → imagined, sentences.
  - Boccato et al. 2026, doi:10.1088/1741-2552/ae8576: cross-subject models, excluding the perceptual
    conditions.
- **GitHub, not papers:**
  - ronnuriel/NeuralSpeechGeometry
  - Dililianxice/jev-inner-speech-bci: shrinkage and diagonal LDA within each behavior.

**Model comparisons are standard practice.** Wilson et al. 2020 (intracortical, linear vs RNN);
Livezey et al. 2019 (ECoG, deep vs linear); Glaser et al. 2020 (decoder benchmark).

## How to frame the work
- **Phase 2:** a single-trial *decoder* test of Kunz's Fig 2 shared-code result, adding the perception
  directions and the drift controls. Say "to our knowledge, the first in human precentral
  intracortical recordings". Avoid "first perception → imagery transfer" (de Borman 2026, Wandelt 2024,
  Xu 2024) and "first attempted → inner transfer" (Kunz Fig 3E).
- **Drop the label "perception → intent".** Silent reading may itself involve inner speech (Wandelt
  2024), so reading → inner transfer may be partly within-behavior.
- **Link the T12 strict-normalization collapse to Kunz's motor-intent offset.** The paper's per-block
  centering removes exactly that offset, using the test block's own mean.
- **Phase 3:** present it as a benchmark. A more novel follow-up is whether the choice of decoder
  changes the *transfer* conclusions, because GNB is sensitive to the across-behavior scaling
  differences in Kunz Fig 2D.
- The subagent flagged checking how T12's interleaved listening trials are aligned (the official
  notebook uses `delayTrialEpochs`). **Already handled:** Phase 2 treats T12 interleaved listening as
  delay-epoch (stimulus) aligned, and Phase 0's Figure B confirms its information is in the delay.

## Not verified
- **Kunz bioRxiv preprint:** bioRxiv rate-limited the subagent, so only the abstract was read.
- **Supplementary material:** Kunz's supplementary figure images and Table S6 (legends only); de Borman
  2026's supplementary figures, including whether imagined → perceived was tested.
- **Xu 2024:** numeric accuracies and peer-review status.
- **Martin 2016 and Card 2024:** their supplementary cross-condition analyses.
- **Search coverage:** Google Scholar was not queried directly, and GitHub search covers only indexed
  default branches.

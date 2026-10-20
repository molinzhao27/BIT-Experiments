# BIT-Experiments

Experiments built on **BIT** (BraIn-to-Text; Zhang et al., [arXiv 2511.21740](https://arxiv.org/abs/2511.21740),
code [yzhang511/bit](https://github.com/yzhang511/bit)). BIT decodes attempted-speech sentences from intracortical
activity with a transformer neural encoder (NDT) connected to an audio LLM (Aero-1-Audio, Qwen2.5-1.5B). This repo
reproduces BIT, then swaps, ablates and adds modules to it.

**Main question:** can the phoneme stage be removed? BIT trains the encoder on phonemes (CTC) before training it with
the LLM, and reports that skipping that stage fails. Here the phoneme stage is replaced by self-supervised
pretraining and contrastive alignment to the LLM's own sentence representations.

## Experiments

Every arm starts from the same SSL-pretrained encoder and ends with the same sentence-level LLM training. They differ in
the stage in between:

| Arm | Stage in between | Status |
|---|---|---|
| **A** phoneme-first (BIT's recipe) | CTC phoneme fine-tuning | running |
| **B** direct (BIT App. C) | none | queued |
| **C** contrastive-first (new) | contrastive alignment to fixed LLM sentence targets, no LLM, no phonemes | Stage 1 done |

Modular changes tested on arm B:

| Change | What is swapped | Code |
|---|---|---|
| LLM: Qwen2.5-1.5B-Instruct | Aero's audio post-training removed | `text_llm.py`, `configs/llms.yaml` |
| LLM: Qwen2.5-1.5B base, no prompt | instruction tuning and the prompt removed | `text_llm.py` |
| LLM: Fast-dLLM v2 1.5B | autoregressive decoder → block-diffusion decoder | `diffusion_llm.py`, `fast_dllm_v2/` |

Planned: a non-contrastive alignment arm (E), a hybrid CTC + contrastive arm (D), random-init controls, batch-size
and label-fraction ablations. Full design: [`docs/DESIGN.md`](docs/DESIGN.md).

## Results so far

Details and caveats: [`results/SUMMARY.md`](results/SUMMARY.md). All metrics: `results/metrics.csv`.
Data: Brain-to-Text '25 (participant T15), with BIT's split of the official val set into BIT-val (713 trials) and
BIT-test (713 trials).

**1. BIT reproduction.** Phoneme error rate 8.91% on BIT-val (paper: 8.87%). Two properties of BIT's released code
change what its sentence-level numbers mean:
- the sentence stage never loads the phoneme encoder (the `--ft_ckpt` key is not read), so the released end-to-end
  model is a randomly initialized encoder trained directly on sentences;
- the default evaluation is teacher-forced (each word is predicted from the true preceding words), giving WER
  19.1% / 20.2% (val / test). This is not free decoding.

Every arm here loads and verifies its starting checkpoint explicitly and is scored by free generation.

**2. SSL pretraining on human data.** BIT's masked modeling on 60 h of speech recordings (T15, T12, and Kunz et al.
sentence sessions). Masked-bin val R² 0.042 (T15) / 0.033 (T12) after removing a smoothing leak in BIT's input path
(0.052 / 0.050 with it); no-learning baselines score ≤ 0.

![SSL training curve](figures/ssl_human_curve.png)

**3. Contrastive alignment (arm C, Stage 1).** With no LLM and no phonemes, the encoder retrieves the right sentence
among 704 candidates 35.8% of the time on BIT-test (top-5: 65.2%; chance 0.14%), equally on sentences never seen in
training (36.1%).

## Layout

```
src/bit_experiments/
  bit_ssl.py, ssl_data.py   SSL pretraining of BIT's NDT encoder, and its corpus
  stage1.py, text_targets.py  arm C: contrastive alignment to cached LLM sentence targets (GradCache)
  stage2.py                 sentence-level LLM training for every arm (BIT's model, padding fix)
  text_llm.py               BIT with a text-only LLM decoder
  diffusion_llm.py          BIT with a block-diffusion LLM decoder (vendored fast_dllm_v2/)
  evaluation.py, textmetrics.py  shared evaluator: WER/CER, BERTScore, seen/unseen split, null controls
scripts/                    one entry point per step (below)
configs/                    every setting; BIT's own configs are patched only by scripts/patch_bit_configs.py
docs/                       design, SSL plan, pretraining-data inventory
results/, figures/          outputs
external/bit                BIT, pinned submodule (f107ae9)
kunz/                       side project: Kunz et al. 2025 isolated-word reproduction (see kunz/README.md)
```

## Setup

```bash
git clone --recurse-submodules git@github.com:molinzhao27/BIT-Experiments.git
pip install -e ".[dev,eval]"                     # this package: evaluator, tests
python -m pytest                                 # CPU, seconds
```

Training runs in BIT's environment: Python 3.10, `configs/bit_requirements.txt`, and the torch build in
`configs/bit_baseline.yaml` (2.7.1 + CUDA 12.8). Then:

```bash
python scripts/patch_bit_configs.py              # apply configs/bit_baseline.yaml to BIT's trainer YAMLs
python scripts/fetch_llms.py                     # pinned LLM snapshots -> checkpoints/hf/
```

Data goes in `data/raw/` (gitignored): Brain-to-Text '25 (`brandman_2024_text`), Brain-to-Text '24
(`willett_2023_text`), and the Kunz et al. sentence sessions (`kunz/scripts/fetch_data.py`). Sources and checksums:
[`docs/PRETRAIN_DATA.md`](docs/PRETRAIN_DATA.md), `scripts/fetch_pretrain_data.py`.

## Pipeline

```bash
# 1. SSL pretraining
python scripts/build_ssl_corpus.py                       # -> data/processed/ssl_human
python scripts/pretrain_ssl.py                           # -> checkpoints/ssl_human/BEST, BEST_core
python scripts/export_finetune_core.py                   # -> BEST_core_ft (masker off for fine-tuning)

# 2. BIT reproduction; arm A's phoneme stage
python scripts/run_bit_baseline.py                       # BIT as released
python scripts/run_bit_baseline.py --stages phoneme eval_phoneme \
    --phoneme-init checkpoints/ssl_human/BEST_core_ft --tag armA_ssl --out results/armA_ssl_phoneme

# 3. Arm C, Stage 1: contrastive alignment
python scripts/cache_text_targets.py                     # fixed LLM sentence targets, cached once
python scripts/train_stage1.py --out checkpoints/stage1/C_l14_b1024_s42

# 4. Stage 2 (every arm), with free-generation evaluation
python scripts/train_stage2.py --name A_ssl_s42 --arm A --init <phoneme BEST>
python scripts/train_stage2.py --name B_ssl_s42 --arm B --init checkpoints/ssl_human/BEST_core
python scripts/train_stage2.py --name C_ssl_s42 --arm C --init checkpoints/stage1/C_l14_b1024_s42
python scripts/train_stage2.py --name B_ssl_qwen25-1.5b-instruct_s42 --arm B \
    --init checkpoints/ssl_human/BEST_core --llm qwen2.5-1.5b-instruct    # LLM swap

# 5. Score and compare
python scripts/evaluate_decodes.py score <decodes.csv>
python scripts/evaluate_decodes.py summarize
```

Checks on the SSL encoder: `scripts/ssl_baselines.py`, `scripts/ssl_leakfree_eval.py`, `scripts/plot_ssl.py`.
Every run is seeded (default 42); configurations are compared over 3 seeds.

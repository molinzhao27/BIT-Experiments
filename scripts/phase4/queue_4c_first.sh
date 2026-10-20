#!/usr/bin/env bash
# Host side: first 4c GPU jobs (approved after the 4b review), each under the shared GPU lock, one after another:
#   1. arm C Stage 1 (contrastive alignment to the fixed LLM targets) from the SSL encoder;
#   2. arm A phoneme stage (CTC) from the SSL encoder, then its PER on BIT-val / BIT-test (reported, never used to
#      choose anything).
#   setsid nohup scripts/phase4/queue_4c_first.sh > ~/.codex-gpu-jobs/kunzdecoder-4c-queue.log 2>&1 < /dev/null &
set -euo pipefail
J=~/.codex-gpu-jobs/kunzdecoder-gpu-job.sh
export KUNZ_PY=/workspace/envs/bit/bin/python
echo "$(date -u +%FT%TZ) stage1 C"
$J stage1-C-l14-b1024-s42 scripts/phase4/train_stage1.py --out checkpoints/stage1/C_l14_b1024_s42 --seed 42
echo "$(date -u +%FT%TZ) armA phoneme from SSL"
docker exec kunzdecoder bash -c "cd /workspace/KunzDecoder && $KUNZ_PY scripts/phase4/export_finetune_core.py" \
  > /dev/null   # SSL core with the task masker (inactive), as the from-scratch run
$J armA-ssl-phoneme scripts/phase4/run_bit_baseline.py --stages phoneme eval_phoneme \
  --phoneme-init checkpoints/ssl_human/BEST_core_ft --tag armA_ssl --out results/phase4/armA_ssl_phoneme
echo "$(date -u +%FT%TZ) done"

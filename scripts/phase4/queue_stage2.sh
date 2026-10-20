#!/usr/bin/env bash
# Host side: Phase 4c Stage 2 runs (scripts/phase4/train_stage2.py), each under the shared GPU lock, after the
# first 4c queue (Stage 1 C, arm A phoneme stage) has finished. Order:
#   1. B_rand_bit4a: free-generation evaluation of the 4a sentence checkpoint (BIT's released pipeline; its encoder
#      was never initialized from the phoneme model, so it is arm B from random init, selected by teacher-forced WER);
#   2. B_ssl_s42: arm B (direct) from the SSL encoder core.
#   (then the LLM-swap runs of the other session, then C and A; appended once their inputs exist)
#   setsid nohup scripts/phase4/queue_stage2.sh > ~/.codex-gpu-jobs/kunzdecoder-stage2-queue.log 2>&1 < /dev/null &
set -euo pipefail
while pgrep -f "kunzdecoder-4c-queu[e].sh" >/dev/null; do sleep 300; done   # [e]: never matches itself
J=~/.codex-gpu-jobs/kunzdecoder-gpu-job.sh
export KUNZ_PY=/workspace/envs/bit/bin/python
S2=scripts/phase4/train_stage2.py
echo "$(date -u +%FT%TZ) B_rand_bit4a (eval only)"
$J stage2-B_rand_bit4a $S2 --name B_rand_bit4a --arm B \
  --eval-only /workspace/KunzDecoder/checkpoints/bit/finetune-ndt-sentence-brandman_2024_text-feat_all/BEST
echo "$(date -u +%FT%TZ) B_ssl_s42"
$J stage2-B_ssl_s42 $S2 --name B_ssl_s42 --arm B --init checkpoints/ssl_human/BEST_core --seed 42
echo "$(date -u +%FT%TZ) done"

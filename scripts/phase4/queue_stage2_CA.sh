#!/usr/bin/env bash
# Host side: Tier-1 Stage 2 runs for arms C and A (SSL-initialized), each under the shared GPU lock. They wait for
# the Stage 2 queue (B_rand_bit4a, B_ssl_s42) to exit and for the other session's two LLM-swap runs to finish
# (RUN_END in their logs; user's decision: swap runs before C and A, Fast-dLLM run 3 after them).
#   setsid nohup scripts/phase4/queue_stage2_CA.sh > ~/.codex-gpu-jobs/kunzdecoder-stage2-CA-queue.log 2>&1 < /dev/null &
set -euo pipefail
L=~/.codex-gpu-jobs
while pgrep -f "kunzdecoder-stage2-queu[e].sh" >/dev/null; do sleep 300; done   # [e]: never matches itself
for run in B_ssl_qwen25-1.5b-instruct_s42 B_ssl_qwen25-1.5b-noprompt_s42; do
  until grep -q "^RUN_END" "$L/kunzdecoder-stage2-$run.log" 2>/dev/null; do sleep 300; done
done
J=$L/kunzdecoder-gpu-job.sh
export KUNZ_PY=/workspace/envs/bit/bin/python
S2=scripts/phase4/train_stage2.py
echo "$(date -u +%FT%TZ) C_ssl_l14_b1024_s42"
$J stage2-C_ssl_l14_b1024_s42 $S2 --name C_ssl_l14_b1024_s42 --arm C \
  --init checkpoints/stage1/C_l14_b1024_s42/best --seed 42
echo "$(date -u +%FT%TZ) A_ssl_s42"
$J stage2-A_ssl_s42 $S2 --name A_ssl_s42 --arm A \
  --init checkpoints/bit/finetune-ndt-phoneme-brandman_2024_text-feat_all/BEST --seed 42
echo "$(date -u +%FT%TZ) done"

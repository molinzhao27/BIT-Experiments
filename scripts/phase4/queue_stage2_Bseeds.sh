#!/usr/bin/env bash
# SUPERSEDED 2026-10-20 by scripts/phase4/queue_phase4_main.sh (one queue, same jobs); kept as a record, do not launch.
# Host side: arm B (Aero, SSL encoder) seeds 43 and 44, the seed spread that the LLM-swap runs are judged against
# (docs/phase4/DESIGN.md §8) and arm B's 2nd / 3rd seeds (§2). Same flags as B_ssl_s42. Per the user (2026-10-20), last
# in the GPU queue: waits until every earlier kunzdecoder queue (4c, Stage 2, LLM swap, C/A, run 3) has exited.
#   cp scripts/phase4/queue_stage2_Bseeds.sh ~/.codex-gpu-jobs/kunzdecoder-stage2-Bseeds-queue.sh   (on the host)
#   setsid nohup ~/.codex-gpu-jobs/kunzdecoder-stage2-Bseeds-queue.sh > ~/.codex-gpu-jobs/kunzdecoder-stage2-Bseeds-queue.log 2>&1 < /dev/null &
set -uo pipefail
J=~/.codex-gpu-jobs/kunzdecoder-gpu-job.sh
export KUNZ_PY=/workspace/envs/bit/bin/python
QUEUES="kunzdecoder-4c-queu[e].sh|kunzdecoder-stage2-queu[e].sh|kunzdecoder-llmswap-queu[e].sh|kunzdecoder-stage2-CA-queu[e].sh|kunzdecoder-llmswap-run3-queu[e].sh"
echo "$(date -u +%FT%TZ) waiting for: $QUEUES"
while pgrep -f "$QUEUES" >/dev/null; do sleep 300; done   # [e]: never matches itself
for seed in 43 44; do
  echo "$(date -u +%FT%TZ) B_ssl_s$seed"
  $J stage2-B_ssl_s$seed scripts/phase4/train_stage2.py --name B_ssl_s$seed --arm B \
    --init checkpoints/ssl_human/BEST_core --seed $seed || echo "B_ssl_s$seed failed (rc=$?)"
done
echo "$(date -u +%FT%TZ) done"

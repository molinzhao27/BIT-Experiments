#!/usr/bin/env bash
# SUPERSEDED 2026-10-20 by scripts/phase4/queue_phase4_main.sh (one queue, same jobs); kept as a record, do not launch.
# Host side: LLM-swap Stage 2 runs (docs/phase4/DESIGN.md §8), arm B from the SSL encoder, seed 42. Starts after the
# Stage 2 queue (4a eval-only, then the arm B Aero reference B_ssl_s42) has exited. Both runs always start: the C/A
# queue (kunzdecoder-stage2-CA-queue.sh) waits for a RUN_END line in each of their logs, so keep these job names.
# One job after another, each under the shared GPU lock:
#   1. B_ssl_qwen25-1.5b-instruct_s42: Qwen2.5-1.5B-Instruct, BIT's prompt (Aero-1-Audio minus its audio training);
#   2. B_ssl_qwen25-1.5b-noprompt_s42: Qwen2.5-1.5B (base), no prompt.
#   cp scripts/phase4/queue_llmswap.sh ~/.codex-gpu-jobs/kunzdecoder-llmswap-queue.sh   (on the host)
#   setsid nohup ~/.codex-gpu-jobs/kunzdecoder-llmswap-queue.sh > ~/.codex-gpu-jobs/kunzdecoder-llmswap-queue.log 2>&1 < /dev/null &
set -uo pipefail
J=~/.codex-gpu-jobs/kunzdecoder-gpu-job.sh
export KUNZ_PY=/workspace/envs/bit/bin/python
INIT=checkpoints/ssl_human/BEST_core
echo "$(date -u +%FT%TZ) waiting for kunzdecoder-stage2-queue.sh"
while pgrep -f "kunzdecoder-stage2-queu[e].sh" >/dev/null; do sleep 300; done
echo "$(date -u +%FT%TZ) Aero reference: $(grep RUN_END ~/.codex-gpu-jobs/kunzdecoder-stage2-B_ssl_s42.log 2>/dev/null | tail -1)"
echo "$(date -u +%FT%TZ) run 1: qwen2.5-1.5b-instruct"
$J stage2-B_ssl_qwen25-1.5b-instruct_s42 scripts/phase4/train_stage2.py --name B_ssl_qwen25-1.5b-instruct_s42 \
  --arm B --init "$INIT" --seed 42 --llm qwen2.5-1.5b-instruct || echo "run 1 failed (rc=$?)"
echo "$(date -u +%FT%TZ) run 2: qwen2.5-1.5b, no prompt"
$J stage2-B_ssl_qwen25-1.5b-noprompt_s42 scripts/phase4/train_stage2.py --name B_ssl_qwen25-1.5b-noprompt_s42 \
  --arm B --init "$INIT" --seed 42 --llm qwen2.5-1.5b --prompt '#' || echo "run 2 failed (rc=$?)"
echo "$(date -u +%FT%TZ) done"

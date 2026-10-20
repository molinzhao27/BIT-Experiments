#!/usr/bin/env bash
# SUPERSEDED 2026-10-20 by scripts/phase4/queue_phase4_main.sh (one queue, same jobs); kept as a record, do not launch.
# Host side: LLM-swap run 3 (docs/phase4/DESIGN.md §8): arm B from the SSL encoder, seed 42, Fast-dLLM v2 1.5B (block
# diffusion), under the shared GPU lock. Per the user, after the C and A Stage 2 runs: waits for the LLM-swap queue
# (runs 1-2) and the C/A queue to exit. Needs the GPU timing smoke (kunzdecoder-llmswap-smoke-gpu) to have passed.
#   cp scripts/phase4/queue_llmswap_run3.sh ~/.codex-gpu-jobs/kunzdecoder-llmswap-run3-queue.sh   (on the host)
#   setsid nohup ~/.codex-gpu-jobs/kunzdecoder-llmswap-run3-queue.sh > ~/.codex-gpu-jobs/kunzdecoder-llmswap-run3-queue.log 2>&1 < /dev/null &
set -uo pipefail
J=~/.codex-gpu-jobs/kunzdecoder-gpu-job.sh
export KUNZ_PY=/workspace/envs/bit/bin/python
echo "$(date -u +%FT%TZ) waiting for kunzdecoder-llmswap-queue.sh and kunzdecoder-stage2-CA-queue.sh"
while pgrep -f "kunzdecoder-llmswap-queu[e].sh" >/dev/null || pgrep -f "kunzdecoder-stage2-CA-queu[e].sh" >/dev/null; do
  sleep 300
done
echo "$(date -u +%FT%TZ) A_ssl_s42: $(grep RUN_END ~/.codex-gpu-jobs/kunzdecoder-stage2-A_ssl_s42.log 2>/dev/null | tail -1)"
if ! grep -q GPU_SMOKE_OK ~/.codex-gpu-jobs/kunzdecoder-llmswap-smoke-gpu.log 2>/dev/null; then
  echo "$(date -u +%FT%TZ) the GPU timing smoke has not passed; not starting run 3"; exit 1
fi
echo "$(date -u +%FT%TZ) run 3: fast-dllm-v2-1.5b"
$J stage2-B_ssl_fast-dllm-v2-1.5b_s42 scripts/phase4/train_stage2.py --name B_ssl_fast-dllm-v2-1.5b_s42 \
  --arm B --init checkpoints/ssl_human/BEST_core --seed 42 --llm fast-dllm-v2-1.5b || echo "run 3 failed (rc=$?)"
echo "$(date -u +%FT%TZ) done"

#!/usr/bin/env bash
# Host side: queue the 4a BIT reproduction as three GPU jobs, each taking the shared lock separately
# (other projects' queued jobs can run between our long stages). Logs: ~/.codex-gpu-jobs/kunzdecoder-bit-*.log
#   nohup scripts/phase4/queue_bit_baseline.sh > ~/.codex-gpu-jobs/kunzdecoder-bit-queue.log 2>&1 &
set -euo pipefail
J=~/.codex-gpu-jobs/kunzdecoder-gpu-job.sh
for stage in phoneme sentence eval; do
  echo "$(date -u +%FT%TZ) queue $stage"
  $J "bit-$stage" scripts/phase4/run_bit_baseline.py --stages "$stage"
  echo "$(date -u +%FT%TZ) done $stage"
done

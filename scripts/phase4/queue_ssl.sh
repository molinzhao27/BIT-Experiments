#!/usr/bin/env bash
# Host side: queue Phase 4b SSL pretraining behind the 4a BIT queue. Each run takes the shared GPU lock itself.
#   setsid nohup scripts/phase4/queue_ssl.sh > ~/.codex-gpu-jobs/kunzdecoder-ssl-queue.log 2>&1 < /dev/null &
set -euo pipefail
while pgrep -f "kunzdecoder-bit-queu[e].sh" >/dev/null; do sleep 300; done   # [e]: never matches itself
echo "$(date -u +%FT%TZ) 4a queue finished; starting SSL"
J=~/.codex-gpu-jobs/kunzdecoder-gpu-job.sh
export KUNZ_PY=/workspace/envs/bit/bin/python
$J ssl-throughput scripts/phase4/pretrain_ssl.py --out checkpoints/ssl_human_tp --max-epochs 1 --seed 42 --device cuda
echo "$(date -u +%FT%TZ) throughput run done"
$J ssl-pretrain scripts/phase4/pretrain_ssl.py --out checkpoints/ssl_human --seed 42 --device cuda
echo "$(date -u +%FT%TZ) SSL pretraining done"

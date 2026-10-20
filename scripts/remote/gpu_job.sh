#!/usr/bin/env bash
# Host side: run one command in the kunzdecoder container under the shared GPU lock,
# following the host convention (~/.codex-gpu-jobs/*-queue.sh):
#   flock ~/.codex-gpu-0.lock -> refuse if any compute process is on the GPU outside
#   the lock (exit 75, retry every 60 s) -> docker exec. Log: ~/.codex-gpu-jobs/kunzdecoder-<name>.log
#
#   nohup scripts/remote/gpu_job.sh <name> <command...> &      (run on the host)
# The command runs with cwd /workspace/KunzDecoder and the project venv, or the interpreter in $KUNZ_PY
# (e.g. KUNZ_PY=/workspace/envs/bit/bin/python for BIT-based training; BIT's env vars are then exported too).
set -uo pipefail
name=$1; shift
log=$HOME/.codex-gpu-jobs/kunzdecoder-$name.log
py=${KUNZ_PY:-.venv/bin/python}
envs=""
[[ $py == /workspace/envs/bit/* ]] && envs="export CC=/opt/worm-build/bin/gcc CUDA_HOME=/opt/worm-build DS_SKIP_CUDA_CHECK=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled; "
cmd="cd /workspace/KunzDecoder && ${envs}$py -u $(printf '%q ' "$@")"
echo "QUEUED=$(date -u +%FT%TZ) cmd=$*" >>"$log"
while true; do
  rc=0
  flock --exclusive "$HOME/.codex-gpu-0.lock" bash -c '
    active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || exit 1
    if [[ -n "$active" ]]; then echo "GPU occupied outside the shared lock: $active" >&2; exit 75; fi
    echo "RUN_START=$(date -u +%FT%TZ)"
    docker exec -i -e KUNZ_GPU_LOCK_HELD=1 kunzdecoder bash -c "$1"
    rc=$?
    echo "RUN_END=$(date -u +%FT%TZ) rc=$rc"
    exit $rc' _ "$cmd" >>"$log" 2>&1 || rc=$?
  [[ $rc == 75 ]] || exit $rc
  sleep 60
done

#!/usr/bin/env bash
# Host side: the Phase 4 Stage 2 queue in one script (user, 2026-10-20). It replaces kunzdecoder-stage2-queue.sh,
# -stage2-CA-queue.sh, -llmswap-queue.sh, -llmswap-run3-queue.sh and -stage2-Bseeds-queue.sh (scripts/phase4/
# queue_stage2.sh, queue_stage2_CA.sh, queue_llmswap.sh, queue_llmswap_run3.sh, queue_stage2_Bseeds.sh): same jobs,
# same flags and job names, new order. Starts after the 4c queue (arm A phoneme stage) and the GPU timing smoke.
#   1. LLM-swap runs (DESIGN.md §8), arm B from the SSL encoder, seed 42:
#      Qwen2.5-1.5B-Instruct; Qwen2.5-1.5B base without prompt; Fast-dLLM v2 (only if the GPU smoke passed);
#   2. the arms: B_rand_bit4a (eval-only, 4a's checkpoint), B_ssl_s42 (Aero reference), C, A;
#   3. arm B seeds 43 and 44.
# One job after another under the shared GPU lock; a failed job is logged and the queue goes on.
#   cp scripts/phase4/queue_phase4_main.sh ~/.codex-gpu-jobs/kunzdecoder-main-queue.sh   (on the host)
#   setsid nohup ~/.codex-gpu-jobs/kunzdecoder-main-queue.sh > ~/.codex-gpu-jobs/kunzdecoder-main-queue.log 2>&1 < /dev/null &
set -uo pipefail
L=~/.codex-gpu-jobs
J=$L/kunzdecoder-gpu-job.sh
export KUNZ_PY=/workspace/envs/bit/bin/python
S2=scripts/phase4/train_stage2.py
INIT=checkpoints/ssl_human/BEST_core
run() {
  local name=$1; shift
  echo "$(date -u +%FT%TZ) start $name"
  $J stage2-$name $S2 --name $name "$@" || echo "$(date -u +%FT%TZ) $name failed (rc=$?)"
}
echo "$(date -u +%FT%TZ) waiting for the 4c queue and the GPU timing smoke"
while pgrep -f "kunzdecoder-4c-queu[e].sh" >/dev/null || pgrep -f "kunzdecoder-gpu-job.sh llmswap-smoke-gp[u]" >/dev/null; do
  sleep 300   # [e], [u]: never match this loop's own pgrep
done

# 1. LLM swaps
run B_ssl_qwen25-1.5b-instruct_s42 --arm B --init $INIT --seed 42 --llm qwen2.5-1.5b-instruct
run B_ssl_qwen25-1.5b-noprompt_s42 --arm B --init $INIT --seed 42 --llm qwen2.5-1.5b --prompt '#'
if grep -q GPU_SMOKE_OK $L/kunzdecoder-llmswap-smoke-gpu.log 2>/dev/null; then
  run B_ssl_fast-dllm-v2-1.5b_s42 --arm B --init $INIT --seed 42 --llm fast-dllm-v2-1.5b
else
  echo "$(date -u +%FT%TZ) the GPU timing smoke has not passed; skipping run 3 (Fast-dLLM v2)"
fi

# 2. arms
run B_rand_bit4a --arm B --eval-only /workspace/KunzDecoder/checkpoints/bit/finetune-ndt-sentence-brandman_2024_text-feat_all/BEST
run B_ssl_s42 --arm B --init $INIT --seed 42
run C_ssl_l14_b1024_s42 --arm C --init checkpoints/stage1/C_l14_b1024_s42/best --seed 42
run A_ssl_s42 --arm A --init checkpoints/bit/finetune-ndt-phoneme-brandman_2024_text-feat_all/BEST --seed 42

# 3. arm B seeds
for seed in 43 44; do
  run B_ssl_s$seed --arm B --init $INIT --seed $seed
done
echo "$(date -u +%FT%TZ) done"

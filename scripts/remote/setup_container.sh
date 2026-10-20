#!/usr/bin/env bash
# Host side (matthewzhao@91.244.199.230): one-time creation of the KunzDecoder
# container. Mirrors the existing per-project containers (`worm-train`,
# `mechinterp`): same image, CDI GPU, user worm, its own named volume at /workspace
# (volumes live on the 1.7 TB /var/lib/docker disk; / has little free space).
# Other projects' containers and volumes are not touched.
set -euo pipefail
docker volume inspect kunzdecoder-data >/dev/null 2>&1 \
  || docker volume create --label project=KunzDecoder kunzdecoder-data
docker container inspect kunzdecoder >/dev/null 2>&1 \
  || docker run -d --name kunzdecoder --label project=KunzDecoder --restart unless-stopped \
       --device nvidia.com/gpu=all --shm-size 16g -v kunzdecoder-data:/workspace \
       worm-train:py312-cu132
docker exec kunzdecoder mkdir -p /workspace/KunzDecoder/data/raw /workspace/KunzDecoder/data/processed

#!/usr/bin/env bash
# Mac side: push code/configs to the container, or pull results/figures back.
#   scripts/remote/sync.sh push     # code, configs, docs -> container
#   scripts/remote/sync.sh data     # needed raw .mat files -> container (allBehaviors skipped: exact unions)
#   scripts/remote/sync.sh pull     # results/, figures/ <- container
# rsync runs through `docker exec`, so files land directly in the kunzdecoder-data volume.
set -euo pipefail
HOST=${KUNZ_HOST:-matthewzhao@91.244.199.230}
RSH="ssh -o BatchMode=yes $HOST docker exec -i"
DEST=kunzdecoder:/workspace/KunzDecoder
cd "$(dirname "$0")/../.."
case "${1:-}" in
  push) rsync -a -e "$RSH" --exclude .venv --exclude data --exclude external --exclude results \
          --exclude figures --exclude checkpoints --exclude logs --exclude __pycache__ --exclude '*.egg-info' --exclude .DS_Store ./ "$DEST/" ;;
  data) rsync -a --partial -e "$RSH" --exclude '*_allBehaviors_raw.mat' \
          data/raw/isolatedVerbalBehaviors data/raw/interleavedVerbalBehaviors "$DEST/data/raw/" ;;
  pull) rsync -a -e "$RSH" "$DEST/results/" results/ && rsync -a -e "$RSH" "$DEST/figures/" figures/ ;;
  *) echo "usage: $0 push|data|pull" >&2; exit 2 ;;
esac

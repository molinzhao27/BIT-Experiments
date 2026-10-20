#!/usr/bin/env bash
# Inside the kunzdecoder container: BIT checkout + its Python environment + the pinned LLM snapshot.
# Separate micromamba env under /workspace (the image's /opt environments are not touched).
set -euo pipefail
cd /workspace/KunzDecoder
CFG=configs/phase4/bit_baseline.yaml
y() { .venv/bin/python -c "import yaml,sys; c=yaml.safe_load(open('$CFG')); print(eval('c'+sys.argv[1]))" "$1"; }

BIT=$(y "['bit']['path']"); COMMIT=$(y "['bit']['commit']")
[[ -d $BIT/.git ]] || git clone -q "$(y "['bit']['repo']")" "$BIT"
git -C "$BIT" fetch -q origin && git -C "$BIT" checkout -q "$COMMIT"
echo "BIT @ $(git -C "$BIT" rev-parse --short HEAD)"

PREFIX=$(y "['env']['prefix']")
export MAMBA_ROOT_PREFIX=/workspace/.micromamba
# Triton (imported via deepspeed) compiles a small C helper on first import: use the image toolchain
export CC=/opt/worm-build/bin/gcc
# deepspeed reads the CUDA toolkit version on import too (no ops are compiled)
export CUDA_HOME=/opt/worm-build DS_SKIP_CUDA_CHECK=1 DS_BUILD_OPS=0
if [[ ! -x $PREFIX/bin/python ]]; then
  micromamba create -y -q -p "$PREFIX" -c conda-forge "python=$(y "['env']['python']")" pip
fi
PY=$PREFIX/bin/python
$PY -m pip install -q --upgrade pip
$PY -m pip install -q $(y "['env']['torch']")
$PY -m pip install -q -r "$(y "['env']['requirements']")"
# deepspeed is imported by BIT's trainer (only used with --ds_config, which we don't pass). BIT pins 0.12.6,
# which imports a torch.distributed symbol removed after torch 2.3 -> 0.17.1 (supports torch 2.7). Its
# setup.py reads the CUDA toolkit version: use the image's nvcc (13.2) and build no ops.
$PY -m pip install -q --no-build-isolation deepspeed==0.17.1
$PY -m pip freeze > configs/phase4/bit_env_freeze.txt
$PY - <<'EOF'
import nltk  # g2p_en's tagger + CMU dict (phoneme labels)
for r in ["averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "cmudict"]:
    nltk.download(r, quiet=True)
import torch, transformers, peft, accelerate, deepspeed, g2p_en
print("torch", torch.__version__, "cuda", torch.version.cuda, "| transformers", transformers.__version__,
      "| peft", peft.__version__, "| accelerate", accelerate.__version__, "| deepspeed", deepspeed.__version__)
EOF

$PY - <<EOF
from huggingface_hub import snapshot_download
p = snapshot_download("$(y "['llm']['repo_id']")", revision="$(y "['llm']['revision']")",
                      local_dir="$(y "['llm']['local_dir']")")
print("LLM snapshot:", p)
EOF
.venv/bin/python scripts/phase4/patch_bit_configs.py

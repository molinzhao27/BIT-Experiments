#!/usr/bin/env bash
# Inside the `kunzdecoder` container: project venv layered on the image's
# /opt/worm-train interpreter (python 3.12, torch 2.14+cu132, numpy/scipy/sklearn).
# --system-site-packages reuses the image's torch; the image itself is never modified.
set -euo pipefail
cd /workspace/KunzDecoder
[[ -x .venv/bin/python ]] || /opt/worm-train/bin/python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -e .
.venv/bin/python - <<'EOF'
import torch, sklearn, numpy, scipy, pandas, matplotlib
print(f"torch {torch.__version__} (cuda {torch.version.cuda}) | sklearn {sklearn.__version__} | "
      f"numpy {numpy.__version__} | scipy {scipy.__version__} | pandas {pandas.__version__}")
EOF
.venv/bin/python -m pip freeze --exclude-editable > .venv/pip-freeze.txt

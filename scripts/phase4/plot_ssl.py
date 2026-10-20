#!/usr/bin/env python
"""Phase 4b figure: SSL training loss and masked-bin val R^2 per epoch, with the no-learning reference predictors.

    python scripts/phase4/plot_ssl.py [--log results/phase4/ssl_human_log.csv]
                                      [--baselines results/phase4/ssl_baselines.json]
                                      [--out figures/phase4/ssl_human_curve.png]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
GROUPS = {"b2t25_T15": "T15 (B2T '25)", "b2t24_T12": "T12 (B2T '24)"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, default=ROOT / "results" / "phase4" / "ssl_human_log.csv")
    ap.add_argument("--baselines", type=Path, default=ROOT / "results" / "phase4" / "ssl_baselines.json")
    ap.add_argument("--out", type=Path, default=ROOT / "figures" / "phase4" / "ssl_human_curve.png")
    args = ap.parse_args()
    log = pd.read_csv(args.log)
    base = json.loads(args.baselines.read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    a.plot(log.epoch, log.train_loss, color="k", lw=1)
    a.set(xlabel="epoch", ylabel="train masked MSE (z-scored TX)", title="Training loss")
    for (g, label), c in zip(GROUPS.items(), ("C0", "C1")):
        b.plot(log.epoch, log[f"val_r2_masked_{g}"], color=c, lw=1.2, label=f"SSL encoder, {label}")
        b.axhline(base[f"r2_masked_{g}_trial_mean"], color=c, ls="--", lw=0.8)
    b.axhline(0, color="0.6", lw=0.5)
    best = log.loc[log.val_r2_masked.idxmax()]
    b.axvline(best.epoch, color="0.4", ls=":", lw=0.8)
    b.set(xlabel="epoch", ylabel="val R² on masked bins", ylim=(-0.05, 0.07),
          title=f"Validation R² (BEST: epoch {int(best.epoch)})")
    b.plot([], [], color="0.3", ls="--", lw=0.8, label="no learning: channel trial mean")
    b.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

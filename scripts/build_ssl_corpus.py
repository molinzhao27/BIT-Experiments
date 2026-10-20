#!/usr/bin/env python
"""Phase 4b: build the human SSL pretraining corpus (threshold crossings, 20 ms, z-scored) for BIT.

    python scripts/build_ssl_corpus.py [--config configs/ssl_human.yaml] [--only kunz b2t25 b2t24]
                                              [--out DIR] [--dry-run]

Writes <out>/<dataset>_<subject>_<split>.h5 (concatenated `spikes` + `offsets`, see ssl_data.write_split),
<out>/norm_<dataset>_<subject>.npz (normalization statistics) and <out>/manifest.json (counts, hours,
channels per stitch group, normalization summary, dedupe decisions, config snapshot). Datasets whose raw files
are absent are skipped and recorded as such. Plan: docs/SSL_PLAN.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bit_experiments import ssl_data  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "ssl_human.yaml")
    ap.add_argument("--only", nargs="+", choices=["kunz", "b2t25", "b2t24"], help="build only these datasets")
    ap.add_argument("--out", type=Path, help="output directory (default: corpus.out_dir)")
    ap.add_argument("--dry-run", action="store_true", help="list the raw sources that would be used, then exit")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())["corpus"]
    out = args.out or ROOT / cfg["out_dir"]
    ds = cfg["datasets"]

    if args.dry_run:
        for name in ("b2t25", "b2t24"):
            if ds[name]["enabled"] and (not args.only or name in args.only):
                print(f"{name}: {ssl_data.first_existing(ds[name]['path'], ROOT) or 'ABSENT -> skipped'}")
        if ds["kunz"]["enabled"] and (not args.only or "kunz" in args.only):
            kz = ds["kunz"]
            kf = ssl_data.kunz_files(ROOT, kz.get("isolated"), kz.get("interleaved"), kz.get("sentences"),
                                     kz.get("archives", ()), kz.get("skip_eval_files", False))
            for k in kf:
                print(f"kunz: {k.participant} {k.date} {k.collection:<11} {k.name}")
            print(f"kunz: {len(kf)} files")
        return 0

    t0 = time.time()
    m = ssl_data.build_corpus(cfg, ROOT, out, only=args.only,
                              log=lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True))
    print(json.dumps({"out": str(out), "total_hours": m["total_hours"], "totals": m["totals"],
                      "stitch_groups": m["stitch_groups"], "minutes": round((time.time() - t0) / 60, 1)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

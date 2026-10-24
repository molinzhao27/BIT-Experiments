#!/usr/bin/env python
"""Build the diagnostic foil set (DESIGN.md §5; docs/PROBES_FOILS.md) from BIT-val / BIT-test sentences. Text only,
CPU, deterministic; run once in the project venv (CMU dictionary: cmudict, frequencies: wordfreq), review, then
copy data/processed/foils/ to the host.

    .venv/bin/python scripts/build_foils.py [--refs results/stage2/B_ssl_s42/decodes] [--tables configs/foils.yaml]
        [--out data/processed/foils]

Sentences: the `reference` column of <refs>/<split>_greedy_real.csv (trial = index into BIT's split, as every
Stage 2 decode CSV). run_probes_foils.py checks each reference against BIT's loader on the host before scoring.
Writes <out>/<sha12>/foils.csv, manifest.json (tables hash, sources, counts per type / variant) and a review sheet
review.md (a seeded sample of each variant), and points <out>/LATEST at it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bit_experiments import foils  # noqa: E402


def _version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refs", type=Path, default=ROOT / "results" / "stage2" / "B_ssl_s42" / "decodes")
    ap.add_argument("--tables", type=Path, default=ROOT / "configs" / "foils.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "foils")
    ap.add_argument("--review-per-variant", type=int, default=12)
    args = ap.parse_args()
    t = foils.check_tables(yaml.safe_load(args.tables.read_text()))
    lex = foils.Lexicon(min_zipf=t["sound_alike_min_zipf"], blocklist=t["blocklist"], stopwords=t["stopwords"])
    rows, sources = [], {}
    for split in t["splits"]:
        csv = args.refs / f"{split}_greedy_real.csv"
        refs = pd.read_csv(csv, keep_default_na=False).sort_values("trial")
        if refs.trial.tolist() != list(range(len(refs))):
            raise ValueError(f"{csv}: trials are not 0..n-1")
        sentences = refs.reference.tolist()
        sources[split] = {"file": rel(csv), "n": len(sentences),
                          "sentences_sha256": hashlib.sha256("\n".join(sentences).encode()).hexdigest()}
        rows += foils.build(sentences, split, t, lex, seed=t["seed"])
    df = pd.DataFrame(rows, columns=foils.COLUMNS)
    body = df.to_csv(index=False).encode()
    sha = hashlib.sha256(body + foils.tables_digest(args.tables).encode()).hexdigest()
    out = args.out / sha[:12]
    out.mkdir(parents=True, exist_ok=True)
    (out / "foils.csv").write_bytes(body)
    counts = df.groupby(["split", "type", "variant"]).size()
    manifest = {"tables": rel(args.tables), "tables_sha256": foils.tables_digest(args.tables),
                "csv_sha256": hashlib.sha256(body).hexdigest(), "seed": t["seed"], "sources": sources,
                "versions": {p: _version(p) for p in ("cmudict", "wordfreq", "pandas")},
                "n_foils": int(len(df)), "counts": {"/".join(k): int(v) for k, v in counts.items()},
                "trials_with_foil": {f"{s}/{k}": int(g.trial.nunique()) for (s, k), g in df.groupby(["split", "type"])}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    lines = ["# Foil review sheet", "", f"`{rel(out)}/foils.csv`, {len(df)} foils. A seeded sample of "
             f"each variant ({args.review_per_variant}); fix a bad rule in {args.tables.name}, then rebuild.", ""]
    for (kind, variant), g in df.groupby(["type", "variant"]):
        lines += [f"## {kind} / {variant} ({len(g)})", "", "| split | trial | true | foil | phon. dist |",
                  "|---|---|---|---|---|"]
        for r in g.sample(min(len(g), args.review_per_variant), random_state=0).itertuples():
            d = "" if r.phon_dist == "" or pd.isna(r.phon_dist) else f"{float(r.phon_dist):.2f}"
            lines.append(f"| {r.split} | {r.trial} | {r.reference} | {r.foil} | {d} |")
        lines.append("")
    (out / "review.md").write_text("\n".join(lines))
    (args.out / "LATEST").write_text(out.name + "\n")
    print(json.dumps({"out": rel(out), **{k: manifest[k] for k in ("n_foils", "trials_with_foil")}},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

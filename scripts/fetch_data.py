#!/usr/bin/env python
"""Phase -1: get the Kunz et al. 2025 Dryad dataset into data/raw/.

    python scripts/fetch_data.py list                 # latest version's files + sizes; pins configs/dryad_manifest.json
    python scripts/fetch_data.py download [--files A.zip B.zip]
                                                      # needs DRYAD_TOKEN, or DRYAD_CLIENT_ID + DRYAD_CLIENT_SECRET
    python scripts/fetch_data.py verify [--quick]     # check data/raw/* against the API's sizes and sha-256
    python scripts/fetch_data.py fetch-url --doi DOI --file NAME --url PRESIGNED --dest data/raw/<dir>
                                                      # any Dryad dataset: download from a presigned link a browser
                                                      # session obtained, verify against that dataset's API manifest
    python scripts/fetch_data.py extract [--required-only]
                                                      # unzip verified archives to data/raw/<name>/ (read-only; zips kept)

Manual route (no API token): download each file from
https://datadryad.org/dataset/doi:10.5061/dryad.gf1vhhn1j into data/raw/ (keep
the .zip files as-is; disable Safari's "open safe files"), then run `verify`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kunzdecoder import dryad  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["list", "download", "verify", "extract", "fetch-url"])
    ap.add_argument("--config", default=ROOT / "configs" / "data.yaml", type=Path)
    ap.add_argument("--files", nargs="+", help="restrict to these file names (default: all)")
    ap.add_argument("--required-only", action="store_true", help="only the archives listed in required_files")
    ap.add_argument("--quick", action="store_true", help="verify sizes only (skip sha-256)")
    ap.add_argument("--doi", help="fetch-url: dataset DOI (default: the Kunz dataset)")
    ap.add_argument("--file", help="fetch-url: file name in the dataset")
    ap.add_argument("--url", help="fetch-url: presigned download link issued to a browser session")
    ap.add_argument("--dest", type=Path, help="fetch-url: destination directory")
    args = ap.parse_args()

    if args.command == "fetch-url":
        cfg = yaml.safe_load(args.config.read_text())
        doi = args.doi or cfg["dryad"]["doi"]
        manifest = dryad.fetch_manifest(cfg["dryad"]["api"], doi)
        slug = doi.split("dryad.")[-1]
        dryad.save_manifest(manifest, ROOT / "configs" / f"dryad_manifest_{slug}.json")
        entry = next(f for f in manifest["files"] if f["path"] == args.file)
        dest = args.dest if args.dest.is_absolute() else ROOT / args.dest
        dryad.download_file(cfg["dryad"]["api"], entry, dest, url=args.url)
        return 0

    cfg = yaml.safe_load(args.config.read_text())
    api, doi = cfg["dryad"]["api"], cfg["dryad"]["doi"]
    manifest_path = ROOT / cfg["dryad"]["manifest"]
    raw = ROOT / cfg["paths"]["raw"]
    raw.mkdir(parents=True, exist_ok=True)

    if args.command == "list" or not manifest_path.exists():
        manifest = dryad.fetch_manifest(api, doi)
        dryad.save_manifest(manifest, manifest_path)
    else:
        manifest = dryad.load_manifest(manifest_path)

    wanted = set(args.files or [])
    if args.required_only:
        wanted |= set(cfg["required_files"])
    entries = [f for f in manifest["files"] if not wanted or f["path"] in wanted]
    if wanted - {f["path"] for f in entries}:
        print(f"unknown file(s): {sorted(wanted - {f['path'] for f in entries})}", file=sys.stderr)
        return 2

    if args.command == "list":
        print(f"{doi}  version {manifest['versionNumber']} (id {manifest['versionId']}), "
              f"published {manifest['publicationDate']}")
        for f in entries:
            print(f"  {f['size'] / 1e6:10.1f} MB  {f['path']:42s} {f['digestType']}:{f['digest'][:12]}…")
        print(f"  {sum(f['size'] for f in entries) / 1e9:10.2f} GB  total ({len(entries)} files)")
        print(f"manifest pinned to {manifest_path.relative_to(ROOT)}")
        return 0

    if args.command == "download":
        token = dryad.get_token()
        if not token:
            print("No Dryad API token. The API requires a bearer token for file downloads.\n"
                  "Set DRYAD_TOKEN (from https://datadryad.org/account) or DRYAD_CLIENT_ID + "
                  "DRYAD_CLIENT_SECRET, or download manually and run `verify`.", file=sys.stderr)
            return 1
        for f in entries:
            dryad.download_file(api, f, raw, token)
        return 0

    if args.command == "verify":
        bad = 0
        for f in entries:
            status = dryad.check_file(raw / f["path"], f, hash_it=not args.quick)
            bad += status not in ("ok", "ok-size", "missing")
            print(f"  {status:14s} {f['path']}")
        missing = [f["path"] for f in entries if not (raw / f["path"]).exists()]
        if missing:
            print(f"{len(missing)} missing: {', '.join(missing)}")
        return 1 if bad else 0

    if args.command == "extract":
        for f in entries:
            src = raw / f["path"]
            if not src.suffix == ".zip" or not src.exists():
                continue
            out = raw / src.stem
            if out.exists() and any(out.iterdir()):
                print(f"  {src.name}: already extracted -> {out.relative_to(ROOT)}")
                continue
            status = dryad.check_file(src, f)
            if status != "ok":
                print(f"  {src.name}: {status}, not extracting", file=sys.stderr)
                continue
            # Archives already wrap everything in <stem>/, so unpack beside the zip
            dest = raw if dryad.zip_top_dir(src) == src.stem else out
            n = dryad.safe_extract(src, dest)
            print(f"  {src.name}: {n} files -> {out.relative_to(ROOT)}")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())

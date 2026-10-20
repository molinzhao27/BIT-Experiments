#!/usr/bin/env python
"""Phase 4b: fetch the approved extra human pretraining files listed in configs/pretrain_datasets.yaml.

    python scripts/fetch_pretrain_data.py --list
    python scripts/fetch_pretrain_data.py --dandi                    # FALCON H2 NWBs (open DANDI API)
    python scripts/fetch_pretrain_data.py --dryad-urls urls.json     # {"<file name>": "<presigned url>"}

Dryad data files are downloaded from presigned storage links that a browser session obtained (Dryad's
web downloads sit behind a browser check; its API needs a token). Every file is verified against the
inventory's size + SHA-256 before it is renamed into place. Resumable (.part files).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEST = {"willett2021": "data/raw/willett_2021_handwriting", "willett2023": "data/raw/willett_2023_text",
        "fan2023": "data/raw/fan_2023_corp", "falcon_h2": "data/raw/falcon_h2"}
CHUNK = 8 * 1024 * 1024


def pending(cfg: dict) -> list[tuple[str, dict]]:
    """(dataset key, file entry) for approved SSL files that are not yet local."""
    out = []
    for d in cfg["datasets"]:
        key = d["id"]
        for f in d.get("files", []):
            if f.get("needed_for_ssl") and not f.get("local_path") and key in DEST:
                out.append((key, f))
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(CHUNK):
            h.update(b)
    return h.hexdigest()


def download(url: str, dest: Path, size: int, digest: str, retries: int = 10) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == size and sha256(dest) == digest:
        return "already ok"
    part = dest.with_name(dest.name + ".part")
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        if have == size:
            break
        try:
            with requests.get(url, headers={"Range": f"bytes={have}-"} if have else {}, stream=True,
                              timeout=(30, 300)) as r:
                r.raise_for_status()
                with open(part, "ab" if have and r.status_code == 206 else "wb") as fh:
                    for chunk in r.iter_content(CHUNK):
                        fh.write(chunk)
        except (requests.RequestException, OSError) as exc:
            print(f"  retry {attempt + 1}: {exc}", flush=True)
            time.sleep(min(60, 2 ** attempt))
    if part.stat().st_size != size or sha256(part) != digest:
        return "MISMATCH (kept .part)"
    part.rename(dest)
    return "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "pretrain_datasets.yaml")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dandi", action="store_true")
    ap.add_argument("--dryad-urls", type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    todo = pending(cfg)
    urls = json.loads(args.dryad_urls.read_text()) if args.dryad_urls else {}
    for key, f in todo:
        name = f.get("name") or f.get("path")
        dest = ROOT / DEST[key] / name
        if args.list:
            print(f"{key:12s} {f['size_bytes'] / 1e6:9.1f} MB  {name}  -> {dest.relative_to(ROOT)}")
            continue
        asset = f.get("asset_id")
        if args.dandi and key == "falcon_h2" and asset:
            url = f"https://api.dandiarchive.org/api/assets/{asset}/download/"
        elif name in urls:
            url = urls[name]
        else:
            continue
        print(f"{key}: {name}: {download(url, dest, int(f['size_bytes']), f['sha256'])}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

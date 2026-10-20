"""Dryad API helpers: manifest, resumable authenticated download, verification, extraction.

Dryad's v2 API lists files anonymously, but `/files/{id}/download` requires an
OAuth bearer token (API account: https://datadryad.org/account). Without a token
the files must be downloaded by hand from the dataset landing page and dropped
into data/raw/, after which `verify` checks them against the API's size + sha-256.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import zipfile
from pathlib import Path
from urllib.parse import quote

import requests

USER_AGENT = "kunzdecoder-repro/0.1 (+https://doi.org/10.5061/dryad.gf1vhhn1j)"
CHUNK = 8 * 1024 * 1024


def _get_json(url: str) -> dict:
    r = requests.get(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}, timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_manifest(api: str, doi: str) -> dict:
    """Latest version of the dataset and its files (id, path, size, sha-256)."""
    ds = _get_json(f"{api}/datasets/{quote(doi, safe='')}")
    version_href = ds["_links"]["stash:version"]["href"]  # latest version
    version_id = int(version_href.rstrip("/").split("/")[-1])
    files, url = [], f"{api}/versions/{version_id}/files?per_page=100"
    while url:
        page = _get_json(url)
        for f in page["_embedded"]["stash:files"]:
            files.append({
                "id": int(f["_links"]["self"]["href"].rstrip("/").split("/")[-1]),
                "path": f["path"],
                "size": int(f["size"]),
                "digest": f.get("digest"),
                "digestType": f.get("digestType"),
            })
        nxt = page["_links"].get("next", {}).get("href")
        url = f"https://datadryad.org{nxt}" if nxt else None
    return {
        "doi": doi,
        "versionId": version_id,
        "versionNumber": ds.get("versionNumber"),
        "publicationDate": ds.get("publicationDate"),
        "storageSize": ds.get("storageSize"),
        "files": sorted(files, key=lambda f: f["path"]),
    }


def get_token() -> str | None:
    """DRYAD_TOKEN, or exchange DRYAD_CLIENT_ID/DRYAD_CLIENT_SECRET for one (valid 10 h)."""
    if tok := os.environ.get("DRYAD_TOKEN"):
        return tok
    cid, secret = os.environ.get("DRYAD_CLIENT_ID"), os.environ.get("DRYAD_CLIENT_SECRET")
    if not (cid and secret):
        return None
    r = requests.post(
        "https://datadryad.org/oauth/token",
        data={"client_id": cid, "client_secret": secret, "grant_type": "client_credentials"},
        headers={"User-Agent": USER_AGENT},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def check_file(path: Path, entry: dict, hash_it: bool = True) -> str:
    """'missing' | 'size-mismatch' | 'sha-mismatch' | 'ok' | 'ok-size' (hash skipped)."""
    if not path.exists():
        return "missing"
    if path.stat().st_size != entry["size"]:
        return "size-mismatch"
    if not hash_it or entry.get("digestType") != "sha-256":
        return "ok-size"
    return "ok" if sha256sum(path) == entry["digest"] else "sha-mismatch"


def download_file(api: str, entry: dict, dest_dir: Path, token: str | None = None, max_retries: int = 20,
                  url: str | None = None) -> Path:
    """Resumable download to <dest>.part via HTTP Range, verified before the final rename.

    Either through the API with a bearer token (requests drops the Authorization header
    when Dryad redirects to its storage host), or from `url`: a presigned storage URL
    that Dryad issued to a browser session (valid 24 h; Dryad's web downloads sit behind
    a browser check, see README).
    """
    dest = dest_dir / entry["path"]
    part = dest.with_name(dest.name + ".part")
    dest_dir.mkdir(parents=True, exist_ok=True)
    if check_file(dest, entry) == "ok":
        print(f"  {entry['path']}: already present and verified")
        return dest
    url = url or f"{api}/files/{entry['id']}/download"
    for attempt in range(1, max_retries + 1):
        have = part.stat().st_size if part.exists() else 0
        if have == entry["size"]:
            break
        headers = {"User-Agent": USER_AGENT}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as r:
                if r.status_code in (401, 403):
                    raise PermissionError(f"Dryad refused the download (HTTP {r.status_code}): {r.text[:200]}")
                r.raise_for_status()
                mode = "ab" if (have and r.status_code == 206) else "wb"
                if mode == "wb":
                    have = 0
                t0, done = time.time(), have
                with open(part, mode) as fh:
                    for chunk in r.iter_content(CHUNK):
                        fh.write(chunk)
                        done += len(chunk)
                        rate = (done - have) / max(time.time() - t0, 1e-6) / 1e6
                        print(f"\r  {entry['path']}: {done / 1e6:9.1f} / {entry['size'] / 1e6:.1f} MB"
                              f"  ({rate:5.1f} MB/s)", end="", flush=True)
                print()
        except PermissionError:
            raise
        except (requests.RequestException, OSError) as exc:
            wait = min(60, 2 ** attempt)
            print(f"\n  attempt {attempt} failed ({exc}); resuming in {wait}s")
            time.sleep(wait)
    status = check_file(part, entry)
    if status not in ("ok", "ok-size"):
        raise RuntimeError(f"{entry['path']}: {status} after download; leaving {part} for inspection")
    part.rename(dest)
    print(f"  {entry['path']}: verified ({status})")
    return dest


def _is_junk(name: str) -> bool:
    return name.startswith("__MACOSX/") or Path(name).name == ".DS_Store"


def safe_extract(zip_path: Path, out_dir: Path) -> int:
    """Extract a zip into out_dir (read-only), refusing traversing paths and skipping macOS junk.

    Returns the number of files extracted.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    root = out_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist() if not _is_junk(m.filename)]
        for m in members:
            if not (out_dir / m.filename).resolve().is_relative_to(root):
                raise RuntimeError(f"unsafe member path in {zip_path.name}: {m.filename}")
        zf.extractall(out_dir, members=members)
    files = [out_dir / m.filename for m in members if not m.is_dir()]
    for f in files:
        f.chmod(0o444)
    return len(files)


def zip_top_dir(zip_path: Path) -> str | None:
    """The single top-level directory every member sits under, if there is one."""
    with zipfile.ZipFile(zip_path) as zf:
        tops = {m.filename.split("/", 1)[0] for m in zf.infolist() if not _is_junk(m.filename)}
    return tops.pop() if len(tops) == 1 else None


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def save_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")

#!/usr/bin/env python
"""Animated replay of the block-diffusion LLM's denoising passes (run 3, Fast-dLLM v2), from decoding_order_sweep traces.

    python scripts/denoise_viewer.py --traces DIR [--split test] --out ops/denoise-viewer/index.html
        [--label TEXT] [--curate 26] [--l2r DIR] [--revisions DIR] [--seed 0] [--standalone]

Reads DIR/<split>_greedy_real_sbs<k>[_remask<t>x<r>].json.gz, one replay mode per file (the format is traced_denoise's
and remask_denoise's docstrings in decoding_order_sweep.py). Every trial's events are replayed and must reproduce the
trace's `out` exactly; every pass must follow the sampler rule (commit each probability > threshold, at least the most
confident one) and lie inside one chunk; every re-mask round must re-open exactly the block's tokens committed below
the re-mask threshold up to its first EOS. Any violation stops the build. Pass counts (one for reading the prompt when
it shares block 0, one per denoising pass and per block's first token) and corpus WER per mode are printed next to the
sweep's summary.json. Writes one self-contained HTML page (data embedded, stdlib only) for the Artifact viewer: per
trial and mode, each pass's proposals, probabilities and commits; the reference, word-level WER (BIT's normalization,
textmetrics.py), seen/unseen, the left-to-right Qwen2.5 decode of the same trial (shown only where its reference string
matches the trace's), and the revision check's leave-one-out edits (--revisions; at the trained chunk size).
Picker: every trial when there are at most --curate, else a curated --curate (perfect / near-miss / bad, seen, short /
long, crossing a block, out-of-order commits, end fixed early, changed by re-masking, fixable by one revision), plus a
random draw over all trials.
The page omits <html>/<head>/<body> (the Artifact viewer adds them); --standalone adds them for opening locally.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
L2R = ROOT / "results" / "stage2" / "B_ssl_qwen25-1.5b-noprompt_s42" / "decodes"
EOS_TEXT = "<|im_end|>"
LIMIT = 15_500_000  # the Artifact page limit is 16 MB
BASE = "8"  # run 3's own chunk size; curation features and the revision check use it


class TraceError(SystemExit):
    pass


def normalize(sentence: str) -> str:
    """BIT's remove_punctuation, as textmetrics.normalize."""
    s = re.sub(r"[^a-zA-Z\- ']", "", sentence or "")
    s = s.replace("- ", " ").lower().replace("--", "").replace(" '", "'").strip()
    return " ".join(w for w in s.split() if w)


def align(ref: list[str], hyp: list[str]) -> str:
    """Word-level edit script, one char per step: = match, s substitution, d deletion, i insertion."""
    n, m = len(ref), len(hyp)
    d = [[i + j if i * j == 0 else 0 for j in range(m + 1)] for i in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]), d[i - 1][j] + 1, d[i][j - 1] + 1)
    ops, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append("=" if ref[i - 1] == hyp[j - 1] else "s")
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            ops.append("d")
            i -= 1
        else:
            ops.append("i")
            j -= 1
    return "".join(reversed(ops))


def scored(ref: str, hyp: str) -> dict:
    r, h = normalize(ref), normalize(hyp)
    ops = align(r.split(), h.split())
    errs = sum(o != "=" for o in ops)
    return {"hyp": hyp, "hw": h, "ops": ops, "errs": errs, "wer": round(errs / max(len(r.split()), 1), 4)}


def replay(t: dict, sampler: dict, eos: int, where: str, remask: float | None) -> dict:
    """Check one trial's events against the sampler (and re-mask rule) and `out`; return its features."""
    bs, sbs, thr = sampler["block_size"], sampler["small_block_size"], sampler["threshold"]
    tail = t["tail"]
    head = bs - tail  # output slots in block 0; the prompt fills the rest

    def fail(msg):
        raise TraceError(f"{where} trial {t['i']}: {msg}")

    def chunk(pos):
        return (0, (pos + tail) // sbs) if pos < head else (1 + (pos - head) // bs, (pos - head) % bs // sbs)

    def block_range(b):
        return (0, head - 1) if b == 0 else (head + (b - 1) * bs, head + b * bs - 1)

    at, seen_at = {}, {}  # pos -> (pass, token, probability when committed), pos -> first pass that listed it
    k = 1 if tail else 0  # reading the prompt is a pass of its own when it shares block 0 (no output)
    reopened = 0
    for e in t["events"]:
        if "r" in e:  # a re-mask round: no forward pass
            if remask is None:
                fail("re-mask event in a plain trace")
            lo, hi = block_range(e["b"])
            block = {p: v for p, v in at.items() if lo <= p <= hi}
            first_eos = min((p for p, v in block.items() if v[1] == eos), default=hi)
            low = {p for p, v in block.items() if v[2] < remask and p <= first_eos}
            edge = {p for p, v in block.items() if v[2] == remask}  # rounded to the threshold: either way
            got = set()
            for pos, tok, p in e["r"]:
                if pos not in at or at[pos][1] != tok or abs(at[pos][2] - p) > 1e-4:
                    fail(f"after pass {k}: re-masked position {pos} ({tok}, p={p}) is not what was committed there")
                got.add(pos)
            if (got ^ low) - edge:
                fail(f"after pass {k}: re-masked {sorted(got)}, the rule gives {sorted(low)} (threshold {remask})")
            for pos in got:
                del at[pos]
            reopened += len(got)
            continue
        k += 1
        if "first" in e:
            pos, tok = e["first"]
            if pos != block_range(e["b"])[0] or (e["b"] == 0 and tail):
                fail(f"pass {k}: first token of block {e['b']} at position {pos}")
            if pos in at:
                fail(f"pass {k}: position {pos} committed twice")
            at[pos] = (k, tok, 1.0)
            seen_at.setdefault(pos, k)
            continue
        m = e["m"]
        if not m or {chunk(p[0]) for p in m} != {(e["b"], chunk(m[0][0])[1])}:
            fail(f"pass {k}: positions {[p[0] for p in m]} are not one chunk of block {e['b']} (chunk size {sbs})")
        top = max(p[2] for p in m)
        if not any(p[3] for p in m):
            fail(f"pass {k}: nothing committed")
        for pos, tok, p, c in m:
            if pos in at:
                fail(f"pass {k}: position {pos} listed while committed")
            if p > thr and not c or c and p < thr and p < top:
                fail(f"pass {k}: position {pos} p={p} committed={c} breaks the sampler rule (threshold {thr})")
            seen_at.setdefault(pos, k)
            if c:
                at[pos] = (k, tok, p)
    seq, pos = [], 0
    while pos in at:
        seq.append(at[pos][1])
        if seq[-1] == eos:
            break
        pos += 1
    out = t["out"]
    if (seq if eos in out else seq[: len(out)]) != out:
        fail(f"replayed commits {seq} do not reproduce out {out}")
    n = out.index(eos) if eos in out else len(out)  # sentence tokens; position n holds the first EOS
    steps = [at[p][0] for p in range(n)]
    end = at[n][0] if n < len(out) else None
    return {
        "n": n,
        "passes": k,
        "tokens": len(out),
        "reopened": reopened,
        "ooo": any(steps[a] > steps[a + 1] for a in range(n - 1)),
        "end_early": end is not None and n > 0 and end < max(steps),
        "end_first": end is not None and end == seen_at[n],
        "straddle": head >= 2 and n - head >= 2,
    }


def pack(t: dict, vix: dict) -> list:
    """Events as compact arrays: [-1, pos, token] for a block's first token, [-2, pos, token, permille]* for a re-mask
    round, else [pos, token, permille, committed]* for a pass."""
    steps = []
    for e in t["events"]:
        if "first" in e:
            steps.append([-1, e["first"][0], vix[e["first"][1]]])
        elif "r" in e:
            steps.append([-2] + [x for pos, tok, p in e["r"] for x in (pos, vix[tok], round(p * 1000))])
        else:
            steps.append([x for pos, tok, p, c in e["m"] for x in (pos, vix[tok], round(p * 1000), c)])
    return steps


def load_l2r(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    with path.open() as f:
        return {int(r["trial"]): r for r in csv.DictReader(f)}


CATS = [  # curation targets: (key, test on chunk-size-8 features, how many sentences should show it)
    ("perfect", lambda f: f["errs"] == 0, 5),
    ("near", lambda f: 0 < f["errs"] <= 2 and f["wer"] <= 0.34, 5),
    ("bad", lambda f: f["wer"] >= 0.6, 4),
    ("seen", lambda f: f["seen"], 4),
    ("unseen", lambda f: not f["seen"], 8),
    ("short", lambda f: f["nref"] <= 4, 3),
    ("long", lambda f: f["nref"] >= 12, 4),
    ("straddle", lambda f: f["straddle"], 5),
    ("ooo", lambda f: f["ooo"], 6),
    ("end_early", lambda f: f["end_early"], 4),
    ("end_first", lambda f: f["end_first"], 4),
    ("remask", lambda f: f["remask_changed"], 3),
    ("revise", lambda f: f["fixable"], 3),
]


def curate(feats: list[dict], k: int, seed: int) -> list[int]:
    """Greedy cover of CATS: repeatedly take the trial meeting the most still-unmet targets (ties: seeded random)."""
    rng = random.Random(seed)
    order = list(range(len(feats)))
    rng.shuffle(order)
    need = {key: q for key, _, q in CATS}
    picked: list[int] = []
    while len(picked) < k:
        best, score = None, 0
        for j in order:
            if j in picked:
                continue
            s = sum(1 for key, test, _ in CATS if need[key] > 0 and test(feats[j]))
            if s > score:
                best, score = j, s
        if best is None:
            break
        picked.append(best)
        for key, test, _ in CATS:
            if test(feats[best]):
                need[key] -= 1
    picked += [j for j in order if j not in picked][: k - len(picked)]
    return sorted(picked)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", type=Path, required=True, help="directory of <split>_greedy_real_sbs<k>*.json.gz")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--out", type=Path, required=True, help="the HTML page to write")
    ap.add_argument("--label", help="banner text (default: '<n> <split> sentences')")
    ap.add_argument("--curate", type=int, default=26, help="picker size when the split has more trials")
    ap.add_argument("--l2r", type=Path, default=L2R, help="left-to-right decodes dir (<split>_greedy_real.csv)")
    ap.add_argument("--revisions", type=Path, help="revision_check dir (default: next to --traces, if present)")
    ap.add_argument("--seed", type=int, default=0, help="curation tie-breaks")
    ap.add_argument("--standalone", action="store_true", help="wrap in a full HTML document (local viewing)")
    args = ap.parse_args()

    files, labels, remasks = {}, {}, {}  # mode -> path, toggle label, re-mask threshold
    pat = re.compile(rf"^{args.split}_greedy_real_sbs(\d+)(?:_remask([\d.]+)x(\d+))?\.json\.gz$")
    for p in args.traces.iterdir():
        if m := pat.match(p.name):
            mode = m[1] + ("r" if m[2] else "")
            files[mode], labels[mode] = p, m[1] + (" + re-mask" if m[2] else "")
    if not files:
        ap.error(f"no {args.split}_greedy_real_sbs*.json.gz in {args.traces}")
    modes = sorted(files, key=lambda s: (s.endswith("r"), int(s.rstrip("r"))))
    base = BASE if BASE in files else modes[0]
    traces = {}
    for mode in modes:
        with gzip.open(files[mode], "rt") as f:
            traces[mode] = json.load(f)
        meta = traces[mode]["meta"]
        if str(meta["sampler"]["small_block_size"]) != mode.rstrip("r"):
            raise TraceError(f"{files[mode].name}: sampler says chunk size {meta['sampler']}")
        if mode.endswith("r"):
            remasks[mode] = {"threshold": meta["remask_threshold"], "rounds": meta["remask_rounds"]}

    rev_dir = args.revisions or args.traces.parent / "revision_check"
    rev_path = rev_dir / f"{args.split}.json.gz"
    revisions = None
    if rev_path.exists():
        with gzip.open(rev_path, "rt") as f:
            revisions = json.load(f)

    # One vocabulary over all files (token id -> text), indexed for the page.
    pieces: dict[int, str] = {}
    for name, d in [(files[m].name, traces[m]) for m in modes] + ([(rev_path.name, revisions)] if revisions else []):
        for tid, text in d["vocab"].items():
            if pieces.setdefault(int(tid), text) != text:
                raise TraceError(f"{name}: token {tid} is {text!r} here but {pieces[int(tid)]!r} elsewhere")
    eos_ids = [tid for tid, text in pieces.items() if text == EOS_TEXT]
    eos = eos_ids[0] if eos_ids else None
    ids = sorted(pieces)
    vix = {tid: j for j, tid in enumerate(ids)}

    by_i = {mode: {t["i"]: t for t in d["trials"]} for mode, d in traces.items()}
    trial_ids = sorted(by_i[base])
    for mode in modes:
        if sorted(by_i[mode]) != trial_ids:
            raise TraceError(f"{files[mode].name} has {len(by_i[mode])} trials vs {len(trial_ids)} at chunk size {base}")
    if revisions and len(revisions["trials"]) != len(trial_ids):
        raise TraceError(f"{rev_path.name}: {len(revisions['trials'])} trials vs {len(trial_ids)} traced")
    l2r = load_l2r(args.l2r / f"{args.split}_greedy_real.csv")
    l2r_bad = 0

    trials, feats, pts = [], [], {mode: [] for mode in modes}
    for i in trial_ids:
        t0 = by_i[base][i]
        rec = {"i": i, "ref": t0["ref"], "rw": normalize(t0["ref"]), "seen": t0["seen"], "prefix": t0["prefix_len"],
               "tail": t0["tail"], "runs": {}}
        reps = {}
        for mode in modes:
            t = by_i[mode][i]
            if (t["ref"], t["seen"], t["prefix_len"], t["tail"]) != (t0["ref"], t0["seen"], t0["prefix_len"], t0["tail"]):
                raise TraceError(f"trial {i}: {files[mode].name} and chunk size {base} disagree on ref/seen/prefix")
            r = reps[mode] = replay(t, traces[mode]["meta"]["sampler"], eos, files[mode].name,
                                    remasks.get(mode, {}).get("threshold"))
            s = scored(t["ref"], t["hyp"])
            rec["runs"][mode] = {**s, "out": [vix[x] for x in t["out"]], "n": r["n"], "steps": pack(t, vix),
                                 "f": [int(r["ooo"]), int(r["end_early"]), int(r["end_first"]), r["reopened"]]}
            pts[mode].append([r["tokens"], r["passes"], int(r["ooo"]), int(r["end_early"]), int(r["end_first"]),
                              s["errs"], len(trials)])
        fixable = False
        if revisions:  # leave-one-out edits on the chunk-size-8 decode (revision_report)
            rv = revisions["trials"][len(trials)]
            if rv["ref"] != t0["ref"] or rv["hyp"] != by_i[base][i]["hyp"]:
                raise TraceError(f"{rev_path.name} trial {i}: ref/hyp differ from the chunk size {base} trace")
            rec["rev"] = [[e["pos"], vix[e["to"]], round(e["p_from"] * 1000), round(e["p_to"] * 1000), e["delta"]]
                          for e in rv["edits"]]
            if any(by_i[base][i]["out"][e["pos"]] != e["from"] for e in rv["edits"]):
                raise TraceError(f"{rev_path.name} trial {i}: an edit's 'from' token is not the decoded one")
            fixable = any(e["delta"] < 0 for e in rv["edits"])
        changed = any(rec["runs"][m]["hyp"] != rec["runs"][base]["hyp"] for m in remasks)
        feats.append({**reps[base], **rec["runs"][base], "seen": t0["seen"], "nref": len(rec["rw"].split()),
                      "remask_changed": changed, "fixable": fixable})
        row = l2r.get(i)
        if row is not None and row["reference"] == t0["ref"]:
            rec["l2r"] = scored(t0["ref"], row["decoded"])
        else:
            rec["l2r"] = None
            l2r_bad += row is not None
        trials.append(rec)

    # Cross-check against the sweep: decoder passes per trial and corpus WER per mode.
    summary_path = args.traces.parent / "summary.json"
    sweep = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    nwords = sum(len(r["rw"].split()) for r in trials)
    print(f"{'mode':<14}{'passes/trial':>13}{'sweep':>8}{'WER':>9}{'sweep':>9}")
    for mode in modes:
        if mode in remasks:
            res = sweep.get("remask", {}).get("results", {}).get(args.split, {})
        else:
            res = sweep.get("results", {}).get(f"{args.split}_greedy_real_sbs{mode}", {})
        passes = sum(p[1] for p in pts[mode]) / len(pts[mode])
        wer = sum(p[5] for p in pts[mode]) / nwords
        print(f"{labels[mode]:<14}{passes:>13.2f}{res.get('decoder_passes_per_trial', float('nan')):>8.2f}"
              f"{100 * wer:>8.2f}%{100 * res.get('wer', float('nan')):>8.2f}%")
    l2r_wer = (sum(r["l2r"]["errs"] for r in trials if r["l2r"]) / max(1, sum(len(r["rw"].split()) for r in trials
                                                                          if r["l2r"])))

    picked = list(range(len(trials))) if len(trials) <= args.curate else curate(feats, args.curate, args.seed)
    start = max(picked, key=lambda j: (feats[j]["errs"] == 0, feats[j]["ooo"], feats[j]["end_early"],
                                       feats[j]["straddle"], 5 <= feats[j]["nref"] <= 12, -j))
    meta = traces[base]["meta"]
    data = {
        "label": args.label or f"{len(picked)} {args.split} sentences",
        "split": args.split, "modes": modes, "labels": labels, "remask": remasks, "base": base,
        "block": meta["sampler"]["block_size"], "threshold": meta["sampler"]["threshold"],
        "vocab": [pieces[t] for t in ids], "eos": vix[eos] if eos is not None else -1, "picked": picked,
        "start": start, "random": len(picked) < len(trials), "trials": trials, "pts": pts,
        "revMode": base if revisions else None, "l2rWer": l2r_wer if l2r else None,
        "meta": {"model": meta.get("source_run") or meta.get("model"), "checkpoint": meta.get("checkpoint"),
                 "l2r": args.l2r.parent.name if l2r else None, "files": [files[m].name for m in modes],
                 "revisions": rev_path.name if revisions else None, "built": time.strftime("%Y-%m-%d %H:%M")},
    }

    def render(d):
        blob = json.dumps(d, separators=(",", ":")).replace("</", "<\\/")
        return TEMPLATE.replace("/*__DATA__*/", blob)

    html = render(data)
    if len(html.encode()) > LIMIT:  # keep every trial at the trained chunk size for the random draw
        for j, rec in enumerate(trials):
            if j not in picked:
                rec["runs"] = {base: rec["runs"][base]}
        html = render(data)
        print(f"over {LIMIT / 1e6:.1f} MB: unpicked trials keep only chunk size {base}")
    if len(html.encode()) > LIMIT:
        data["trials"] = [trials[j] for j in picked]
        remap = {j: n for n, j in enumerate(picked)}
        data["pts"] = {k: [p[:-1] + [remap.get(p[-1], -1)] for p in v] for k, v in data["pts"].items()}
        data["start"], data["picked"], data["random"] = remap[start], list(range(len(picked))), False
        html = render(data)
        print("still over: random draw dropped, picked trials only")
    if args.standalone:
        html = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,'
                ' initial-scale=1, viewport-fit=cover"><style>[hidden]{display:none!important}body{margin:0}</style></head>'
                f"<body>{html}</body></html>")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    n_l2r = sum(r["l2r"] is not None for r in trials)
    print(f"{len(trials)} trials x modes {[labels[m] for m in modes]}: every replay reproduces `out` and follows the "
          f"sampler{' and re-mask' if remasks else ''} rule")
    print(f"picker {len(picked)}, start trial {trials[start]['i']}, left-to-right decodes matched {n_l2r}"
          f" (reference mismatch {l2r_bad}, WER {100 * l2r_wer:.2f}%), revision edits "
          f"{sum(len(r.get('rev', [])) for r in trials) if revisions else 'none'}")
    print(f"wrote {args.out} ({len(html.encode()) / 1e6:.2f} MB)")
    return 0


TEMPLATE = r"""<title>Brain-to-Text Diffusion Replay</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@75..100,500..800&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">
<style>
/* Layout: one column (max 1040px). Header, then the replay stage (tiles, pass strips, controls), then the
   reference/decode comparison, the legend, and (with enough sentences) a summary over the whole split. */
:root {
  --paper: #edf0f3; --card: #fafbfc; --ink: #141b22; --ink-2: #4a5562; --ink-3: #6f7a87; --rule: #c9d1da;
  --mask: #d6dde5; --solid: #1b2430; --solid-fg: #f3f5f8; --flash: #e39b12; --flash-fg: #1d1404;
  --cobalt: #2b4fd8; --on-cobalt: #ffffff; --cobalt-tint: rgba(43, 79, 216, .13); --reopen: #a8327f;
  --bad: #b8382a; --good: #22744a; --warn: #9a6400;
  --noise: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='90' height='90'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.8' numOctaves='2' stitchTiles='stitch'/%3E%3CfeColorMatrix values='0 0 0 0 .08 0 0 0 0 .1 0 0 0 0 .14 1.5 0 0 0 -.5'/%3E%3C/filter%3E%3Crect width='90' height='90' filter='url(%23n)'/%3E%3C/svg%3E");
  --f-display: "Archivo", "Arial Narrow", system-ui, sans-serif;
  --f-body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --f-mono: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper: #0d1116; --card: #141a21; --ink: #e4e9ee; --ink-2: #a9b4bf; --ink-3: #808c98; --rule: #2b3540;
    --mask: #1f2731; --solid: #dde4eb; --solid-fg: #0f151b; --flash: #f0ae2e; --flash-fg: #1d1404;
    --cobalt: #8199ff; --on-cobalt: #0d1116; --cobalt-tint: rgba(129, 153, 255, .17); --reopen: #e77fcb;
    --bad: #f07a68; --good: #62c48f; --warn: #e2b04f;
    --noise: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='90' height='90'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.8' numOctaves='2' stitchTiles='stitch'/%3E%3CfeColorMatrix values='0 0 0 0 .85 0 0 0 0 .9 0 0 0 0 1 1.3 0 0 0 -.5'/%3E%3C/filter%3E%3Crect width='90' height='90' filter='url(%23n)'/%3E%3C/svg%3E");
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --paper: #0d1116; --card: #141a21; --ink: #e4e9ee; --ink-2: #a9b4bf; --ink-3: #808c98; --rule: #2b3540;
  --mask: #1f2731; --solid: #dde4eb; --solid-fg: #0f151b; --flash: #f0ae2e; --flash-fg: #1d1404;
  --cobalt: #8199ff; --on-cobalt: #0d1116; --cobalt-tint: rgba(129, 153, 255, .17); --reopen: #e77fcb;
  --bad: #f07a68; --good: #62c48f; --warn: #e2b04f;
  --noise: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='90' height='90'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.8' numOctaves='2' stitchTiles='stitch'/%3E%3CfeColorMatrix values='0 0 0 0 .85 0 0 0 0 .9 0 0 0 0 1 1.3 0 0 0 -.5'/%3E%3C/filter%3E%3Crect width='90' height='90' filter='url(%23n)'/%3E%3C/svg%3E");
  color-scheme: dark;
}
* { box-sizing: border-box; }
body { background: var(--paper); color: var(--ink); font: 15px/1.55 var(--f-body); }
.wrap { max-width: 1040px; margin: 0 auto; padding-block: 28px 56px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  display: grid; grid-template-columns: minmax(0, 1fr); gap: 22px; }
header { display: grid; grid-template-columns: minmax(0, 1fr); gap: 10px; }
.eyebrow { margin: 0; font: 500 11.5px/1.4 var(--f-mono); letter-spacing: .06em; text-transform: uppercase; color: var(--ink-3); }
h1 { margin: 0; font: 760 clamp(30px, 5.4vw, 50px)/1.02 var(--f-display); font-stretch: 82%; letter-spacing: -.012em; text-wrap: balance; }
h2 { margin: 0; font: 700 19px/1.2 var(--f-display); font-stretch: 88%; text-wrap: balance; }
.banner { justify-self: start; margin: 2px 0 0; padding: 5px 11px; border-radius: 4px; border: 1px solid var(--flash);
  background: color-mix(in srgb, var(--flash) 14%, transparent); font: 500 13px/1.45 var(--f-mono); color: var(--ink); max-width: 100%; }
.lede { margin: 0; max-width: 68ch; color: var(--ink-2); }
.lede b { color: var(--ink); font-weight: 600; }

.stage { background: var(--card); border: 1px solid var(--rule); border-radius: 10px; padding: 18px 18px 16px; display: grid; grid-template-columns: minmax(0, 1fr); gap: 16px; min-width: 0; }
.toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 14px; }
.field { display: flex; align-items: center; gap: 8px; min-width: 0; flex: 1 1 320px; }
.field label, .lbl { font: 500 11px/1 var(--f-mono); letter-spacing: .07em; text-transform: uppercase; color: var(--ink-3); }
select, button { font: inherit; color: var(--ink); }
select { min-width: 0; width: 100%; flex: 1; max-width: 520px; padding: 7px 10px; border: 1px solid var(--rule); border-radius: 6px; background: var(--card); font: 13px/1.3 var(--f-mono); }
.btn { display: inline-flex; align-items: center; gap: 7px; padding: 7px 12px; border: 1px solid var(--rule); border-radius: 6px; background: var(--card); cursor: pointer; font: 500 13px/1.2 var(--f-body); }
.btn:hover { border-color: var(--ink-3); }
.btn:disabled { opacity: .45; cursor: default; }
.btn svg { width: 15px; height: 15px; fill: currentColor; }
:focus-visible { outline: 2px solid var(--cobalt); outline-offset: 2px; }
.seg { display: inline-flex; flex-wrap: wrap; align-items: center; gap: 8px; }
.seg .opts { display: inline-flex; border: 1px solid var(--rule); border-radius: 6px; overflow: hidden; }
.seg button { border: 0; border-left: 1px solid var(--rule); background: transparent; padding: 7px 11px; cursor: pointer; font: 500 13px/1.2 var(--f-mono); white-space: nowrap; }
.seg button:first-child { border-left: 0; }
.seg button[aria-checked="true"] { background: var(--cobalt); color: var(--on-cobalt); }
.seg button:disabled { opacity: .4; cursor: not-allowed; }

.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chip { padding: 3px 9px; border-radius: 999px; border: 1px solid var(--rule); font: 500 12px/1.4 var(--f-mono); color: var(--ink-2); }
.chip.good { border-color: var(--good); color: var(--good); }
.chip.warn { border-color: var(--warn); color: var(--warn); }
.chip.bad { border-color: var(--bad); color: var(--bad); }
.chip.note { border-style: dashed; }
.chip.ro { border-color: var(--reopen); color: var(--reopen); border-style: dashed; }

.tiles { display: flex; flex-wrap: wrap; row-gap: 12px; align-items: flex-start; min-height: 84px; }
.slot { position: relative; display: grid; grid-template-rows: 16px auto 15px auto; padding: 0 2px; }
.slot.word { padding-left: 9px; }
.slot.cut { padding-left: 13px; }
.slot.cut::before { content: ""; position: absolute; left: 5px; top: 4px; bottom: 14px; border-left: 1.5px dashed var(--ink-3); opacity: .55; }
.slot.block::before { border-left: 2px solid var(--ink-2); opacity: .9; top: 0; }
.slot.active::after { content: ""; position: absolute; left: 0; right: 0; top: 11px; height: 3px; background: var(--cobalt); opacity: .85; }
.slot.cut.active::after { left: 5px; }
.lane { position: relative; font: 600 9.5px/12px var(--f-mono); letter-spacing: .08em; text-transform: uppercase; color: var(--ink-3); white-space: nowrap; overflow: visible; }
.tile { position: relative; height: 46px; width: calc(var(--w, 3) * 1ch + 18px); min-width: 34px; border-radius: 5px;
  font: 500 15.5px/1 var(--f-mono); display: grid; place-items: center; overflow: hidden;
  background: var(--mask); color: var(--ink); transition: background-color .45s, color .45s, box-shadow .45s; }
.tile::before { content: ""; position: absolute; inset: 0; background-image: var(--noise); background-size: 90px 90px; opacity: .9; transition: opacity .4s; }
.tile .tok { position: relative; white-space: pre; opacity: 0; transition: opacity .35s; }
.tile .bar { position: absolute; left: 6px; right: 6px; bottom: 5px; height: 3px; border-radius: 2px; background: color-mix(in srgb, var(--ink) 14%, transparent); opacity: 0; }
.tile .bar i { position: absolute; left: 0; top: 0; bottom: 0; width: calc(var(--p, 0) * 100%); border-radius: 2px; background: var(--ink-2); transition: width .35s; }
.tile .bar::after { content: ""; position: absolute; left: 90%; top: -2px; bottom: -2px; width: 1px; background: var(--ink); opacity: .55; }
.tile.guess::before { opacity: .55; }
.tile.guess .tok { opacity: calc(.18 + .62 * var(--p, 0)); font-style: italic; }
.tile.guess .bar { opacity: 1; }
.tile.done, .tile.new { background: var(--solid); color: var(--solid-fg); }
.tile.done::before, .tile.new::before { opacity: 0; }
.tile.done .tok, .tile.new .tok { opacity: 1; }
.tile.new { background: var(--flash); color: var(--flash-fg); box-shadow: 0 0 0 3px color-mix(in srgb, var(--flash) 35%, transparent); animation: pop .5s cubic-bezier(.3, 1.5, .5, 1); }
.tile.new .bar { opacity: 1; background: color-mix(in srgb, var(--flash-fg) 20%, transparent); }
.tile.new .bar i { background: var(--flash-fg); }
.tile.new .bar::after { background: var(--flash-fg); }
.tile.first .bar { display: none; }
.tile.reopen { color: var(--reopen); box-shadow: inset 0 0 0 2px var(--reopen); animation: shake .45s ease-out; }
.tile.reopen::before { opacity: .35; }
.tile.reopen .tok { opacity: 1; text-decoration: line-through 2px; }
.tile.live::before { animation: static .42s steps(3) infinite; }
.tile.prompt { background: transparent; border: 1px dashed var(--ink-3); color: var(--ink-3); width: auto; padding: 0 10px; font: 500 12px/1.2 var(--f-mono); }
.tile.prompt::before { opacity: 0; }
.tile.prompt .tok { opacity: 1; }
.tile.prompt.reading { border: 1.5px solid var(--cobalt); color: var(--cobalt); background: var(--cobalt-tint); }
.tile.end { width: auto; min-width: 74px; padding: 0 10px; grid-template-rows: 1fr auto; gap: 3px; padding-block: 6px 6px; }
.tile.end .tok { font-size: 14px; }
.tile.end.part { background: color-mix(in srgb, var(--solid) 55%, var(--mask)); color: var(--solid-fg); }
.tile.end.part::before { opacity: .3; }
.tile.end.part .tok { opacity: 1; }
.tile.end.reopen .tok { text-decoration: none; }
.dots { position: relative; display: flex; flex-wrap: wrap; gap: 2px 5px; max-width: 132px; justify-content: center; }
.dots span { display: flex; gap: 2px; }
.dots i { width: 4px; height: 4px; border-radius: 1px; background: color-mix(in srgb, var(--ink) 22%, transparent); }
.tile.done .dots i, .tile.new .dots i, .tile.part .dots i { background: color-mix(in srgb, currentColor 30%, transparent); }
.dots i.g { background: color-mix(in srgb, var(--ink) 50%, transparent); }
.dots i.w { background: var(--cobalt); }
.dots i.c { background: currentColor; }
.tile.end:not(.done):not(.new):not(.part) .dots i.c { background: var(--ink); }
.dots i.n { background: var(--flash); }
.tile.new .dots i.n { background: var(--flash-fg); }
.dots i.o, .tile .dots i.o { background: var(--reopen); }
.meta { font: 500 10.5px/15px var(--f-mono); color: var(--ink-3); text-align: center; white-space: nowrap; font-variant-numeric: tabular-nums; }
.meta.c { color: var(--ink-2); }
.meta.n { color: var(--ink); font-weight: 600; }
.meta.o { color: var(--reopen); }
.rev { width: 0; justify-self: center; display: flex; justify-content: center; white-space: nowrap; font: 600 10.5px/14px var(--f-mono); }
.tiles.has-rev .rev { min-height: 14px; }
.rev.fix { color: var(--good); }
.rev.brk { color: var(--bad); }
.rev.neu { color: var(--ink-3); }
@keyframes pop { 0% { transform: scale(.8); } 60% { transform: scale(1.09); } 100% { transform: scale(1); } }
@keyframes shake { 0%, 100% { transform: translateX(0); } 25% { transform: translateX(-3px); } 60% { transform: translateX(3px); } }
@keyframes static { from { background-position: 0 0; } to { background-position: 90px 45px; } }

.status { margin: 0; min-height: 2.9em; color: var(--ink-2); font-size: 14px; max-width: 82ch; }
.status b { color: var(--ink); font-weight: 600; }
.status code { font: 500 13px var(--f-mono); color: var(--ink); background: var(--mask); padding: 1px 5px; border-radius: 3px; }
.status .fix { color: var(--good); } .status .brk { color: var(--bad); }

.passes { display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 14px 22px; align-items: center; }
.counter { display: grid; gap: 2px; min-width: 150px; }
.counter .big { font: 760 40px/1 var(--f-display); font-stretch: 80%; font-variant-numeric: tabular-nums; }
.counter .big small { font-size: 18px; color: var(--ink-3); font-weight: 600; }
.counter .sub { font: 500 12px/1.4 var(--f-mono); color: var(--ink-2); }
.strips { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 7px 10px; align-items: center; min-width: 0; }
.strip { display: grid; grid-template-columns: repeat(var(--cols), minmax(3px, 15px)); gap: 2px; min-width: 0; }
.cell { height: 26px; border-radius: 2px; border: 1px solid var(--rule); background: transparent; padding: 0; position: relative; cursor: pointer; overflow: hidden; }
.cell i { position: absolute; left: 0; right: 0; bottom: 0; height: calc(var(--h, 1) * 100%); background: var(--ink-3); opacity: .35; }
.cell.past i, .cell.cur i { opacity: 1; background: var(--solid); }
.cell.cur { border-color: var(--flash); box-shadow: 0 0 0 1.5px var(--flash); }
.cell.cur i { background: var(--flash); }
.cell.f i { background-image: repeating-linear-gradient(135deg, transparent 0 2px, color-mix(in srgb, var(--card) 70%, transparent) 2px 3px); }
.cell.pr { border-style: dashed; }
.cell.pr.past i, .cell.pr.cur i { background: var(--cobalt); }
.cell.rm { border-style: dashed; border-color: var(--reopen); }
.cell.rm i, .cell.rm.past i, .cell.rm.cur i { background: var(--reopen); }
.cell.rm i { opacity: .35; } .cell.rm.past i, .cell.rm.cur i { opacity: 1; }
.cell.ar { cursor: default; height: 14px; }
.cell.ar.past { background: var(--ink-3); border-color: var(--ink-3); }
.strip-l { font: 500 11px/1.25 var(--f-mono); color: var(--ink-3); white-space: nowrap; }

.controls { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
.controls .play { min-width: 92px; justify-content: center; background: var(--ink); color: var(--card); border-color: var(--ink); }
.controls .gap { flex: 1; }
.speed { display: inline-flex; align-items: center; gap: 8px; }
.speed select { flex: 0 0 auto; width: auto; min-width: 64px; }
.keys { font: 12px var(--f-mono); color: var(--ink-3); }

.compare { display: grid; grid-template-columns: minmax(0, 1fr); gap: 0; border-top: 1px solid var(--rule); }
.row { display: grid; grid-template-columns: 230px minmax(0, 1fr) 92px; gap: 6px 18px; padding: 12px 0; border-bottom: 1px solid var(--rule); align-items: baseline; }
.row .who { font: 500 12px/1.4 var(--f-mono); color: var(--ink-3); }
.row .who b { display: block; font: 600 13px/1.4 var(--f-body); color: var(--ink); }
.row .words { font: 17px/1.5 var(--f-body); min-width: 0; overflow-wrap: anywhere; }
.row .words .s, .row .words .i { color: var(--bad); text-decoration: underline 1.5px; text-underline-offset: 3px; }
.row .words .i { text-decoration-style: dotted; }
.row .words .d { color: var(--bad); font-family: var(--f-mono); font-size: 14px; }
.row .w { text-align: right; font: 600 15px/1.4 var(--f-mono); font-variant-numeric: tabular-nums; }
.row .w small { display: block; font: 500 11px/1.3 var(--f-mono); color: var(--ink-3); }
.row.na .words { color: var(--ink-3); font-style: italic; font-size: 14px; }

.legend { display: flex; flex-wrap: wrap; gap: 12px 22px; align-items: center; font-size: 13px; color: var(--ink-2); }
.legend > span { display: inline-flex; align-items: center; gap: 8px; }
.legend .tile { height: 26px; width: 40px; min-width: 0; font-size: 12px; animation: none; }
.legend .tile.guess .tok { opacity: .5; }
.legend .tile .bar { bottom: 3px; }
.legend .rev { width: auto; }
.foot-note { margin: 0; font-size: 13px; color: var(--ink-3); max-width: 82ch; }

.summary { display: grid; grid-template-columns: minmax(0, 1fr); gap: 14px; }
.summary .grid { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr); gap: 22px; align-items: center; }
.plot { position: relative; min-width: 0; }
.plot svg { width: 100%; height: auto; display: block; overflow: visible; }
.plot text { font: 11px var(--f-mono); fill: var(--ink-3); }
.plot .axis { stroke: var(--rule); }
.plot .grid line { stroke: var(--rule); stroke-dasharray: 2 3; }
.plot .ar { stroke: var(--ink-3); stroke-dasharray: 5 4; fill: none; }
.plot circle { fill: var(--cobalt); fill-opacity: .55; stroke: var(--card); stroke-width: 1.5; cursor: pointer; }
.plot circle:hover { fill-opacity: 1; }
.plot .cur-pt { fill: var(--flash); fill-opacity: 1; }
.tip { position: absolute; pointer-events: none; background: var(--solid); color: var(--solid-fg); font: 12px/1.4 var(--f-mono); padding: 5px 8px; border-radius: 4px; white-space: nowrap; transform: translate(-50%, -130%); }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; }
.tbl { overflow-x: auto; }
th, td { text-align: right; padding: 7px 8px; border-bottom: 1px solid var(--rule); font-variant-numeric: tabular-nums; white-space: nowrap; }
th { font: 500 10.5px/1.3 var(--f-mono); letter-spacing: .03em; color: var(--ink-3); text-transform: uppercase; vertical-align: bottom; white-space: normal; min-width: 64px; }
th:first-child, td:first-child { text-align: left; }
tr.cur td { background: var(--cobalt-tint); }
tr.ref td { color: var(--ink-3); }
td { font-family: var(--f-mono); }
footer { font: 12px/1.6 var(--f-mono); color: var(--ink-3); overflow-wrap: anywhere; }

@media (max-width: 720px) {
  .row { grid-template-columns: minmax(0, 1fr) auto; }
  .row .who { grid-column: 1 / -1; }
  .passes { grid-template-columns: minmax(0, 1fr); }
  .summary .grid { grid-template-columns: minmax(0, 1fr); }
  .controls .gap { display: none; }
}
@media (prefers-reduced-motion: reduce) {
  .tile, .tile .tok, .tile .bar i, .tile::before { transition: none; }
  .tile.new, .tile.reopen, .tile.live::before { animation: none; }
}
</style>

<div class="wrap">
  <header>
    <p class="eyebrow">BIT-Experiments · run 3 · Fast-dLLM v2 1.5B block diffusion · T15 attempted speech</p>
    <h1>Brain to text, one pass at a time</h1>
    <p class="banner" id="banner"></p>
    <p class="lede">Each sentence here is decoded from intracortical recordings of attempted speech, with no phoneme labels anywhere. The neural encoder turns the recording into tokens inside the prompt, and the diffusion LLM then fills <b>32-token blocks of masks</b>, one chunk at a time. On every forward pass it guesses every masked slot of the chunk and locks in each guess it is <b>more than 90% sure of</b> (always at least its most confident one). Locked tokens never change, except in the re-mask mode.</p>
  </header>

  <section class="stage" aria-label="Replay">
    <div class="toolbar">
      <div class="field"><label for="trial">Sentence</label><select id="trial"></select></div>
      <button class="btn" id="random" type="button" hidden><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M11 2h3v3l-1.1-1.1-2.6 2.6-1-1L11.9 3 11 2zM2 4h2.6l6.3 7.1L12 10v4H8l1.1-1.1L3.9 6H2V4zm0 8h1.9l1.8-2 1 1.1L4.6 14H2v-2z"/></svg>Random sentence</button>
      <div class="seg"><span class="lbl" id="chunk-l">Chunk size</span><div class="opts" id="chunks" role="radiogroup" aria-labelledby="chunk-l"></div></div>
    </div>
    <div class="chips" id="chips"></div>
    <div class="tiles" id="tiles" aria-live="off"></div>
    <p class="status" id="status" aria-live="polite"></p>
    <div class="passes">
      <div class="counter"><span class="big" id="count"></span><span class="sub" id="countsub"></span></div>
      <div class="strips" id="strips"></div>
    </div>
    <div class="controls">
      <button class="btn" id="restart" type="button" aria-label="Back to the start"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 3h2v10H3zM13 3v10L6 8z"/></svg></button>
      <button class="btn" id="back" type="button" aria-label="Previous step"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M11 3v10L4 8z"/></svg>Back</button>
      <button class="btn play" id="play" type="button"></button>
      <button class="btn" id="fwd" type="button" aria-label="Next step">Next<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M5 3v10l7-5z"/></svg></button>
      <span class="gap"></span>
      <span class="speed"><label class="lbl" for="speed">Speed</label><select id="speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select></span>
      <span class="keys">space · ← →</span>
    </div>
  </section>

  <section class="compare" id="compare" aria-label="Reference and decodes"></section>

  <section class="legend" aria-label="Legend">
    <span><span class="tile"></span>mask</span>
    <span><span class="tile guess" style="--p:.55"><span class="tok">is</span><span class="bar"><i></i></span></span>guess: faint, bar = probability, tick = 0.9</span>
    <span><span class="tile new" style="animation:none"><span class="tok">is</span></span>locked in this pass</span>
    <span><span class="tile done"><span class="tok">is</span></span>locked earlier; #n under it = the pass</span>
    <span id="lg-reopen" hidden><span class="tile reopen" style="animation:none"><span class="tok">is</span></span>re-opened by re-masking</span>
    <span id="lg-rev" hidden><span class="rev fix">→ was</span><span class="rev brk">→ as</span>final frame: the token the model would pick with the whole sentence visible (green fixes a word, red adds an error, grey neither)</span>
    <span><span class="meta">▸</span>a block's first token, predicted left to right from the block before</span>
    <span><span class="meta">┆ │</span>chunk and block boundaries</span>
  </section>
  <p class="foot-note">Everything after the sentence is end-of-text padding; it is folded into one end tile, with a dot per slot to the end of the block. Pass counts include the pass that reads the prompt, as the decoding sweep counts them; the left-to-right strip counts one pass per token, as a standard LLM needs (its first comes from reading the prompt).</p>

  <section class="summary" id="summary" hidden></section>
  <footer id="foot"></footer>
</div>

<script type="application/json" id="data">/*__DATA__*/</script>
<script>
(() => {
"use strict";
const D = JSON.parse(document.getElementById("data").textContent);
const $ = (id) => document.getElementById(id);
const V = D.vocab, EOS = D.eos, BS = D.block;
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
const pct = (x) => Math.round(x * 100) + "%";
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const show = (v) => v === EOS ? "⏹" : (V[v].replace(/^ +/, "").replace(/\n/g, "↵") || "␣");
const wordStart = (v) => /^\s/.test(V[v]);
const firstMode = (t) => (t.runs[D.base] ? D.base : Object.keys(t.runs)[0]);
const plural = (n, w) => `${n} ${w}${n === 1 ? "" : w.endsWith("s") ? "es" : "s"}`;

let S = { t: D.start, sbs: firstMode(D.trials[D.start]), k: 0, playing: false, timer: 0 };
let M = null;

function build(trial, mode) {
  const run = trial.runs[mode], tail = trial.tail, head = BS - tail, sb = parseInt(mode, 10);
  const blockOf = (p) => (p < head ? 0 : 1 + Math.floor((p - head) / BS));
  const chunkOf = (p) => blockOf(p) + ":" + Math.floor((p < head ? p + tail : (p - head) % BS) / sb);
  const blockEnd = (p) => (blockOf(p) === 0 ? head - 1 : head + blockOf(p) * BS - 1);
  const frames = [];
  if (tail) frames.push({ kind: "prompt", list: [] });
  for (const s of run.steps) {
    const list = [];
    if (s[0] === -1) { frames.push({ kind: "first", list: [[s[1], s[2], -1, 1]] }); continue; }
    if (s[0] === -2) { for (let j = 1; j < s.length; j += 3) list.push([s[j], s[j + 1], s[j + 2] / 1000, 0]); frames.push({ kind: "r", list }); continue; }
    for (let j = 0; j < s.length; j += 4) list.push([s[j], s[j + 1], s[j + 2] / 1000, s[j + 3]]);
    frames.push({ kind: "m", list });
  }
  const n = run.n, hasEnd = n < run.out.length, hi = hasEnd ? blockEnd(n) : n - 1;
  const hist = {}, width = {}, listed = [{}], passNo = [0];
  let pass = 0;
  frames.forEach((f, i) => {
    const k = i + 1, map = {};
    if (f.kind !== "r") pass++;
    passNo.push(pass);
    for (const [p, v, pr, c] of f.list) {
      if (f.kind === "r") (hist[p] = hist[p] || []).push({ k, c: false, v, p: pr });
      else {
        map[p] = [v, pr, c];
        if (c) (hist[p] = hist[p] || []).push({ k, c: true, v, p: pr, first: f.kind === "first" });
      }
      if (p < n) width[p] = Math.max(width[p] || 0, show(v).length);
    }
    listed.push(map);
  });
  for (let p = 0; p < n; p++) width[p] = Math.max(width[p] || 0, show(run.out[p]).length);
  const rev = {};
  if (mode === D.revMode && trial.rev) for (const [p, v, pf, pt, d] of trial.rev) rev[p] = { v, pf: pf / 1000, pt: pt / 1000, d };
  return { trial, run, sbs: mode, sb, frames, n, hasEnd, hi, hist, width, listed, passNo, rev, blockOf, chunkOf, head,
    K: frames.length, N: pass };
}

function stateAt(p, k) {
  const h = M.hist[p];
  let e = null;
  if (h) for (const x of h) { if (x.k <= k) e = x; else break; }
  if (e && e.c) return { st: e.k === k ? "new" : "done", v: e.v, pr: e.p, at: e.k, first: e.first };
  if (e && e.k === k) return { st: "reopen", v: e.v, pr: e.p };
  const g = M.listed[k][p];
  if (g) return { st: "guess", v: g[0], pr: g[1] };
  return { st: "" };
}

function layout() {
  const box = $("tiles");
  box.textContent = "";
  box.classList.toggle("has-rev", Object.keys(M.rev).length > 0);
  const { n, run, blockOf, chunkOf, width, trial } = M;
  const el = (cls, html) => { const d = document.createElement("div"); d.className = cls; d.innerHTML = html; return d; };
  M.slots = [];
  M.promptTile = null;
  if (trial.tail) {
    const s = el("slot block", `<span class="lane">block 0</span><div class="tile prompt" title="The last ${trial.tail} of the ${trial.prefix} prompt tokens (neural activity and instruction) share block 0, leaving ${M.head} output slots."><span class="tok">prompt ×${trial.tail}</span></div><span class="meta">${trial.prefix} tok</span><span class="rev"></span>`);
    box.appendChild(s);
    M.promptTile = s.children[1];
  }
  const cutAt = (p) => (p > 0 ? chunkOf(p) !== chunkOf(p - 1) : trial.tail > 0 && trial.tail % M.sb === 0);
  const newBlockAt = (p) => (p === 0 ? !trial.tail : blockOf(p) !== blockOf(p - 1));
  for (let p = 0; p < n; p++) {
    const nb = newBlockAt(p), cls = ["slot", nb ? (p ? "block cut" : "block") : cutAt(p) ? "cut" : "", (p > 0 && wordStart(run.out[p])) ? "word" : ""].join(" ");
    const s = el(cls, `<span class="lane">${nb ? "block " + blockOf(p) : ""}</span><div class="tile" style="--w:${width[p]}"><span class="tok"></span><span class="bar"><i></i></span></div><span class="meta"></span><span class="rev"></span>`);
    box.appendChild(s);
    M.slots.push(s);
  }
  M.endSlot = null;
  if (M.hasEnd) {
    const p = n, nb = newBlockAt(p);
    const groups = [];
    for (let q = n; q <= M.hi; q++) { if (q === n || chunkOf(q) !== chunkOf(q - 1)) groups.push([]); groups[groups.length - 1].push(q); }
    const dots = groups.map((g) => "<span>" + g.map((q) => `<i data-q="${q}"></i>`).join("") + "</span>").join("");
    const s = el(["slot word", nb ? (p ? "block cut" : "block") : cutAt(p) ? "cut" : ""].join(" "), `<span class="lane">${nb ? "block " + blockOf(p) : ""}</span><div class="tile end" title="End of the sentence: positions ${n} to ${M.hi}, the rest of block ${blockOf(n)}"><span class="tok"></span><span class="dots">${dots}</span></div><span class="meta"></span><span class="rev"></span>`);
    box.appendChild(s);
    M.endSlot = s;
    M.dots = [...s.querySelectorAll(".dots i")];
  }
}

const KIND = { prompt: " pr", first: " f", r: " rm", m: "" };
function strips() {
  const box = $("strips"), { K, N, frames } = M, T = M.run.out.length;
  const count = (f) => (f.kind === "r" ? f.list.length : f.list.filter((x) => x[3]).length);
  const cols = Math.max(K, T), maxc = Math.max(1, ...frames.map(count));
  let a = `<span class="strip-l">diffusion<br>${plural(N, "pass")}</span><div class="strip" style="--cols:${cols}">`;
  frames.forEach((f, i) => {
    const c = count(f), k = i + 1, pn = M.passNo[k];
    const what = f.kind === "r" ? `Re-mask: ${c} re-opened (not a pass)` : f.kind === "prompt" ? `Pass ${pn}: reads the prompt` : `Pass ${pn}: ${c} locked in` + (f.kind === "first" ? " (a block's first token)" : "");
    const h = f.kind === "prompt" ? 0.3 : 0.25 + 0.75 * c / maxc;
    a += `<button type="button" class="cell${KIND[f.kind]}" data-k="${k}" style="--h:${h.toFixed(3)}" aria-label="${what}" title="${what}"><i></i></button>`;
  });
  a += `</div><span class="strip-l">left to right<br>${plural(T, "pass")}</span><div class="strip" style="--cols:${cols}">`;
  for (let j = 1; j <= T; j++) a += `<span class="cell ar" title="Pass ${j}: token ${j} of ${T}"></span>`;
  box.innerHTML = a + "</div>";
  M.cells = [...box.querySelectorAll(".cell:not(.ar)")];
  M.arcells = [...box.querySelectorAll(".cell.ar")];
}

function setTile(tile, cls, restart) {
  if (tile.className === cls) return;
  if (restart) { tile.className = "tile"; void tile.offsetWidth; }
  tile.className = cls;
}

function setRev(el, p, last) {
  const r = M.rev[p];
  if (!r || !last) { el.textContent = ""; el.className = "rev"; el.removeAttribute("title"); return; }
  el.className = "rev " + (r.d < 0 ? "fix" : r.d > 0 ? "brk" : "neu");
  el.textContent = "→ " + show(r.v) + " " + pct(r.pt);
  el.title = `With the whole sentence visible, the model prefers "${show(r.v)}" (p = ${r.pt.toFixed(3)}) to "${show(M.run.out[p])}" (p = ${r.pf.toFixed(3)}). Swapping only this token ${r.d < 0 ? "removes " + plural(-r.d, "word error") : r.d > 0 ? "adds " + plural(r.d, "word error") : "leaves the word errors unchanged"}.`;
}

function frame(k) {
  S.k = k = Math.max(0, Math.min(M.K, k));
  const f = k ? M.frames[k - 1] : null;
  const active = f && (f.kind === "m" || f.kind === "first") ? M.chunkOf(f.list[0][0]) : null;
  const last = k === M.K;
  if (M.promptTile) M.promptTile.classList.toggle("reading", !!f && f.kind === "prompt");
  for (let p = 0; p < M.n; p++) {
    const s = M.slots[p], tile = s.children[1], meta = s.children[2], t = tile.firstChild, x = stateAt(p, k);
    let txt = "", m = "", mc = "meta", pv = 0, title = "mask";
    if (x.st === "done" || x.st === "new") {
      txt = show(x.v); m = (x.first ? "▸" : "") + "#" + M.passNo[x.at]; mc += x.st === "new" ? " n" : " c";
      pv = x.st === "new" ? Math.max(0, x.pr) : 0;
      title = `"${txt}", locked in at pass ${M.passNo[x.at]}` + (x.pr > 0 ? ` (p = ${x.pr.toFixed(3)})` : "");
    } else if (x.st === "reopen") {
      txt = show(x.v); m = "re-mask"; mc += " o";
      title = `"${txt}" was locked in at p = ${x.pr.toFixed(3)}, below the re-mask threshold, so it is a mask again`;
    } else if (x.st === "guess") {
      txt = show(x.v); pv = x.pr; m = pct(pv); title = `guess "${txt}" at p = ${pv.toFixed(3)}`;
    }
    const st = x.st === "new" && x.first ? "new first" : x.st;
    const live = !x.st && active && M.chunkOf(p) === active;
    setTile(tile, "tile" + (st ? " " + st : "") + (live ? " live" : ""), x.st === "new" || x.st === "reopen");
    t.textContent = txt;
    tile.style.setProperty("--p", pv);
    tile.title = title;
    meta.textContent = m; meta.className = mc;
    s.classList.toggle("active", !!active && M.chunkOf(p) === active);
    setRev(s.children[3], p, last);
  }
  if (M.endSlot) {
    const tile = M.endSlot.children[1], meta = M.endSlot.children[2], t = tile.firstChild;
    let eos = 0, listedAny = false;
    M.dots.forEach((d) => {
      const x = stateAt(+d.dataset.q, k);
      let c = "";
      if (x.st === "done" || x.st === "new") { c = x.st === "new" ? "n" : "c"; if (x.v === EOS) eos++; }
      else if (x.st === "reopen") c = "o";
      else if (x.st === "guess") { c = x.v === EOS ? "g" : "w"; listedAny = true; }
      d.className = c;
    });
    const xe = stateAt(M.n, k);
    let st = "", txt, m = "";
    if ((xe.st === "done" || xe.st === "new") && xe.v === EOS) { st = xe.st; txt = "⏹ ×" + eos; m = "end #" + M.passNo[xe.at]; }
    else if (xe.st === "reopen") { st = "reopen"; txt = "⏹ ×" + eos; m = "re-mask"; }
    else if (eos) { st = "part"; txt = "⏹ ×" + eos; m = xe.st === "guess" ? pct(xe.pr) : ""; }
    else if (listedAny) { st = "guess"; txt = xe.st === "guess" ? (xe.v === EOS ? "⏹ ?" : show(xe.v) + " …") : "×" + M.dots.length; m = xe.st === "guess" ? pct(xe.pr) : ""; }
    else txt = "×" + M.dots.length;
    const inChunk = !!active && M.dots.some((d) => M.chunkOf(+d.dataset.q) === active);
    setTile(tile, "tile end" + (st ? " " + st : "") + (!st && inChunk ? " live" : ""), st === "new" || st === "reopen");
    t.textContent = txt;
    tile.style.setProperty("--p", xe.st === "guess" ? xe.pr : 0);
    meta.textContent = m;
    meta.className = "meta" + (st === "new" ? " n" : st === "done" ? " c" : st === "reopen" ? " o" : "");
    M.endSlot.classList.toggle("active", inChunk);
    setRev(M.endSlot.children[3], M.n, last);
  }
  M.cells.forEach((c, i) => { c.className = "cell" + KIND[M.frames[i].kind] + (i + 1 < k ? " past" : i + 1 === k ? " cur" : ""); });
  const pn = M.passNo[k];
  M.arcells.forEach((c, i) => c.classList.toggle("past", i < pn));
  $("count").innerHTML = `${pn}<small> / ${M.N}</small>`;
  $("countsub").textContent = `passes for ${M.run.out.length} tokens`;
  $("status").innerHTML = statusText(k);
  $("play").innerHTML = S.playing ? '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 3h3v10H4zM9 3h3v10H9z"/></svg>Pause' : '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5v11L13 8z"/></svg>' + (k >= M.K ? "Replay" : "Play");
  $("back").disabled = k === 0; $("fwd").disabled = k >= M.K;
}

function statusText(k) {
  const T = M.run.out.length, words = M.trial.rw.split(" ").length;
  if (k === 0) return `<b>Before the first pass.</b> Every output slot is a mask; the prompt holds the neural activity. The model will fill chunks of ${plural(M.sb, "slot")}${M.sbs.endsWith("r") ? ", then re-open its least confident tokens" : ""}.`;
  const f = M.frames[k - 1], pn = M.passNo[k];
  let s;
  if (f.kind === "prompt") {
    s = `<b>Pass ${pn}: reading the prompt.</b> The model runs the ${M.trial.prefix}-token prompt (neural activity and instruction) once and caches it. Its last ${M.trial.tail} tokens share block 0 with the ${plural(M.head, "output slot")}, so this pass writes nothing yet.`;
  } else if (f.kind === "first") {
    const b = M.blockOf(f.list[0][0]);
    s = b === 0 ? `<b>Pass ${pn}: reading the prompt.</b> The prompt fills whole blocks, so this pass also predicts block 0's first token, <code>${esc(show(f.list[0][1]))}</code>, left to right.`
      : `<b>Pass ${pn}: block ${b} opens.</b> Its first token, <code>${esc(show(f.list[0][1]))}</code>, is predicted left to right from the finished block ${b - 1} and locked in at once.`;
  } else if (f.kind === "r") {
    const b = M.blockOf(f.list[0][0]);
    const round = M.frames.slice(0, k).filter((g) => g.kind === "r" && M.blockOf(g.list[0][0]) === b).length;
    const thr = D.remask[M.sbs] ? D.remask[M.sbs].threshold : "";
    const items = f.list.map((x) => `<code>${esc(show(x[1]))}</code> ${x[2].toFixed(2)}`);
    s = `<b>Re-mask round ${round}, block ${b}.</b> The block is filled, so every token it locked in below ${thr} goes back to a mask: ${items.join(", ")}. The same sampler re-fills them with the rest of the block visible. This step is not a forward pass.`;
  } else {
    const lock = f.list.filter((x) => x[3]), ps = f.list.map((x) => x[0]), p0 = ps[0];
    const lo = Math.min(...ps), hi = Math.max(...ps);
    const shown = lock.filter((x) => x[1] !== EOS).map((x) => `<code>${esc(show(x[1]))}</code>`);
    const ends = lock.length - shown.length;
    let lockTxt = shown.length > 8 ? shown.slice(0, 8).join(" ") + ` and ${shown.length - 8} more` : shown.join(" ");
    if (ends) lockTxt += (shown.length ? " + " : "") + `${plural(ends, "end token")} <code>⏹</code>`;
    const why = lock.length === 1 && lock[0][2] <= D.threshold ? " (none cleared 0.9, so only its most confident guess)" : "";
    const where = lo === hi ? `position ${lo}` : `positions ${lo}–${hi}`;
    s = `<b>Pass ${pn}</b> · block ${M.blockOf(p0)}, chunk ${+M.chunkOf(p0).split(":")[1] + 1} of ${BS / M.sb} · ${plural(f.list.length, "masked slot")} guessed (${where}), ${lock.length} locked in${why}: ${lockTxt}.`;
  }
  if (k === M.K) {
    s += ` <b>Done:</b> ${plural(M.N, "pass")} for ${T} tokens (${plural(words, "word")} + end); left to right needs ${T}.`;
    const rv = Object.entries(M.rev);
    if (rv.length) {
      const items = rv.map(([p, r]) => `<code>${esc(show(M.run.out[+p]))}</code> → <code>${esc(show(r.v))}</code> (${r.d < 0 ? '<span class="fix">fixes ' + plural(-r.d, "word") + "</span>" : r.d > 0 ? '<span class="brk">adds ' + plural(r.d, "error") + "</span>" : "same errors"})`);
      s += ` With the whole sentence visible, the model would change ${plural(rv.length, "token")}: ${items.join(", ")}.`;
    }
  }
  return s;
}

function words(rw, hw, ops) {
  const r = rw ? rw.split(" ") : [], h = hw ? hw.split(" ") : [];
  let i = 0, j = 0;
  const out = [];
  for (const o of ops) {
    if (o === "=") { out.push(esc(h[j])); i++; j++; }
    else if (o === "s") { out.push(`<span class="s" title="reference: ${esc(r[i])}">${esc(h[j])}</span>`); i++; j++; }
    else if (o === "d") { out.push(`<span class="d" title="missing: ${esc(r[i])}">‸</span>`); i++; }
    else { out.push(`<span class="i" title="extra word">${esc(h[j])}</span>`); j++; }
  }
  return out.join(" ");
}

function compare() {
  const t = M.trial, r = M.run, nref = t.rw.split(" ").length;
  const wer = (x) => `<span class="w">${pct(x.wer)}<small>${x.errs} of ${nref}</small></span>`;
  const who = (k) => (k.endsWith("r") ? "same model, re-masking sampler" : k === M.sbs ? "Fast-dLLM v2 1.5B, this replay" : "same model, other chunk size");
  let h = `<div class="row"><span class="who"><b>Reference</b>what was attempted</span><span class="words">${esc(t.rw)}</span><span class="w"><small>WER</small></span></div>`;
  h += `<div class="row"><span class="who"><b>Diffusion, chunk ${esc(D.labels[M.sbs])}</b>${who(M.sbs)}</span><span class="words">${words(t.rw, r.hw, r.ops)}</span>${wer(r)}</div>`;
  for (const k of D.modes) {
    if (k === M.sbs || !t.runs[k] || t.runs[k].hw === r.hw) continue;
    h += `<div class="row"><span class="who"><b>Diffusion, chunk ${esc(D.labels[k])}</b>${who(k)}</span><span class="words">${words(t.rw, t.runs[k].hw, t.runs[k].ops)}</span>${wer(t.runs[k])}</div>`;
  }
  if (t.l2r) h += `<div class="row"><span class="who"><b>Left to right</b>Qwen2.5-1.5B, same pipeline</span><span class="words">${words(t.rw, t.l2r.hw, t.l2r.ops)}</span>${wer(t.l2r)}</div>`;
  else h += `<div class="row na"><span class="who"><b>Left to right</b>Qwen2.5-1.5B</span><span class="words">No matching decode for this trial.</span><span></span></div>`;
  $("compare").innerHTML = h;
}

function chips() {
  const t = M.trial, r = M.run, f = r.f, sev = r.errs === 0 ? "good" : r.wer <= 0.34 ? "warn" : "bad";
  const c = [`<span class="chip">${D.split} #${t.i}</span>`,
    `<span class="chip ${sev}">${r.errs === 0 ? "exact" : "WER " + pct(r.wer)}</span>`,
    `<span class="chip">${t.seen ? "seen in training" : "unseen sentence"}</span>`,
    `<span class="chip">${plural(M.N, "pass")} · ${r.out.length} tokens</span>`];
  const lastBlock = M.blockOf(Math.max(0, M.n - 1));
  if (lastBlock > 0) c.push(`<span class="chip note">spans ${lastBlock + 1} blocks</span>`);
  if (f[0]) c.push(`<span class="chip note">locked out of left-to-right order</span>`);
  if (f[1]) c.push(`<span class="chip note">end fixed before the last word</span>`);
  if (f[2]) c.push(`<span class="chip note">end fixed the first time it was guessed</span>`);
  if (f[3]) c.push(`<span class="chip ro">${plural(f[3], "token")} re-opened</span>`);
  const fixes = Object.values(M.rev).filter((x) => x.d < 0).length;
  if (fixes) c.push(`<span class="chip note">${plural(fixes, "revision")} would fix a word</span>`);
  $("chips").innerHTML = c.join("");
}

function load(t, sbs, autoplay) {
  stop();
  S.t = t; S.sbs = sbs;
  M = build(D.trials[t], sbs);
  layout(); strips(); chips(); compare();
  $("chunks").querySelectorAll("button").forEach((b) => {
    b.setAttribute("aria-checked", String(b.dataset.s === sbs));
    b.disabled = !D.trials[t].runs[b.dataset.s];
    b.tabIndex = b.dataset.s === sbs ? 0 : -1;
  });
  if ($("trial").value !== String(t)) ensureOption(t);
  summaryMark();
  frame(reduce && !autoplay ? M.K : 0);
  if (autoplay && !reduce) play();
  else if (reduce) frame(M.K);
}

function ensureOption(t) {
  let o = $("trial").querySelector(`option[value="${t}"]`);
  if (!o) {
    let g = $("trial").querySelector("optgroup[data-r]");
    if (!g) { g = document.createElement("optgroup"); g.label = "Random draws"; g.dataset.r = "1"; $("trial").appendChild(g); }
    o = optionFor(t); g.appendChild(o);
  }
  $("trial").value = String(t);
}

function optionFor(j) {
  const t = D.trials[j], r = t.runs[firstMode(t)];
  const ref = t.ref.length > 46 ? t.ref.slice(0, 44) + "…" : t.ref;
  const o = document.createElement("option");
  o.value = String(j);
  o.textContent = `#${t.i}  ${r.errs === 0 ? "exact" : pct(r.wer)}  “${ref}”${t.seen ? "  · seen" : ""}`;
  return o;
}

const DUR = 1150;
function delay() {
  const f = M.frames[S.k];  // the frame about to show
  return (S.k === 0 ? 500 : f && f.kind === "r" ? DUR * 1.3 : DUR) / +$("speed").value;
}
function tick() {
  if (!S.playing) return;
  if (S.k >= M.K) { stop(); frame(S.k); return; }
  frame(S.k + 1);
  S.timer = setTimeout(tick, S.k >= M.K ? 300 : delay());
}
function play() { if (S.k >= M.K) frame(0); S.playing = true; frame(S.k); clearTimeout(S.timer); S.timer = setTimeout(tick, delay()); }
function stop() { S.playing = false; clearTimeout(S.timer); }
function toggle() { if (S.playing) { stop(); frame(S.k); } else play(); }
function step(d) { stop(); frame(S.k + d); }

function randomTrial() {
  const pool = D.trials.map((t, j) => j).filter((j) => D.trials[j].runs[S.sbs] && j !== S.t);
  if (pool.length) load(pool[Math.floor(Math.random() * pool.length)], S.sbs, true);
}

function summary() {
  const n = D.pts[D.base].length;
  if (n < 12) return;
  const sec = $("summary");
  sec.hidden = false;
  const nw = D.trials.reduce((a, t) => a + t.rw.split(" ").length, 0), all = D.trials.length === n;
  let tb = `<div class="tbl"><table><thead><tr><th>Chunk size</th><th>Mean passes</th><th>Passes per token</th><th>Out-of-order commits</th><th>End fixed before last word</th><th>End fixed on first guess</th><th>WER</th></tr></thead><tbody>`;
  for (const s of D.modes) {
    const P = D.pts[s], sum = (f) => P.reduce((a, p) => a + f(p), 0);
    tb += `<tr data-s="${s}"><td>${esc(D.labels[s])}${s === D.base ? " (trained)" : ""}</td><td>${(sum((p) => p[1]) / P.length).toFixed(2)}</td><td>${(sum((p) => p[1]) / sum((p) => p[0])).toFixed(2)}</td><td>${pct(sum((p) => p[2]) / P.length)}</td><td>${pct(sum((p) => p[3]) / P.length)}</td><td>${pct(sum((p) => p[4]) / P.length)}</td><td>${all ? (100 * sum((p) => p[5]) / nw).toFixed(2) + "%" : "–"}</td></tr>`;
  }
  if (D.l2rWer !== null) tb += `<tr class="ref"><td>Left to right (Qwen2.5-1.5B)</td><td></td><td>1.00</td><td>0%</td><td>0%</td><td></td><td>${(100 * D.l2rWer).toFixed(2)}%</td></tr>`;
  tb += `</tbody></table></div>`;
  sec.innerHTML = `<h2>All ${n} ${D.split} sentences</h2>${tb}<div class="grid"><div class="plot" id="plot"></div><p class="foot-note">Each dot is a group of sentences with the same length and pass count at the selected chunk size (bigger dot, more sentences); below the dashed line, diffusion needed fewer passes than left-to-right decoding would. Click a dot to replay one of its sentences; the amber dot is the current one. Passes include the one that reads the prompt; tokens include the end token; re-mask rounds are not passes.</p></div>`;
  scatter();
}

function scatter() {
  const box = document.getElementById("plot");
  if (!box) return;
  const P = D.pts[S.sbs] || [], W = 460, H = 300, L = 40, B = 34, R = 10, T = 12;
  const maxX = Math.max(4, ...P.map((p) => p[0])), maxY = Math.max(4, ...P.map((p) => p[1]), maxX);
  const x = (v) => L + (v / maxX) * (W - L - R), y = (v) => H - B - (v / maxY) * (H - B - T);
  const cells = new Map();
  for (const p of P) { const key = p[0] + "," + p[1]; if (!cells.has(key)) cells.set(key, []); cells.get(key).push(p); }
  const big = Math.max(...[...cells.values()].map((c) => c.length));
  const ticks = (m) => { const out = []; const st = m > 30 ? 10 : 5; for (let v = 0; v <= m; v += st) out.push(v); return out; };
  let g = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Passes against tokens per sentence, chunk size ${esc(D.labels[S.sbs])}"><g class="grid">`;
  for (const v of ticks(maxY)) g += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${v}</text>`;
  for (const v of ticks(maxX)) g += `<text x="${x(v)}" y="${H - B + 16}" text-anchor="middle">${v}</text>`;
  const e = Math.min(maxX, maxY);
  g += `</g><line class="axis" x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}"/>`;
  g += `<line class="ar" x1="${x(0)}" y1="${y(0)}" x2="${x(e)}" y2="${y(e)}"/>`;
  g += `<text x="${x(e * 0.9) - 10}" y="${y(e * 0.9) - 4}" text-anchor="end">left to right: 1 pass per token</text>`;
  g += `<text x="${(L + W - R) / 2}" y="${H - 2}" text-anchor="middle">tokens in the sentence (with end)</text>`;
  g += `<text transform="translate(11 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">diffusion passes (chunk ${esc(D.labels[S.sbs])})</text>`;
  for (const [key, c] of cells) {
    const [a, b] = key.split(",").map(Number);
    g += `<circle data-key="${key}" cx="${x(a)}" cy="${y(b)}" r="${(2.5 + 7.5 * Math.sqrt(c.length / big)).toFixed(1)}"/>`;
  }
  if (M && M.sbs === S.sbs) g += `<circle class="cur-pt" cx="${x(M.run.out.length)}" cy="${y(M.N)}" r="4.5" pointer-events="none"/>`;
  box.innerHTML = g + `</svg><div class="tip" hidden></div>`;
  const tip = box.querySelector(".tip");
  box.querySelectorAll("circle[data-key]").forEach((el) => {
    const c = cells.get(el.dataset.key), [a, b] = el.dataset.key.split(",");
    el.addEventListener("mouseenter", () => {
      tip.hidden = false; tip.textContent = `${a} tokens, ${b} passes: ${plural(c.length, "sentence")}`;
      const bb = box.getBoundingClientRect(), eb = el.getBoundingClientRect();
      tip.style.left = (eb.left + eb.width / 2 - bb.left) + "px"; tip.style.top = (eb.top - bb.top) + "px";
    });
    el.addEventListener("mouseleave", () => { tip.hidden = true; });
    el.addEventListener("click", () => {
      const js = c.map((p) => p[6]).filter((j) => j >= 0 && D.trials[j] && D.trials[j].runs[S.sbs]);
      if (js.length) load(js[Math.floor(Math.random() * js.length)], S.sbs, true);
    });
  });
}
function summaryMark() {
  if ($("summary").hidden) return;
  $("summary").querySelectorAll("tr[data-s]").forEach((tr) => tr.classList.toggle("cur", tr.dataset.s === S.sbs));
  scatter();
}

function init() {
  $("banner").textContent = D.label;
  $("lg-reopen").hidden = !Object.keys(D.remask).length;
  $("lg-rev").hidden = !D.revMode;
  const sel = $("trial");
  if (D.picked.length === D.trials.length) D.picked.forEach((j) => sel.appendChild(optionFor(j)));
  else {
    const bands = [["Exact", (r) => r.errs === 0], ["Close (WER ≤ 34%)", (r) => r.errs > 0 && r.wer <= 0.34], ["Partly wrong", (r) => r.wer > 0.34 && r.wer < 0.6], ["Mostly wrong (WER ≥ 60%)", (r) => r.wer >= 0.6]];
    for (const [name, test] of bands) {
      const js = D.picked.filter((j) => test(D.trials[j].runs[firstMode(D.trials[j])]));
      if (!js.length) continue;
      const g = document.createElement("optgroup"); g.label = name;
      js.forEach((j) => g.appendChild(optionFor(j))); sel.appendChild(g);
    }
  }
  sel.addEventListener("change", () => load(+sel.value, D.trials[+sel.value].runs[S.sbs] ? S.sbs : firstMode(D.trials[+sel.value]), true));
  $("random").hidden = !D.random;
  $("random").addEventListener("click", randomTrial);
  const ch = $("chunks");
  for (const s of D.modes) {
    const b = document.createElement("button"), sb = parseInt(s, 10), rm = D.remask[s];
    b.type = "button"; b.dataset.s = s; b.setAttribute("role", "radio"); b.textContent = D.labels[s];
    b.title = rm ? `chunk ${sb}, then re-open every token locked in below ${rm.threshold} and re-fill (up to ${rm.rounds} rounds per block)` : sb === 1 ? "one token per pass, strictly left to right" : sb === BS ? "the whole block at once" : s === D.base ? "the trained setting" : `${sb} slots at a time`;
    b.addEventListener("click", () => load(S.t, s, true));
    ch.appendChild(b);
  }
  ch.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    e.preventDefault(); e.stopPropagation();
    const ok = D.modes.filter((s) => D.trials[S.t].runs[s]);
    const i = ok.indexOf(S.sbs) + (e.key === "ArrowRight" ? 1 : -1);
    if (i >= 0 && i < ok.length) { load(S.t, ok[i], true); ch.querySelector(`[data-s="${ok[i]}"]`).focus(); }
  });
  $("play").addEventListener("click", toggle);
  $("back").addEventListener("click", () => step(-1));
  $("fwd").addEventListener("click", () => step(1));
  $("restart").addEventListener("click", () => { stop(); frame(0); });
  $("strips").addEventListener("click", (e) => { const c = e.target.closest("[data-k]"); if (c) { stop(); frame(+c.dataset.k); } });
  $("speed").addEventListener("change", () => { if (S.playing) { clearTimeout(S.timer); S.timer = setTimeout(tick, delay()); } });
  document.addEventListener("keydown", (e) => {
    if (e.target.closest("select, input, textarea") || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === " " && !e.target.closest("button")) { e.preventDefault(); toggle(); }
    else if (e.key === "ArrowRight") { e.preventDefault(); step(1); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); step(-1); }
  });
  const m = D.meta;
  $("foot").innerHTML = `Traces: ${m.files.map(esc).join(", ")} (scripts/decoding_order_sweep.py)${m.revisions ? " · revision check: " + esc(m.revisions) : ""} · run ${esc(m.model || "")} · checkpoint ${esc(m.checkpoint || "")} · sampler: block ${BS}, threshold ${D.threshold} · left-to-right decodes: ${esc(m.l2r || "none")} · every replay verified against its trace's output · built ${esc(m.built)}`;
  summary();
  load(S.t, S.sbs, true);
}
init();
})();
</script>
"""


if __name__ == "__main__":
    sys.exit(main())

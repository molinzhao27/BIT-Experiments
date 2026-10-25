#!/usr/bin/env python
"""Build or extend BIT's noise-floor JSON (PerfLab CONTRACT.md) from compare.py results.

    python noise_floor.py --out <PerfLab>/noise_floor/bit.json --pair <name> --kind same_machine|cross_machine \
        CMP.json [CMP.json ...] [--merge] [--provisional] [--note TEXT] [--baseline-commit SHA]

Each CMP.json is one compare.py result (two runs of one case). A pair groups the comparisons of one machine pair
(e.g. two repeats on one machine, or one machine's runs against another's). --merge keeps the file's other pairs, so
the cross-machine pair can be added later. Per pair and output ("<case>/<output>"): the largest max_abs, max_rel and
rel_l2 over its comparisons, whether all were exact, and for discrete outputs the largest numbers of mismatches,
near-ties and other mismatches in one comparison. The acceptance floor ("floor") is the maximum over the
cross-machine pairs if any exist, else over the same-machine pairs (then "floor_basis" says so: it is a lower bound).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import golden_lib as gl  # noqa: E402

SCHEMA = 1


def fold(acc: dict | None, v: dict) -> dict:
    """The running worst case of one output over comparisons."""
    kind = v.get("kind", "discrete")
    if acc is None:
        acc = {"kind": kind, "n_comparisons": 0, "all_exact": True}
        if kind == "continuous":
            acc.update(max_abs=0.0, max_rel=0.0, rel_l2=0.0, comparable=True)
        else:
            acc.update(mismatches=0)
            if "near_tie" in v:
                acc.update(near_tie=0, other=0)
    acc["n_comparisons"] += 1
    acc["all_exact"] &= bool(v.get("exact", False))
    if kind == "continuous":
        for k in ("max_abs", "max_rel", "rel_l2"):
            acc[k] = max(acc[k], float(v.get(k) or 0.0))
        acc["comparable"] &= bool(v.get("comparable", True))
    else:
        acc["mismatches"] = max(acc["mismatches"], int(v.get("mismatches") or 0))
        if "near_tie" in v:
            acc["near_tie"] = max(acc.get("near_tie", 0), int(v["near_tie"]))
            acc["other"] = max(acc.get("other", 0), int(v["other"]))
    return acc


def pair_entry(name: str, kind: str, cmps: list[tuple[str, dict]]) -> dict:
    outputs, comparisons, machines, gpus, info = {}, [], set(), {}, {}
    for path, c in cmps:
        if not c.get("comparable"):
            raise SystemExit(f"{path}: not comparable ({c.get('problems')})")
        case = c["baseline"]["case"]
        for who in ("baseline", "candidate"):
            machines.add(c[who]["machine"])
            gpus[c[who]["machine"]] = c[who]["gpu"]
        comparisons.append({"file": str(path), "case": case, "shapes": c.get("shapes"),
                            "baseline": c["baseline"]["dir"], "candidate": c["candidate"]["dir"],
                            "same_code": c.get("same_code"), "same_harness": c.get("same_harness"),
                            "n_outputs": c["summary"]["n_outputs"], "n_exact": c["summary"]["n_exact"]})
        for k, v in c["outputs"].items():
            if v.get("kind") == "info":  # per-process RNGs BIT does not seed: recorded, never judged
                info.setdefault(f"{case}/{k}", []).append(bool(v.get("exact")))
                continue
            outputs[f"{case}/{k}"] = fold(outputs.get(f"{case}/{k}"), v)
    if kind == "same_machine" and len(machines) != 1:
        raise SystemExit(f"--kind same_machine but the comparisons span {sorted(machines)}")
    if kind == "cross_machine" and len(machines) < 2:
        raise SystemExit(f"--kind cross_machine but only {sorted(machines)}")
    return {"kind": kind, "machines": sorted(machines), "gpus": gpus, "comparisons": comparisons, "outputs": outputs,
            "informational": info}


def acceptance(pairs: dict) -> tuple[dict, str]:
    cross = [p for p in pairs.values() if p["kind"] == "cross_machine"]
    use = cross or list(pairs.values())
    basis = ("cross-machine: " if cross else "same-machine only (a lower bound; provisional): ") + \
        ", ".join(n for n, p in pairs.items() if p in use)
    floor = {}
    for p in use:
        for k, v in p["outputs"].items():
            f = floor.get(k)
            if f is None:
                floor[k] = dict(v)
                continue
            f["n_comparisons"] += v["n_comparisons"]
            f["all_exact"] &= v["all_exact"]
            for x in ("max_abs", "max_rel", "rel_l2", "mismatches", "near_tie", "other"):
                if x in v:
                    f[x] = max(f.get(x, 0), v[x])
    return floor, basis


def machines_from(cmps) -> dict:
    out = {}
    for path, c in cmps:
        for who in ("baseline", "candidate"):
            d = Path(c[who]["dir"])
            m = c[who]["machine"]
            info = {"gpu": c[who]["gpu"], "torch": c[who].get("torch"), "cuda": c[who].get("cuda")}
            p = d / "main" / "phase.json"
            if p.is_file():
                env = gl.read_json(p).get("env", {})
                info.update(gpu_sm_count=env.get("gpu_sm_count"), gpu_capability=env.get("gpu_capability"),
                            gpu_memory_gib=env.get("gpu_memory_gib"), nvidia_smi=env.get("nvidia_smi"),
                            transformers=env.get("transformers"), cudnn=env.get("cudnn"))
            out.setdefault(m, info)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("comparisons", nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--pair", required=True, help="pair name, e.g. <machine>~<machine>")
    ap.add_argument("--kind", required=True, choices=("same_machine", "cross_machine"))
    ap.add_argument("--merge", action="store_true", help="keep the existing file's other pairs")
    ap.add_argument("--provisional", action="store_true")
    ap.add_argument("--final", action="store_true", help="clear the provisional flag")
    ap.add_argument("--note", action="append", default=[])
    ap.add_argument("--baseline-note", help="where the baseline trees came from (their files are hashed anyway)")
    args = ap.parse_args(argv)
    cmps = [(str(p), gl.read_json(p)) for p in args.comparisons]
    doc = gl.read_json(args.out) if (args.merge and args.out.is_file()) else {}
    if doc and doc.get("schema") != SCHEMA:
        raise SystemExit(f"{args.out}: schema {doc.get('schema')} != {SCHEMA}")
    taus = {(c["near_tie"]["tau_logit"], c["near_tie"]["tau_prob"]) for _, c in cmps}
    if doc.get("near_tie"):
        taus.add((doc["near_tie"]["tau_logit"], doc["near_tie"]["tau_prob"]))
    if len(taus) != 1:
        raise SystemExit(f"comparisons use different near-tie thresholds: {sorted(taus)}")
    tau_logit, tau_prob = taus.pop()
    # one baseline tree per case (the AR and the diffusion trees differ); all runs of a case must share it
    codes = {}
    for _, c in cmps:
        for w in ("baseline", "candidate"):
            codes.setdefault(c["baseline"]["case"], set()).add(c[w]["code_sha256"])
    for case, entry in (doc.get("baseline_code") or {}).items():
        if isinstance(entry, dict) and entry.get("training_files_sha256") and case in codes:
            codes[case].add(entry["training_files_sha256"])
    bad = {k: sorted(v) for k, v in codes.items() if len(v) != 1}
    if bad:
        raise SystemExit(f"the floor's runs of a case must all use the same baseline code; got {bad}")
    harness = {c[w]["harness_sha256"] for _, c in cmps for w in ("baseline", "candidate")}
    if doc.get("harness", {}).get("combined_sha256"):
        harness.add(doc["harness"]["combined_sha256"])
    if len(harness) != 1:
        raise SystemExit(f"the floor's runs must all use the same harness; got {sorted(harness)}")
    first = gl.read_json(Path(cmps[0][1]["baseline"]["dir"]) / "manifest.json")
    doc.setdefault("project", "bit")
    doc["schema"] = SCHEMA
    doc["near_tie"] = {"tau_logit": tau_logit, "tau_prob": tau_prob}
    doc["harness"] = {"combined_sha256": harness.pop(), "git": first["harness"].get("git"),
                      "source": first["harness"].get("source"), "files_sha256": first["harness"].get("files_sha256")}
    bc = doc.get("baseline_code") or {}
    for path, c in cmps:
        case = c["baseline"]["case"]
        man = gl.read_json(Path(c["baseline"]["dir"]) / "manifest.json")
        bc[case] = {"training_files_sha256": next(iter(codes[case])), "tree": man["code_root"].get("path"),
                    "layout": man["code_root"].get("layout"), "git": man["code_root"].get("git"),
                    "n_files": len(man["code_root"].get("files", {}))}
    doc["baseline_code"] = bc
    if args.baseline_note:
        doc["baseline_note"] = args.baseline_note
    doc.setdefault("machines", {}).update(machines_from(cmps))
    doc.setdefault("pairs", {})[args.pair] = pair_entry(args.pair, args.kind, cmps)
    doc["pairs"][args.pair]["added_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    doc["floor"], doc["floor_basis"] = acceptance(doc["pairs"])
    if args.final:
        doc["provisional"] = False
    elif args.provisional or "provisional" not in doc:
        doc["provisional"] = bool(args.provisional) or not any(p["kind"] == "cross_machine"
                                                               for p in doc["pairs"].values())
    doc["notes"] = doc.get("notes", []) + args.note
    doc["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    gl.write_json(args.out, doc)
    n_exact = sum(1 for v in doc["floor"].values() if v["all_exact"])
    print(f"{args.out}: pair {args.pair} ({args.kind}, {len(cmps)} comparisons); floor basis: {doc['floor_basis']}; "
          f"{n_exact}/{len(doc['floor'])} outputs exact in the floor; provisional {doc['provisional']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Compare two golden-harness runs of the same case (run.py outputs): per output, max abs / max rel differences for
continuous values, exactness flags, and discrete mismatches (batch indices, counts, digests, argmaxes, decoded text
and token ids), each decode mismatch classified as a near-tie or not from the BASELINE's top-1 / top-2 margin.

    python compare.py BASELINE_DIR CANDIDATE_DIR [--out cmp.json] [--tau-logit 0.5] [--tau-prob 0.05]
        [--floor noise_floor.json]

Writes the JSON (default: CANDIDATE_DIR/compare_vs_<baseline name>.json) and prints a short summary. With --floor,
adds a verdict per output against the noise floor's entry for this case (README.md, "Pass rule") and exits 3 if any
output is outside it. Exit 2: the runs are not comparable (different case, or a phase failed).
Definitions of max_abs, max_rel, rel_l2 and near-tie: golden_lib.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import golden_lib as gl  # noqa: E402

# BIT seeds torch only; Python's and numpy's global RNGs start from OS entropy in every process. Nothing in Stage 2
# draws from them (the samplers use their own seeded generators), so they are recorded but never judged.
INFO = ("rng_python", "rng_numpy")


def load_run(d: Path) -> dict:
    d = Path(d)
    run = {"dir": str(d), "manifest": gl.read_json(d / "manifest.json")}
    for phase in ("main", "resume"):
        p = d / phase / "phase.json"
        run[phase] = gl.read_json(p) if p.is_file() else None
    p = d / "main" / "decodes.json"
    run["decodes"] = gl.read_json(p) if p.is_file() else None
    return run


def torch_load(path: Path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def run_label(run) -> dict:
    m, env = run["manifest"], (run["main"] or {}).get("env", {})
    return {"dir": run["dir"], "machine": m.get("machine"), "gpu": env.get("gpu"), "case": m.get("case"),
            "family": (m.get("case_config") or {}).get("family", m.get("case")), "train_args": m.get("train_args"),
            "shapes": m.get("shapes"), "status": m.get("status"),
            "code_sha256": m.get("code_root", {}).get("training_files_sha256"),
            "code_git": m.get("code_root", {}).get("git"),
            "harness_sha256": m.get("harness", {}).get("combined_sha256"),
            "torch": env.get("torch"), "cuda": env.get("cuda")}


def digest_outputs(o: dict, prefix: str, b: dict, c: dict) -> None:
    """One discrete output per digest key; Python / numpy RNG keys are informational."""
    for k in sorted(set(b or {}) | set(c or {})):
        v = gl.compare_discrete((b or {}).get(k), (c or {}).get(k))
        if k in INFO:
            v["kind"] = "info"
        o[f"{prefix}/{k}"] = v


def rng_file_digests(path: Path) -> dict:
    return {f"rng_{k}": gl.nested_digest(v) for k, v in torch_load(path).items()}


def flat_eval(ev: dict) -> tuple[list, list]:
    ids, margins = [], []
    for row in ev.get("batches", []):
        ids += row.get("argmax", [])
        margins += row.get("margin", [])
    return ids, margins


def compare_eval(be: dict, ce: dict, tau_logit: float) -> dict:
    out = {"loss": gl.compare_scalars([r["loss"] for r in be["batches"]] + [be["avg_loss"]],
                                      [r["loss"] for r in ce["batches"]] + [ce["avg_loss"]]),
           "n": gl.compare_discrete([r["n"] for r in be["batches"]], [r["n"] for r in ce["batches"]]),
           "inputs": gl.compare_discrete([r.get("inputs") for r in be["batches"]], [r.get("inputs") for r in ce["batches"]])}
    for when in ("after_seed", "after_eval"):
        b, c = (be.get("rng") or {}).get(when), (ce.get("rng") or {}).get(when)
        if b is not None or c is not None:
            tmp = {}
            digest_outputs(tmp, f"rng_{when}", b, c)
            out.update(tmp)
    b_ids, b_m = flat_eval(be)
    c_ids, c_m = flat_eval(ce)
    if b_ids or c_ids:
        out["argmax"] = gl.classify_argmax(b_ids, c_ids, b_m, tau_logit)
        if len(b_ids) == len(c_ids):
            same = [i for i, (a, b) in enumerate(zip(b_ids, c_ids)) if a == b]
            out["argmax_margin"] = gl.compare_scalars([b_m[i] for i in same], [c_m[i] for i in same])
    return out


def compare_decodes(bd: dict, cd: dict, tau_logit: float, tau_prob: float) -> dict:
    out = {}
    if bd["trials"] != cd["trials"]:
        out["trials"] = gl.compare_discrete(bd["trials"], cd["trials"])
        return out
    out["trials"] = gl.compare_discrete(bd["trials"], cd["trials"])
    out["text"] = gl.compare_discrete([r["text"] for r in bd["rows"]], [r["text"] for r in cd["rows"]])
    mism, margins_b, margins_c = [], [], []
    for br, cr in zip(bd["rows"], cd["rows"]):
        if br.get("ids") is None or cr.get("ids") is None:
            mism.append({"trial": br["trial"], "near_tie": False, "why": "token ids missing"})
            continue
        if bd.get("diffusion"):
            res = gl.classify_diffusion(br, cr, bd.get("threshold"), tau_logit, tau_prob)
        else:
            res = gl.classify_ar(br, cr, tau_logit)
        if res is not None:
            res.update(trial=br["trial"], base_text=br["text"], cand_text=cr["text"])
            mism.append(res)
        elif br.get("margins") and cr.get("margins") and len(br["margins"]) == len(cr["margins"]):
            for x, y in zip(br["margins"], cr["margins"]):
                if x is not None and y is not None:
                    margins_b.append(x)
                    margins_c.append(y)
    near = sum(1 for m in mism if m.get("near_tie"))
    out["ids"] = {"kind": "discrete", "n": len(bd["rows"]), "mismatches": len(mism), "near_tie": near,
                  "other": len(mism) - near, "exact": not mism, "details": mism}
    out["wer"] = gl.compare_scalars([bd["score"]["wer"]], [cd["score"]["wer"]])
    out["margins_matching_decodes"] = gl.compare_scalars(margins_b, margins_c)
    return out


def compare_runs(base_dir, cand_dir, tau_logit: float = gl.TAU_LOGIT, tau_prob: float = gl.TAU_PROB) -> dict:
    base, cand = load_run(base_dir), load_run(cand_dir)
    res = {"baseline": run_label(base), "candidate": run_label(cand),
           "near_tie": {"tau_logit": tau_logit, "tau_prob": tau_prob}, "outputs": {}, "problems": []}
    lb, lc = res["baseline"], res["candidate"]
    res["same_machine"] = lb["machine"] == lc["machine"]
    res["same_code"] = lb["code_sha256"] == lc["code_sha256"]
    res["same_harness"] = lb["harness_sha256"] == lc["harness_sha256"]
    if lb["family"] != lc["family"]:
        res["problems"].append(f"different case families: {lb['case']} vs {lc['case']}")
        res["comparable"] = False
        return res
    res["same_case"] = lb["case"] == lc["case"]
    res["same_train_args"] = lb.get("train_args") == lc.get("train_args")
    for who, run in (("baseline", base), ("candidate", cand)):
        if run["main"] is None or run["main"].get("status") != "ok":
            res["problems"].append(f"{who}: main phase {None if run['main'] is None else run['main'].get('status')}")
    res["comparable"] = not res["problems"]
    if not res["comparable"]:
        return res
    o = res["outputs"]
    bm, cm = base["main"], cand["main"]
    digest_outputs(o, "init/digests", bm["initial_digests"], cm["initial_digests"])
    o["rng_fingerprint"] = gl.compare_discrete(sorted(bm.get("rng_fingerprint", {}).items()),
                                               sorted(cm.get("rng_fingerprint", {}).items()))
    shapes = [s for s in bm["shape_results"] if s in cm["shape_results"]]
    res["shapes"] = shapes
    for s in shapes:
        b, c = bm["shape_results"][s], cm["shape_results"][s]
        bd, cd = Path(base["dir"]) / "main" / s, Path(cand["dir"]) / "main" / s
        sel_b, sel_c = b["selection"], c["selection"]
        o[f"{s}/batches/train"] = gl.compare_discrete([x["indices"] for x in sel_b["train"]],
                                                      [x["indices"] for x in sel_c["train"]])
        o[f"{s}/batches/train_data"] = gl.compare_discrete([x["data_sha256"] for x in sel_b["train"]],
                                                           [x["data_sha256"] for x in sel_c["train"]])
        o[f"{s}/batches/val"] = gl.compare_discrete([x["indices"] for x in sel_b["val"]],
                                                    [x["indices"] for x in sel_c["val"]])
        o[f"{s}/batches/val_data"] = gl.compare_discrete([x["data_sha256"] for x in sel_b["val"]],
                                                         [x["data_sha256"] for x in sel_c["val"]])
        tb, tc = b["train"], c["train"]
        o[f"{s}/train/loss"] = gl.compare_scalars(tb["loss"], tc["loss"])
        o[f"{s}/train/n"] = gl.compare_discrete(tb["n"], tc["n"])
        o[f"{s}/train/shapes"] = gl.compare_discrete([sorted(x.items()) for x in tb["shapes"]],
                                                     [sorted(x.items()) for x in tc["shapes"]])
        o[f"{s}/train/inputs"] = gl.compare_discrete(tb.get("inputs"), tc.get("inputs"))  # tensors the forward got
        o[f"{s}/train/delivered_as_selected"] = {"kind": "self", "baseline": b.get("delivered_as_selected"),
                                                 "candidate": c.get("delivered_as_selected"),
                                                 "exact": b.get("delivered_as_selected") == c.get("delivered_as_selected"),
                                                 "mismatches": int(b.get("delivered_as_selected") is True
                                                                   and c.get("delivered_as_selected") is not True)}
        o[f"{s}/train/lr"] = gl.compare_scalars(tb["lr"], tc["lr"])
        o[f"{s}/train/grad_norm"] = gl.compare_scalars(tb["grad_norm"], tc["grad_norm"])
        o[f"{s}/grads_step1"] = gl.compare_tensors(torch_load(bd / "grads_step1.pt"), torch_load(cd / "grads_step1.pt"))
        o[f"{s}/updates"] = gl.compare_tensors(torch_load(bd / "updates.pt"), torch_load(cd / "updates.pt"))
        ob, oc = torch_load(bd / "optimizer.pt"), torch_load(cd / "optimizer.pt")
        for key in ("exp_avg", "exp_avg_sq", "step"):
            o[f"{s}/optimizer/{key}"] = gl.compare_tensors({n: v[key] for n, v in ob["state"].items() if key in v},
                                                           {n: v[key] for n, v in oc["state"].items() if key in v})
        o[f"{s}/optimizer/groups"] = gl.compare_discrete(gl.nested_digest(ob["groups"]), gl.nested_digest(oc["groups"]))
        for key in ("rng_k", "rng_end"):
            digest_outputs(o, f"{s}/{key}", rng_file_digests(bd / f"{key}.pt"), rng_file_digests(cd / f"{key}.pt"))
        o[f"{s}/scheduler_end"] = gl.compare_discrete(gl.nested_digest(b["scheduler_end"]),
                                                      gl.nested_digest(c["scheduler_end"]))
        for key in ("digests_k", "digests_end"):
            digest_outputs(o, f"{s}/{key}", b[key], c[key])
        o[f"{s}/grads_none"] = gl.compare_discrete(b.get("grads_none") or [], c.get("grads_none") or [])
        for key, val in compare_eval(b["eval_after_train"], c["eval_after_train"], tau_logit).items():
            o[f"{s}/eval_after_train/{key}"] = val
        br = (base["resume"] or {}).get("shape_results", {}).get(s)
        cr = (cand["resume"] or {}).get("shape_results", {}).get(s)
        if br and cr:
            o[f"{s}/resume/loss"] = gl.compare_scalars(br["train"]["loss"], cr["train"]["loss"])
            o[f"{s}/resume/inputs"] = gl.compare_discrete(br["train"].get("inputs"), cr["train"].get("inputs"))
            digest_outputs(o, f"{s}/resume/digests_end", br["digests_end"], cr["digests_end"])
            o[f"{s}/resume/accelerate_restored_rng"] = gl.compare_discrete(
                sorted((br.get("accelerate_restored_rng") or {}).items()),
                sorted((cr.get("accelerate_restored_rng") or {}).items()))
            o[f"{s}/resume/self_exact"] = {"kind": "self", "baseline": bool(br.get("exact")),
                                           "candidate": bool(cr.get("exact")),
                                           "exact": bool(br.get("exact")) == bool(cr.get("exact")),
                                           "mismatches": int(bool(br.get("exact")) and not cr.get("exact")),
                                           "candidate_load_matches_step_k": cr.get("load_matches_step_k"),
                                           "candidate_vs_uninterrupted": {
                                               k: {x: v.get(x) for x in ("exact", "max_abs", "max_rel")}
                                               if isinstance(v, dict) else v
                                               for k, v in cr.get("vs_uninterrupted", {}).items()}}
        elif br or cr:
            res["problems"].append(f"{s}: resume phase in one run only")
    bb, cb = bm.get("best"), cm.get("best")
    if bb and cb:
        o["best/digest"] = gl.compare_discrete(bb["digest"], cb["digest"])
        for s in shapes:
            if s in bb.get("eval", {}) and s in cb.get("eval", {}):
                for key, val in compare_eval(bb["eval"][s], cb["eval"][s], tau_logit).items():
                    o[f"best/eval/{s}/{key}"] = val
        if base["decodes"] and cand["decodes"]:
            for key, val in compare_decodes(base["decodes"], cand["decodes"], tau_logit, tau_prob).items():
                o[f"best/decode/{key}"] = val
    elif bb or cb:
        res["problems"].append("best-checkpoint eval/decode in one run only")
    res["summary"] = summarize(o)
    return res


def is_exact(v: dict) -> bool:
    return bool(v.get("exact", False))


def summarize(outputs: dict) -> dict:
    exact, cont, disc, info = [], {}, {}, {}
    for k, v in outputs.items():
        if v.get("kind") == "info":
            info[k] = bool(v.get("exact"))
            continue
        if is_exact(v):
            exact.append(k)
        elif v.get("kind") == "continuous":
            cont[k] = {x: v.get(x) for x in ("max_abs", "max_rel", "rel_l2", "n_tensors_differ", "worst", "comparable")}
        else:
            disc[k] = {x: v.get(x) for x in ("n", "mismatches", "near_tie", "other", "baseline", "candidate")
                       if x in v}
    judged = len(outputs) - len(info)
    return {"n_outputs": judged, "n_exact": len(exact), "all_exact": len(exact) == judged,
            "continuous_differences": cont, "discrete_mismatches": disc, "informational": info}


# ---------------------------------------------------------------- verdict against a noise floor

def verdict(res: dict, floor: dict) -> dict:
    """Per output: within the floor's entry for this case (README.md "Pass rule"). Outputs without an entry must
    be exact. Continuous: max_abs <= floor max_abs and max_rel <= floor max_rel. Discrete with near-ties: no more
    non-near-tie mismatches than the floor's, and no more mismatches in all. Other discrete: no more mismatches."""
    case = res["baseline"]["case"]
    entries = floor.get("floor", {})
    out, ok = {}, True
    for k, v in res["outputs"].items():
        f = entries.get(f"{case}/{k}")
        if v.get("kind") == "info":
            out[k] = {"pass": True, "why": "informational"}
            continue
        if is_exact(v):
            out[k] = {"pass": True, "why": "exact"}
            continue
        if f is None:
            out[k] = {"pass": False, "why": "differs; no floor entry (must be exact)"}
        elif v.get("kind") == "continuous":
            good = (v.get("comparable", True) and v["max_abs"] <= f.get("max_abs", 0.0)
                    and v["max_rel"] <= f.get("max_rel", 0.0))
            out[k] = {"pass": bool(good), "max_abs": v["max_abs"], "floor_max_abs": f.get("max_abs"),
                      "max_rel": v["max_rel"], "floor_max_rel": f.get("max_rel")}
        elif "near_tie" in v:
            good = v["other"] <= f.get("other", 0) and v["mismatches"] <= f.get("mismatches", 0)
            out[k] = {"pass": bool(good), "mismatches": v["mismatches"], "near_tie": v["near_tie"],
                      "other": v["other"], "floor": {x: f.get(x) for x in ("mismatches", "near_tie", "other")}}
        else:
            good = v.get("mismatches", 1) <= f.get("mismatches", 0)
            out[k] = {"pass": bool(good), "mismatches": v.get("mismatches"), "floor_mismatches": f.get("mismatches")}
        ok &= out[k]["pass"]
    return {"pass": bool(ok), "floor_basis": floor.get("floor_basis"), "provisional": floor.get("provisional"),
            "outputs": out}


def fmt(x) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.3g}"
    return str(x)


def human(res: dict) -> str:
    lb, lc = res["baseline"], res["candidate"]
    lines = [f"case {lb['case']}: baseline {lb['machine']} ({lb['gpu']}) {lb['dir']}",
             f"          candidate {lc['machine']} ({lc['gpu']}) {lc['dir']}",
             f"same machine {res.get('same_machine')}, same code {res.get('same_code')}, "
             f"same harness {res.get('same_harness')}, same case {res.get('same_case')}, "
             f"train args {lb.get('train_args')} / {lc.get('train_args')}"]
    if res["problems"]:
        lines += [f"PROBLEM: {p}" for p in res["problems"]]
    if not res.get("comparable", False):
        return "\n".join(lines)
    s = res["summary"]
    lines.append(f"{s['n_exact']}/{s['n_outputs']} outputs bitwise exact")
    for k, v in s["continuous_differences"].items():
        lines.append(f"  ~ {k}: max_abs {fmt(v['max_abs'])}, max_rel {fmt(v['max_rel'])}, rel_l2 {fmt(v['rel_l2'])}"
                     + (f", worst {v['worst']}" if v.get("worst") else ""))
    for k, v in s["discrete_mismatches"].items():
        extra = f" ({v['near_tie']} near-tie, {v['other']} other)" if "near_tie" in v else ""
        if "baseline" in v:
            lines.append(f"  ! {k}: baseline {v['baseline']}, candidate {v['candidate']}")
        else:
            lines.append(f"  ! {k}: {v.get('mismatches')}/{v.get('n')} differ{extra}")
    if "verdict" in res:
        bad = [k for k, v in res["verdict"]["outputs"].items() if not v["pass"]]
        lines.append(f"verdict vs floor ({res['verdict'].get('floor_basis')}): "
                     + ("PASS" if res["verdict"]["pass"] else f"FAIL ({len(bad)}: {', '.join(bad[:8])})"))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--tau-logit", type=float, default=gl.TAU_LOGIT)
    ap.add_argument("--tau-prob", type=float, default=gl.TAU_PROB)
    ap.add_argument("--floor", type=Path, help="a noise-floor JSON (noise_floor.py): add a verdict")
    args = ap.parse_args(argv)
    res = compare_runs(args.baseline, args.candidate, args.tau_logit, args.tau_prob)
    rc = 0 if res.get("comparable") else 2
    if args.floor is not None and res.get("comparable"):
        res["verdict"] = verdict(res, gl.read_json(args.floor))
        rc = 0 if res["verdict"]["pass"] else 3
    out = args.out or args.candidate / f"compare_vs_{args.baseline.resolve().name}.json"
    gl.write_json(out, res)
    print(human(res))
    print(f"-> {out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())

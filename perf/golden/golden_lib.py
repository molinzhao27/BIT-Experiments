"""Shared pieces of BIT's golden harness: digests, numeric comparisons and near-tie classification.

Used by run.py (BIT's pinned GPU env, Python 3.10) and by compare.py / noise_floor.py (the project .venv). Only numpy
and the standard library are imported at module level; torch is imported where tensors are handled, so the runner's
parent process never touches CUDA.

Definitions (also in README.md):
- max_abs: the largest elementwise |candidate - baseline| over every element of an output.
- max_rel: per tensor ||candidate - baseline||_inf / ||baseline||_inf (normwise, so values near zero do not blow it
  up), the largest over the output's tensors. A list of scalars is a list of one-element tensors, so for losses this
  is the elementwise relative difference.
- rel_l2: ||candidate - baseline||_2 / ||baseline||_2 over the whole output.
- exact: bitwise equal (same dtype, shape and bytes; NaN payloads and signed zeros included).
- near-tie (decodes, argmaxes): the BASELINE's top-1 minus top-2 logit at the first differing token is <= tau_logit
  (default 0.5 logit, about 4 bf16 ulps at typical top-1 logits of 16-32). For the block-diffusion sampler, a commit
  decision is a near-tie when the committed token's probability is within tau_prob (default 0.05) of the sampler's
  threshold, or, for the forced commit of the most confident position, when the two highest probabilities in that
  pass are within tau_prob.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

TAU_LOGIT = 0.5
TAU_PROB = 0.05


# ---------------------------------------------------------------- JSON and digests

def jsonable(value):
    """A JSON-safe copy: tensors and arrays as lists, numpy scalars as Python scalars, other objects as repr."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is in both environments
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return repr(value)


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(obj), indent=1, sort_keys=False) + "\n")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def tensor_bytes(t) -> bytes:
    """The raw bytes of a tensor (CPU, contiguous), dtype-independent of interpretation."""
    import torch

    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bool:
        t = t.to(torch.uint8)
    return t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b""


def tensor_digest(t) -> str:
    h = hashlib.sha256(f"{str(t.dtype)}:{tuple(t.shape)};".encode())
    h.update(tensor_bytes(t))
    return h.hexdigest()


def nested_digest(value) -> str:
    """sha256 over a nested structure of tensors / arrays / scalars, keys sorted, bitwise for tensors."""
    import torch

    h = hashlib.sha256()

    def visit(v, path):
        if isinstance(v, torch.Tensor):
            h.update(f"{path}=T{str(v.dtype)}{tuple(v.shape)};".encode())
            h.update(tensor_bytes(v))
        elif isinstance(v, np.ndarray):
            h.update(f"{path}=A{v.dtype.str}{v.shape};".encode())
            h.update(np.ascontiguousarray(v).tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str):
                visit(v[k], f"{path}.{k}")
        elif isinstance(v, (list, tuple)):
            h.update(f"{path}=L{len(v)};".encode())
            for i, x in enumerate(v):
                visit(x, f"{path}[{i}]")
        elif isinstance(v, float):
            h.update(f"{path}=F{float(v).hex()};".encode())
        elif callable(v):  # e.g. a scheduler's anneal function: by name, never by address
            h.update(f"{path}=C{getattr(v, '__module__', '')}.{getattr(v, '__qualname__', type(v).__name__)};".encode())
        else:
            h.update(f"{path}={type(v).__name__}:{v!r};".encode())

    visit(value, "")
    return h.hexdigest()


# ---------------------------------------------------------------- numeric comparison

def _empty_stats():
    return {"kind": "continuous", "exact": True, "comparable": True, "n": 0, "max_abs": 0.0, "max_rel": 0.0,
            "rel_l2": 0.0, "n_tensors": 0, "n_tensors_differ": 0, "worst": None, "problems": []}


def _finalize(stats, sq_diff, sq_base):
    if sq_base > 0:
        stats["rel_l2"] = math.sqrt(sq_diff / sq_base)
    elif sq_diff > 0:
        stats["rel_l2"] = math.inf
    return stats


def compare_tensors(base: dict, cand: dict) -> dict:
    """Compare two {name: tensor} dicts (any nesting of one level). Floating tensors: max_abs / max_rel / rel_l2;
    integer and bool tensors: count of differing elements. Missing, extra or reshaped tensors make the output
    non-comparable (and not exact)."""
    import torch

    stats = _empty_stats()
    stats["discrete_mismatches"] = 0
    sq_diff = sq_base = 0.0
    keys = sorted(set(base) | set(cand))
    worst_rel = -1.0
    for k in keys:
        if k not in base or k not in cand:
            stats["exact"] = stats["comparable"] = False
            stats["problems"].append(f"{k}: only in {'baseline' if k in base else 'candidate'}")
            continue
        a, b = base[k], cand[k]
        if a is None or b is None:
            if (a is None) != (b is None):
                stats["exact"] = stats["comparable"] = False
                stats["problems"].append(f"{k}: None in one run only")
            continue
        a, b = torch.as_tensor(a).detach().cpu(), torch.as_tensor(b).detach().cpu()
        stats["n_tensors"] += 1
        if a.shape != b.shape or a.dtype != b.dtype:
            stats["exact"] = stats["comparable"] = False
            stats["n_tensors_differ"] += 1
            stats["problems"].append(f"{k}: {a.dtype}{tuple(a.shape)} vs {b.dtype}{tuple(b.shape)}")
            continue
        stats["n"] += a.numel()
        exact = tensor_bytes(a) == tensor_bytes(b)
        if not exact:
            stats["exact"] = False
            stats["n_tensors_differ"] += 1
        if a.numel() == 0:
            continue
        if a.is_floating_point():
            a64, b64 = a.to(torch.float64), b.to(torch.float64)
            fa, fb = torch.isfinite(a64), torch.isfinite(b64)
            nf = ~fa
            code = lambda x: torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)  # noqa: E731
            if not torch.equal(fa, fb) or (bool(nf.any()) and not torch.equal(code(a64[nf]), code(b64[nf]))):
                stats["problems"].append(f"{k}: non-finite values differ")
                stats["max_abs"] = math.inf
                stats["max_rel"] = math.inf
                worst_rel = math.inf
                stats["worst"] = k
                continue
            d = (a64[fa] - b64[fa]).abs()
            if d.numel() == 0:
                continue
            mx = float(d.max())
            scale = float(a64[fa].abs().max())
            rel = mx / scale if scale > 0 else (0.0 if mx == 0 else math.inf)
            stats["max_abs"] = max(stats["max_abs"], mx)
            stats["max_rel"] = max(stats["max_rel"], rel)
            sq_diff += float((d ** 2).sum())
            sq_base += float((a64[fa].abs() ** 2).sum())
            if rel > worst_rel and not exact:
                worst_rel, stats["worst"] = rel, k
        else:
            n_bad = int((a != b).sum())
            stats["discrete_mismatches"] += n_bad
            if n_bad:
                d = (a.to(torch.float64) - b.to(torch.float64)).abs()
                stats["max_abs"] = max(stats["max_abs"], float(d.max()))
                if stats["worst"] is None:
                    stats["worst"] = k
    return _finalize(stats, sq_diff, sq_base)


def compare_scalars(base, cand) -> dict:
    """Two equally long lists of numbers (or nested lists of equal shape), elementwise."""
    import torch

    def flat(x):
        out = []

        def visit(v):
            if isinstance(v, (list, tuple)):
                for y in v:
                    visit(y)
            else:
                out.append(math.nan if v is None else float(v))
        visit(x)
        return out

    fa, fb = flat(base), flat(cand)
    if len(fa) != len(fb):
        s = _empty_stats()
        s.update(exact=False, comparable=False, problems=[f"length {len(fa)} vs {len(fb)}"])
        return s
    s = compare_tensors({f"[{i}]": torch.tensor([x], dtype=torch.float64) for i, x in enumerate(fa)},
                        {f"[{i}]": torch.tensor([x], dtype=torch.float64) for i, x in enumerate(fb)})
    s.pop("discrete_mismatches", None)
    return s


def compare_discrete(base, cand) -> dict:
    """Two lists of hashable items (or two values): count of differing positions."""
    if not isinstance(base, (list, tuple)) or not isinstance(cand, (list, tuple)):
        same = base == cand
        return {"kind": "discrete", "n": 1, "mismatches": 0 if same else 1, "exact": same}
    n = max(len(base), len(cand))
    bad = [i for i in range(n) if i >= len(base) or i >= len(cand) or base[i] != cand[i]]
    return {"kind": "discrete", "n": n, "mismatches": len(bad), "exact": not bad, "first": bad[:10]}


# ---------------------------------------------------------------- near ties

def first_diff(a, b):
    """First index where two sequences differ (a strict prefix differs at its end); None if equal."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def classify_ar(base: dict, cand: dict, tau_logit: float = TAU_LOGIT):
    """One AR greedy decode: None if the token ids match, else the first differing token, the baseline's top-1 minus
    top-2 logit there, and whether that is a near-tie."""
    j = first_diff(base["ids"], cand["ids"])
    if j is None:
        return None
    margins = base.get("margins") or []
    m = margins[j] if j < len(margins) and margins[j] is not None else None
    return {"pos": j, "base_token": base["ids"][j] if j < len(base["ids"]) else None,
            "cand_token": cand["ids"][j] if j < len(cand["ids"]) else None, "margin": m,
            "near_tie": m is not None and m <= tau_logit, "how": "token"}


def _commits(event):
    """What a sampler event commits: ('first', pos, token) or ('m', {pos: token})."""
    if "first" in event:
        return ("first", int(event["first"][0]), int(event["first"][1]))
    return ("m", {int(e[0]): int(e[1]) for e in event.get("m", []) if int(e[3]) == 1})


def _decision_near_tie(entries, pos, threshold, tau_prob):
    """Whether committing (or not) masked position `pos` in a pass was a near-tie in the baseline."""
    probs = {int(e[0]): float(e[2]) for e in entries}
    if pos not in probs:
        return False, None
    p = probs[pos]
    gap = abs(p - threshold) if threshold is not None else math.inf
    above = [q for q in probs.values() if threshold is not None and q > threshold]
    if not above and len(probs) >= 2:  # only the forced commit of the most confident position decides
        top = sorted(probs.values(), reverse=True)[:2]
        gap = min(gap, top[0] - top[1])
    return gap <= tau_prob, gap


def classify_diffusion(base: dict, cand: dict, threshold, tau_logit: float = TAU_LOGIT, tau_prob: float = TAU_PROB):
    """One block-diffusion greedy decode. With both traces: the first sampler event whose commits differ (the state
    is identical before it), and at that event whether every differing decision was a near-tie in the baseline
    (token: top-1 minus top-2 logit <= tau_logit; commit or not: probability within tau_prob of the threshold, or of
    the runner-up for the forced commit). Without a usable candidate trace: the first differing output token and the
    baseline's margins from the pass that committed it."""
    j = first_diff(base["ids"], cand["ids"])
    if j is None:
        return None
    res = {"pos": j, "base_token": base["ids"][j] if j < len(base["ids"]) else None,
           "cand_token": cand["ids"][j] if j < len(cand["ids"]) else None}
    be, ce = base.get("events"), cand.get("events")
    if be and ce:
        for e, (x, y) in enumerate(zip(be, ce)):
            cx, cy = _commits(x), _commits(y)
            if cx == cy:
                continue
            res.update(event=e, how="trace")
            if cx[0] != cy[0] or (cx[0] == "first" and cx[1] != cy[1]):
                res.update(near_tie=False, why="sampler structure differs")
                return res
            if cx[0] == "first":
                m = x["first"][2] if len(x["first"]) > 2 else None
                res.update(margin=m, near_tie=m is not None and m <= tau_logit, why="block first token")
                return res
            entries = {int(z[0]): z for z in x.get("m", [])}
            cand_entries = {int(z[0]): z for z in y.get("m", [])}
            if set(entries) != set(cand_entries):
                res.update(near_tie=False, why="masked positions differ")
                return res
            decisions, ok = [], True
            for pos in sorted(q for q in set(cx[1]) | set(cy[1]) if cx[1].get(q) != cy[1].get(q)):
                bz, cz = entries[pos], cand_entries[pos]
                if int(bz[3]) == 1 and int(cz[3]) == 1:  # both committed, different tokens
                    m = bz[4] if len(bz) > 4 else None
                    near = m is not None and m <= tau_logit
                    decisions.append({"pos": pos, "kind": "token", "margin": m, "near_tie": near})
                else:  # committed in one run only
                    near, gap = _decision_near_tie(x.get("m", []), pos, threshold, tau_prob)
                    decisions.append({"pos": pos, "kind": "commit", "prob_gap": gap, "near_tie": near})
                ok &= decisions[-1]["near_tie"]
            margins = [d["margin"] for d in decisions if d.get("margin") is not None]
            res.update(decisions=decisions, near_tie=bool(decisions) and ok,
                       margin=min(margins) if margins else None, why="pass commits differ")
            return res
        if len(be) != len(ce):
            res.update(how="trace", near_tie=False, why="trace lengths differ after equal events")
            return res
    # fallback: output level
    margins = base.get("margins") or []
    probs = base.get("commit_probs") or []
    m = margins[j] if j < len(margins) else None
    p = probs[j] if j < len(probs) else None
    near = (m is not None and m <= tau_logit) or (p is not None and threshold is not None
                                                     and abs(p - threshold) <= tau_prob)
    res.update(how="output", margin=m, commit_prob=p, near_tie=bool(near))
    return res


def classify_argmax(base_ids, cand_ids, base_margins, tau_logit: float = TAU_LOGIT) -> dict:
    """Teacher-forced argmax tokens at supervised positions: mismatches and how many are near-ties."""
    n = max(len(base_ids), len(cand_ids))
    if len(base_ids) != len(cand_ids):
        return {"kind": "discrete", "n": n, "mismatches": n, "near_tie": 0, "other": n, "comparable": False,
                "exact": False}
    bad = [i for i, (a, b) in enumerate(zip(base_ids, cand_ids)) if a != b]
    near = [i for i in bad if base_margins[i] is not None and base_margins[i] <= tau_logit]
    return {"kind": "discrete", "n": n, "mismatches": len(bad), "near_tie": len(near), "other": len(bad) - len(near),
            "comparable": True, "exact": not bad,
            "max_mismatch_margin": max((base_margins[i] for i in bad), default=None)}


# ---------------------------------------------------------------- block-diffusion trace alignment

def align_diffusion_trace(rec: dict, calls: list, block_size: int) -> dict:
    """Attach logit margins to one decode's sampler trace (BCI_DiffusionLLM._denoise's `trace` record).

    rec: {prefix_len, tail, events, out}. calls: every lm_head output of that decode in order, each
    {L, top1: [L], margin: [L]} (top-1 minus top-2 logit per position, the mask token excluded, as the sampler sees
    them). Each sampler event comes from one forward: a block's first token from the last position of the forward
    before it (the prefix prefill, or the previous block's commit), a fill pass from the token-shifted logits of
    [tail, block] (slot s reads position max(tail + s - 1, 0)). A partial-prefix prefill projects logits that no
    event uses (one extra call, absent when the runtime skips that projection).

    Returns events with margins appended ("first": [pos, token, margin]; "m" entries: [pos, proposal, probability,
    committed, margin]), per output token the margin and probability of the pass that committed it (None for a
    block's first token: committed without a threshold), and any alignment problems (then margins are None)."""
    events, tail0 = rec.get("events") or [], int(rec.get("tail", 0))
    n_out = len(rec.get("out") or [])
    problems = []
    calls = list(calls)
    if len(calls) == len(events) + 1:
        calls = calls[1:]
    elif len(calls) != len(events):
        problems.append(f"{len(calls)} lm_head calls for {len(events)} sampler events")
        calls = [None] * len(events)

    def start(b):
        return 0 if b == 0 else (block_size - tail0) + (b - 1) * block_size

    def at(call, r, token):
        if call is None or not 0 <= r < call["L"]:
            return None
        if call["top1"][r] != token and call["margin"][r] != 0:  # a zero margin is an exact tie: either may win
            problems.append(f"position {r}: top-1 {call['top1'][r]} != sampler's {token}")
            return None
        return float(call["margin"][r])

    out_events, margins, probs = [], [None] * n_out, [None] * n_out
    for ev, call in zip(events, calls):
        b = int(ev.get("b", 0))
        if "first" in ev:
            pos, tok = int(ev["first"][0]), int(ev["first"][1])
            m = at(call, (call["L"] - 1) if call else -1, tok)
            out_events.append({"b": b, "first": [pos, tok, m]})
            if 0 <= pos < n_out:
                margins[pos], probs[pos] = m, None
            continue
        tail = tail0 if b == 0 else 0
        entries = []
        for e in ev.get("m", []):
            pos, prop, prob, committed = int(e[0]), int(e[1]), float(e[2]), int(e[3])
            m = at(call, max(tail + pos - start(b) - 1, 0), prop)
            entries.append([pos, prop, prob, committed, m])
            if committed and 0 <= pos < n_out:
                margins[pos], probs[pos] = m, prob
        new = {"b": b, "m": entries}
        for key in ("e", "r"):
            if key in ev:
                new[key] = ev[key]
        out_events.append(new)
    return {"events": out_events, "margins": margins, "commit_probs": probs, "problems": problems}

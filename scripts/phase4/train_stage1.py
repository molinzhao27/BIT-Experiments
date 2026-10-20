#!/usr/bin/env python
"""Phase 4 Stage 1 (arms C, E): align encoder + projector to fixed LLM sentence targets, no LLM, no phonemes.

    <bit env>/python scripts/phase4/train_stage1.py --out checkpoints/stage1/C_l14_b1024_s42 \
        [--config configs/phase4/stage1.yaml] [--init DIR|none] [--target layer14|emb|layers5|...]
        [--loss infonce|cosine|mse] [--batch-size N] [--chunk-size N] [--seed N] [--device auto]
        [--max-epochs N] [--limit N]

Selection: val top-1 sentence retrieval among the distinct BIT-val sentences; early stop on its plateau.
Logs (out/log.jsonl): per step loss, logit scale, distinct sentences in the InfoNCE denominator, masked false
negatives; per eval retrieval top-1/top-5 on all BIT-val trials and on trials whose sentence is unseen in train.
At the end the best checkpoint is also scored on BIT-test (report only, never used for any choice).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "bit"))

from kunzdecoder import stage1, text_targets  # noqa: E402


def load_splits(data_dir: Path, limit: int | None) -> dict[str, list[dict]]:
    from data.brandman_2024_text.prepare_data import load_data

    ds = load_data(data_dir=str(data_dir), features=["tx1", "spikePow"])  # BIT's sentence-stage input (512 features)
    ds.pop("holdout", None)
    if limit:
        ds = {k: v[:limit] for k, v in ds.items()}
    return ds


def embed_all(items, ndt, projector, head, device, chunk, bf16) -> np.ndarray:
    ndt.eval(), projector.eval(), head.eval()
    out = []
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda"):
        for i in range(0, len(items), chunk):
            b = stage1.collate(items[i:i + chunk], device)
            x, m, *_ = ndt.encoder(b["spikes"], b["spikes_mask"], b["spikes_timestamp"], b["dataset_name"],
                                   force_mask=False)
            out.append(head.embed_neural(projector(x), m).float().cpu())
    ndt.train(), projector.train(), head.train()
    return torch.cat(out).numpy()


def evaluate(split_items, split_rows, train_rows, ndt, projector, head, targets, device, cfg) -> dict:
    zn = embed_all(split_items, ndt, projector, head, device, cfg["chunk_size"], cfg["training"]["bf16"])
    cand = np.unique(split_rows)
    with torch.no_grad():
        zt = head.embed_text(torch.as_tensor(targets[cand], device=device)).cpu().numpy()
    unseen = ~np.isin(split_rows, train_rows)
    return {"all": stage1.retrieval(zn, split_rows, cand, zt), "unseen": stage1.retrieval(zn, split_rows, cand, zt, unseen)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "phase4" / "stage1.yaml")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--init")
    ap.add_argument("--target")
    ap.add_argument("--loss")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--chunk-size", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-epochs", type=int)
    ap.add_argument("--limit", type=int, help="first N trials per split (smoke tests)")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    for k, v in {"init": args.init, "loss": args.loss, "batch_size": args.batch_size, "chunk_size": args.chunk_size,
                 "seed": args.seed}.items():
        if v is not None:
            cfg[k] = None if v == "none" else v
    if args.target:
        cfg["targets"]["name"] = args.target
    if args.max_epochs:
        cfg["training"]["max_epochs"] = args.max_epochs
    seed = cfg["seed"]
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "a")
    t0 = time.time()

    # data + fixed targets
    rows_of, targets, tmeta = text_targets.load(ROOT / cfg["targets"]["cache"], cfg["targets"]["name"])
    ds = load_splits(ROOT / cfg["data_dir"], args.limit)
    items = {s: stage1.masked_items(v, rows_of) for s, v in ds.items()}
    rows = {s: np.array([rows_of[r["sentence"]] for r in v]) for s, v in ds.items()}
    targets_t = torch.as_tensor(targets, device=device)

    # model
    config = stage1.bit_sentence_config()
    init = None if cfg["init"] is None else (ROOT / cfg["init"])
    ndt, report = stage1.build_encoder(config, init)
    llm_hidden = targets.shape[1]  # the LLM's embedding width (1536): projector output = target width
    projector = stage1.build_projector(config, ndt.encoder.hidden_size, llm_hidden)
    head = stage1.AlignHead(llm_hidden, llm_hidden, cfg["head"]["proj_dim"], cfg["loss"], cfg["head"]["init_temp"],
                            cfg["head"]["max_scale"])
    for m in (ndt, projector, head):
        m.to(device).train()
    o = cfg["optimizer"]
    opt = torch.optim.AdamW([{"params": list(ndt.parameters()) + list(projector.parameters()), "lr": o["lr_encoder"]},
                             {"params": head.parameters(), "lr": o["lr_head"]}], weight_decay=o["weight_decay"])
    base_lrs = [g["lr"] for g in opt.param_groups]
    run_meta = {"config": cfg, "init_report": report, "targets_meta": tmeta, "device": str(device),
                "n": {s: len(v) for s, v in items.items()}, "llm_hidden": llm_hidden}
    (out / "run.json").write_text(json.dumps(run_meta, indent=1, default=str))
    print(json.dumps({k: run_meta[k] for k in ("n", "device")} | {"init": str(init), "loaded": report and report["loaded"],
                     "skipped_shape": report and report["skipped_shape"]}), flush=True)

    def embed(ch):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg["training"]["bf16"] and device.type == "cuda"):
            x, m, *_ = ndt.encoder(ch["spikes"], ch["spikes_mask"], ch["spikes_timestamp"], ch["dataset_name"],
                                   force_mask=False)
            return head.embed_neural(projector(x), m)

    B, C = cfg["batch_size"], cfg["chunk_size"]
    n_train = len(items["train"])
    best, since, step = -1.0, 0, 0
    for epoch in range(1, cfg["training"]["max_epochs"] + 1):
        perm = np.random.permutation(n_train)
        for i in range(0, n_train - B + 1 if n_train >= B else 1, B):  # full batches only (fixed denominator)
            idx = perm[i:i + B]
            chunks = [stage1.collate([items["train"][j] for j in idx[c:c + C]], device) for c in range(0, len(idx), C)]
            sid = torch.as_tensor(rows["train"][idx], device=device)
            step += 1
            for g, lr in zip(opt.param_groups, base_lrs):
                g["lr"] = lr * min(1.0, step / max(1, o["warmup_steps"]))
            opt.zero_grad(set_to_none=True)
            zt = head.embed_text(targets_t[sid])
            loss, stats = stage1.gradcache_step(embed, chunks, zt, sid, head, device)
            torch.nn.utils.clip_grad_norm_([p for m in (ndt, projector, head) for p in m.parameters()], o["grad_clip"])
            opt.step()
            log.write(json.dumps({"step": step, "epoch": epoch, "loss": loss, "batch": len(idx)} | stats) + "\n")
        if epoch % cfg["training"]["eval_every_epochs"]:
            continue
        ev = evaluate(items["val"], rows["val"], rows["train"], ndt, projector, head, targets, device, cfg)
        log.write(json.dumps({"epoch": epoch, "step": step, "val": ev, "minutes": round((time.time() - t0) / 60, 2)}) + "\n")
        log.flush()
        print(f"epoch {epoch:3d} step {step:5d} loss {loss:.3f} | val top1 {ev['all']['top1']:.3f} top5 "
              f"{ev['all']['top5']:.3f} (unseen top1 {ev['unseen']['top1']:.3f}, n={ev['unseen']['n']})", flush=True)
        if ev["all"]["top1"] > best:
            best, since = ev["all"]["top1"], 0
            stage1.save_checkpoint(out / "best", ndt, projector, head, config,
                                   {"epoch": epoch, "step": step, "val": ev, "config": cfg})
        else:
            since += 1
            if since >= cfg["training"]["patience_evals"]:
                break
    # report-only evaluation of the selected checkpoint on BIT-test
    ckpt = torch.load(out / "best" / "align_head.pt", map_location=device)
    ndt.encoder.load_state_dict(torch.load(out / "best" / "encoder.bin", map_location=device))
    projector.load_state_dict(torch.load(out / "best" / "projector.bin", map_location=device))
    head.load_state_dict(ckpt["state"])
    test = evaluate(items["test"], rows["test"], rows["train"], ndt, projector, head, targets, device, cfg)
    summary = {"best_epoch": ckpt["epoch"], "val": ckpt["val"], "test": test, "steps": step,
               "minutes": round((time.time() - t0) / 60, 2), "batch_size": B, "loss": cfg["loss"],
               "target": cfg["targets"]["name"], "init": str(init)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

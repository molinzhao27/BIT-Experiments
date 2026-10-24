#!/usr/bin/env python
"""How well does each LLM fill blanks in BIT sentences out of the box, before any brain training? (CPU, text only.)

    <bit env>/python scripts/text_blank_fill_check.py [--n 100] [--threads 6]      (run from a host code copy)

Question (2026-10-23): Aero block diffusion learned far slower than the Instruct + diffusion control, with identical
configs apart from the LM weights (val WER 94.1% vs 35.6% at epoch 10). Is Aero's LM worse at the diffusion task
before training starts?

For each of the first --n BIT-val reference sentences, with BIT's text prompt and no neural input
("neural activity:" + "-> sentence:" + sentence + EOS, padded with EOS to the 32-token block like Stage 2), it
reports, in nats per supervised token:
- diffusion: block_diffusion_ce, the Stage 2 objective (complementary masks, U(eps, 1) ratio per block), with the
  same masks for every model (one seed per sentence). Prompt tokens are context; sentence, EOS and pad are targets.
- causal: next-token CE of the sentence + EOS given the prompt, with the LM as an ordinary causal Qwen2 (not for
  Fast-dLLM v2, which is not a causal LM). A sanity check that the LM itself reads BIT text normally.
Models: Qwen2.5-1.5B-Instruct (the control's LM), Aero-1-Audio's LM (loaded as Stage 2 loads it), Fast-dLLM v2 1.5B
(NVIDIA's converted model, the reference). Paired bootstrap CIs over sentences for Aero - Instruct.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external" / "bit")]

from bit_experiments.diffusion_llm import BCI_DiffusionLLM, aero_language_model, block_diffusion_ce  # noqa: E402

HF = ROOT / "checkpoints" / "hf"
MASK_ID, EOS_ID, BLOCK = 151665, 151645, 32
SENTENCES = ROOT / "results" / "stage2" / "B_ssl_qwen25-1.5b-instruct-diffusion_s42" / "decodes" / "val_greedy_real.csv"


def rows(tok, n):
    a, b = tok("neural activity:")["input_ids"], tok("-> sentence:")["input_ids"]  # BIT's prompt, split at "#"
    refs = [r["reference"] for r in csv.DictReader(open(SENTENCES))][:n]
    out = []
    for s in refs:
        prompt, target = a + b, tok(" " + s)["input_ids"] + [EOS_ID]
        pad = (-(len(prompt) + len(target))) % BLOCK
        out.append((prompt, target + [EOS_ID] * pad, len(target)))
    return refs, out


@torch.no_grad()
def diffusion_ce(llm, data):
    decoder, lm_head, embed = llm.model, llm.lm_head, llm.get_input_embeddings()
    res = []
    for i, (prompt, target, _) in enumerate(data):
        ids = torch.tensor([prompt + target])
        labels = torch.tensor([[-100] * len(prompt) + target])
        torch.manual_seed(1000 + i)  # the same masks for every model
        ce, n, _, _ = block_diffusion_ce(decoder, lm_head, embed, embed(ids), labels, BLOCK, MASK_ID)
        res.append((float(ce), int(n)))
    return res


@torch.no_grad()
def causal_ce(lm, data):
    res = []
    for prompt, target, n_real in data:
        ids = torch.tensor([prompt + target[:n_real]])
        logits = lm(input_ids=ids).logits[0, len(prompt) - 1:-1].float()
        res.append((float(torch.nn.functional.cross_entropy(logits, ids[0, len(prompt):], reduction="sum")), n_real))
    return res


def causal_aero():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg, state = aero_language_model(HF / "Aero-1-Audio")
    lm = Qwen2ForCausalLM(Qwen2Config(**cfg)).float()
    missing, unexpected = lm.load_state_dict(state, strict=False)
    assert set(missing) <= {"lm_head.weight"} and not unexpected, (missing, unexpected)
    lm.tie_weights()
    return lm.eval()


def summary(name, res):
    ce = np.array([c for c, _ in res]); n = np.array([k for _, k in res])
    return f"{name}: {ce.sum() / n.sum():.4f} nats/token over {n.sum()} tokens"


def paired(a, b, seed=0, n_boot=5000):
    ca, cb = np.array([c for c, _ in a]), np.array([c for c, _ in b]); n = np.array([k for _, k in a])
    d = ca.sum() / n.sum() - cb.sum() / n.sum()
    idx = np.random.default_rng(seed).integers(0, len(n), (n_boot, len(n)))
    bs = (ca[idx].sum(1) - cb[idx].sum(1)) / n[idx].sum(1)
    return f"{d:+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(HF / "Qwen2.5-1.5B-Instruct")
    refs, data = rows(tok, args.n)
    print(f"{len(data)} BIT-val sentences, {sum(len(t) for _, t, _ in data)} target tokens incl. EOS padding", flush=True)
    diff, caus = {}, {}
    for name, path in (("instruct", "Qwen2.5-1.5B-Instruct"), ("aero", "Aero-1-Audio"), ("fast-dllm-v2", "Fast_dLLM_v2_1.5B")):
        llm = BCI_DiffusionLLM.from_pretrained(HF / path).eval()
        diff[name] = diffusion_ce(llm, data)
        print("diffusion", summary(name, diff[name]), flush=True)
        del llm
    for name in ("instruct", "aero"):
        lm = (AutoModelForCausalLM.from_pretrained(HF / "Qwen2.5-1.5B-Instruct", torch_dtype=torch.float32).eval()
              if name == "instruct" else causal_aero())
        caus[name] = causal_ce(lm, data)
        print("causal", summary(name, caus[name]), flush=True)
        del lm
    print("diffusion aero - instruct:", paired(diff["aero"], diff["instruct"]))
    print("diffusion fast-dllm-v2 - instruct:", paired(diff["fast-dllm-v2"], diff["instruct"]))
    print("causal aero - instruct:", paired(caus["aero"], caus["instruct"]))


if __name__ == "__main__":
    main()

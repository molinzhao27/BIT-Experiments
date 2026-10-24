"""Diagnostic sentence contrasts (DESIGN.md §5; docs/PROBES_FOILS.md): forced choice between a held-out trial's true
sentence and a constructed foil, scored given the trial's neural data. The neural data are fixed, so every contrast is
built on the text side.

**Construction** (CPU, deterministic: one generator per (seed, split, trial, foil type), so adding trials or types
changes no other foil; tables in configs/foils.yaml):
- `paraphrase`:
  - synonym: one content word replaced by a curated near-synonym;
  - contraction: "do not" <-> "don't" and the like.
  A semantic code should accept these.
- `sound_alike`: one content word replaced by a CMU-dictionary neighbour, a common real word at phoneme edit
  distance 1 (stress ignored). Homophones and inflections of the same word are excluded.
- `order_swap`: two adjacent words swapped, so the words and phonemes are kept and only the order changes.
- `negation`: the first negation removed, or else "not" inserted after the first copula / modal.
- `number_name`: a number word, or a day / month / place / first name, replaced by another of its category with the
  same number of syllables where possible.

Each row records what changed and the sentence-level phoneme edit distance (CMU dictionary), so effects can be
compared at matched phonetic distance.

**Scores** (higher = the decoder prefers the sentence):
- `loglik` for autoregressive decoders: the teacher-forced log-likelihood of sentence + EOS given [prompt, neural
  tokens], as BIT's training CE computes it.
- `denoise` for block-diffusion decoders: minus the training loss of block_diffusion_ce, per sentence. Draws:
  - per 32-token block a mask ratio t ~ U(eps, 1); each target token is masked with probability t;
  - a complementary copy masks the other target tokens;
  - the CE is summed over the masked tokens of both copies (sentence + EOS + the block's EOS padding, each token
    exactly once per pair), averaged over n_draws pairs.

  It is the expected value of the objective the decoder was trained on, estimated by Monte Carlo. The draws are
  common random numbers: the same masks for every sentence of a trial and in every condition. That cancels most of
  the noise in a difference, and the Monte Carlo standard error is reported. It is not a calibrated likelihood: the
  1/t-weighted ELBO has unbounded variance with complementary masks (weight 1/(1 - t) for the complement).
- `aligner`: cosine of BIT's in-loop contrastive head (ModalityAligner: mean of projector outputs vs mean of the text
  embeddings of [prompt, sentence, EOS]); every Stage 2 run has one.
- `stage1`: cosine of arm C's Stage 1 head (neural embedding vs the sentence's fixed LLM target, recomputed for
  foils with text_targets.compute_targets).

**Conditions:**
- `real`;
- `null_shuffled`: another trial's neural data, from stage2's seeded derangement (the pairing of its decodes);
- `null_prompt_only`: no neural tokens, so the score is the decoder's text prior.

The neural evidence for a contrast is the real margin minus the null margin.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

TYPES = ("paraphrase", "sound_alike", "order_swap", "negation", "number_name")
SPLIT_CODE = {"train": 0, "val": 1, "test": 2}
SUFFIXES = ("s", "es", "ed", "d", "ing", "er", "ers", "est", "ly", "'s", "n", "en", "y")
COLUMNS = ("split", "trial", "type", "variant", "reference", "foil", "position", "from", "to", "n_words_ref",
           "n_words_foil", "phon_dist")


# ---------------------------------------------------------------- lexicon (CMU dictionary + word frequency)

class Lexicon:
    """CMU pronunciations (stress removed) and wordfreq Zipf frequencies, with an edit-distance-1 neighbour index."""

    def __init__(self, cmu: dict[str, list[list[str]]] | None = None, zipf=None, min_zipf: float = 3.0,
                 blocklist=(), stopwords=()):
        if cmu is None:
            import cmudict

            cmu = cmudict.dict()
        if zipf is None:
            from wordfreq import zipf_frequency

            zipf = lambda w: zipf_frequency(w, "en")  # noqa: E731
        self.prons = {w: sorted({tuple(p.rstrip("012") for p in pr) for pr in prs}) for w, prs in cmu.items()}
        self.by_pron: dict[tuple, list[str]] = {}
        for w, prs in self.prons.items():
            for p in prs:
                self.by_pron.setdefault(p, []).append(w)
        self.phones = sorted({p for prs in self.prons.values() for pr in prs for p in pr})
        self.zipf, self.min_zipf = zipf, min_zipf
        self.blocklist, self.stopwords = set(blocklist), set(stopwords)
        self._neighbours: dict[str, list[str]] = {}

    def pron(self, word: str):
        p = self.prons.get(word)
        return p[0] if p else None

    def syllables(self, word: str) -> int | None:
        p = self.pron(word)
        return None if p is None else sum(ph[0] in "AEIOU" for ph in p)

    def _edits(self, pron: tuple) -> set[tuple]:
        out = set()
        for i in range(len(pron) + 1):
            for ph in self.phones:
                out.add(pron[:i] + (ph,) + pron[i:])  # insertion
                if i < len(pron) and ph != pron[i]:
                    out.add(pron[:i] + (ph,) + pron[i + 1:])  # substitution
            if i < len(pron):
                out.add(pron[:i] + pron[i + 1:])  # deletion
        out.discard(pron)
        return out

    def neighbours(self, word: str) -> list[str]:
        """Real words at phoneme edit distance exactly 1 (no shared pronunciation), most frequent first; not the
        word's inflections, not stopwords or blocked words, Zipf >= min_zipf."""
        if word in self._neighbours:
            return self._neighbours[word]
        own = set(self.prons.get(word, ()))
        cands = set()
        for p in own:
            for e in self._edits(p):
                cands.update(self.by_pron.get(e, ()))
        keep = [v for v in cands if v != word and v.isalpha() and not own & set(self.prons[v])
                and v not in self.blocklist and v not in self.stopwords and not inflection(word, v)]
        scored = sorted(((self.zipf(v), v) for v in keep), key=lambda x: (-x[0], x[1]))
        self._neighbours[word] = [v for z, v in scored if z >= self.min_zipf]
        return self._neighbours[word]

    def sentence_phones(self, words: list[str]) -> list[str] | None:
        out = []
        for w in words:
            p = self.pron(w)
            if p is None:
                return None
            out += list(p)
        return out


def inflection(a: str, b: str) -> bool:
    """b is a regular inflection / derivation of a (or a of b): use / used, run / running, big / bigger."""
    for x, y in ((a, b), (b, a)):
        for s in SUFFIXES:
            if y in (x + s, x[:-1] + s if x.endswith("e") else None, x + x[-1:] + s,
                     x[:-1] + "i" + s if x.endswith("y") else None):
                return True
    return False


# ---------------------------------------------------------------- foil rules (each: words, tables, rng -> list[...])

def _pick(rng, options):
    return options[int(rng.integers(len(options)))] if options else None


def foil_synonym(words, t, rng, lex=None):
    syn, stop = t["synonyms"], set(t["stopwords"])
    pos = [i for i, w in enumerate(words) if w in syn and w not in stop]
    i = _pick(rng, pos)
    if i is None:
        return None
    return words[:i] + [syn[words[i]]] + words[i + 1:], {"variant": "synonym", "position": i, "from": words[i],
                                                         "to": syn[words[i]]}


def foil_contraction(words, t, rng, lex=None):
    contract = {tuple(k.split()): v for k, v in t["contractions"].items()}
    expand = {v: k.split() for k, v in t["contractions"].items()}
    options = []
    for i, w in enumerate(words):
        if w in expand:
            exp = list(expand[w])
            if w.endswith("'s") and i + 1 < len(words) and words[i + 1] in t["has_before"]:
                exp = [exp[0], "has"]
            options.append((i, 1, exp))
        for n in (1, 2):
            key = tuple(words[i:i + n])
            if len(key) != n or key not in contract:
                continue
            after = words[i + n] if i + n < len(words) else None
            if n == 2 and key[1] in ("will", "have") and after == "not":
                continue  # "won't" / "haven't", never "they'll not"
            if n == 2 and key[1] == "have" and after not in t["contract_have_before"]:
                continue  # "i've got", never "you've really strong views"
            options.append((i, n, [contract[key]]))
    o = _pick(rng, options)
    if o is None:
        return None
    i, n, new = o
    return words[:i] + new + words[i + n:], {"variant": "contraction", "position": i, "from": " ".join(words[i:i + n]),
                                             "to": " ".join(new)}


def foil_sound_alike(words, t, rng, lex):
    stop = set(t["stopwords"])
    pos = [i for i, w in enumerate(words) if w not in stop and len(w) >= 3 and w.isalpha() and lex.neighbours(w)]
    i = _pick(rng, pos)
    if i is None:
        return None
    v = lex.neighbours(words[i])[0]
    return words[:i] + [v] + words[i + 1:], {"variant": "neighbour", "position": i, "from": words[i], "to": v}


def foil_order_swap(words, t, rng, lex=None):
    pos = [i for i in range(len(words) - 1) if words[i] != words[i + 1]]
    i = _pick(rng, pos)
    if i is None:
        return None
    new = list(words)
    new[i], new[i + 1] = new[i + 1], new[i]
    return new, {"variant": "adjacent", "position": i, "from": f"{words[i]} {words[i + 1]}",
                 "to": f"{words[i + 1]} {words[i]}"}


def foil_negation(words, t, rng, lex=None):
    remove = t["negation_remove"]
    pos = [i for i, w in enumerate(words) if w in remove]
    if pos:
        i = pos[0]
        new = words[:i] + ([remove[words[i]]] if remove[words[i]] else []) + words[i + 1:]
        if new:
            return new, {"variant": "remove", "position": i, "from": words[i], "to": remove[words[i]]}
    pos = [i for i, w in enumerate(words) if w in set(t["negation_insert_after"])]
    if not pos:
        return None
    i = pos[0]
    return words[:i + 1] + ["not"] + words[i + 1:], {"variant": "insert", "position": i + 1, "from": "", "to": "not"}


def _categories(t) -> dict[str, tuple[str, str]]:
    cat = {}
    for c, ws in t["numbers"].items():
        cat.update({w: ("number", c) for w in ws})
    for c, ws in t["names"].items():
        cat.update({w: ("name", c) for w in ws})
    return cat


def foil_number_name(words, t, rng, lex):
    cat = _categories(t)

    def eligible(i, w):
        if w not in cat:
            return False
        variant, c = cat[w]
        prev, nxt = (words[i - 1] if i else None), (words[i + 1] if i + 1 < len(words) else None)
        if variant == "number" and t.get("number_compound") and any(
                x in cat and cat[x][0] == "number" for x in (prev, nxt)):
            return False  # "two thousand", "eighty seven"
        if c == "ordinal" and prev not in t["ordinal_after"]:
            return False  # "the first year", not "do first"
        return not (variant == "name" and prev in t.get("name_not_after", ()))

    pos = [i for i, w in enumerate(words) if eligible(i, w)]
    i = _pick(rng, pos)
    if i is None:
        return None
    w = words[i]
    variant, c = cat[w]
    group = t["numbers"][c] if variant == "number" else t["names"][c]
    cands = [v for v in group if v != w]
    same = [v for v in cands if lex is not None and lex.syllables(v) is not None and lex.syllables(v) == lex.syllables(w)]
    v = _pick(rng, same or cands)
    return words[:i] + [v] + words[i + 1:], {"variant": f"{variant}:{c}", "position": i, "from": w, "to": v}


def check_tables(t: dict) -> dict:
    """Every word in the tables is a string (YAML 1.1 reads bare yes / no / on / off as booleans)."""
    bad = []

    def walk(x, path):
        if isinstance(x, dict):
            for k, v in x.items():
                if not isinstance(k, str):
                    bad.append((path, k))
                walk(v, f"{path}.{k}")
        elif isinstance(x, list):
            for v in x:
                walk(v, path)
        elif not isinstance(x, str):
            bad.append((path, x))

    for key in ("stopwords", "synonyms", "contractions", "has_before", "contract_have_before", "negation_remove",
                "negation_insert_after", "numbers", "names", "ordinal_after", "name_not_after", "blocklist"):
        walk(t.get(key, []), key)
    if bad:
        raise ValueError(f"non-string entries in the foil tables (quote them): {bad[:5]}")
    return t


RULES = {"paraphrase": (foil_synonym, foil_contraction), "sound_alike": (foil_sound_alike,),
         "order_swap": (foil_order_swap,), "negation": (foil_negation,), "number_name": (foil_number_name,)}


def build(sentences: list[str], split: str, tables: dict, lex: Lexicon, seed: int = 0) -> list[dict]:
    """Every applicable foil of every sentence (at most one per rule)."""
    from bit_experiments.phoneme_frames import edit_distance
    from bit_experiments.textmetrics import normalize

    rows = []
    for trial, ref in enumerate(sentences):
        words = ref.split()
        ref_ph = lex.sentence_phones(words)
        for ti, kind in enumerate(TYPES):
            for ri, rule in enumerate(RULES[kind]):
                rng = np.random.default_rng([seed, SPLIT_CODE[split], trial, ti, ri])
                out = rule(words, tables, rng, lex)
                if out is None:
                    continue
                new, info = out
                foil = " ".join(new)
                if normalize(foil) == normalize(ref):
                    continue
                foil_ph = lex.sentence_phones(new)
                dist = (None if ref_ph is None or foil_ph is None
                        else edit_distance(ref_ph, foil_ph) / max(len(ref_ph), len(foil_ph)))
                rows.append({"split": split, "trial": trial, "type": kind, "reference": ref, "foil": foil,
                             "n_words_ref": len(words), "n_words_foil": len(new), "phon_dist": dist, **info})
    return [{k: r.get(k) for k in COLUMNS} for r in rows]


def tables_digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_foils(path: Path):
    """data/processed/foils/<sha12>/ (or its parent with LATEST) -> (DataFrame, manifest)."""
    import pandas as pd

    path = Path(path)
    if (path / "LATEST").exists():
        path = path / (path / "LATEST").read_text().strip()
    df = pd.read_csv(path / "foils.csv", keep_default_na=False, dtype={"from": str, "to": str})
    manifest = json.loads((path / "manifest.json").read_text())
    if hashlib.sha256((path / "foils.csv").read_bytes()).hexdigest() != manifest["csv_sha256"]:
        raise ValueError(f"{path}/foils.csv was edited after it was built (edit configs/foils.yaml and rebuild)")
    return df, manifest | {"dir": str(path)}


# ---------------------------------------------------------------- scoring: batches of one trial's sentences

def encode_texts(texts: list[str], tokenizer, prompt: str) -> dict:
    """BIT's create_llm_labels for several sentences of one trial: [prompt_a, prompt_b, sentence + EOS], targets -100
    on the prompt; right-padded with 0 (BIT's pad value for input_ids)."""
    a = list(tokenizer(prompt.split("#")[0])["input_ids"])
    b = list(tokenizer(prompt.split("#")[1])["input_ids"])
    rows = [a + b + list(tokenizer(t + tokenizer.eos_token)["input_ids"]) for t in texts]
    n = max(map(len, rows))
    ids = torch.zeros(len(rows), n, dtype=torch.long)
    att = torch.zeros(len(rows), n, dtype=torch.long)
    tgt = torch.full((len(rows), n), -100, dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, :len(r)], att[i, :len(r)] = torch.tensor(r), 1
        tgt[i, len(a) + len(b):len(r)] = torch.tensor(r[len(a) + len(b):])
    return {"input_ids": ids, "attention_mask": att, "targets": tgt, "input_split": torch.full((len(rows),), len(a)),
            "n_prompt": len(a) + len(b)}


def neural_inputs(spikes: np.ndarray, n_rows: int) -> dict:
    """One trial's neural input repeated for n_rows sentences (no padding: every row is the lone trial)."""
    x = torch.as_tensor(np.asarray(spikes), dtype=torch.float32)
    t = x.shape[0]
    return {"spikes": x.expand(n_rows, t, x.shape[1]).contiguous(),
            "spikes_mask": torch.ones(n_rows, t, dtype=torch.long),
            "spikes_timestamp": torch.arange(t).expand(n_rows, t).contiguous(),
            "spikes_lengths": torch.full((n_rows,), t), "dataset_name": torch.zeros(n_rows, dtype=torch.long)}


@contextmanager
def no_neural_tokens(model):
    """prepare_embeds without neural tokens (null_prompt_only): the decoder sees the prompt and the sentence only."""
    def prepare_embeds(input_ids, spikes, spikes_mask, spikes_timestamp, dataset_name=None, day_idx=None):
        text = model.llm.get_input_embeddings()(input_ids)
        empty = text.new_zeros(text.shape[0], 0, text.shape[-1])
        return empty, text, spikes_mask.new_zeros(text.shape[0], 0), None

    model.prepare_embeds = prepare_embeds
    try:
        yield
    finally:
        del model.prepare_embeds


def _to(d: dict, device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in d.items()}


@torch.no_grad()
def ar_loglik(model, spikes, texts, tokenizer, prompt, device="cpu", dtype=None, prompt_only=False) -> np.ndarray:
    """Teacher-forced log-likelihood of each text (sentence + EOS) given the trial's neural tokens, from BIT's own
    forward (packed merge: every row equals its lone trial)."""
    enc = encode_texts(texts, tokenizer, prompt)
    inputs = _to({k: v for k, v in enc.items() if k != "n_prompt"} | neural_inputs(spikes, len(texts)), device)
    device = torch.device(device)
    ctx = no_neural_tokens(model) if prompt_only else _null()
    with ctx, torch.autocast(device.type, dtype=dtype or torch.bfloat16, enabled=dtype is not None):
        out = model(**inputs)
    logp = F.log_softmax(out.preds[:, :-1].float(), -1)
    tgt = out.targets[:, 1:]
    ok = tgt != -100
    ll = logp.gather(-1, tgt.clamp_min(0)[..., None])[..., 0]
    return (ll * ok).sum(1).cpu().numpy()


@contextmanager
def _null():
    yield


class _Captured(Exception):
    pass


def rowwise_block_ce(decoder, lm_head, embed, clean, labels, mask_id: int, masked: torch.Tensor) -> torch.Tensor:
    """block_diffusion_ce (diffusion_llm) with given masks, summed per row instead of over the batch."""
    from bit_experiments.diffusion_llm import diffusion_layout

    b, length = labels.shape
    is_target = labels != -100
    mask_embed = embed(torch.tensor([mask_id], device=labels.device)).to(clean.dtype)

    def noisy_copy(m):
        m = m & is_target
        return torch.cat([torch.where(m[..., None], mask_embed, clean), clean], dim=1), torch.where(m, labels, -100)

    (x1, l1), (x2, l2) = noisy_copy(masked), noisy_copy(~masked)
    x, lab = torch.cat([x1, x2]), torch.cat([l1, l2])
    with diffusion_layout(decoder, training=True):
        hidden = decoder(inputs_embeds=x, labels=lab).last_hidden_state[:, :length]
    shifted = lab[:, 1:]
    supervised = shifted != -100
    ce = F.cross_entropy(lm_head(hidden[:, :-1][supervised]).float(), shifted[supervised], reduction="none")
    rows = torch.arange(2 * b, device=labels.device)[:, None].expand_as(shifted)[supervised] % b
    return torch.zeros(b, device=labels.device, dtype=ce.dtype).index_add_(0, rows, ce)


def draw_masks(length: int, block_size: int, generator: torch.Generator) -> torch.Tensor:
    """block_diffusion_ce's draws for one row: a ratio per block, a uniform per position."""
    from bit_experiments.diffusion_llm import EPS

    n_blocks = length // block_size
    ratio = (1 - EPS) * torch.rand(n_blocks, generator=generator) + EPS
    return (torch.rand(n_blocks, block_size, generator=generator) < ratio[:, None]).reshape(length)


@torch.no_grad()
def diffusion_denoise(model, spikes, texts, tokenizer, prompt, n_draws: int, seed: int, device="cpu", dtype=None,
                      prompt_only=False) -> tuple[np.ndarray, np.ndarray]:
    """(score, Monte Carlo SE) per text: minus the mean over n_draws complementary mask pairs of the summed training
    CE. The prefix and targets are built by the model's own forward (captured before its loss), and the same masks
    are used for every text (seeded by `seed`)."""
    enc = encode_texts(texts, tokenizer, prompt)
    inputs = _to({k: v for k, v in enc.items() if k != "n_prompt"} | neural_inputs(spikes, len(texts)), device)
    device = torch.device(device)
    captured = {}

    def capture(clean, labels):
        captured.update(clean=clean, labels=labels)
        raise _Captured

    model.denoising_ce = capture
    try:
        with (no_neural_tokens(model) if prompt_only else _null()), \
                torch.autocast(device.type, dtype=dtype or torch.bfloat16, enabled=dtype is not None):
            try:
                model(**inputs)
            except _Captured:
                pass
            clean, labels = captured["clean"], captured["labels"]
            decoder, lm_head, embed = model._parts()
            g = torch.Generator().manual_seed(int(seed))
            draws = []
            for _ in range(n_draws):
                m = draw_masks(labels.shape[1], model.block_size, g).to(device)
                draws.append(rowwise_block_ce(decoder, lm_head, embed, clean, labels, model.mask_id,
                                              m[None].expand_as(labels)))
    finally:
        del model.denoising_ce
    d = -torch.stack(draws).float().cpu().numpy()  # [n_draws, n_texts]
    return d.mean(0), d.std(0, ddof=1) / np.sqrt(n_draws) if n_draws > 1 else np.zeros(len(texts))


@torch.no_grad()
def aligner_cosine(model, spikes, texts, tokenizer, prompt, device="cpu", dtype=None) -> np.ndarray:
    """BIT's contrastive head (ModalityAligner): cosine between the trial's mean projector output and each row's mean
    text embedding over its own tokens [prompt, sentence, EOS]."""
    enc = encode_texts(texts, tokenizer, prompt)
    nin = _to(neural_inputs(spikes, 1), device)
    device = torch.device(device)
    with torch.autocast(device.type, dtype=dtype or torch.bfloat16, enabled=dtype is not None):
        h, _ = model.prepare_neural_embeds(nin["spikes"], nin["spikes_mask"], nin["spikes_timestamp"],
                                           nin["dataset_name"])
        z = F.normalize(model.modality_aligner.spike_proj(model.projector(h).mean(1)).float(), dim=-1)
        emb = model.llm.get_input_embeddings()
        out = []
        for i in range(len(texts)):
            ids = enc["input_ids"][i, enc["attention_mask"][i].bool()].to(device)
            t = F.normalize(model.modality_aligner.text_proj(emb(ids).mean(0)).float(), dim=-1)
            out.append(float(z[0] @ t))
    return np.asarray(out)


# ---------------------------------------------------------------- forced choice

def forced_choice(scores, n_boot: int = 2000, seed: int = 0) -> dict:
    """scores: DataFrame with columns trial, type, variant, condition, score_true, score_foil (one row per foil and
    condition). Per (type, variant) and pooled per type: accuracy (true preferred; ties 1/2) and mean margin in every
    condition, and the neural evidence real - null (accuracy and margin), with paired trial-bootstrap 95% CIs."""
    import pandas as pd

    df = scores.copy()
    df["margin"] = df.score_true - df.score_foil
    df["win"] = (df.margin > 0) + 0.5 * (df.margin == 0)
    out = {}
    groups = [("type", t, df[df.type == t]) for t in sorted(df.type.unique())]
    groups += [("variant", f"{t}/{v}", g) for (t, v), g in df.groupby(["type", "variant"])]
    rng = np.random.default_rng(seed)
    for level, key, g in groups:
        wide = g.pivot_table(index=["trial", "variant"], columns="condition", values=["win", "margin"])
        res = {"level": level, "n": int(len(wide))}
        for c in sorted(g.condition.unique()):
            res[f"acc_{c}"] = float(wide["win"][c].mean())
            res[f"margin_{c}"] = float(wide["margin"][c].mean())
        trials = wide.index.get_level_values(0).to_numpy()
        ids, inv = np.unique(trials, return_inverse=True)
        for c in [c for c in g.condition.unique() if c != "real"]:
            for stat in ("win", "margin"):
                diff = (wide[stat]["real"] - wide[stat][c]).to_numpy()
                ok = ~np.isnan(diff)
                per = np.bincount(inv[ok], weights=diff[ok], minlength=len(ids))
                cnt = np.bincount(inv[ok], minlength=len(ids))
                boot = [per[s].sum() / max(1, cnt[s].sum())
                        for s in (rng.integers(0, len(ids), len(ids)) for _ in range(n_boot))]
                name = "acc" if stat == "win" else "margin"
                res[f"{name}_real_minus_{c}"] = float(diff[ok].mean()) if ok.any() else float("nan")
                res[f"{name}_real_minus_{c}_ci"] = np.percentile(boot, [2.5, 97.5]).tolist() if ok.any() else None
        out[key] = res
    return out


def neural_evidence_table(scores, null: str = "null_shuffled"):
    """One row per foil: its phoneme distance, type and the real - null margin."""
    s = scores.assign(margin=scores.score_true - scores.score_foil)
    w = s.pivot_table(index=["trial", "type", "variant", "foil", "phon_dist"], columns="condition",
                      values="margin").reset_index()
    w["evidence"] = w["real"] - w[null]
    return w


def type_effects(scores, null: str = "null_shuffled", reference: str = "sound_alike", n_boot: int = 2000,
                 seed: int = 0) -> dict:
    """OLS: evidence ~ 1 + phon_dist + type dummies (reference type left out). Effects of each type relative to
    `reference` at matched phoneme distance, with trial-bootstrap 95% CIs."""
    w = neural_evidence_table(scores, null)
    w = w[w.phon_dist.astype(str) != ""]
    w = w.assign(phon_dist=w.phon_dist.astype(float)).dropna(subset=["phon_dist", "evidence"])
    types = [t for t in sorted(w.type.unique()) if t != reference]
    x = np.column_stack([np.ones(len(w)), w.phon_dist] + [(w.type == t).to_numpy(float) for t in types])
    y = w.evidence.to_numpy()
    names = ["intercept", "phon_dist"] + [f"{t}_vs_{reference}" for t in types]
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    trials = w.trial.to_numpy()
    ids = np.unique(trials)
    rows = {t: np.flatnonzero(trials == t) for t in ids}
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[t] for t in rng.choice(ids, len(ids))])
        boot.append(np.linalg.lstsq(x[idx], y[idx], rcond=None)[0])
    ci = np.percentile(np.asarray(boot), [2.5, 97.5], axis=0)
    return {"null": null, "reference": reference, "n_foils": int(len(w)), "n_trials": int(len(ids)),
            "coef": {n: {"est": float(c), "ci": [float(ci[0, i]), float(ci[1, i])]} for i, (n, c) in
                     enumerate(zip(names, coef))}}

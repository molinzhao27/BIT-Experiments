"""Probes and RSA of frozen representations (DESIGN.md §5; docs/PROBES_FOILS.md). Also the held-out erasure check of
docs/RETRAINING_ABLATIONS.md.

- **Frame-level phoneme probe:** multinomial logistic regression on standardized frame features. It is full batch,
  L-BFGS, float32, with an L2 penalty chosen on the val split by val log-loss (calibrated probabilities, so the
  information estimate is meaningful); test is reported.
  - Labels: phoneme_frames' segment labels (40 classes: 39 phonemes + SIL).
  - Reported: accuracy, balanced accuracy, log-loss, and information = prior log-loss - probe log-loss (bits per
    frame; the prior is the train class frequencies).
  - Chance: the majority class.
  - CIs: a cluster bootstrap over trials, since frames within a trial are not independent.
  - An MLP probe (one hidden layer) measures what a non-linear reader recovers, e.g. after linear erasure.
- **RSA** (BIT App. L.1):
  - Each trial's sequence is cut into 10 segments, and the segment means are concatenated.
  - RDMs are 1 - cosine. They are compared by the Pearson correlation of their upper triangles.
  - The model RDM comes from the cached LLM sentence embeddings (bit_experiments.text_targets: mean-pooled, L2).
  - Partial RSA regresses covariate RDMs (phonetic: normalized phoneme edit distance; optionally length and duration)
    out of both RDMs, by least squares on the upper triangles, and correlates the residuals.
  - p-values: Mantel permutation (the neural RDM's rows and columns permuted together).
  - CIs: a trial bootstrap with the pairs of a resampled trial with itself dropped (Nili et al. 2014).

Python 3.10-compatible; torch and numpy only (runs in BIT's environment and in the project venv).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from bit_experiments.phoneme_frames import edit_distance

LOG2 = float(np.log(2))


# ---------------------------------------------------------------- linear and MLP probes

def _standardize(train: torch.Tensor):
    mean = train.mean(0)
    sd = train.std(0).clamp_min(1e-6)
    return mean, sd


def _fit_lbfgs(x: torch.Tensor, y: torch.Tensor, n_classes: int, l2: float, max_iter: int) -> torch.Tensor:
    w = torch.zeros(x.shape[1] + 1, n_classes, dtype=x.dtype, device=x.device, requires_grad=True)
    opt = torch.optim.LBFGS([w], lr=1, max_iter=max_iter, history_size=20, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-6, tolerance_change=1e-9)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(x @ w[:-1] + w[-1], y) + l2 * (w[:-1] ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach()


def fit_linear_probe(x_train, y_train, x_val, y_val, n_classes: int, l2_grid=(1e-5, 1e-4, 1e-3, 1e-2),
                     max_iter: int = 300, device="cpu") -> dict:
    """Multinomial logistic regression; the L2 weight (on the mean CE) chosen by val log-loss. Returns the probe."""
    xt, xv = (torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device) for a in (x_train, x_val))
    yt, yv = (torch.as_tensor(np.asarray(a), dtype=torch.long, device=device) for a in (y_train, y_val))
    mean, sd = _standardize(xt)
    xt, xv = (xt - mean) / sd, (xv - mean) / sd
    best, grid = None, []
    for l2 in l2_grid:
        w = _fit_lbfgs(xt, yt, n_classes, l2, max_iter)
        logits = xv @ w[:-1] + w[-1]
        acc = float((logits.argmax(1) == yv).float().mean())
        loss = float(F.cross_entropy(logits, yv))
        grid.append({"l2": l2, "val_acc": acc, "val_loss": loss})
        if best is None or (-loss, acc) > (-best[1], best[0]):
            best = (acc, loss, l2, w)
    return {"kind": "linear", "mean": mean, "sd": sd, "w": best[3], "l2": best[2], "grid": grid,
            "prior": np.bincount(np.asarray(y_train), minlength=n_classes) / len(y_train), "n_classes": n_classes}


def fit_mlp_probe(x_train, y_train, x_val, y_val, n_classes: int, hidden: int = 256, epochs: int = 30,
                  batch_size: int = 4096, lr: float = 1e-3, patience: int = 3, seed: int = 0, device="cpu") -> dict:
    """One hidden layer (ReLU), Adam, early stopping on val loss; seeded."""
    g = torch.Generator().manual_seed(seed)
    xt, xv = (torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device) for a in (x_train, x_val))
    yt, yv = (torch.as_tensor(np.asarray(a), dtype=torch.long, device=device) for a in (y_train, y_val))
    mean, sd = _standardize(xt)
    xt, xv = (xt - mean) / sd, (xv - mean) / sd
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = torch.nn.Sequential(torch.nn.Linear(xt.shape[1], hidden), torch.nn.ReLU(),
                                  torch.nn.Linear(hidden, n_classes)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    best, state, bad = float("inf"), None, 0
    for _ in range(epochs):
        net.train()
        for idx in torch.randperm(len(xt), generator=g).split(batch_size):
            idx = idx.to(device)
            opt.zero_grad()
            F.cross_entropy(net(xt[idx]), yt[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            loss = float(F.cross_entropy(net(xv), yv))
        if loss < best - 1e-4:
            best, state, bad = loss, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state)
    return {"kind": "mlp", "mean": mean, "sd": sd, "net": net.eval(), "val_loss": best,
            "prior": np.bincount(np.asarray(y_train), minlength=n_classes) / len(y_train), "n_classes": n_classes}


@torch.no_grad()
def probe_logits(probe: dict, x) -> torch.Tensor:
    x = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=probe["mean"].device)
    x = (x - probe["mean"]) / probe["sd"]
    if probe["kind"] == "linear":
        return x @ probe["w"][:-1] + probe["w"][-1]
    return probe["net"](x)


def score_probe(probe: dict, x, y, trials=None, n_boot: int = 1000, seed: int = 0) -> dict:
    """Accuracy, balanced accuracy, log-loss and information gain (bits / frame) on (x, y); cluster-bootstrap 95% CIs
    of accuracy and information over `trials` (trial id per frame)."""
    y = np.asarray(y)
    logits = probe_logits(probe, x)
    logp = F.log_softmax(logits.float(), -1).cpu().numpy()
    pred = logp.argmax(1)
    correct = pred == y
    nll = -logp[np.arange(len(y)), y]
    prior = np.clip(probe["prior"], 1e-12, None)
    prior_nll = -np.log(prior[y])
    classes = np.unique(y)
    res = {"n_frames": int(len(y)), "accuracy": float(correct.mean()),
           "balanced_accuracy": float(np.mean([correct[y == c].mean() for c in classes])),
           "log_loss": float(nll.mean()), "prior_log_loss": float(prior_nll.mean()),
           "info_bits": float((prior_nll.mean() - nll.mean()) / LOG2),
           "majority": float((y == int(np.argmax(probe["prior"]))).mean())}
    if trials is not None and n_boot:
        trials = np.asarray(trials)
        ids, inv = np.unique(trials, return_inverse=True)
        per = np.stack([np.bincount(inv, weights=w, minlength=len(ids))
                        for w in (correct.astype(float), prior_nll - nll, np.ones(len(y)))])
        rng = np.random.default_rng(seed)
        boot = []
        for _ in range(n_boot):
            s = per[:, rng.integers(0, len(ids), len(ids))].sum(1)
            boot.append((s[0] / s[2], s[1] / s[2] / LOG2))
        boot = np.asarray(boot)
        res["accuracy_ci"] = np.percentile(boot[:, 0], [2.5, 97.5]).tolist()
        res["info_bits_ci"] = np.percentile(boot[:, 1], [2.5, 97.5]).tolist()
        res["n_trials"] = int(len(ids))
    return res


def strip(probe: dict) -> dict:
    """The JSON-able part of a fitted probe."""
    return {k: v for k, v in probe.items() if k in ("kind", "l2", "grid", "val_loss", "n_classes")}


# ---------------------------------------------------------------- RSA

def segment_pool(feats: np.ndarray, n_segments: int = 10) -> np.ndarray:
    """BIT App. L.1: the sequence [T, d] cut into n_segments contiguous parts, their means concatenated."""
    feats = np.asarray(feats, np.float64)
    if len(feats) < n_segments:
        raise ValueError(f"{len(feats)} frames < {n_segments} segments")
    return np.concatenate([p.mean(0) for p in np.array_split(feats, n_segments)])


def cosine_rdm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float64)
    v = v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-12)
    return 1 - v @ v.T


def upper(m: np.ndarray) -> np.ndarray:
    return m[np.triu_indices(len(m), 1)]


def sequence_rdm(seqs: list) -> np.ndarray:
    """Normalized edit distance (edits / longer length) between sequences (e.g. phoneme label sequences)."""
    n = len(seqs)
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            m[i, j] = m[j, i] = edit_distance(seqs[i], seqs[j]) / max(1, len(seqs[i]), len(seqs[j]))
    return m


def abs_diff_rdm(x) -> np.ndarray:
    """|x_i - x_j| / max(x_i, x_j) (e.g. lengths, durations)."""
    x = np.asarray(x, np.float64)
    return np.abs(x[:, None] - x[None]) / np.maximum(x[:, None], x[None]).clip(1e-12)


def _residualizer(covariates: list[np.ndarray]):
    """v -> v minus its least-squares fit on [1, covariates] (upper-triangle vectors)."""
    if not covariates:
        return lambda v: v - v.mean()
    q, _ = np.linalg.qr(np.column_stack([np.ones(len(covariates[0]))] + list(covariates)))
    return lambda v: v - q @ (q.T @ v)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)).clip(1e-300))


def rsa(neural: np.ndarray, model: np.ndarray, covariates: list[np.ndarray] = (), n_perm: int = 1000,
        n_boot: int = 1000, seed: int = 0) -> dict:
    """(Partial) RSA of two RDMs [N, N]: Pearson r of the upper triangles after regressing `covariates` (RDMs) out of
    both; one-sided Mantel p (r > permuted r); trial-bootstrap 95% CI."""
    n = len(neural)
    iu = np.triu_indices(n, 1)
    covs = [np.asarray(c)[iu] for c in covariates]
    res_fn = _residualizer(covs)
    mod = res_fn(model[iu])
    r = _pearson(res_fn(neural[iu]), mod)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_perm):
        p = rng.permutation(n)
        exceed += _pearson(res_fn(neural[np.ix_(p, p)][iu]), mod) >= r
    out = {"r": r, "p_mantel": (exceed + 1) / (n_perm + 1) if n_perm else None, "n_items": n, "n_pairs": len(iu[0]),
           "n_perm": n_perm}
    if n_boot:
        boot = []
        for _ in range(n_boot):
            s = rng.integers(0, n, n)
            i, j = np.triu_indices(n, 1)
            keep = s[i] != s[j]
            a, b = s[i][keep], s[j][keep]
            cv = [np.asarray(c)[a, b] for c in covariates]
            f = _residualizer(cv)
            boot.append(_pearson(f(neural[a, b]), f(model[a, b])))
        out["ci"] = np.percentile(boot, [2.5, 97.5]).tolist()
        out["n_boot"] = n_boot
    return out

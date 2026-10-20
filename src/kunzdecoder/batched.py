"""Pooled-variance Gaussian NB (diagonal LDA) batched over label permutations, in torch.

Every function carries a leading permutation axis P: labels are (P, n) and models are
(P, W, K, F) over W candidate windows. Row 0 is normally the unpermuted labels, so the
observed statistic and its permutation null come out of the same code path. float64
throughout, so P = 1 reproduces the numpy implementation in `models.py` exactly.

`nested_cv` uses one fixed partition for every row. Its permutation null is
`nested_cv_refold`, which re-stratifies the folds on each permuted row, as the observed
pipeline does on the true labels. A transfer null keeps the decoder fixed and permutes
the test labels instead (`transfer_predict` + `permuted_labels(..., groups=blocks)`).
"""

from __future__ import annotations

import numpy as np
import torch


def onehot(labels: torch.Tensor, k: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(labels, k).to(torch.float64)


def fit(X: torch.Tensor, Y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """X (n, W, F), Y (P, n, K) one-hot -> means (P, W, K, F), pooled var (P, W, F)."""
    n, k = X.shape[0], Y.shape[-1]
    counts = Y.sum(1).clamp_min(1.0)  # (P, K)
    means = torch.einsum("pnk,nwf->pwkf", Y, X) / counts[:, None, :, None]
    ss = (X * X).sum(0)  # (W, F)
    var = (ss[None] - torch.einsum("pk,pwkf->pwf", counts, means * means)) / max(n - k, 1)
    return means, var.clamp_min(1e-12)


def scores(means: torch.Tensor, var: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Discriminant (monotone in the class log-likelihood) -> (P, n, W, K)."""
    inv = 1.0 / var
    q1 = torch.einsum("nwf,pwf->pnw", X * X, inv)
    q2 = torch.einsum("nwf,pwkf->pnwk", X, means * inv[:, :, None, :])
    q3 = torch.einsum("pwkf,pwf->pwk", means * means, inv)
    return 2 * q2 - q1[..., None] - q3[:, None]


def select_window(X: torch.Tensor, labels: torch.Tensor, k: int, folds: list) -> torch.Tensor:
    """Inner-CV window choice per permutation -> (P,) window indices (first maximum on ties)."""
    correct = torch.zeros(labels.shape[0], X.shape[1], dtype=torch.float64, device=X.device)
    Y = onehot(labels, k)
    for tr, va in folds:
        m, v = fit(X[tr], Y[:, tr])
        pred = scores(m, v, X[va]).argmax(-1)  # (P, n_va, W)
        correct += (pred == labels[:, va, None]).sum(1)
    return correct.argmax(1)


def fit_at(X: torch.Tensor, labels: torch.Tensor, k: int, w: torch.Tensor):
    """Fit each permutation's model only at its own window w[p] -> means (P, K, F), var (P, F)."""
    Xp = X[:, w].permute(1, 0, 2)  # (P, n, F)
    Y = onehot(labels, k)
    counts = Y.sum(1).clamp_min(1.0)
    means = torch.einsum("pnk,pnf->pkf", Y, Xp) / counts[..., None]
    ss = (Xp * Xp).sum(1)
    var = (ss - torch.einsum("pk,pkf->pf", counts, means * means)) / max(X.shape[0] - k, 1)
    return means, var.clamp_min(1e-12)


def predict_at(means: torch.Tensor, var: torch.Tensor, X: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Predict X (n, W, F) at each permutation's window w[p] -> (P, n) class indices."""
    Xp = X[:, w].permute(1, 0, 2)  # (P, n, F)
    inv = 1.0 / var
    s = (2 * torch.einsum("pnf,pkf->pnk", Xp, means * inv[:, None])
         - torch.einsum("pnf,pf->pn", Xp * Xp, inv)[..., None]
         - torch.einsum("pkf,pf->pk", means * means, inv)[:, None])
    return s.argmax(-1)


def nested_cv(X: torch.Tensor, labels: torch.Tensor, k: int, outer: list, inner: list) -> dict:
    """Nested CV (inner folds pick the window, outer folds evaluate) for every permutation row.

    outer: [(tr, te)], inner[i]: [(tr, va)] indices relative to outer[i]'s tr.
    Returns correct counts (P,) and chosen windows (P, n_outer).
    """
    P = labels.shape[0]
    correct = torch.zeros(P, dtype=torch.long, device=X.device)
    chosen = torch.empty(P, len(outer), dtype=torch.long, device=X.device)
    for i, (tr, te) in enumerate(outer):
        best = select_window(X[tr], labels[:, tr], k, inner[i])
        m, v = fit_at(X[tr], labels[:, tr], k, best)
        correct += (predict_at(m, v, X[te], best) == labels[:, te]).sum(1)
        chosen[:, i] = best
    return {"correct": correct, "windows": chosen}


def nested_cv_refold(X: torch.Tensor, labels: torch.Tensor, k: int, folds_of) -> dict:
    """`nested_cv` where every row gets its own partition: folds_of(labels[p]) -> (outer, inner),
    as `decoding.nested_folds` returns them. A permuted row thus goes through the same
    label-dependent stratified splitting as the true labels.

    Stratified fold sizes depend only on n, so the rows' folds stack into (P, size) index arrays;
    training sets enter as 0/1 masks over all n trials.
    """
    P, n = labels.shape
    dev = X.device
    parts = [folds_of(row) for row in labels.cpu().numpy()]
    Y = onehot(labels, k)
    XX = X * X
    rows = torch.arange(P, device=dev)[:, None]
    idx = lambda a: torch.as_tensor(np.stack(a), device=dev)  # noqa: E731

    def masked_fit(tr):  # (P, n_tr) absolute indices -> means (P, W, K, F), var (P, W, F)
        M = torch.zeros(P, n, dtype=torch.float64, device=dev)
        M[rows, tr] = 1.0
        YM = Y * M[..., None]
        counts = YM.sum(1).clamp_min(1.0)
        means = torch.einsum("pnk,nwf->pwkf", YM, X) / counts[:, None, :, None]
        ss = torch.einsum("pn,nwf->pwf", M, XX)
        var = (ss - torch.einsum("pk,pwkf->pwf", counts, means * means)) / max(tr.shape[1] - k, 1)
        return means, var.clamp_min(1e-12)

    def row_scores(means, var, Xv):  # Xv (P, n_v, W, F) -> (P, n_v, W, K)
        inv = 1.0 / var
        q1 = torch.einsum("pnwf,pwf->pnw", Xv * Xv, inv)
        q2 = torch.einsum("pnwf,pwkf->pnwk", Xv, means * inv[:, :, None, :])
        q3 = torch.einsum("pwkf,pwf->pwk", means * means, inv)
        return 2 * q2 - q1[..., None] - q3[:, None]

    n_outer = len(parts[0][0])
    correct = torch.zeros(P, dtype=torch.long, device=dev)
    chosen = torch.empty(P, n_outer, dtype=torch.long, device=dev)
    for i in range(n_outer):
        tr = idx([o[i][0] for o, _ in parts])
        te = idx([o[i][1] for o, _ in parts])
        win = torch.zeros(P, X.shape[1], dtype=torch.float64, device=dev)
        for j in range(len(parts[0][1][i])):
            itr = idx([o[i][0][inn[i][j][0]] for o, inn in parts])
            iva = idx([o[i][0][inn[i][j][1]] for o, inn in parts])
            m, v = masked_fit(itr)
            pred = row_scores(m, v, X[iva]).argmax(-1)  # (P, n_va, W)
            win += (pred == labels[rows, iva][..., None]).sum(1)
        best = win.argmax(1)
        m, v = masked_fit(tr)
        m, v = m[rows[:, 0], best][:, None], v[rows[:, 0], best][:, None]  # (P, 1, K, F), (P, 1, F)
        Xt = X[te, best[:, None]][:, :, None]  # (P, n_te, 1, F)
        correct += (row_scores(m, v, Xt)[:, :, 0].argmax(-1) == labels[rows, te]).sum(1)
        chosen[:, i] = best
    return {"correct": correct, "windows": chosen}


def transfer_predict(Xa: torch.Tensor, labels_a: torch.Tensor, k: int, inner: list,
                     tests: dict[str, torch.Tensor]) -> dict:
    """Train on behavior A (window chosen by inner CV on A only), predict each behavior B.

    tests[name] = Xb (n_b, W_a, F) evaluated at A's candidate windows transferred to B's event.
    Returns predicted classes per test (P, n_b) and chosen windows (P,).
    """
    best = select_window(Xa, labels_a, k, inner)
    m, v = fit_at(Xa, labels_a, k, best)
    return {"windows": best} | {name: predict_at(m, v, Xb, best) for name, Xb in tests.items()}


def transfer(Xa: torch.Tensor, labels_a: torch.Tensor, k: int, inner: list,
             tests: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> dict:
    """`transfer_predict` scored against yb: tests[name] = (Xb, yb (n_b,) true labels).
    Returns correct counts per test (P,) and chosen windows (P,).
    """
    out = transfer_predict(Xa, labels_a, k, inner, {name: Xb for name, (Xb, _) in tests.items()})
    return {"windows": out["windows"]} | {name: (out[name] == yb[None]).sum(1) for name, (_, yb) in tests.items()}


def permuted_labels(y: np.ndarray, n_perm: int, rng: np.random.Generator,
                    groups: np.ndarray | None = None) -> np.ndarray:
    """(1 + n_perm, n): row 0 = true labels, rows 1.. = independent permutations
    (within each group, e.g. recording block, if groups is given)."""
    if groups is None:
        return np.vstack([y] + [rng.permutation(y) for _ in range(n_perm)])
    out = np.tile(y, (1 + n_perm, 1))
    for g in np.unique(groups):
        cols = np.flatnonzero(groups == g)
        out[1:, cols] = rng.permuted(np.tile(y[cols], (n_perm, 1)), axis=1)
    return out


def device_of(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)

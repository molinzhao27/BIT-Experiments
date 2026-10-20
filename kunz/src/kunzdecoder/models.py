"""Gaussian naive-Bayes-style classifiers, vectorized over a leading 'window' axis.

All functions take X of shape (n, W, F) — trials x candidate windows x features —
so one call fits/predicts every candidate window at once.

diag_lda: MATLAB fitcdiscr(..., 'DiscrimType','diaglinear') with a uniform prior,
          i.e. Gaussian NB with one pooled (class-shared) variance per feature.
          This is the official Kunz et al. "Gaussian Naive Bayes".
gnb:      sklearn GaussianNB-style per-class variances (var_smoothing=1e-9), uniform prior.
"""

from __future__ import annotations

import numpy as np


def fit(X: np.ndarray, y: np.ndarray, classes: np.ndarray, kind: str = "diag_lda") -> dict:
    X = X.astype(np.float64, copy=False)
    onehot = (y[:, None] == classes[None, :]).astype(np.float64)  # (n, K)
    counts = onehot.sum(0)  # (K,)
    means = np.einsum("nk,nwf->wkf", onehot, X) / counts[None, :, None]  # (W, K, F)
    resid = X - np.einsum("nk,wkf->nwf", onehot, means)
    if kind == "diag_lda":
        var = (resid**2).sum(0) / max(len(y) - len(classes), 1)  # (W, F) pooled
        var = np.maximum(var, 1e-12)[:, None, :]  # (W, 1, F)
    elif kind == "gnb":
        var = np.einsum("nk,nwf->wkf", onehot, resid**2) / counts[None, :, None]  # (W, K, F)
        var = var + 1e-9 * X.var(0).max(-1)[:, None, None]
    else:
        raise ValueError(kind)
    return {"means": means, "var": var, "classes": classes, "kind": kind}


def log_likelihood(model: dict, X: np.ndarray) -> np.ndarray:
    """(n, W, K) class log-likelihoods (uniform prior => argmax is the MAP class)."""
    X = X.astype(np.float64, copy=False)
    mu, var = model["means"], model["var"]  # (W,K,F), (W,1|K,F)
    inv = 1.0 / np.broadcast_to(var, mu.shape)  # (W, K, F)
    # sum_f (x - mu)^2 / var, expanded to avoid an (n, W, K, F) temporary
    quad = (np.einsum("nwf,wkf->nwk", X**2, inv) - 2 * np.einsum("nwf,wkf->nwk", X, mu * inv)
            + (mu**2 * inv).sum(-1)[None])
    ll = -0.5 * quad
    if model["kind"] == "gnb":
        ll = ll - 0.5 * np.log(var).sum(-1)[None]
    return ll


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    """(n, W) predicted class labels."""
    return model["classes"][log_likelihood(model, X).argmax(-1)]

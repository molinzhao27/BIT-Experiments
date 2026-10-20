"""Checks for the vectorized classifiers and window extraction (run: python -m pytest tests)."""

import sys
from pathlib import Path

import numpy as np
from sklearn.naive_bayes import GaussianNB

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kunzdecoder import features, models  # noqa: E402


def _data(seed=0, n=140, w=3, f=20, k=7):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(2, 2 + k), n // k)
    X = rng.normal(size=(n, w, f)) + 0.8 * (y[:, None, None] % 3)
    return X, y


def test_gnb_matches_sklearn():
    X, y = _data()
    classes = np.unique(y)
    m = models.fit(X, y, classes, "gnb")
    for wi in range(X.shape[1]):
        ref = GaussianNB(priors=np.full(len(classes), 1 / len(classes))).fit(X[:, wi], y)
        assert np.array_equal(models.predict(m, X)[:, wi], ref.predict(X[:, wi]))


def test_diag_lda_matches_naive():
    X, y = _data(1)
    classes = np.unique(y)
    m = models.fit(X, y, classes, "diag_lda")
    for wi in range(X.shape[1]):
        Z = X[:, wi]
        mu = np.stack([Z[y == c].mean(0) for c in classes])
        var = sum(((Z[y == c] - mu[i]) ** 2).sum(0) for i, c in enumerate(classes)) / (len(y) - len(classes))
        ll = -0.5 * (((Z[:, None] - mu[None]) ** 2) / var).sum(-1)
        assert np.array_equal(models.predict(m, X)[:, wi], classes[ll.argmax(1)])


def test_window_means_clip_to_block():
    x = np.arange(20, dtype=float)[:, None]
    block = np.array([1] * 10 + [2] * 10)
    out = features.window_means(x, np.array([2, 12]), block, np.array([0, 5, 20]), 4)
    assert out[0, 0, 0] == np.mean([2, 3, 4, 5])
    assert out[0, 1, 0] == np.mean([7, 8, 9])  # clipped at block end, not into block 2
    assert out[0, 2, 0] == np.mean([6, 7, 8, 9])  # fully outside -> last 4 bins of block
    assert out[1, 1, 0] == np.mean([17, 18, 19])

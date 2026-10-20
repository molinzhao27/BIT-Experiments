"""Window-average decoders from scikit-learn under the Phase 1 nested CV.

The inner folds pick the (window, hyperparameter) pair jointly; the outer folds evaluate.
Ties go to the earliest window, then the first grid value (grids are ordered strongest
regularization first).
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits


def make_model(name: str, hp, n_classes: int = 7):
    if name == "lda_shrinkage":  # Ledoit-Wolf shrinkage covariance, uniform prior (like the baseline)
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=np.full(n_classes, 1 / n_classes))
    if name == "logreg":  # multinomial, L2
        return LogisticRegression(C=hp, max_iter=5000)
    if name == "svm":  # linear SVM, one-vs-rest, squared hinge, L2
        return LinearSVC(C=hp, max_iter=50000)
    raise ValueError(name)


def outer_fold_job(name: str, grid: list, X: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray,
                   inner: list) -> dict:
    """One outer fold: inner-CV score for every (window, hp), refit the best, predict the test fold."""
    with threadpool_limits(1), warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        Xtr, ytr = X[tr], y[tr]
        correct = np.zeros((X.shape[1], len(grid)))
        for itr, iva in inner:
            for w in range(X.shape[1]):
                for h, hp in enumerate(grid):
                    m = make_model(name, hp).fit(Xtr[itr, w], ytr[itr])
                    correct[w, h] += (m.predict(Xtr[iva, w]) == ytr[iva]).sum()
        w, h = np.unravel_index(np.argmax(correct), correct.shape)  # first max: earliest window, first hp
        m = make_model(name, grid[h]).fit(Xtr[:, w], ytr)
        return {"te": te, "pred": m.predict(X[te][:, w]), "window_idx": int(w), "hp": grid[h],
                "inner_acc": correct / len(tr)}

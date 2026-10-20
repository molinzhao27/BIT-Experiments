"""Nested cross-validated window selection + decoding, and binomial statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest
from sklearn.model_selection import StratifiedKFold

from . import models


@dataclass
class NestedCVResult:
    y_true: np.ndarray  # (n,)
    y_pred: np.ndarray  # (n,)
    fold: np.ndarray  # (n,) outer fold of each trial
    window_idx: np.ndarray  # (n,) index into the candidate starts chosen for that trial's fold
    inner_acc: np.ndarray  # (n_folds, W) inner-CV accuracy of every candidate window

    @property
    def n_correct(self) -> int:
        return int((self.y_true == self.y_pred).sum())


def stratified_folds(y: np.ndarray, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    return list(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def nested_folds(y: np.ndarray, outer_folds: int, inner_folds: int, seed: int):
    """Outer folds (seed) and, per outer fold k, inner folds on its training set (seed + 1000 + k).

    Inner indices are relative to the outer training set.
    """
    outer = stratified_folds(y, outer_folds, seed)
    inner = [stratified_folds(y[tr], inner_folds, seed + 1000 + k) for k, (tr, _) in enumerate(outer)]
    return outer, inner


def select_window(X: np.ndarray, y: np.ndarray, classes: np.ndarray, n_folds: int, seed: int,
                  kind: str) -> tuple[int, np.ndarray]:
    """Inner k-fold CV accuracy for every candidate window; returns (best index, accuracies).

    One stratified partition is shared by all windows. Ties go to the earliest window
    (the official code keeps the first window with strictly lower loss).
    """
    correct = np.zeros(X.shape[1])
    for tr, va in stratified_folds(y, n_folds, seed):
        m = models.fit(X[tr], y[tr], classes, kind)
        correct += (models.predict(m, X[va]) == y[va][:, None]).sum(0)
    acc = correct / len(y)
    return int(np.argmax(acc)), acc


def nested_cv(X: np.ndarray, y: np.ndarray, outer_folds: int = 10, inner_folds: int = 10, seed: int = 42,
              kind: str = "diag_lda") -> NestedCVResult:
    """Outer k-fold evaluation; the window is chosen on each outer-training set only.

    X: (n_trials, n_candidate_windows, n_features) window-averaged features.
    """
    classes = np.unique(y)
    n = len(y)
    y_pred = np.empty_like(y)
    fold = np.empty(n, int)
    widx = np.empty(n, int)
    inner = np.empty((outer_folds, X.shape[1]))
    for k, (tr, te) in enumerate(stratified_folds(y, outer_folds, seed)):
        best, inner[k] = select_window(X[tr], y[tr], classes, inner_folds, seed + 1000 + k, kind)
        m = models.fit(X[tr][:, [best]], y[tr], classes, kind)
        y_pred[te] = models.predict(m, X[te][:, [best]])[:, 0]
        fold[te], widx[te] = k, best
    return NestedCVResult(y, y_pred, fold, widx, inner)


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    ci = binomtest(k, n).proportion_ci(confidence_level=level, method="exact")
    return float(ci.low), float(ci.high)


def confusion(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Row-normalized confusion matrix (rows = true class)."""
    c = np.zeros((len(classes), len(classes)))
    idx = {v: i for i, v in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        c[idx[t], idx[p]] += 1
    return c / np.maximum(c.sum(1, keepdims=True), 1)

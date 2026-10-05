"""Numerical helpers for the PS3-C ladder. numpy only, deterministic, CPU.

Nothing here decides a causal state; it computes numbers that the rungs in
``ps3c`` gate. Every estimator is small and closed-form so its behaviour can be
read and tested against planted worlds.
"""

from __future__ import annotations

import math

import numpy as np


def as_matrix(columns, n):
    """Stack 1-D columns into an (n, k) float matrix; k may be zero."""
    if not columns:
        return np.zeros((n, 0))
    return np.column_stack([np.asarray(c, dtype=float) for c in columns])


def standardize(w):
    """Column z-scores (constant columns stay zero); for scale-invariant penalised fits."""
    w = np.asarray(w, dtype=float)
    if w.ndim != 2 or w.shape[1] == 0:
        return w
    sd = w.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return (w - w.mean(0)) / sd


def add_const(x):
    return np.column_stack([np.ones(len(x)), x])


def ols(x, y, ridge=0.0):
    """OLS (optional tiny ridge on non-constant columns) with HC1 standard errors.

    ``x`` must already contain a constant column when one is wanted.
    Returns dict(beta, resid, se, df, fitted).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = x.shape
    xtx = x.T @ x
    if ridge:
        pen = np.eye(k) * ridge
        pen[0, 0] = 0.0
        xtx = xtx + pen
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ x.T @ y
    fitted = x @ beta
    resid = y - fitted
    df = max(n - k, 1)
    meat = (x * resid[:, None] ** 2).T @ x
    cov = xtx_inv @ meat @ xtx_inv * (n / df)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return {"beta": beta, "resid": resid, "se": se, "df": df, "fitted": fitted}


def logistic(x, t, ridge=1e-3, iters=100, tol=1e-10):
    """Ridge-penalised logistic regression by IRLS; ``x`` includes the constant."""
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    k = x.shape[1]
    beta = np.zeros(k)
    pen = np.eye(k) * ridge
    pen[0, 0] = 0.0
    for _ in range(iters):
        eta = np.clip(x @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-9, None)
        grad = x.T @ (t - p) - pen @ beta
        hess = (x * w[:, None]).T @ x + pen
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def chrono_folds(n, k=5, purge=0, min_train=20):
    """Expanding-window chronological folds over rows already sorted by time.

    Fold j trains on rows [0, start_j - purge) and tests on [start_j, end_j).
    ``purge`` drops the training rows whose outcome window may overlap the test
    block. Folds whose training part is smaller than ``min_train`` are skipped.
    """
    bounds = np.linspace(0, n, k + 2).astype(int)
    folds = []
    for j in range(1, k + 1):
        start, end = bounds[j], bounds[j + 1]
        tr_end = max(start - purge, 0)
        if tr_end < min_train or end <= start:
            continue
        folds.append((np.arange(0, tr_end), np.arange(start, end)))
    return folds


def crossfit_predict(x, y, k=5, ridge=0.0, logistic_model=False):
    """Out-of-fold predictions by contiguous blocks (both directions), no row predicts itself."""
    n = len(y)
    pred = np.empty(n)
    edges = np.linspace(0, n, k + 1).astype(int)
    for j in range(k):
        test = np.arange(edges[j], edges[j + 1])
        train = np.setdiff1d(np.arange(n), test)
        if logistic_model:
            beta = logistic(x[train], y[train])
            pred[test] = sigmoid(x[test] @ beta)
        else:
            beta = ols(x[train], y[train], ridge=ridge)["beta"]
            pred[test] = x[test] @ beta
    return pred


def block_indices(n, rng, block=None):
    """Moving-block bootstrap row indices for time-ordered episodes."""
    block = block or max(1, int(round(math.sqrt(n))))
    starts = rng.integers(0, max(n - block + 1, 1), size=int(math.ceil(n / block)))
    idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
    return idx


def rankdata(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x), dtype=float)
    # average ties
    vals = x[order]
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return corr(rankdata(x), rankdata(y))


def residualize(v, h):
    """Residual of ``v`` on [1, h] (h may have zero columns)."""
    x = add_const(h)
    return v - x @ ols(x, v)["beta"]


def bh_q(pvals):
    """Benjamini-Hochberg q-values; None entries stay None."""
    idx = [i for i, p in enumerate(pvals) if p is not None]
    q = [None] * len(pvals)
    if not idx:
        return q
    p = np.array([pvals[i] for i in idx], dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(ranked, 0, 1)
    for j, i in enumerate(idx):
        q[i] = float(out[j])
    return q


def smd(x, t, w=None):
    """Standardised mean difference of ``x`` between t==1 and t==0 (optionally weighted)."""
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=bool)
    w = np.ones(len(x)) if w is None else np.asarray(w, dtype=float)
    if t.sum() == 0 or (~t).sum() == 0:
        return None
    m1 = np.average(x[t], weights=w[t])
    m0 = np.average(x[~t], weights=w[~t])
    v1 = np.var(x[t])
    v0 = np.var(x[~t])
    s = math.sqrt((v1 + v0) / 2.0)
    return 0.0 if s == 0 else float(abs(m1 - m0) / s)


def robustness_value(t_stat, df, q=1.0, alpha=None):
    """Cinelli-Hazlett (2020) robustness value of an OLS coefficient.

    Minimum partial R2 an unobserved confounder would need with both treatment and
    outcome to reduce the estimate by 100*q percent (alpha=None) or to make it
    statistically indistinguishable from that reduction at level alpha.
    """
    if df <= 0:
        return None
    f = q * abs(t_stat) / math.sqrt(df)
    if alpha is not None:
        from statistics import NormalDist

        crit = NormalDist().inv_cdf(1 - alpha / 2) / math.sqrt(max(df - 1, 1))
        f = f - crit
        if f <= 0:
            return 0.0
    return float(0.5 * (math.sqrt(f ** 4 + 4 * f ** 2) - f ** 2))


def normal_two_sided_p(z):
    from statistics import NormalDist

    return float(2 * (1 - NormalDist().cdf(abs(z))))

"""Residual-correlation causal screening with iterative peeling.

Implements the two-model residual construction (Frisch-Waugh-Lovell partial
correlation) with an iterative peeling schedule, as described in:
Mandal, R.K. (2026). "Residual-Correlation Causal Screening with Localized
Supervised Modeling" (working paper).

Optimized implementation: each screening round computes ALL candidate
partial correlations from a single precision (inverse-covariance) matrix
of the remaining design -- O(m^3) per round -- instead of 2m separate
least-squares solves -- O(m * n * m^2).  Results are numerically identical
to the residual construction (FWL theorem); a per-candidate residual
fallback guards ill-conditioned designs.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from scipy import stats

_EPS = 1e-12


@dataclass
class ScreenResult:
    ranking: list = field(default_factory=list)       # feature names, decreasing causal power
    causal_power: dict = field(default_factory=dict)  # name -> partial correlation at selection round
    p_value: dict = field(default_factory=dict)       # name -> p-value at selection round
    missing_causal_power: float = np.nan              # share of response variance unexplained by ranked factors


def _validate(X: np.ndarray, y: np.ndarray) -> None:
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D after ravel, got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y have mismatched rows: {X.shape[0]} vs {y.shape[0]}")
    if X.shape[0] < 3:
        raise ValueError("need at least 3 samples")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or inf")
    if not np.isfinite(y).all():
        raise ValueError("y contains NaN or inf")


def _residuals(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """OLS residuals of y on X (with intercept)."""
    if X.shape[1] == 0:
        return y - y.mean()
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def _partial_corr(y: np.ndarray, xj: np.ndarray, Xrest: np.ndarray) -> tuple[float, float]:
    """Reference path: corr(resid(y|Xrest), resid(xj|Xrest)) and its p-value."""
    ey = _residuals(y, Xrest)
    ej = _residuals(xj, Xrest)
    if ey.std() < _EPS or ej.std() < _EPS:
        return 0.0, 1.0
    r = float(np.corrcoef(ey, ej)[0, 1])
    return r, _pval(r, len(y), Xrest.shape[1])


def _pval(r: float, n: int, k: int) -> float:
    df = max(n - k - 2, 1)
    t = r * np.sqrt(df / max(_EPS, 1.0 - r * r))
    return float(2 * stats.t.sf(abs(t), df))


def _round_partial_corrs(y: np.ndarray, Xr: np.ndarray) -> np.ndarray | None:
    """Partial corr of y with each column of Xr, given the other columns.

    Single precision-matrix computation for the whole round:
        Z = [Xr, y] centered;  P = inv(cov(Z));
        pcorr(y, x_j | rest) = -P[j, m] / sqrt(P[j, j] * P[m, m]).
    Returns None if the design is too ill-conditioned to trust (caller
    falls back to the per-candidate residual construction).
    """
    n, m = Xr.shape
    Z = np.column_stack([Xr, y])
    Z = Z - Z.mean(axis=0)
    sd = Z.std(axis=0)
    if (sd < _EPS).any():
        return None                      # constant column: use reference path
    Z /= sd                              # correlation scale for conditioning
    C = (Z.T @ Z) / n
    # reciprocal condition number guard
    if np.linalg.cond(C) > 1e10:
        return None
    P = np.linalg.inv(C)
    d = np.sqrt(np.diag(P))
    r = -P[:m, m] / (d[:m] * d[m])
    if not np.isfinite(r).all() or (np.abs(r) > 1.0 + 1e-8).any():
        return None
    return np.clip(r, -1.0, 1.0)


class CausalScreen:
    """Iterative residual-correlation screen ('peeling').

    Each round computes, for every remaining candidate j, the partial
    correlation of X_j with y given all other remaining candidates; the
    strongest significant candidate is recorded and REMOVED from all
    subsequent conditioning sets (so a discovered factor's influence cannot
    re-enter through the variables it confounds).
    """

    def __init__(self, alpha: float = 0.05, max_factors: int | None = None):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.max_factors = max_factors

    def fit(self, X, y, feature_names=None) -> ScreenResult:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        _validate(X, y)
        n, p = X.shape
        names = list(feature_names) if feature_names is not None else [f"x{j}" for j in range(p)]
        if len(names) != p:
            raise ValueError(f"feature_names has {len(names)} entries for {p} columns")
        remaining = list(range(p))
        res = ScreenResult()
        budget = self.max_factors or p
        y_cur = y.copy()          # response, progressively peeled of discovered factors
        for _ in range(budget):
            if not remaining:
                break
            m = len(remaining)
            k = m - 1                       # conditioning-set size per candidate
            rs = _round_partial_corrs(y_cur, X[:, remaining])
            if rs is None:
                # ill-conditioned round: exact per-candidate reference path
                best = None
                for j in remaining:
                    rest = [t for t in remaining if t != j]
                    r, pv = _partial_corr(y_cur, X[:, j], X[:, rest])
                    if best is None or abs(r) > abs(best[1]):
                        best = (j, r, pv)
                j, r, pv = best
            else:
                i = int(np.argmax(np.abs(rs)))
                j, r = remaining[i], float(rs[i])
                pv = _pval(r, n, k)
            if pv > self.alpha:
                break
            res.ranking.append(names[j])
            res.causal_power[names[j]] = r
            res.p_value[names[j]] = pv
            remaining.remove(j)
            # PEEL: strip the discovered factor's contribution out of the
            # response so its effect cannot re-enter the screen through
            # variables it confounds (paper Sec. 4.3).
            y_cur = _residuals(y_cur, X[:, [j]])
        res.missing_causal_power = float(np.var(y_cur) / np.var(y)) if np.var(y) > 0 else np.nan
        self.result_ = res
        return res

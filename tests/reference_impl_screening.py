"""Reference implementation (naive residual construction). Test oracle only.

Implements the two-model residual construction (Frisch-Waugh-Lovell partial
correlation) with an iterative peeling schedule, as described in:
Mandal, R.K. (2026). "Residual-Correlation Causal Screening with Localized
Supervised Modeling" (working paper).
"""
import numpy as np
from dataclasses import dataclass, field
from scipy import stats


@dataclass
class ScreenResult:
    ranking: list = field(default_factory=list)       # feature names, decreasing causal power
    causal_power: dict = field(default_factory=dict)  # name -> partial correlation at selection round
    p_value: dict = field(default_factory=dict)       # name -> p-value at selection round
    missing_causal_power: float = np.nan              # share of response variance unexplained by ranked factors


def _residuals(y, X):
    """OLS residuals of y on X (with intercept)."""
    if X.shape[1] == 0:
        return y - y.mean()
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def _partial_corr(y, xj, Xrest):
    """corr(resid(y|Xrest), resid(xj|Xrest)) and its p-value."""
    ey = _residuals(y, Xrest)
    ej = _residuals(xj, Xrest)
    if ey.std() < 1e-12 or ej.std() < 1e-12:
        return 0.0, 1.0
    r = float(np.corrcoef(ey, ej)[0, 1])
    n, k = len(y), Xrest.shape[1]
    df = max(n - k - 2, 1)
    t = r * np.sqrt(df / max(1e-12, 1 - r * r))
    p = 2 * stats.t.sf(abs(t), df)
    return r, float(p)


class CausalScreen:
    """Iterative residual-correlation screen ('peeling').

    Each round computes, for every remaining candidate j, the partial
    correlation of X_j with y given all other remaining candidates; the
    strongest significant candidate is recorded and REMOVED from all
    subsequent conditioning sets (so a discovered factor's influence cannot
    re-enter through the variables it confounds).
    """

    def __init__(self, alpha=0.05, max_factors=None):
        self.alpha = alpha
        self.max_factors = max_factors

    def fit(self, X, y, feature_names=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        p = X.shape[1]
        names = list(feature_names) if feature_names is not None else [f"x{j}" for j in range(p)]
        remaining = list(range(p))
        res = ScreenResult()
        budget = self.max_factors or p
        y_cur = y.copy()          # response, progressively peeled of discovered factors
        for _ in range(budget):
            if not remaining:
                break
            best = None
            for j in remaining:
                rest = [k for k in remaining if k != j]
                r, pv = _partial_corr(y_cur, X[:, j], X[:, rest])
                if best is None or abs(r) > abs(best[1]):
                    best = (j, r, pv)
            j, r, pv = best
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

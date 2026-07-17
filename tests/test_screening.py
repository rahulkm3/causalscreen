import numpy as np
from causalscreen import CausalScreen, PartitionedRegressor

def _synth(n=4000, seed=0):
    """Ground truth: x1 true cause; x2 = confounded proxy of x1 (no own effect);
    x3 pure noise; x4 weaker true cause."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = 0.9 * x1 + 0.3 * rng.normal(size=n)   # rides on x1
    x3 = rng.normal(size=n)
    x4 = rng.normal(size=n)
    y = 2.0 * x1 + 0.7 * x4 + 0.5 * rng.normal(size=n)
    X = np.column_stack([x1, x2, x3, x4])
    return X, y

def test_ranking_recovers_true_causes():
    X, y = _synth()
    res = CausalScreen().fit(X, y, ["x1", "x2", "x3", "x4"])
    assert set(res.ranking[:2]) == {"x1", "x4"}   # both true causes found
    assert "x2" not in res.ranking                  # confounded proxy rejected
    assert "x3" not in res.ranking                  # noise rejected

def test_raw_corr_vs_causal_power_exposes_proxy():
    X, y = _synth()
    raw2 = abs(np.corrcoef(X[:, 1], y)[0, 1])
    res = CausalScreen().fit(X, y, ["x1", "x2", "x3", "x4"])
    assert raw2 > 0.8                      # proxy looks great naively...
    assert "x2" not in res.ranking          # ...but is rejected by the screen

def test_missing_causal_power_bounds():
    X, y = _synth()
    res = CausalScreen().fit(X, y, ["x1", "x2", "x3", "x4"])
    assert 0.0 <= res.missing_causal_power < 0.2

def test_partitioned_regressor_runs_and_beats_nothing():
    rng = np.random.default_rng(1)
    n = 3000
    g = rng.integers(0, 4, n).astype(float)      # categorical causal factor
    x = rng.normal(size=n)
    slopes = np.array([2.0, -1.0, 0.5, 3.0])
    y = slopes[g.astype(int)] * x + 0.2 * rng.normal(size=n)
    X = np.column_stack([g, x])
    pr = PartitionedRegressor(partition_features=["g"], min_cell=50).fit(X, y, ["g", "x"])
    Xq = X[:200]
    err_local = np.mean((pr.predict(Xq) - y[:200]) ** 2)
    err_global = np.mean((pr.global_.predict(Xq) - y[:200]) ** 2)
    assert err_local < 0.25 * err_global    # local models slash error in-cell

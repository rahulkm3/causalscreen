"""Robustness suite: equivalence with the reference residual construction,
edge cases, batch/loop consistency, and input validation."""
import sys, pathlib
import numpy as np
import pytest

import importlib.util

_HERE = pathlib.Path(__file__).parent

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Reference implementations: the naive residual (FWL) construction and the
# per-query local regressor, kept as the mathematical spec the optimized
# code must reproduce exactly.
ref_screen = _load("reference_impl_screening", _HERE / "reference_impl_screening.py")
ref_local = _load("reference_impl_local", _HERE / "reference_impl_local.py")

from causalscreen import CausalScreen, PartitionedRegressor
from causalscreen.screening import _round_partial_corrs, _partial_corr


# ---------- screening: numerical equivalence with reference ----------

@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("n,p", [(200, 5), (500, 10), (120, 8)])
def test_round_pcorrs_match_residual_construction(seed, n, p):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(p, p))
    X = rng.normal(size=(n, p)) @ A          # correlated design
    y = X @ rng.normal(size=p) + rng.normal(size=n)
    rs = _round_partial_corrs(y, X)
    assert rs is not None
    for j in range(p):
        rest = np.delete(np.arange(p), j)
        r_ref, _ = _partial_corr(y, X[:, j], X[:, rest])
        assert rs[j] == pytest.approx(r_ref, abs=1e-8)

@pytest.mark.parametrize("seed", range(5))
def test_full_screen_matches_reference(seed):
    rng = np.random.default_rng(seed)
    n, p = 800, 7
    X = rng.normal(size=(n, p))
    X[:, 1] = 0.8 * X[:, 0] + 0.3 * rng.normal(size=n)     # proxy
    y = 1.5 * X[:, 0] - 0.9 * X[:, 4] + 0.5 * rng.normal(size=n)
    names = [f"f{j}" for j in range(p)]
    new = CausalScreen().fit(X, y, names)
    old = ref_screen.CausalScreen().fit(X, y, names)
    assert new.ranking == old.ranking
    for f in new.ranking:
        assert new.causal_power[f] == pytest.approx(old.causal_power[f], abs=1e-7)
        assert new.p_value[f] == pytest.approx(old.p_value[f], rel=1e-4, abs=1e-12)
    assert new.missing_causal_power == pytest.approx(old.missing_causal_power, abs=1e-9)

def test_constant_feature_falls_back_and_matches_reference():
    rng = np.random.default_rng(3)
    n = 300
    X = rng.normal(size=(n, 4)); X[:, 2] = 5.0             # constant column
    y = 2 * X[:, 0] + rng.normal(size=n)
    names = list("abcd")
    new = CausalScreen().fit(X, y, names)
    old = ref_screen.CausalScreen().fit(X, y, names)
    assert new.ranking == old.ranking
    assert "c" not in new.ranking

def test_perfectly_collinear_features_do_not_crash():
    rng = np.random.default_rng(4)
    n = 300
    x = rng.normal(size=n)
    X = np.column_stack([x, 2 * x, rng.normal(size=n)])    # exact collinearity
    y = x + 0.1 * rng.normal(size=n)
    res = CausalScreen().fit(X, y, ["a", "a2", "b"])
    assert isinstance(res.ranking, list)                    # no crash; sane output
    assert np.isfinite(res.missing_causal_power)

def test_max_factors_budget_respected():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(500, 6))
    y = X @ np.ones(6) + 0.1 * rng.normal(size=500)
    res = CausalScreen(max_factors=2).fit(X, y)
    assert len(res.ranking) <= 2

def test_screen_input_validation():
    with pytest.raises(ValueError):
        CausalScreen(alpha=0.0)
    with pytest.raises(ValueError):
        CausalScreen().fit(np.ones((5, 2)), np.ones(4))
    Xbad = np.ones((10, 2)); Xbad[0, 0] = np.nan
    with pytest.raises(ValueError):
        CausalScreen().fit(Xbad, np.ones(10))
    with pytest.raises(ValueError):
        CausalScreen().fit(np.random.default_rng(0).normal(size=(10, 3)),
                           np.zeros(10), ["a", "b"])       # name-count mismatch

# ---------- partitioned regressor: equivalence + batching ----------

def _cells_data(seed=1, n=2500):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 5, n).astype(float)
    h = rng.integers(0, 3, n).astype(float)
    x = rng.normal(size=n)
    y = (g - 2) * x + 0.5 * h + 0.2 * rng.normal(size=n)
    return np.column_stack([g, h, x]), y, ["g", "h", "x"]

def test_predictions_match_reference_implementation():
    X, y, names = _cells_data()
    Xq = X[:300]
    new = PartitionedRegressor(partition_features=["g", "h"], min_cell=40).fit(X, y, names)
    old = ref_local.PartitionedRegressor(partition_features=["g", "h"], min_cell=40).fit(X, y, names)
    np.testing.assert_allclose(new.predict(Xq), old.predict(Xq), atol=1e-8)

def test_batch_predict_equals_loop_predict():
    X, y, names = _cells_data(seed=2)
    pr = PartitionedRegressor(partition_features=["g"], min_cell=40).fit(X, y, names)
    Xq = np.vstack([X[:100], [[99.0, 0.0, 0.3]]])          # incl. unseen cell
    loop = np.array([pr.predict_one(x) for x in Xq])
    np.testing.assert_allclose(pr.predict(Xq), loop, atol=1e-10)

def test_out_of_support_uses_global_model():
    X, y, names = _cells_data(seed=3)
    pr = PartitionedRegressor(partition_features=["g"], min_cell=40).fit(X, y, names)
    xq = np.array([42.0, 1.0, 0.5])                        # cell never seen
    assert pr.predict_one(xq) == pytest.approx(float(pr.global_.predict(xq.reshape(1, -1))[0]))

def test_small_cell_falls_back_to_global():
    X, y, names = _cells_data(seed=4)
    pr = PartitionedRegressor(partition_features=["g"], min_cell=10**6).fit(X, y, names)
    np.testing.assert_allclose(pr.predict(X[:50]), pr.global_.predict(X[:50]), atol=1e-10)

def test_no_partition_features_is_pure_global():
    X, y, names = _cells_data(seed=5)
    pr = PartitionedRegressor().fit(X, y, names)
    np.testing.assert_allclose(pr.predict(X[:50]), pr.global_.predict(X[:50]), atol=1e-10)

def test_local_models_cached_one_fit_per_cell():
    X, y, names = _cells_data(seed=6)
    pr = PartitionedRegressor(partition_features=["g"], min_cell=40).fit(X, y, names)
    pr.predict(X[:500]); n_models = len(pr._models)
    pr.predict(X[:500])                                     # second pass: no refits
    assert len(pr._models) == n_models
    assert n_models <= len(pr._cells)

def test_partition_validation():
    X, y, names = _cells_data(seed=7)
    with pytest.raises(ValueError):
        PartitionedRegressor(partition_features=["nope"]).fit(X, y, names)
    with pytest.raises(ValueError):
        PartitionedRegressor(min_cell=1)

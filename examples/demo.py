"""Demo: confounded-proxy rejection + partitioned local modeling error reduction.
Run: python examples/demo.py
"""
import numpy as np
from causalscreen import CausalScreen, PartitionedRegressor

rng = np.random.default_rng(7)
n = 3000

# --- Part 1: screening on synthetic ground truth ---
x1 = rng.normal(size=n)                       # true cause (strong)
x2 = 0.9 * x1 + 0.3 * rng.normal(size=n)      # confounded proxy of x1
x3 = rng.normal(size=n)                       # noise
x4 = rng.normal(size=n)                       # true cause (weak)
y = 2.0 * x1 + 0.7 * x4 + 0.5 * rng.normal(size=n)
X = np.column_stack([x1, x2, x3, x4]); names = ["x1", "x2", "x3", "x4"]

res = CausalScreen().fit(X, y, names)
print("== Screening ==")
print(f"naive |corr| with y : " + ", ".join(f"{nm}={abs(np.corrcoef(X[:,j],y)[0,1]):.2f}" for j, nm in enumerate(names)))
print(f"causal ranking      : {res.ranking}")
print(f"causal powers       : " + ", ".join(f"{k}={v:.2f}" for k, v in res.causal_power.items()))
print(f"missing causal power: {res.missing_causal_power:.1%}  (noise floor of the DGP is ~5%)")

# --- Part 2: partition on a causal categorical factor; local vs global model ---
g = rng.integers(0, 5, n).astype(float)       # causal regime variable
xr = rng.normal(size=n)
slopes = np.array([2.0, -1.0, 0.5, 3.0, -2.5])
yr = slopes[g.astype(int)] * xr + 0.2 * rng.normal(size=n)
Xr = np.column_stack([g, xr])
tr, te = slice(0, 2500), slice(2500, None)

pr = PartitionedRegressor(partition_features=["g"], min_cell=50)   # linear base model
pr.fit(Xr[tr], yr[tr], ["g", "x"])
mse_local = np.mean((pr.predict(Xr[te]) - yr[te]) ** 2)
mse_global = np.mean((pr.global_.predict(Xr[te]) - yr[te]) ** 2)
print("\n== Partitioned local modeling (out-of-sample) ==")
print(f"global model MSE : {mse_global:.4f}")
print(f"local  models MSE : {mse_local:.4f}   (reduction: {1 - mse_local/mse_global:.1%})")

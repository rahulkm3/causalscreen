"""Causal-factor partitioning with inference-time localized training."""
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LinearRegression


class PartitionedRegressor:
    """Route each query to its causal-factor cell; train ONE local model there
    (interpolation). Out-of-support queries fall back to the global model
    (extrapolation), preserving generalization -- see paper Sec. 5.1.
    """

    def __init__(self, base_estimator=None, partition_features=None, min_cell=30):
        self.base = base_estimator if base_estimator is not None else LinearRegression()
        self.partition_features = partition_features or []
        self.min_cell = min_cell

    def fit(self, X, y, feature_names):
        self.names_ = list(feature_names)
        self.X_, self.y_ = np.asarray(X, float), np.asarray(y, float).ravel()
        self.idx_ = [self.names_.index(f) for f in self.partition_features]
        self.global_ = clone(self.base).fit(self.X_, self.y_)
        return self

    def _cell_mask(self, xq):
        mask = np.ones(len(self.X_), bool)
        for i in self.idx_:
            mask &= (self.X_[:, i] == xq[i])
            if mask.sum() < self.min_cell:      # stop before losing significance
                return None
        return mask

    def predict_one(self, xq):
        xq = np.asarray(xq, float)
        mask = self._cell_mask(xq) if self.idx_ else None
        if mask is not None and mask.sum() >= self.min_cell:
            local = clone(self.base).fit(self.X_[mask], self.y_[mask])
            return float(local.predict(xq.reshape(1, -1))[0])
        return float(self.global_.predict(xq.reshape(1, -1))[0])

    def predict(self, Xq):
        return np.array([self.predict_one(x) for x in np.asarray(Xq, float)])

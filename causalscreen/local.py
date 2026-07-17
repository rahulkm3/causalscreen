"""Causal-factor partitioning with inference-time localized training.

Optimized implementation: cells are indexed ONCE at fit time (dict of
cell-key -> row indices), local models are fitted at most once per cell
and cached, and batch prediction groups queries by cell so each cell's
model predicts its queries in a single vectorized call.  Semantics are
identical to the per-query construction: a query gets a local model iff
its exact cell holds >= min_cell training rows, else the global model.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LinearRegression


class PartitionedRegressor:
    """Route each query to its causal-factor cell; train ONE local model there
    (interpolation). Out-of-support queries fall back to the global model
    (extrapolation), preserving generalization -- see paper Sec. 5.1.

    Note: cell membership is exact-match on the partition features, so
    partition features should be discrete/categorical.  Continuous partition
    features will rarely match and queries will fall back to the global model.
    """

    def __init__(self, base_estimator=None, partition_features=None, min_cell: int = 30):
        if min_cell < 2:
            raise ValueError("min_cell must be >= 2")
        self.base = base_estimator if base_estimator is not None else LinearRegression()
        self.partition_features = partition_features or []
        self.min_cell = min_cell

    def fit(self, X, y, feature_names):
        self.names_ = list(feature_names)
        self.X_, self.y_ = np.asarray(X, float), np.asarray(y, float).ravel()
        if self.X_.ndim != 2 or self.X_.shape[0] != self.y_.shape[0]:
            raise ValueError("X must be 2-D with rows matching y")
        if not np.isfinite(self.X_).all() or not np.isfinite(self.y_).all():
            raise ValueError("X or y contains NaN or inf")
        missing = [f for f in self.partition_features if f not in self.names_]
        if missing:
            raise ValueError(f"partition features not in feature_names: {missing}")
        self.idx_ = [self.names_.index(f) for f in self.partition_features]
        self.global_ = clone(self.base).fit(self.X_, self.y_)
        # Index cells once: cell key -> training-row indices.
        self._cells: dict[tuple, np.ndarray] = {}
        if self.idx_:
            keys = self.X_[:, self.idx_]
            order = np.lexsort(keys.T[::-1])
            sorted_keys = keys[order]
            boundaries = np.flatnonzero(
                np.r_[True, (sorted_keys[1:] != sorted_keys[:-1]).any(axis=1)]
            )
            for s, e in zip(boundaries, np.r_[boundaries[1:], len(order)]):
                if e - s >= self.min_cell:
                    self._cells[tuple(sorted_keys[s])] = order[s:e]
        self._models: dict[tuple, object] = {}   # lazy per-cell model cache
        return self

    def _cell_model(self, key: tuple):
        """Fit (once) and return the cached local model for a qualifying cell."""
        mdl = self._models.get(key)
        if mdl is None:
            rows = self._cells[key]
            mdl = clone(self.base).fit(self.X_[rows], self.y_[rows])
            self._models[key] = mdl
        return mdl

    def predict_one(self, xq) -> float:
        xq = np.asarray(xq, float)
        if self.idx_:
            key = tuple(xq[self.idx_])
            if key in self._cells:
                return float(self._cell_model(key).predict(xq.reshape(1, -1))[0])
        return float(self.global_.predict(xq.reshape(1, -1))[0])

    def predict(self, Xq) -> np.ndarray:
        Xq = np.asarray(Xq, float)
        if Xq.ndim == 1:
            Xq = Xq.reshape(1, -1)
        out = np.empty(len(Xq), float)
        if not self.idx_:
            return self.global_.predict(Xq).astype(float)
        # Group queries by cell so each model predicts once, vectorized.
        keys = [tuple(row) for row in Xq[:, self.idx_]]
        groups: dict[tuple, list[int]] = {}
        fallback: list[int] = []
        for i, key in enumerate(keys):
            (groups.setdefault(key, []) if key in self._cells else fallback).append(i)
        for key, idxs in groups.items():
            out[idxs] = self._cell_model(key).predict(Xq[idxs])
        if fallback:
            out[fallback] = self.global_.predict(Xq[fallback])
        return out

"""Instance-based and neural candidate scorers: distance-weighted KNN and an MLP."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import config
from .common import N_JOBS


class SubsampledKNN(BaseEstimator, ClassifierMixin):
    """Distance-weighted KNN fitted on a random subsample (full data is too slow to query)."""

    def __init__(self, n_neighbors=50, max_train=100_000, random_state=config.SEED):
        self.n_neighbors = n_neighbors
        self.max_train = max_train
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(X), size=min(self.max_train, len(X)), replace=False)
        self.model_ = make_pipeline(StandardScaler(), KNeighborsClassifier(
            n_neighbors=self.n_neighbors, weights="distance", n_jobs=N_JOBS))
        self.model_.fit(X[idx], y[idx])
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X)


class EpochMLP(BaseEstimator, ClassifierMixin):
    """Standardised MLP trained epoch by epoch with partial_fit so train/val loss can be recorded."""

    def __init__(self, hidden=(64, 32), alpha=1e-4, lr=1e-3, batch_size=1024, epochs=20,
                 max_train=400_000, random_state=config.SEED):
        self.hidden = hidden
        self.alpha = alpha
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_train = max_train
        self.random_state = random_state

    def fit(self, X, y, X_val=None, y_val=None):
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(X), size=min(self.max_train, len(X)), replace=False)
        X, y = X[idx], y[idx]
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        Xv = self.scaler_.transform(X_val) if X_val is not None else None
        self.net_ = MLPClassifier(hidden_layer_sizes=self.hidden, alpha=self.alpha, learning_rate_init=self.lr,
                                  batch_size=self.batch_size, random_state=self.random_state)
        self.history_ = {"epoch": [], "train_loss": [], "val_loss": []}
        for ep in range(self.epochs):
            perm = rng.permutation(len(Xs))
            self.net_.partial_fit(Xs[perm], y[perm], classes=np.array([0, 1]))
            self.history_["epoch"].append(ep + 1)
            self.history_["train_loss"].append(float(log_loss(y, self.net_.predict_proba(Xs)[:, 1], labels=[0, 1])))
            if Xv is not None:
                self.history_["val_loss"].append(float(log_loss(y_val, self.net_.predict_proba(Xv)[:, 1], labels=[0, 1])))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self.net_.predict_proba(self.scaler_.transform(X))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

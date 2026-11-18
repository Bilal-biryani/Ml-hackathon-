"""Model zoo for candidate scoring (binary: is this candidate the reference arc?).

One file per algorithm family; model_zoo() lists every model compared in scripts/03_train_compare.py.
"""
from .boosting import make_lgbm
from .common import N_JOBS
from .linear import make_gaussian_nb, make_logistic_regression
from .neighbors_nn import EpochMLP, SubsampledKNN
from .trees import make_extra_trees, make_random_forest


def model_zoo():
    """name -> (family, short config string, factory)."""
    return {
        "LogisticRegression": ("Linear", "StandardScaler + L2, C=1.0, lbfgs, max_iter=1000", make_logistic_regression),
        "GaussianNB": ("Probabilistic", "var_smoothing=1e-9", make_gaussian_nb),
        "KNN": ("Instance-based", "k=50, distance-weighted, scaled, fit on 100k-row subsample",
                lambda: SubsampledKNN(n_neighbors=50, max_train=100_000)),
        "RandomForest": ("Ensemble (bagging)", "200 trees, min_samples_leaf=5, max_features=sqrt", make_random_forest),
        "ExtraTrees": ("Ensemble (randomised trees)", "200 trees, min_samples_leaf=5, max_features=sqrt", make_extra_trees),
        "LightGBM": ("Boosting", "400 trees, lr=0.05, num_leaves=63, min_child_samples=20", make_lgbm),
        "MLP": ("Neural network", "(64,32) ReLU, Adam lr=1e-3, alpha=1e-4, batch 1024, 20 epochs, 400k-row subsample",
                lambda: EpochMLP()),
    }


__all__ = ["model_zoo", "make_lgbm", "make_logistic_regression", "make_gaussian_nb", "make_random_forest",
           "make_extra_trees", "SubsampledKNN", "EpochMLP", "N_JOBS"]

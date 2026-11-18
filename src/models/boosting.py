"""Gradient boosting candidate scorer: LightGBM."""
from lightgbm import LGBMClassifier

from .. import config
from .common import N_JOBS


def make_lgbm(**kw):
    params = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=20,
                  subsample=1.0, colsample_bytree=1.0, reg_lambda=0.0, random_state=config.SEED,
                  n_jobs=N_JOBS, verbose=-1)
    params.update(kw)
    return LGBMClassifier(**params)

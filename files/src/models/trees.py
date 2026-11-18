"""Bagging-style tree ensembles: Random Forest and Extra Trees."""
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from .. import config
from .common import N_JOBS


def make_random_forest(**kw):
    params = dict(n_estimators=200, min_samples_leaf=5, max_features="sqrt", n_jobs=N_JOBS, random_state=config.SEED)
    params.update(kw)
    return RandomForestClassifier(**params)


def make_extra_trees(**kw):
    params = dict(n_estimators=200, min_samples_leaf=5, max_features="sqrt", n_jobs=N_JOBS, random_state=config.SEED)
    params.update(kw)
    return ExtraTreesClassifier(**params)

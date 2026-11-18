"""Loading of the processed modelling tables."""
import numpy as np
import pandas as pd

from . import config
from .io import read_arcs
from .metrics import RELIABLE_M


def load_partition(part):
    cands = pd.read_pickle(config.PROCESSED_DIR / f"{part}_cands.pkl")
    obs = pd.read_pickle(config.PROCESSED_DIR / f"{part}_obs.pkl")
    base = config.TRAIN_DIR if part == "train" else config.STUDENT_TEST_DIR
    arcs = {t: read_arcs(base / t) for t in obs["trip_id"].unique()}
    return cands, obs, arcs


def training_mask(cands, obs):
    """Rows usable for fitting: fixes with a reliable label that is present in the candidate set."""
    good = obs[(obs["align_dist"] <= RELIABLE_M) & obs["label_in_cands"]][["trip_id", "obs"]]
    key = pd.MultiIndex.from_frame(cands[["trip_id", "obs"]])
    return key.isin(pd.MultiIndex.from_frame(good))


def eval_rows_mask(cands, obs):
    """Candidate rows belonging to evaluation fixes (reliable reference label)."""
    good = obs[obs["align_dist"] <= RELIABLE_M][["trip_id", "obs"]]
    key = pd.MultiIndex.from_frame(cands[["trip_id", "obs"]])
    return key.isin(pd.MultiIndex.from_frame(good))


def xy(cands):
    return cands[config.FEATURES].to_numpy(dtype=np.float64), cands["y"].to_numpy(dtype=np.int64)

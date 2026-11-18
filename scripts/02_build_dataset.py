"""Build the candidate-level modelling tables for the train and student-test partitions.

Outputs (data/processed/):
  {part}_cands.pkl  one row per (GPS fix, candidate arc): features + binary label y
  {part}_obs.pkl    one row per GPS fix: reference arc, alignment distance, candidate coverage
  cleaning_log.csv  per-trip cleaning counts
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src import config
from src.io import list_trips
from src.pipeline import prepare_trip, reference_labels


def build_one(trip_dir, part):
    t0 = time.time()
    p = prepare_trip(trip_dir)
    lab, route = reference_labels(trip_dir, p)
    c = p.cands.merge(lab[["obs", "label_arc"]], on="obs", how="left")
    c["y"] = (c["arc"] == c["label_arc"]).astype(np.int8)
    c = c.drop(columns="label_arc")
    obs = lab.copy()
    obs.insert(0, "trip_id", p.inp.trip_id)
    obs["t"] = p.t
    obs["lon"] = p.inp.track[:, 0]
    obs["lat"] = p.inp.track[:, 1]
    obs["dt_prev"] = np.r_[np.nan, np.diff(p.t)]
    ncand = c.groupby("obs").size()
    obs["n_cand"] = ncand.reindex(obs["obs"], fill_value=0).to_numpy()
    hit = c.groupby("obs")["y"].max()
    obs["label_in_cands"] = hit.reindex(obs["obs"], fill_value=0).to_numpy().astype(bool)
    obs["partition"] = part
    log = dict(p.inp.cleaning_log, trip_id=p.inp.trip_id, partition=part,
               n_nodes=len(p.inp.nodes), n_arcs=len(p.inp.arcs), n_sub_arcs=len(p.net.sub_arcs),
               route_len=len(route), seconds=time.time() - t0)
    return c, obs, log


def main():
    logs = []
    for part, d in (("train", config.TRAIN_DIR), ("student_test", config.STUDENT_TEST_DIR)):
        trips = list_trips(d)
        if not trips:
            raise FileNotFoundError(
                f"No trip folders found in {d}. Place the unchanged college-provided "
                f"{part} partition there before building processed tables."
            )
        res = Parallel(n_jobs=config.BUILD_JOBS, verbose=0)(delayed(build_one)(t, part) for t in trips)
        cands = pd.concat([r[0] for r in res], ignore_index=True)
        obs = pd.concat([r[1] for r in res], ignore_index=True)
        logs += [r[2] for r in res]
        cands.to_pickle(config.PROCESSED_DIR / f"{part}_cands.pkl")
        obs.to_pickle(config.PROCESSED_DIR / f"{part}_obs.pkl")
        print(f"{part}: trips={len(trips)} fixes={len(obs)} candidate rows={len(cands)} "
              f"positives={int(cands.y.sum())} candidate-miss-rate={1 - obs.label_in_cands.mean():.4f} "
              f"align>50m={np.mean(obs.align_dist > 50):.4f}")
    pd.DataFrame(logs).to_csv(config.RESULTS_DIR / "cleaning_log.csv", index=False)


if __name__ == "__main__":
    main()

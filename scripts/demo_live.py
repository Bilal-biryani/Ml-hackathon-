"""Review 1 live demo: train two lightweight models on one trip-grouped fold and print the scores.

Usage (from the project folder):  python scripts/demo_live.py
Needs data/processed/ (run scripts/02_build_dataset.py first). It writes NO files, so the
committed results stay untouched.
"""
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

import numpy as np
from sklearn.model_selection import GroupKFold

from src import config
from src.data import eval_rows_mask, load_partition, training_mask, xy
from src.metrics import candidate_metrics, observation_metrics, top1_per_obs
from src.models import make_lgbm, make_logistic_regression

print(f"Project: {root}")
cands, obs, arcs = load_partition("train")
X, y = xy(cands)
groups = cands["trip_id"].to_numpy()
print(f"Train table: {len(cands):,} candidate rows, {len(obs):,} GPS points, {cands.trip_id.nunique()} trips, "
      f"{len(config.FEATURES)} features, {y.mean():.1%} positives")

tri, vai = next(GroupKFold(n_splits=5).split(X, y, groups))
tri = tri[training_mask(cands, obs)[tri]]
val_trips = np.unique(groups[vai])
print(f"Fold 0: train on {len(np.unique(groups[tri]))} trips, validate on {len(val_trips)} unseen trips\n")

c_val = cands[cands.trip_id.isin(val_trips)]
o_val = obs[obs.trip_id.isin(val_trips)]
ev = eval_rows_mask(c_val, o_val)

demo_models = (
    ("Logistic Regression", make_logistic_regression),
    ("LightGBM", lambda: make_lgbm(n_estimators=50, num_leaves=31, n_jobs=min(4, config.N_JOBS))),
)

for name, factory in demo_models:
    t0 = time.time()
    m = factory().fit(X[tri], y[tri])
    fit_s = time.time() - t0
    p = m.predict_proba(X[vai])[:, 1]
    cm = candidate_metrics(c_val.loc[ev, "y"].to_numpy(), p[ev])
    om, _ = observation_metrics(o_val, top1_per_obs(c_val, p), arcs)
    print(f"{name:20s} fit {fit_s:5.1f}s | macro F1 {cm['macro_f1']:.4f} | Top-1 accuracy {om['top1_acc']:.4f} | "
          f"log loss {cm['log_loss']:.4f} | route continuity {om['route_continuity']:.4f}")

print("\nFull 5-fold results for all 7 models: results/model_comparison.csv and figures/cmp_*.png")

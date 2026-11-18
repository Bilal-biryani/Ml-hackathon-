"""End-to-end inference used by the API and the map scripts.

MapMatcher.predict() sees only the GPS track and road network. Scoring against the reference route
happens in score_against_reference(), which the API calls separately on the server.
"""
import json
import time

import joblib
import numpy as np

from . import config, hmm
from .metrics import RELIABLE_M, continuity
from .pipeline import prepare_trip, reference_labels


class MapMatcher:
    def __init__(self, models_dir=config.MODELS_DIR):
        self.cfg = json.load(open(models_dir / "final_config.json"))
        self.model = joblib.load(models_dir / self.cfg["model_file"])
        hp = json.load(open(models_dir / "hmm_config.json"))
        self.hmm_params = hmm.HMMParams(**hp["params"])
        self.features = self.cfg["features"]

    def predict(self, trip_dir):
        timings = {}
        t0 = time.perf_counter()
        prep = prepare_trip(trip_dir)
        timings["preprocess_s"] = time.perf_counter() - t0
        n = len(prep.xy)
        cands = prep.cands.copy()
        t0 = time.perf_counter()
        cands["proba"] = self.model.predict_proba(cands[self.features].to_numpy(dtype=np.float64))[:, 1] if len(cands) else []
        timings["classifier_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        steps = hmm.build_steps(cands, n)
        trans = hmm.transition_distances(prep.net, prep.xy, prep.t, steps, self.hmm_params.gap_break_s,
                                         self.hmm_params.backtrack_tol_m)
        pred_hmm = hmm.decode(steps, trans, self.hmm_params, n)
        timings["hmm_s"] = time.perf_counter() - t0
        pred_clf = hmm.classifier_only(steps, n)
        return {"prep": prep, "cands": cands, "pred_hmm": pred_hmm, "pred_clf": pred_clf, "timings": timings}


def score_against_reference(trip_dir, prep, pred_by_name):
    """Server-side evaluation with the reference route (never sent to the client)."""
    try:
        ref = reference_labels(trip_dir, prep)
    except Exception:  # unreadable or empty reference file: still return the prediction, without metrics
        return None
    if ref is None or len(ref[1]) == 0:
        return None
    lab, _ = ref
    lab_arc = lab["label_arc"].to_numpy()
    ev = lab["align_dist"].to_numpy() <= RELIABLE_M
    out = {"n_fixes": int(len(lab_arc)), "n_eval_fixes": int(ev.sum())}
    for name, pred in pred_by_name.items():
        ref_set, pred_set = set(lab_arc[ev].tolist()), set(pred[ev & (pred >= 0)].tolist())
        tp = len(ref_set & pred_set)
        p = tp / len(pred_set) if pred_set else 0.0
        r = tp / len(ref_set) if ref_set else 0.0
        out[name] = {
            "top1_accuracy": float((pred[ev] == lab_arc[ev]).mean()) if ev.any() else None,
            "route_continuity": continuity(pred, prep.net.arcs),
            "route_precision": p, "route_recall": r, "route_f1": 2 * p * r / (p + r) if p + r else 0.0,
        }
    return out

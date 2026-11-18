"""Evaluation metrics: candidate-level classification + observation/route-level map matching.

Evaluation fixes are those whose reference label is reliable (fix within RELIABLE_M of the aligned
reference arc). Fixes with no candidates, or whose reference arc is not among the candidates, count
as errors for Top-1 accuracy.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score

RELIABLE_M = 50.0
EPS = 1e-6


def candidate_metrics(y, p, thr=0.5):
    y = np.asarray(y)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    yh = (p >= thr).astype(int)
    return {
        "macro_f1": f1_score(y, yh, average="macro"),
        "macro_precision": precision_score(y, yh, average="macro", zero_division=0),
        "macro_recall": recall_score(y, yh, average="macro", zero_division=0),
        "cand_accuracy": accuracy_score(y, yh),
        "log_loss": log_loss(y, p, labels=[0, 1]),
    }


def top1_per_obs(cands, score):
    """Highest-scoring candidate per (trip, fix)."""
    d = cands[["trip_id", "obs", "arc"]].copy()
    d["score"] = np.asarray(score)
    idx = d.groupby(["trip_id", "obs"], sort=False)["score"].idxmax()
    return d.loc[idx, ["trip_id", "obs", "arc", "score"]].rename(columns={"arc": "pred_arc"}).reset_index(drop=True)


def continuity(pred_arcs, arcs):
    """Share of *adjacent GPS fixes* whose predicted arcs are identical or directly connected.

    Missing predictions are represented by a negative arc id. They invalidate only the adjacent
    pair(s) that contain them; they must not be dropped first, because doing so would incorrectly
    compare non-consecutive GPS fixes.
    """
    p = np.asarray(pred_arcs, dtype=np.int64)
    if len(p) < 2:
        return np.nan
    left, right = p[:-1], p[1:]
    valid = (left >= 0) & (right >= 0) & (left < len(arcs)) & (right < len(arcs))
    if not np.any(valid):
        return np.nan
    a, b = left[valid], right[valid]
    return float(np.mean((a == b) | (arcs[a, 1] == arcs[b, 0])))


def observed_arc_set_metrics(obs_eval):
    """Arc-set precision/recall/F1 over evaluated GPS fixes.

    This is not an ordered full-route metric: it compares only the unique reference/predicted arcs
    attached to evaluation fixes.
    """
    ref = set(obs_eval["label_arc"].tolist())
    pred = set(obs_eval["pred_arc"].dropna().astype(np.int64).tolist())
    tp = len(ref & pred)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(ref) if ref else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def route_level(obs_eval):
    """Backward-compatible alias for :func:`observed_arc_set_metrics`."""
    return observed_arc_set_metrics(obs_eval)


def decision_metrics(cands, pred):
    """Candidate-level macro F1 when the decision is 'chosen candidate = 1, others = 0'."""
    m = cands[["trip_id", "obs", "arc", "y"]].merge(pred[["trip_id", "obs", "pred_arc"]], on=["trip_id", "obs"], how="left")
    yh = (m["arc"] == m["pred_arc"]).astype(int)
    return {"decision_macro_f1": f1_score(m["y"], yh, average="macro"),
            "decision_macro_precision": precision_score(m["y"], yh, average="macro", zero_division=0),
            "decision_macro_recall": recall_score(m["y"], yh, average="macro", zero_division=0)}


def observation_metrics(obs, pred, arcs_by_trip):
    """Top-1 accuracy, miss rate, continuity and route-level scores.

    obs: per-fix table with trip_id, obs, label_arc, align_dist, label_in_cands
    pred: per-fix predictions (trip_id, obs, pred_arc)
    """
    m = obs.merge(pred[["trip_id", "obs", "pred_arc"]], on=["trip_id", "obs"], how="left")
    ev = m[m["align_dist"] <= RELIABLE_M]
    correct = (ev["pred_arc"] == ev["label_arc"])
    cont, rp, rr, rf, per_trip = [], [], [], [], []
    for tid, g in m.sort_values(["trip_id", "obs"]).groupby("trip_id"):
        c = continuity(g["pred_arc"].fillna(-1).astype(np.int64).to_numpy(), arcs_by_trip[tid])
        e = g[g["align_dist"] <= RELIABLE_M]
        p, r, f = observed_arc_set_metrics(e)
        acc = float((e["pred_arc"] == e["label_arc"]).mean()) if len(e) else np.nan
        cont.append(c); rp.append(p); rr.append(r); rf.append(f)
        per_trip.append({
            "trip_id": tid, "top1": acc, "continuity": c,
            "observed_arc_set_f1": f,
            "route_f1": f,  # legacy alias retained for existing result readers
            "n_eval": len(e),
        })
    arc_p, arc_r, arc_f = float(np.mean(rp)), float(np.mean(rr)), float(np.mean(rf))
    return {
        "top1_acc": float(correct.mean()),
        "top1_acc_given_hit": float(correct[ev["label_in_cands"]].mean()),
        "candidate_miss_rate": float(1 - ev["label_in_cands"].mean()),
        "route_continuity": float(np.nanmean(cont)),
        "observed_arc_set_precision": arc_p,
        "observed_arc_set_recall": arc_r,
        "observed_arc_set_f1": arc_f,
        # Legacy names are kept so old notebooks / committed CSV readers continue to work.
        "route_precision": arc_p, "route_recall": arc_r, "route_f1": arc_f,
        "n_eval_fixes": int(len(ev)),
    }, pd.DataFrame(per_trip)

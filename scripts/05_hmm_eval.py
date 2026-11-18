"""HMM / Viterbi route reconstruction: decoder tuning and classifier-only vs HMM comparison.

Decoder parameters are tuned on the 80 training trips using OUT-OF-FOLD probabilities of the final
model (so emissions are as calibrated as on unseen trips). The chosen decoder is then applied to the
student-testing trips for every model from the comparison and for the tuned final model.
Outputs: results/hmm_grid.csv, models/hmm_config.json, results/hmm_comparison.csv,
         results/hmm_per_trip.csv, figures/hmm_*.png
"""
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src import config, hmm
from src.data import eval_rows_mask, load_partition
from src.io import load_trip_input
from src.metrics import RELIABLE_M, continuity, decision_metrics
from src.network import RoadNetwork

GRID = {"beta": [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0], "emission_weight": [0.5, 1.0, 2.0, 3.0, 5.0],
        "unreachable_logp": [-10.0, -25.0, -50.0, -100.0]}


def precompute(trip_dir, c, gap_break_s=config.GAP_BREAK_S):
    inp = load_trip_input(trip_dir)
    net = RoadNetwork(inp.nodes, inp.arcs, inp.track[:, :2])
    xy = net.to_xy(inp.track[:, :2])
    steps = hmm.build_steps(c.assign(proba=0.0), len(xy))
    trans = hmm.transition_distances(net, xy, inp.track[:, 2], steps, gap_break_s)
    return {"trip_id": inp.trip_id, "n": len(xy), "steps": steps, "trans": trans, "arcs": net.arcs}


def attach(pre, cands, proba):
    """Steps of every trip with the given candidate probabilities (same row order as build_steps)."""
    c = cands.assign(proba=proba)
    out = []
    for P in pre:
        ct = c[c.trip_id == P["trip_id"]].sort_values(["obs", "dist_rank"])
        pv = ct["proba"].to_numpy()
        steps, k = [], 0
        for (o, a, t, _) in P["steps"]:
            steps.append((o, a, t, pv[k:k + len(a)]))
            k += len(a)
        out.append(steps)
    return out


def evaluate(pre, obs, cands, steps_all, params, use_hmm=True):
    """Metrics over a partition for one set of probabilities and one decoder setting."""
    preds, per_trip = [], []
    obs_by_trip = {t: g.sort_values("obs") for t, g in obs.groupby("trip_id")}
    for P, steps in zip(pre, steps_all):
        pred = hmm.decode(steps, P["trans"], params, P["n"]) if use_hmm else hmm.classifier_only(steps, P["n"])
        o = obs_by_trip[P["trip_id"]]
        ev = o.align_dist.to_numpy() <= RELIABLE_M
        lab = o.label_arc.to_numpy()
        ref, pr = set(lab[ev].tolist()), set(pred[ev & (pred >= 0)].tolist())
        tp = len(ref & pr)
        rp = tp / len(pr) if pr else 0
        rr = tp / len(ref) if ref else 0
        per_trip.append({"trip_id": P["trip_id"], "top1": float((pred[ev] == lab[ev]).mean()) if ev.any() else np.nan,
                         "n_eval": int(ev.sum()), "correct": int((pred[ev] == lab[ev]).sum()),
                         "continuity": continuity(pred, P["arcs"]),
                         "route_f1": 2 * rp * rr / (rp + rr) if rp + rr else 0.0})
        preds.append(pd.DataFrame({"trip_id": P["trip_id"], "obs": np.arange(P["n"]), "pred_arc": pred}))
    pt = pd.DataFrame(per_trip)
    pred = pd.concat(preds)
    dm = decision_metrics(cands[eval_rows_mask(cands, obs)], pred[pred.pred_arc >= 0])
    return {"top1_acc": pt.correct.sum() / pt.n_eval.sum(), "route_continuity": pt.continuity.mean(),
            "route_f1": pt.route_f1.mean(), "trips_top1_ge_0.9": float((pt.top1 >= 0.9).mean()),
            **dm}, pt


def main():
    final_cfg = json.load(open(config.MODELS_DIR / "final_config.json"))
    tr_c, tr_o, _ = load_partition("train")
    st_c, st_o, _ = load_partition("student_test")
    t0 = time.time()
    tr_pre = Parallel(n_jobs=8)(delayed(precompute)(config.TRAIN_DIR / t, tr_c[tr_c.trip_id == t])
                                for t in sorted(tr_o.trip_id.unique()))
    st_pre = Parallel(n_jobs=8)(delayed(precompute)(config.STUDENT_TEST_DIR / t, st_c[st_c.trip_id == t])
                                for t in sorted(st_o.trip_id.unique()))
    print(f"transition pre-computation: {time.time() - t0:.0f}s", flush=True)
    unreach = np.mean([np.isinf(x[0]).mean() for P in tr_pre for x in P["trans"] if x is not None])
    gaps = sum(x is None for P in tr_pre for x in P["trans"][1:])

    oof = np.load(config.PROCESSED_DIR / "oof_final.npy")
    tr_steps = attach(tr_pre, tr_c, oof)
    base_tr, _ = evaluate(tr_pre, tr_o, tr_c, tr_steps, None, use_hmm=False)
    print("train OOF classifier-only:", base_tr, flush=True)

    keys = list(GRID)
    combos = list(itertools.product(*GRID.values()))

    grid = []
    for combo in combos:
        m, _ = evaluate(tr_pre, tr_o, tr_c, tr_steps, hmm.HMMParams(**dict(zip(keys, combo))))
        grid.append({**dict(zip(keys, combo)), **m})
    grid = pd.DataFrame(grid)
    grid.to_csv(config.RESULTS_DIR / "hmm_grid.csv", index=False)
    best = grid.sort_values(["top1_acc", "route_continuity"], ascending=False).iloc[0]
    params = hmm.HMMParams(beta=float(best.beta), emission_weight=float(best.emission_weight),
                           unreachable_logp=float(best.unreachable_logp))
    print("best decoder:", params, best.top1_acc, flush=True)
    json.dump({"params": params.to_dict(), "selected_on": "train trips, out-of-fold probabilities of the final model",
               "objective": "observation-level Top-1 accuracy (reliable fixes)", "train_oof_top1": float(best.top1_acc),
               "train_oof_classifier_only_top1": float(base_tr["top1_acc"]),
               "unreachable_pair_fraction_train": float(unreach), "gap_breaks_train": int(gaps)},
              open(config.MODELS_DIR / "hmm_config.json", "w"), indent=1)

    # student-test comparison for every model + the tuned final model
    rows, per_trip = [], []
    names = pd.read_csv(config.RESULTS_DIR / "model_comparison.csv")["model"].tolist()
    sources = [(n, config.PROCESSED_DIR / f"st_proba_{n}.npy") for n in names]
    sources.append((f"{final_cfg['model']} (tuned, final)", config.PROCESSED_DIR / "st_proba_final.npy"))
    for name, path in sources:
        st_steps = attach(st_pre, st_c, np.load(path))
        for mode in ("classifier-only", "classifier+HMM"):
            m, pt = evaluate(st_pre, st_o, st_c, st_steps, params, use_hmm=(mode == "classifier+HMM"))
            rows.append({"model": name, "mode": mode, **m})
            per_trip.append(pt.assign(model=name, mode=mode))
            print(name, mode, {k: round(v, 4) for k, v in m.items()}, flush=True)
    # also train OOF for the final model
    m_tr, _ = evaluate(tr_pre, tr_o, tr_c, tr_steps, params)
    rows.append({"model": f"{final_cfg['model']} (tuned, final) [train OOF]", "mode": "classifier-only", **base_tr})
    rows.append({"model": f"{final_cfg['model']} (tuned, final) [train OOF]", "mode": "classifier+HMM", **m_tr})
    res = pd.DataFrame(rows)
    res.to_csv(config.RESULTS_DIR / "hmm_comparison.csv", index=False)
    pt = pd.concat(per_trip)
    pt.to_csv(config.RESULTS_DIR / "hmm_per_trip.csv", index=False)
    plots(res, pt, grid, final_cfg)


def plots(res, pt, grid, final_cfg):
    F = config.FIGURES_DIR
    st = res[~res.model.str.contains("train OOF")]
    piv = st.pivot(index="model", columns="mode", values="top1_acc").sort_values("classifier+HMM")
    pc = st.pivot(index="model", columns="mode", values="route_continuity").loc[piv.index]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    yy = np.arange(len(piv))
    ax[0].barh(yy - 0.2, piv["classifier-only"], 0.4, color="#bbbbbb", label="classifier-only")
    ax[0].barh(yy + 0.2, piv["classifier+HMM"], 0.4, color="#dc3912", label="classifier + HMM/Viterbi")
    ax[0].set_yticks(yy); ax[0].set_yticklabels(piv.index); ax[0].set_xlim(0.5, 1); ax[0].legend(loc="lower right")
    ax[0].set_title("Student-test observation-level Top-1 accuracy")
    ax[1].barh(yy - 0.2, pc["classifier-only"], 0.4, color="#bbbbbb", label="classifier-only")
    ax[1].barh(yy + 0.2, pc["classifier+HMM"], 0.4, color="#dc3912", label="classifier + HMM/Viterbi")
    ax[1].set_yticks(yy); ax[1].set_yticklabels([]); ax[1].set_xlim(0.5, 1); ax[1].set_title("Route continuity")
    fig.tight_layout(); fig.savefig(F / "hmm_models_comparison.png"); plt.close(fig)

    fin = f"{final_cfg['model']} (tuned, final)"
    p = pt[pt.model == fin].pivot(index="trip_id", columns="mode", values="top1")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(p))
    ax.bar(x - 0.2, p["classifier-only"], 0.4, color="#bbbbbb", label="classifier-only")
    ax.bar(x + 0.2, p["classifier+HMM"], 0.4, color="#dc3912", label="classifier + HMM")
    ax.set_xticks(x); ax.set_xticklabels(p.index, rotation=30); ax.set_ylim(0.5, 1.0); ax.legend()
    ax.set_title(f"Per-trip Top-1 accuracy on student-test trips ({fin})")
    fig.tight_layout(); fig.savefig(F / "hmm_per_trip.png"); plt.close(fig)

    g = grid[grid.unreachable_logp == grid.sort_values("top1_acc").iloc[-1].unreachable_logp]
    hm = g.pivot(index="emission_weight", columns="beta", values="top1_acc")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    im = ax.imshow(hm.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(hm.shape[1])); ax.set_xticklabels(hm.columns); ax.set_xlabel("beta (m)")
    ax.set_yticks(range(hm.shape[0])); ax.set_yticklabels(hm.index); ax.set_ylabel("emission weight")
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            ax.text(j, i, f"{hm.values[i, j]:.4f}", ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im); ax.set_title("Decoder grid on train OOF probabilities: Top-1 accuracy")
    fig.tight_layout(); fig.savefig(F / "hmm_grid.png"); plt.close(fig)


if __name__ == "__main__":
    main()

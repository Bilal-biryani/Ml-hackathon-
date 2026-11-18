"""Train and compare candidate-scoring models.

Protocol
  * 5-fold GroupKFold over the 80 training trips (a trip never appears in two folds).
  * Fit on fixes with reliable labels; evaluate every reliable fix of the held-out fold.
  * Refit on all training trips, evaluate on the 10 student-testing trips.
Outputs: results/model_comparison.csv, results/cv_fold_scores.csv, results/per_trip_student_test.csv,
         results/training_histories.json, results/selected_model.json, data/processed/oof_<model>.npy,
         figures/cmp_*.png, models/<model>.joblib (small models)
"""
import inspect
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold

from src import config
from src.data import eval_rows_mask, load_partition, training_mask, xy
from src.metrics import candidate_metrics, decision_metrics, observation_metrics, top1_per_obs
from src.models import EpochMLP, make_lgbm, model_zoo

def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


QUICK = _env_flag("PATHSNAP_QUICK")
RESULTS_DIR = config.RESULTS_DIR / "smoke" if QUICK else config.RESULTS_DIR
MODELS_DIR = config.MODELS_DIR / "smoke" if QUICK else config.MODELS_DIR
FIGURES_DIR = config.FIGURES_DIR / "smoke" if QUICK else config.FIGURES_DIR
ARRAYS_DIR = RESULTS_DIR / "arrays" if QUICK else config.PROCESSED_DIR
for _d in (RESULTS_DIR, MODELS_DIR, FIGURES_DIR, ARRAYS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
SMALL_MODELS = {"LogisticRegression", "GaussianNB", "LightGBM", "MLP"}


def fit(model, X, y, Xv=None, yv=None):
    if isinstance(model, EpochMLP):
        return model.fit(X, y, Xv, yv)
    if model.__class__.__name__ == "LGBMClassifier" and Xv is not None:
        kwargs = {"eval_names": ["train", "valid"], "eval_metric": "binary_logloss"}
        # LightGBM 4.7 deprecated eval_set in favour of eval_X/eval_y. Keep compatibility
        # with earlier 4.x releases so the repository remains reproducible across machines.
        if "eval_X" in inspect.signature(model.fit).parameters:
            kwargs.update(eval_X=(X, Xv), eval_y=(y, yv))
        else:
            kwargs["eval_set"] = [(X, y), (Xv, yv)]
        return model.fit(X, y, **kwargs)
    return model.fit(X, y)


def score_split(cands, obs, arcs, proba):
    ev = eval_rows_mask(cands, obs)
    cm = candidate_metrics(cands.loc[ev, "y"].to_numpy(), proba[ev])
    pred = top1_per_obs(cands, proba)
    om, per_trip = observation_metrics(obs, pred, arcs)
    dm = decision_metrics(cands[ev], pred)
    return {**cm, **om, **dm}, pred, per_trip


def main():
    tr_c, tr_o, tr_arcs = load_partition("train")
    st_c, st_o, st_arcs = load_partition("student_test")
    X, y = xy(tr_c)
    Xs, ys = xy(st_c)
    fit_mask = training_mask(tr_c, tr_o)
    groups = tr_c["trip_id"].to_numpy()
    folds = list(GroupKFold(n_splits=N_FOLDS).split(X, y, groups))
    with open(RESULTS_DIR / "cv_folds.json", "w") as f:
        json.dump({"folds": [sorted(np.unique(groups[v]).tolist()) for _, v in folds]}, f, indent=1)
    if QUICK:
        print(f"Smoke mode enabled; outputs will be written under {RESULTS_DIR}", flush=True)

    zoo = model_zoo()
    if QUICK:  # smoke test: 3 cheap models
        zoo = {k: zoo[k] for k in ("GaussianNB", "LightGBM", "MLP")}
        zoo["LightGBM"] = (zoo["LightGBM"][0], zoo["LightGBM"][1], lambda: make_lgbm(n_estimators=10))
        zoo["MLP"] = (zoo["MLP"][0], zoo["MLP"][1], lambda: EpochMLP(epochs=2, max_train=20000))
    rows, fold_rows, hist, per_trip_all, cms = [], [], {}, [], {}
    for name, (family, cfg, factory) in zoo.items():
        print(f"\n=== {name} ===", flush=True)
        oof = np.zeros(len(X))
        fold_scores = []
        fit_times = []
        for k, (tri, vai) in enumerate(folds):
            tri = tri[fit_mask[tri]]
            m = factory()
            t0 = time.time()
            if k == 0 and name in ("MLP", "LightGBM"):
                vfit = vai[fit_mask[vai]]
                fit(m, X[tri], y[tri], X[vfit], y[vfit])
                if name == "MLP":
                    hist[name] = m.history_
                else:
                    r = m.evals_result_
                    hist[name] = {"epoch": list(range(1, len(r["train"]["binary_logloss"]) + 1)),
                                  "train_loss": r["train"]["binary_logloss"], "val_loss": r["valid"]["binary_logloss"]}
            else:
                fit(m, X[tri], y[tri])
            fit_times.append(time.time() - t0)
            oof[vai] = m.predict_proba(X[vai])[:, 1]
            vt = np.unique(groups[vai])
            cm_ = tr_c["trip_id"].isin(vt).to_numpy()
            om_ = tr_o["trip_id"].isin(vt).to_numpy()
            s, _, _ = score_split(tr_c[cm_], tr_o[om_], tr_arcs, oof[cm_])
            # training-fold score (overfitting check) on reliable rows of training trips
            sub = np.random.default_rng(config.SEED).choice(tri, size=min(100_000, len(tri)), replace=False)
            s_tr = candidate_metrics(y[sub], m.predict_proba(X[sub])[:, 1])
            fold_scores.append(s)
            fold_rows.append({"model": name, "fold": k, **s, "train_macro_f1": s_tr["macro_f1"],
                              "train_log_loss": s_tr["log_loss"], "fit_seconds": fit_times[-1]})
            print(f"  fold {k}: macroF1={s['macro_f1']:.4f} top1={s['top1_acc']:.4f} logloss={s['log_loss']:.4f} "
                  f"train_macroF1={s_tr['macro_f1']:.4f} ({fit_times[-1]:.1f}s)", flush=True)
        np.save(ARRAYS_DIR / f"oof_{name}.npy", oof)

        # refit on all training trips -> student test
        m = factory()
        t0 = time.time()
        fit(m, X[fit_mask], y[fit_mask])
        train_time = time.time() - t0
        t0 = time.time()
        ps = m.predict_proba(Xs)[:, 1]
        inf_time = time.time() - t0
        st, pred, per_trip = score_split(st_c, st_o, st_arcs, ps)
        per_trip.insert(0, "model", name)
        per_trip_all.append(per_trip)
        ev = eval_rows_mask(st_c, st_o)
        cms[name] = confusion_matrix(ys[ev], (ps[ev] >= 0.5).astype(int), labels=[0, 1]).tolist()
        np.save(ARRAYS_DIR / f"st_proba_{name}.npy", ps)
        if name in SMALL_MODELS:
            joblib.dump(m, MODELS_DIR / f"{name}.joblib")

        fs = pd.DataFrame(fold_scores)
        rows.append({
            "model": name, "family": family, "config": cfg,
            "cv_macro_f1": fs.macro_f1.mean(), "cv_macro_f1_std": fs.macro_f1.std(),
            "cv_top1": fs.top1_acc.mean(), "cv_top1_std": fs.top1_acc.std(),
            "cv_log_loss": fs.log_loss.mean(), "cv_continuity": fs.route_continuity.mean(),
            "cv_precision": fs.macro_precision.mean(), "cv_recall": fs.macro_recall.mean(),
            "st_macro_f1": st["macro_f1"], "st_precision": st["macro_precision"], "st_recall": st["macro_recall"],
            "st_top1": st["top1_acc"], "st_log_loss": st["log_loss"], "st_continuity": st["route_continuity"],
            "st_observed_arc_set_f1": st["observed_arc_set_f1"],
            "st_route_f1": st["route_f1"],  # legacy alias; see src.metrics
            "st_decision_macro_f1": st["decision_macro_f1"],
            "st_candidate_miss_rate": st["candidate_miss_rate"],
            "train_time_s": train_time, "inference_ms_per_fix": inf_time / len(st_o) * 1000,
            "mean_fold_fit_s": float(np.mean(fit_times)),
        })
        print(pd.Series(rows[-1]).to_string(), flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(RESULTS_DIR / "cv_fold_scores.csv", index=False)
    pd.concat(per_trip_all).to_csv(RESULTS_DIR / "per_trip_student_test.csv", index=False)
    json.dump(hist, open(RESULTS_DIR / "training_histories.json", "w"))
    json.dump(cms, open(RESULTS_DIR / "confusion_matrices.json", "w"))
    best = res.sort_values(["cv_top1", "cv_macro_f1"], ascending=False).iloc[0]["model"]
    json.dump({"selected_model": best, "criterion": "highest mean CV observation-level Top-1 accuracy "
               "(tie-break: CV macro F1)"}, open(RESULTS_DIR / "selected_model.json", "w"), indent=1)
    print("\nSelected:", best)
    plot(res, pd.DataFrame(fold_rows), hist, cms)


def plot(res, folds, hist, cms):
    F = FIGURES_DIR
    order = res.sort_values("cv_top1", ascending=False)["model"].tolist()
    r = res.set_index("model").loc[order]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(order, r.cv_macro_f1, yerr=r.cv_macro_f1_std, capsize=4, color="#3366cc")
    for i, f in enumerate(order):
        pts = folds[folds.model == f].macro_f1
        ax.scatter([i] * len(pts), pts, color="k", s=10, zorder=3)
    ax.set_ylabel("macro F1 (candidate level)"); ax.set_title("5-fold GroupKFold CV macro F1 (train, 80 trips); dots = folds")
    ax.set_ylim(max(0, r.cv_macro_f1.min() - 0.1), 1); plt.xticks(rotation=20)
    fig.tight_layout(); fig.savefig(F / "cmp_cv_macro_f1.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(order)); w = 0.27
    ax.bar(x - w, r.st_top1, w, label="Top-1 accuracy", color="#109618")
    ax.bar(x, r.st_macro_f1, w, label="Macro F1 (thr 0.5)", color="#3366cc")
    ax.bar(x + w, r.cv_top1, w, label="CV Top-1 (train)", color="#bbbbbb")
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=20); ax.set_ylim(0.75, 0.96); ax.legend(loc="upper right", ncol=3)
    ax.set_title("Student-testing (10 trips) Top-1 accuracy and macro F1")
    fig.tight_layout(); fig.savefig(F / "cmp_student_test.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].bar(order, r.cv_log_loss, color="#dc3912", alpha=0.6, label="CV")
    ax[0].plot(order, r.st_log_loss, "ko-", label="student-test")
    ax[0].set_yscale("log"); ax[0].set_title("Log loss (lower is better)"); ax[0].legend(); ax[0].tick_params(axis="x", rotation=20)
    ax[1].bar(order, r.inference_ms_per_fix, color="#ff9900"); ax[1].set_yscale("log")
    ax[1].set_title("Inference time per GPS fix (ms, student-test)"); ax[1].tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(F / "cmp_logloss_time.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - 0.2, r.cv_continuity, 0.4, label="CV", color="#990099")
    ax.bar(x + 0.2, r.st_continuity, 0.4, label="student-test", color="#dd77dd")
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=20); ax.set_ylim(0, 1); ax.legend()
    ax.set_title("Route continuity of classifier-only Top-1 predictions")
    fig.tight_layout(); fig.savefig(F / "cmp_continuity.png"); plt.close(fig)

    n = len(order)
    fig, axs = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 7.5))
    for a, name in zip(axs.ravel(), order):
        c = np.array(cms[name])
        cn = c / c.sum(axis=1, keepdims=True)
        a.imshow(cn, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                a.text(j, i, f"{c[i, j]}\n({cn[i, j]:.2f})", ha="center", va="center",
                       color="white" if cn[i, j] > 0.5 else "black", fontsize=9)
        a.set_xticks([0, 1]); a.set_yticks([0, 1]); a.set_xticklabels(["pred 0", "pred 1"]); a.set_yticklabels(["true 0", "true 1"])
        a.set_title(name)
    for a in axs.ravel()[n:]:
        a.axis("off")
    fig.suptitle("Candidate-level confusion matrices (student-test, threshold 0.5)")
    fig.tight_layout(); fig.savefig(F / "cmp_confusion_matrices.png"); plt.close(fig)

    if hist:
        fig, axs = plt.subplots(1, len(hist), figsize=(6 * len(hist), 4))
        axs = np.atleast_1d(axs)
        for a, (name, h) in zip(axs, hist.items()):
            a.plot(h["epoch"], h["train_loss"], label="train (fold-0 training trips)")
            a.plot(h["epoch"], h["val_loss"], label="validation (fold-0 held-out trips)")
            a.set_xlabel("epoch" if name == "MLP" else "boosting iteration"); a.set_ylabel("log loss")
            a.set_title(f"{name} training history"); a.legend()
        fig.tight_layout(); fig.savefig(F / "cmp_training_curves.png"); plt.close(fig)


if __name__ == "__main__":
    if "--plots-only" in sys.argv:  # redraw figures from the saved result files
        R = RESULTS_DIR
        plot(pd.read_csv(R / "model_comparison.csv"), pd.read_csv(R / "cv_fold_scores.csv"),
             json.load(open(R / "training_histories.json")), json.load(open(R / "confusion_matrices.json")))
    else:
        main()

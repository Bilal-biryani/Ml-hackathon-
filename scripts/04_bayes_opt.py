"""Bayesian hyper-parameter optimisation (Optuna TPE) of the selected model.

Objective: mean observation-level Top-1 accuracy over the same 5-fold GroupKFold split of the 80
training trips used in the model comparison (student-test trips are never seen by the search).
Outputs: results/optuna_trials.csv, results/optuna_best.json, results/tuning_comparison.csv,
         models/final_model.joblib, models/final_config.json, data/processed/oof_final.npy,
         data/processed/st_proba_final.npy, figures/opt_*.png
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold

from src import config
from src.data import eval_rows_mask, load_partition, training_mask, xy
from src.metrics import candidate_metrics, observation_metrics, top1_per_obs
from src.models import N_JOBS, make_lgbm

N_TRIALS = 40
N_FOLDS = 5


def space(trial, name):
    if name == "LightGBM":
        return make_lgbm(
            n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 300, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        )
    cls = {"RandomForest": RandomForestClassifier, "ExtraTrees": ExtraTreesClassifier}[name]
    return cls(n_estimators=trial.suggest_int("n_estimators", 100, 400, step=50),
               max_depth=trial.suggest_int("max_depth", 8, 40),
               min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 50, log=True),
               max_features=trial.suggest_float("max_features", 0.2, 1.0),
               n_jobs=N_JOBS, random_state=config.SEED)


def build_from_params(name, params):
    return space(optuna.trial.FixedTrial(params), name)


def cv_eval(model_factory, X, y, tr_c, tr_o, tr_arcs, folds, fit_mask, trial=None):
    groups = tr_c["trip_id"].to_numpy()
    oof = np.zeros(len(X))
    scores = []
    for k, (tri, vai) in enumerate(folds):
        tri = tri[fit_mask[tri]]
        m = model_factory()
        m.fit(X[tri], y[tri])
        oof[vai] = m.predict_proba(X[vai])[:, 1]
        vt = np.unique(groups[vai])
        cm_ = tr_c["trip_id"].isin(vt).to_numpy()
        om_ = tr_o["trip_id"].isin(vt).to_numpy()
        c, o = tr_c[cm_], tr_o[om_]
        ev = eval_rows_mask(c, o)
        s = candidate_metrics(c.loc[ev, "y"].to_numpy(), oof[cm_][ev])
        om, _ = observation_metrics(o, top1_per_obs(c, oof[cm_]), tr_arcs)
        scores.append({**s, **om})
        if trial is not None:
            trial.report(float(np.mean([z["top1_acc"] for z in scores])), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return pd.DataFrame(scores), oof


def evaluate_student(model, Xs, st_c, st_o, st_arcs):
    t0 = time.time()
    p = model.predict_proba(Xs)[:, 1]
    inf = (time.time() - t0) / len(st_o) * 1000
    ev = eval_rows_mask(st_c, st_o)
    s = candidate_metrics(st_c.loc[ev, "y"].to_numpy(), p[ev])
    om, _ = observation_metrics(st_o, top1_per_obs(st_c, p), st_arcs)
    return {**s, **om, "inference_ms_per_fix": inf}, p


def main():
    sel = json.load(open(config.RESULTS_DIR / "selected_model.json"))["selected_model"]
    print("Selected model from comparison:", sel)
    tr_c, tr_o, tr_arcs = load_partition("train")
    st_c, st_o, st_arcs = load_partition("student_test")
    X, y = xy(tr_c)
    Xs, _ = xy(st_c)
    fit_mask = training_mask(tr_c, tr_o)
    folds = list(GroupKFold(n_splits=N_FOLDS).split(X, y, tr_c["trip_id"].to_numpy()))

    def objective(trial):
        t0 = time.time()
        fs, _ = cv_eval(lambda: space(trial, sel), X, y, tr_c, tr_o, tr_arcs, folds, fit_mask, trial)
        trial.set_user_attr("cv_macro_f1", float(fs.macro_f1.mean()))
        trial.set_user_attr("cv_log_loss", float(fs.log_loss.mean()))
        trial.set_user_attr("cv_top1_std", float(fs.top1_acc.std()))
        trial.set_user_attr("seconds", time.time() - t0)
        print(f"trial {trial.number}: top1={fs.top1_acc.mean():.4f} f1={fs.macro_f1.mean():.4f} "
              f"({time.time() - t0:.0f}s) {trial.params}", flush=True)
        return float(fs.top1_acc.mean())

    t_start = time.time()
    study = optuna.create_study(direction="maximize", study_name=f"pathsnap_{sel}",
                                sampler=optuna.samplers.TPESampler(seed=config.SEED, n_startup_trials=10),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    study.optimize(objective, n_trials=N_TRIALS)
    search_seconds = time.time() - t_start
    trials = study.trials_dataframe()
    trials.to_csv(config.RESULTS_DIR / "optuna_trials.csv", index=False)
    best = study.best_trial
    print("Best:", best.value, best.params)

    # untuned (comparison defaults) vs tuned, identical protocol
    cmp = pd.read_csv(config.RESULTS_DIR / "model_comparison.csv").set_index("model").loc[sel]
    tuned_cv, oof = cv_eval(lambda: build_from_params(sel, best.params), X, y, tr_c, tr_o, tr_arcs, folds, fit_mask)
    final = build_from_params(sel, best.params)
    t0 = time.time()
    final.fit(X[fit_mask], y[fit_mask])
    train_time = time.time() - t0
    st, ps = evaluate_student(final, Xs, st_c, st_o, st_arcs)
    untuned_model = joblib.load(config.MODELS_DIR / f"{sel}.joblib") if (config.MODELS_DIR / f"{sel}.joblib").exists() else None
    if untuned_model is not None:
        st_u, _ = evaluate_student(untuned_model, Xs, st_c, st_o, st_arcs)
        untuned_inf = st_u["inference_ms_per_fix"]
    else:
        untuned_inf = cmp.inference_ms_per_fix
    comp = pd.DataFrame([
        {"model": f"{sel} (untuned)", "cv_macro_f1": cmp.cv_macro_f1, "cv_top1": cmp.cv_top1, "cv_log_loss": cmp.cv_log_loss,
         "st_top1": cmp.st_top1, "st_macro_f1": cmp.st_macro_f1, "st_log_loss": cmp.st_log_loss,
         "st_continuity": cmp.st_continuity, "train_time_s": cmp.train_time_s, "inference_ms_per_fix": untuned_inf},
        {"model": f"{sel} (tuned)", "cv_macro_f1": tuned_cv.macro_f1.mean(), "cv_top1": tuned_cv.top1_acc.mean(),
         "cv_log_loss": tuned_cv.log_loss.mean(), "st_top1": st["top1_acc"], "st_macro_f1": st["macro_f1"],
         "st_log_loss": st["log_loss"], "st_continuity": st["route_continuity"], "train_time_s": train_time,
         "inference_ms_per_fix": st["inference_ms_per_fix"]},
    ])
    comp.to_csv(config.RESULTS_DIR / "tuning_comparison.csv", index=False)
    print(comp.T)

    np.save(config.PROCESSED_DIR / "oof_final.npy", oof)
    np.save(config.PROCESSED_DIR / "st_proba_final.npy", ps)
    joblib.dump(final, config.MODELS_DIR / "final_model.joblib")
    json.dump({"model": sel, "params": best.params, "features": config.FEATURES,
               "best_cv_top1": best.value, "n_trials": len(study.trials),
               "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
               "search_seconds": search_seconds, "sampler": "TPE (seed 42, 10 random start-up trials)",
               "pruner": "MedianPruner (8 start-up trials, warm-up 1 fold)",
               "objective": "maximise mean 5-fold GroupKFold observation-level Top-1 accuracy (train trips only)"},
              open(config.RESULTS_DIR / "optuna_best.json", "w"), indent=1)
    json.dump({"model_file": "final_model.joblib", "model": sel, "params": best.params, "features": config.FEATURES,
               "candidate_radius_steps": list(config.CANDIDATE_RADIUS_STEPS), "max_candidates": config.MAX_CANDIDATES},
              open(config.MODELS_DIR / "final_config.json", "w"), indent=1)

    # plots
    done = trials[trials.state == "COMPLETE"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].scatter(done.number, done.value, color="#3366cc", label="completed trial")
    pr = trials[trials.state == "PRUNED"]
    ax[0].scatter(pr.number, [done.value.min()] * len(pr), marker="x", color="#999999", label="pruned (plotted at min)")
    ax[0].plot(done.number, done.value.cummax(), color="#dc3912", label="best so far")
    ax[0].axhline(cmp.cv_top1, color="k", ls="--", label="untuned CV Top-1")
    ax[0].set_xlabel("trial"); ax[0].set_ylabel("CV Top-1 accuracy"); ax[0].set_title(f"Optuna TPE history ({sel})"); ax[0].legend()
    try:
        imp = optuna.importance.get_param_importances(study)
        ax[1].barh(list(imp.keys())[::-1], list(imp.values())[::-1], color="#109618")
        ax[1].set_title("Hyper-parameter importance (fANOVA)")
    except Exception as e:  # importance needs enough completed trials
        ax[1].text(0.1, 0.5, f"importance unavailable: {e}")
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "opt_history.png"); plt.close(fig)

    metrics = ["cv_macro_f1", "cv_top1", "st_top1", "st_macro_f1"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    xx = np.arange(len(metrics))
    ax[0].bar(xx - 0.2, comp.iloc[0][metrics].astype(float), 0.4, label="untuned", color="#bbbbbb")
    ax[0].bar(xx + 0.2, comp.iloc[1][metrics].astype(float), 0.4, label="tuned", color="#3366cc")
    ax[0].set_xticks(xx); ax[0].set_xticklabels(metrics, rotation=15); ax[0].set_ylim(0.5, 1); ax[0].legend()
    ax[0].set_title("Accuracy / F1")
    ax[1].bar(["CV", "student-test"], [comp.iloc[0].cv_log_loss, comp.iloc[0].st_log_loss], width=-0.4, align="edge", color="#bbbbbb", label="untuned")
    ax[1].bar(["CV", "student-test"], [comp.iloc[1].cv_log_loss, comp.iloc[1].st_log_loss], width=0.4, align="edge", color="#3366cc", label="tuned")
    ax[1].set_title("Log loss"); ax[1].legend()
    ax[2].bar(["untuned", "tuned"], comp.inference_ms_per_fix, color=["#bbbbbb", "#3366cc"]); ax[2].set_title("Inference ms per fix")
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "opt_tuned_vs_untuned.png"); plt.close(fig)


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    main()

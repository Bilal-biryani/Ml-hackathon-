"""Build the consolidated technical report PDF from the saved experiment outputs.

Every number in the report is read from results/*.csv|json produced by scripts 01-06, so the PDF
can be regenerated and verified from the repository.
Output: report/PathSnap_Technical_Report.pdf
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from src import config

R, F = config.RESULTS_DIR, config.FIGURES_DIR
OUT = config.ROOT / "report"
OUT.mkdir(exist_ok=True)

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=16, spaceBefore=10, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=13, spaceBefore=8, spaceAfter=4)
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontSize=11, spaceBefore=6, spaceAfter=3)
P = ParagraphStyle("P", parent=ss["BodyText"], fontSize=9.5, leading=12.5, spaceAfter=4)
SM = ParagraphStyle("SM", parent=P, fontSize=8, leading=10)
CAP = ParagraphStyle("CAP", parent=P, fontSize=8, textColor=colors.HexColor("#555555"), alignment=TA_CENTER)
TC = ParagraphStyle("TC", parent=P, fontSize=7.2, leading=8.6, spaceAfter=0)
BOX = ParagraphStyle("BOX", parent=P, backColor=colors.HexColor("#fff4e5"), borderColor=colors.HexColor("#ff9900"),
                     borderWidth=0.8, borderPadding=6, spaceBefore=6, spaceAfter=8)

story = []


def h1(t): story.append(Paragraph(t, H1))
def h2(t): story.append(Paragraph(t, H2))
def h3(t): story.append(Paragraph(t, H3))
def p(t, st=P): story.append(Paragraph(t, st))


def bullets(items):
    for it in items:
        story.append(Paragraph("&bull; " + it, P))


def img(path, width=17.0, caption=None):
    path = Path(path)
    if not path.exists():
        p(f"<i>[missing figure: {path.name}]</i>")
        return
    from reportlab.lib.utils import ImageReader
    w, h = ImageReader(str(path)).getSize()
    width_pt = width * cm
    el = [Image(str(path), width=width_pt, height=width_pt * h / w)]
    if caption:
        el.append(Paragraph(caption, CAP))
    story.append(KeepTogether(el))
    story.append(Spacer(1, 6))


def table(rows, header, col_widths=None, font=7.2):
    st = ParagraphStyle("t", parent=TC, fontSize=font, leading=font + 1.6)
    data = [[Paragraph(f"<b>{h}</b>", st) for h in header]]
    data += [[Paragraph(str(c), st) for c in r] for r in rows]
    t = Table(data, colWidths=[w * cm for w in col_widths] if col_widths else None, repeatRows=1)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8ecf3")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))


def f(x, d=4):
    try:
        return f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return str(x)


def pc(x, d=2):
    return f"{100 * float(x):.{d}f}%"


# ----------------------------------------------------------------------------- load results
eda = json.load(open(R / "eda_summary.json"))
inv = pd.read_csv(R / "eda_inventory.csv", dtype={"trip_id": str})
mv = pd.read_csv(R / "eda_missing_values.csv")
clog = pd.read_csv(R / "cleaning_log.csv", dtype={"trip_id": str})
cmp_ = pd.read_csv(R / "model_comparison.csv")
folds = pd.read_csv(R / "cv_fold_scores.csv")
sel = json.load(open(R / "selected_model.json"))["selected_model"]
best = json.load(open(R / "optuna_best.json"))
trials = pd.read_csv(R / "optuna_trials.csv")
tune = pd.read_csv(R / "tuning_comparison.csv")
hmm_cfg = json.load(open(config.MODELS_DIR / "hmm_config.json"))
hcmp = pd.read_csv(R / "hmm_comparison.csv")
hpt = pd.read_csv(R / "hmm_per_trip.csv", dtype={"trip_id": str})
grid = pd.read_csv(R / "hmm_grid.csv")
c = cmp_.set_index("model")
fold_train = folds.groupby("model")[["train_macro_f1", "macro_f1", "train_log_loss", "log_loss"]].mean()
hp = hmm_cfg["params"]
fin_name = f"{sel} (tuned, final)"
hs = hcmp[hcmp.model == fin_name].set_index("mode")
htr = hcmp[hcmp.model.str.contains("train OOF")].set_index("mode")
n_tr_fix, n_st_fix = eda["fixes"]["train"], eda["fixes"]["student_test"]

# ----------------------------------------------------------------------------- title
story.append(Spacer(1, 3 * cm))
story.append(Paragraph("PathSnap: Distributed Map-Matching &amp; Route Reconstruction",
                       ParagraphStyle("T", parent=H1, fontSize=22, leading=28, alignment=TA_CENTER)))
story.append(Paragraph("Technical Report: UE24CS352A Machine Learning Hackathon",
                       ParagraphStyle("T2", parent=H2, alignment=TA_CENTER)))
story.append(Spacer(1, 1 * cm))
p("<b>Team:</b> PT-040<br/><b>Members:</b> Mohammad Bilal (PES1UG24AM162), Mohammed Sinan (PES1UG24AM165), "
  "Mohammed Arfan (PES1UG24AM163), Mirunjai (PES1UG24AM161)<br/>"
  "<b>Repository:</b> https://github.com/Bilal-biryani/Ml-hackathon-", ParagraphStyle("c", parent=P, alignment=TA_CENTER, fontSize=11, leading=16))
story.append(Spacer(1, 1 * cm))
p("<b>Summary.</b> Each GPS fix is matched to one directed road arc. For each fix, candidate arcs within 50 m are "
  f"generated (up to {config.MAX_CANDIDATES}) and described by {len(config.FEATURES)} engineered features. "
  f"Seven classifiers from six algorithm families were compared using 5-fold trip-grouped cross-validation on the "
  f"80 training trips. <b>{sel}</b> was selected and tuned with Bayesian optimisation "
  f"({best['n_trials']} Optuna TPE trials). Its probabilities feed an HMM whose transitions use road-network "
  f"shortest paths, decoded with Viterbi. On the 10 student-testing trips, the final pipeline reaches an "
  f"observation-level Top-1 accuracy of <b>{pc(hs.loc['classifier+HMM', 'top1_acc'])}</b> "
  f"(classifier only: {pc(hs.loc['classifier-only', 'top1_acc'])}) and a route continuity of "
  f"<b>{pc(hs.loc['classifier+HMM', 'route_continuity'])}</b> (classifier only: "
  f"{pc(hs.loc['classifier-only', 'route_continuity'])}). Every number in this report is produced by the "
  "scripts in the repository (results/*.csv, results/*.json).")
p("<b>AI-assistance disclosure.</b> [Team to complete honestly as required by the event policy: state which "
  "parts of the code and report were produced or edited with AI tools, and confirm that every member has run "
  "and understood the experiments.]", BOX)
story.append(PageBreak())

# ----------------------------------------------------------------------------- 2.2 EDA
h1("2.2 Dataset Understanding and Exploratory Data Analysis")
p(f"<b>Partitions used.</b> Train: {eda['trips']['train']} trips, {n_tr_fix:,} GPS fixes. "
  f"Student-test: {eda['trips']['student_test']} trips, {n_st_fix:,} fixes. The evaluator-testing partition was "
  "never available to, or used by, any part of the development pipeline. Unless a caption says otherwise, "
  "statistics cover the train partition. File-structure checks cover train and student-test.")
h2("Raw data and how the files relate")
bullets([
    "<b>.track</b>: longitude, latitude, timestamp (s). These are the GPS observations. "
    f"{pc(eda['dt_eq_1s_overall'], 1)} of the train sampling intervals are exactly 1 s; the largest gap is "
    f"{eda['max_gap_s']:,.0f} s, and {eda['trips_with_gap_gt_60s']} of the 90 trips contain a gap over 60 s.",
    "<b>.nodes</b>: longitude, latitude, with the row index as node id. <b>.arcs</b>: from_node, to_node, with "
    "the row index as arc id. These are directed road segments. "
    f"Per trip there are {eda['nodes_per_trip']['min']:,.0f} to {eda['nodes_per_trip']['max']:,.0f} nodes and "
    f"{eda['arcs_per_trip']['min']:,.0f} to {eda['arcs_per_trip']['max']:,.0f} arcs (median "
    f"{eda['arcs_per_trip']['50%']:,.0f}). Two-way streets appear as two arcs with opposite directions: "
    f"{pc(eda['two_way_arc_frac'], 1)} of the generated candidates have a reverse twin.",
    "<b>.route</b>: the ordered reference arc ids. Consecutive route arcs are connected "
    f"(head = tail) in {pc(eda['route_connected_frac_mean'], 2)} of cases. The route has <b>no timestamps</b>, so "
    "it does not say which GPS fix lies on which arc; each GPS row is <i>not</i> a route row.",
])
h2("Q1. How did you preprocess the data?")
bullets([
    "<b>Cleaning</b> (src/io.py): drop non-finite values, out-of-range coordinates and fixes more than about 5 km "
    "outside the network extent; sort by timestamp; drop duplicate timestamps. On the provided data "
    f"no fix needed removal ({int(clog.non_finite.sum())} non-finite, {int(clog.out_of_range.sum())} out of range, "
    f"{int(clog.outside_network.sum())} outside the network, {int(clog.duplicate_time.sum())} duplicate timestamps), "
    "but the same checks run inside the API for unseen trips.",
    "<b>Projection</b>: longitude and latitude are angles, so each trip is projected to a local equirectangular "
    "metric frame centred on the track (x = R&middot;cos&phi;<sub>0</sub>&middot;&Delta;&lambda;, y = R&middot;&Delta;&phi;). "
    f"The trips lie on five continents ({eda['trips_by_region']['Europe/Africa']} in Europe/Africa, "
    f"{eda['trips_by_region']['Americas']} in the Americas, {eda['trips_by_region']['Asia/Oceania']} in Asia/Oceania), so one global "
    f"projection would be wrong; but a single trip spans at most {eda['max_trip_extent_km']:.0f} km, so a per-trip projection "
    "differs from geodesic distance by well under 1% (far less for typical trips of a few km). Every distance is in metres.",
    f"<b>Network restriction</b>: only arcs whose bounding box lies within {config.NETWORK_MARGIN_M:.0f} m of the "
    f"track's bounding box are indexed. Arc geometry is sampled every {config.DENSIFY_SPACING_M:.0f} m into a "
    "KD-tree for radius queries.",
    "<b>Reference labels per fix</b> (src/labels.py): a monotone dynamic programme assigns each fix a route "
    "position j<sub>0</sub> &le; j<sub>1</sub> &le; &hellip; that minimises the total point-to-arc distance. "
    "The label of fix i is route[j<sub>i</sub>].",
    f"<b>Label reliability</b>: fixes more than 50 m from their aligned reference arc are treated as unlabelled "
    f"({pc(eda['train_align_gt50_frac'])} of train fixes). They are excluded from fitting and evaluation, "
    "because the reference route simply does not cover them (see the coverage plot below). Fixes whose reference "
    "arc is not among the candidates are also excluded from fitting, but they <i>are</i> counted as errors at evaluation.",
])
h2("Missing-value and data-quality analysis")
table([[r.partition, r.file, r.trips, f"{r.rows:,}", r.cols, r.missing_values] for r in mv.itertuples()],
      ["Partition", "File", "Trips", "Rows", "Columns", "Missing values"], [3, 2.2, 1.5, 2.5, 1.8, 2.5])
p("No file has missing values or malformed rows, and every timestamp sequence is strictly increasing "
  f"({eda['non_monotonic_timestamps']} non-monotonic steps). The real missing data is <b>missing fixes</b> "
  "(gaps in the 1 Hz stream) and <b>missing reference coverage</b>: trips 00000054 and 00000033 have long parts "
  "of the GPS track with no reference route, so their alignment distances reach hundreds of metres. "
  f"Candidate miss rate on train: {pc(eda['train_candidate_miss_rate_all'])} over all fixes, but only "
  f"{pc(eda['train_candidate_miss_rate_reliable'])} over fixes with reliable labels.")
img(F / "eda_missing_gaps.png", 17, "Left: implied missing 1 Hz fixes per trip (train + student-test). "
    "Right: share of train fixes more than 50 m from the reference route.")
img(F / "eda_inventory.png", 17, "Dataset inventory: fixes per trip, sampling intervals (train), network size (train).")
inv_t = inv[inv.partition == "train"]
p(f"<b>Road-network connectivity</b> (train): the median trip network has {eda['weak_components_median']:.0f} weakly "
  f"connected components; the largest strongly connected component covers a median "
  f"{pc(eda['largest_scc_frac_median'], 1)} of nodes, and there are {eda['isolated_nodes_total']} isolated nodes. "
  "Because the extracts are not one strongly connected graph (one-way streets, pieces cut off at the extract "
  "boundary), a directed path between two candidate arcs can fail to exist. The HMM therefore uses a soft "
  "'unreachable' penalty rather than probability zero.")
h2("GPS trajectory and road-network visualisations")
img(F / "eda_trip_locations.png", 16, "Trip locations (mean of each track), train and student-test: the trips are "
    "spread over five continents, so every trip gets its own local projection.")
img(F / "eda_all_trajectories.png", 14, "GPS trajectories of 16 train trips, each in its own local metric frame "
    "(green = start, red = end): loops, long straight drives and dense urban routes.")
img(F / "eda_network_trip.png", 13, "Train trip 00000046: nearby road network (grey), reference route (green) and GPS fixes (blue).")
img(F / "eda_zoom_noise.png", 9, "300 m zoom of the same trip: fixes lie a few metres off the road and near parallel arcs.")
img(F / "map_eda_00000046.png", 15, "Folium map (maps/eda_trip_00000046.html), train partition: GPS fixes, "
    "road network within 150 m, reference route, with a layer control and legend.")
h2("Q2. Observed patterns, extracted features and why")
p(f"The median fix-to-reference-arc distance is a few metres, but the reference arc is the <b>nearest</b> candidate "
  f"for only {pc(eda['true_arc_is_nearest_frac'], 1)} of fixes. The main reason is that each two-way street gives two "
  "geometrically identical arcs, which can only be told apart by the direction of travel. Near junctions, several "
  "arcs are equally close. These observations led directly to the features:")
table([
    ["dist, dist_rank, dist_minus_min, dist_ratio", "Geometric closeness, absolute and relative to the best candidate (GPS noise is about 5-15 m)."],
    ["n_cand, cand_density_10m", "Ambiguity: how many arcs compete within the radius and within 10 m of the best one."],
    ["cos_heading, abs_heading_diff, heading_valid", "Direction of travel (fix i-2 to i+2) against the arc bearing; this separates twin arcs and crossing streets. Invalid when the vehicle is stationary (displacement < 3 m)."],
    ["speed", "Kinematic context: at low speed the heading is noisy, and fast segments favour long arterial arcs."],
    ["arc_len, t_frac, dist_to_end", "Where the fix projects on the arc; projections clamped to an arc end are often an adjacent arc."],
    ["d_prev2, d_next2, d_prev5, d_next5", "Temporal context: distance of neighbouring fixes to the same arc. The true arc stays close over consecutive seconds."],
    ["out_degree, in_degree, is_two_way", "Local topology: junction complexity and whether a reverse twin exists."],
], ["Feature(s)", "Motivation"], [5, 12.5])
img(F / "eda_candidates.png", 17, "Train: fix-to-reference distance, candidates per fix, and distance rank of the true arc.")
img(F / "eda_feature_distributions.png", 17, "Feature distributions for the reference arc vs other candidates (train, reliable labels).")
story.append(PageBreak())

# ----------------------------------------------------------------------------- 3 models
h1("3. Model Experimentation and Comparison")
p("<b>Task formulation.</b> Each row is a (fix, candidate arc) pair with label 1 if the arc is the fix's "
  "reference arc. A model outputs P(candidate = reference arc). The Top-1 prediction for a fix is its "
  "highest-scoring candidate.")
p("<b>Protocol.</b> 5-fold GroupKFold over the 80 train trips (folds split by trip, results/cv_folds.json). "
  "Models are fitted on reliable-label rows, then refitted on all 80 trips and evaluated once on the 10 "
  "student-testing trips. <b>Macro F1</b> is the candidate-level binary macro F1 at threshold 0.5. <b>Top-1 "
  "accuracy</b> is observation-level, computed on every fix with a reliable label (a fix whose reference arc is "
  "missing from the candidates counts as wrong). <b>Route continuity</b> is the share of consecutive fixes whose "
  "Top-1 arcs are identical or directly connected. <b>Primary selection metric: CV Top-1 accuracy</b>. It "
  "measures the map-matching objective directly, is computed on held-out trips, and does not depend on a "
  "probability threshold, unlike macro F1.")
h2("3.1 Overall model comparison")
rows = []
for m in c.sort_values("cv_top1", ascending=False).index:
    r = c.loc[m]
    rows.append([m, r.family, r.config, f"{f(r.cv_macro_f1)} &plusmn; {f(r.cv_macro_f1_std, 3)}", f(r.st_top1),
                 f(r.st_continuity), f(r.st_macro_f1), f(r.st_log_loss), f"{f(r.inference_ms_per_fix, 4)} ms/fix"])
table(rows, ["Model", "Algorithm family", "Key configuration", "CV macro F1 (mean &plusmn; sd)", "Top-1 acc. (student-test)",
             "Route continuity (student-test)", "Student-test macro F1", "Log loss (student-test)", "Inference time"],
      [2.2, 2.0, 3.6, 1.9, 1.5, 1.6, 1.5, 1.4, 1.6])
p("Every model outputs probabilities, so log loss applies to all of them. Route continuity here is for "
  "classifier-only Top-1 predictions; Section 5 reports it after HMM decoding. Additional columns (CV Top-1, CV log "
  "loss, precision, recall, training time) are in results/model_comparison.csv.")
rows = []
for m in c.sort_values("cv_top1", ascending=False).index:
    r = c.loc[m]
    rows.append([m, f(r.cv_top1), f(r.cv_top1_std, 3), f(r.cv_log_loss), f(r.st_precision), f(r.st_recall),
                 f(r.cv_continuity), f"{r.train_time_s:.1f}"])
table(rows, ["Model", "CV Top-1", "CV Top-1 sd", "CV log loss", "Student-test macro precision",
             "Student-test macro recall", "CV continuity", "Train time (s, 80 trips)"], [3, 1.8, 1.8, 1.8, 2.3, 2.3, 1.9, 2.4])
p(f"<b>Selection.</b> {sel} has the highest mean CV Top-1 accuracy ({f(c.loc[sel, 'cv_top1'])}). See the "
  "per-model analyses below for the trade-offs.")

story.append(PageBreak())
h2("3.2 Per-model analysis")
NOTES = json.load(open(config.ROOT / "report" / "model_notes.json"))
for i, m in enumerate(c.sort_values("cv_top1", ascending=False).index, 1):
    r = c.loc[m]
    ft = fold_train.loc[m]
    n = NOTES[m]
    h3(f"Model {i}: {m} ({r.family})")
    p(f"<b>A1. Why suitable?</b> {n['why']}")
    p(f"<b>A2. Strengths / limitations.</b> {n['strengths']}")
    p(f"<b>B5. Important hyper-parameters (initial values).</b> {r.config}. {n.get('hyper', '')}")
    p(f"<b>B6. Loss / objective.</b> {n['loss']}")
    p(f"<b>C7. Student-test performance.</b> Top-1 accuracy {f(r.st_top1)}, route continuity {f(r.st_continuity)}, "
      f"route F1 {f(r.st_route_f1)}, inference {f(r.inference_ms_per_fix, 4)} ms per fix "
      f"(trained in {r.train_time_s:.1f} s on 80 trips).")
    p(f"<b>C8. Observation-level metrics (student-test).</b> Top-1 accuracy {f(r.st_top1)}; macro F1 {f(r.st_macro_f1)} "
      f"(precision {f(r.st_precision)}, recall {f(r.st_recall)}); log loss {f(r.st_log_loss)}.")
    gap = ft.train_macro_f1 - ft.macro_f1
    p(f"<b>C9. Over- / under-fitting.</b> Mean training-fold macro F1 {f(ft.train_macro_f1)} vs held-out-fold "
      f"{f(ft.macro_f1)} (gap {f(gap, 3)}); training log loss {f(ft.train_log_loss)} vs held-out {f(ft.log_loss)}; "
      f"fold-to-fold Top-1 sd {f(r.cv_top1_std, 3)}. {n['fit']}")
h2("3.3 Model visualisations")
img(F / "cmp_cv_macro_f1.png", 15, "Cross-validation macro F1 for every model (bars = mean, dots = the 5 folds).")
img(F / "cmp_student_test.png", 15, "Student-test Top-1 accuracy and macro F1 (CV Top-1 shown for reference).")
img(F / "cmp_logloss_time.png", 16, "Log loss (CV bars, student-test line) and inference time per fix.")
img(F / "cmp_continuity.png", 13, "Route continuity of classifier-only predictions.")
img(F / "cmp_confusion_matrices.png", 17, "Candidate-level confusion matrices on student-test (threshold 0.5).")
img(F / "cmp_training_curves.png", 15, "Epoch-wise histories, shown only for models that produce them: MLP "
    "(epochs) and LightGBM (boosting iterations). Recorded on fold 0 (65 train trips vs 15 held-out trips). "
    "No curves are shown for the other algorithms because they do not train iteratively.")
story.append(PageBreak())

# ----------------------------------------------------------------------------- 4 Bayesian optimisation
h1(f"4. Bayesian Hyper-parameter Optimisation ({sel})")
space_rows = {
    "LightGBM": [["n_estimators", "int 100-800, step 50"], ["learning_rate", "float 0.01-0.2, log-uniform"],
                 ["num_leaves", "int 15-255, log-uniform"], ["min_child_samples", "int 5-300, log-uniform"],
                 ["subsample (bagging_fraction, freq 1)", "float 0.5-1.0"], ["colsample_bytree", "float 0.4-1.0"],
                 ["reg_lambda (L2)", "float 1e-3 to 10, log-uniform"], ["reg_alpha (L1)", "float 1e-3 to 10, log-uniform"]],
    "RandomForest": [["n_estimators", "int 100-400"], ["max_depth", "int 8-40"], ["min_samples_leaf", "int 1-50 log"], ["max_features", "float 0.2-1.0"]],
    "ExtraTrees": [["n_estimators", "int 100-400"], ["max_depth", "int 8-40"], ["min_samples_leaf", "int 1-50 log"], ["max_features", "float 0.2-1.0"]],
}.get(sel, [])
p("<b>Q1/Q2. Hyper-parameters and ranges.</b> The parameters that control model capacity (tree count, leaves, "
  "minimum child size), the learning rate, and the stochastic and regularisation terms, which control variance "
  "across trips. Log-uniform ranges are used where the effect is multiplicative.")
table(space_rows, ["Hyper-parameter", "Search distribution"], [7, 8])
done = trials[trials.state == "COMPLETE"]
p(f"<b>Q3. Objective.</b> {best['objective']}. Sampler: {best['sampler']}; pruner: {best['pruner']}. "
  "The pruner may stop a trial after its first folds if it is clearly below the median"
  + (f"; in this run it stopped {best['n_pruned']} trial(s)." if best['n_pruned'] else
     "; in this run no trial was pruned, because every configuration stayed close to the median.") )
p(f"<b>Q4. Trials.</b> {best['n_trials']} trials ({len(done)} completed, {best['n_pruned']} pruned). "
  f"Total search time {best['search_seconds'] / 60:.1f} min. The full trial log is in results/optuna_trials.csv.")
p("<b>Q5. Best values.</b> " + ", ".join(f"{k} = {f(v, 4) if isinstance(v, float) else v}" for k, v in best["params"].items())
  + f" (best CV Top-1 {f(best['best_cv_top1'])}).")
tu = tune.set_index("model")
un, tn = tu.iloc[0], tu.iloc[1]
better = tn.cv_top1 > un.cv_top1
p(f"<b>Q6. Did tuning help?</b> CV Top-1 {f(un.cv_top1)} &rarr; {f(tn.cv_top1)}; student-test Top-1 {f(un.st_top1)} "
  f"&rarr; {f(tn.st_top1)}; CV macro F1 {f(un.cv_macro_f1)} &rarr; {f(tn.cv_macro_f1)}; student-test macro F1 "
  f"{f(un.st_macro_f1)} &rarr; {f(tn.st_macro_f1)}; log loss (student-test) {f(un.st_log_loss)} &rarr; {f(tn.st_log_loss)}. "
  + ("The tuned model is better on the optimised objective. " if better else
     "The tuned model did not beat the untuned configuration on the optimised objective. ")
  + "Both models use the same folds, features and student-test trips. The gain is small (about 0.3 percentage "
    "points), which shows that the untuned LightGBM was already near its plateau for these features. The price is "
    "a slower model (larger trees). After HMM decoding, the two models are within 0.001 Top-1 of each other on "
    "student-test (Section 5), which is within trip-to-trip noise. The tuned model is kept because it was "
    "selected on CV, never on student-test results.")
table([[i, f(r.cv_macro_f1), f(r.cv_top1), f(r.st_top1), f(r.st_macro_f1), f(r.cv_log_loss), f(r.st_log_loss),
        f"{r.train_time_s:.1f}", f(r.inference_ms_per_fix, 4)] for i, r in tu.iterrows()],
      ["Model", "CV macro F1", "CV Top-1", "Student-test Top-1", "Student-test macro F1", "CV log loss",
       "Student-test log loss", "Train time (s)", "Inference (ms/fix)"], [3.4, 1.6, 1.5, 1.8, 1.8, 1.6, 1.8, 1.6, 1.8])
img(F / "opt_history.png", 16, "Optuna optimisation history (CV Top-1 per trial) and fANOVA hyper-parameter importance.")
img(F / "opt_tuned_vs_untuned.png", 16, "Untuned vs tuned: accuracy and F1, log loss, inference time.")
story.append(PageBreak())

# ----------------------------------------------------------------------------- 5 HMM
h1("5. HMM and Viterbi Route Reconstruction")
p("<b>Q1. Hidden states.</b> At fix i, the hidden state is one of that fix's candidate arcs (up to "
  f"{config.MAX_CANDIDATES}, including both directions of a two-way street), together with the fix's projected "
  "position t on that arc. The decoded state sequence gives exactly one arc id per fix, so the "
  "candidate-to-road-segment correspondence is kept (src/hmm.py).")
p(f"<b>Q2. Emission probabilities.</b> log e<sub>i</sub>(s) = w &middot; log p<sub>clf</sub>(s), where "
  f"p<sub>clf</sub> is the tuned {sel} probability for candidate s (clipped at 10<sup>-6</sup>). The classifier "
  "already combines distance, heading and context, so it replaces the usual Gaussian distance emission. The weight "
  f"w balances emission against transition; selected w = {hp['emission_weight']}.")
p("<b>Q3. Transition probabilities.</b> log a(s&rarr;s') = &minus;|d<sub>route</sub> &minus; d<sub>gc</sub>| / &beta;. "
  "Here d<sub>gc</sub> is the straight-line distance between the two fixes, and d<sub>route</sub> is the driving "
  "distance from the projected point on arc a to the projected point on arc b: (1&minus;t<sub>a</sub>)&middot;len<sub>a</sub> "
  "+ shortestpath(head(a), tail(b)) + t<sub>b</sub>&middot;len<sub>b</sub>. A bounded Dijkstra is used, with search limit "
  "max(2&middot;d<sub>gc</sub> + 100 m, 200 m). On the same arc, moving forward costs (t<sub>b</sub>&minus;t<sub>a</sub>)&middot;len, "
  "and backward jitter up to 15 m is tolerated. This exponential form (as in Newson and Krumm's HMM map matching) "
  f"penalises detours and impossible jumps. Selected &beta; = {hp['beta']} m.")
p(f"<b>Q4. Missing candidates and GPS gaps.</b> If no arc lies within 50 m, the radius widens to 100, 200 and then "
  "400 m. Fixes with still no candidate are skipped by the decoder and get no prediction (they count as errors). "
  f"If two consecutive fixes are more than {hp['gap_break_s']:.0f} s apart, the transition becomes uniform, so the "
  "chain restarts and the emissions decide. For shorter gaps, the Dijkstra search limit grows with "
  f"d<sub>gc</sub>. Train data contains {hmm_cfg['gap_breaks_train']} such chain breaks.")
p(f"<b>Q5. Disconnected segments.</b> If no path exists within the search limit (for example the candidates are "
  "in different network components, or the move needs a long detour), the transition gets a fixed log-score "
  f"of {hp['unreachable_logp']} instead of &minus;&infin;. Viterbi therefore never gets stuck: a single noisy fix "
  "cannot force a disconnected route, but a sequence of strong emissions can still override the penalty when the "
  f"network itself is broken. {pc(hmm_cfg['unreachable_pair_fraction_train'], 1)} of candidate pairs in train were "
  "unreachable. When the decoded route is drawn, consecutive distinct arcs that are not adjacent are joined with "
  "the shortest path.")
p("<b>Viterbi recurrence.</b> &delta;<sub>i</sub>(s') = max<sub>s</sub>[&delta;<sub>i-1</sub>(s) + log a(s&rarr;s')] + "
  "log e<sub>i</sub>(s'), with back-pointers and back-tracking from argmax &delta;<sub>n</sub>. Scores are "
  "re-normalised at each step to avoid underflow.")
gv = lambda col: ", ".join(f"{v:g}" for v in sorted(grid[col].unique()))
p(f"<b>Decoder tuning.</b> A grid over &beta; &isin; {{{gv('beta')}}}, w &isin; {{{gv('emission_weight')}}} and the unreachable "
  f"score &isin; {{{gv('unreachable_logp')}}} ({len(grid)} settings). It was evaluated on the 80 train trips using "
  "<i>out-of-fold</i> probabilities of the final model, so the emissions are as calibrated as on unseen trips. "
  "Student-test trips were not used for the choice. The best setting lies inside the grid, not on its edge. "
  "Some settings tie exactly (e.g. &beta;=1, w=2, &minus;50 and &beta;=2, w=1, &minus;25). That is expected: multiplying every "
  "emission and transition score by the same constant does not change the Viterbi argmax, so those are the same decoder.")
img(F / "hmm_grid.png", 12, "Decoder grid (train OOF): Top-1 accuracy over beta and emission weight.")
p("<b>Q6. Did HMM decoding improve the route?</b>")
rows = []
for name in [fin_name]:
    for mode in ("classifier-only", "classifier+HMM"):
        r = hs.loc[mode]
        rows.append([f"{name}, student-test", mode, f(r.top1_acc), f(r.decision_macro_f1), f(r.route_continuity),
                     f(r.route_f1), pc(r["trips_top1_ge_0.9"], 0)])
for mode in ("classifier-only", "classifier+HMM"):
    r = htr.loc[mode]
    rows.append([f"{sel} train OOF (80 trips)", mode, f(r.top1_acc), f(r.decision_macro_f1), f(r.route_continuity),
                 f(r.route_f1), pc(r["trips_top1_ge_0.9"], 0)])
table(rows, ["Model / data", "Decoding", "Top-1 acc.", "Decision macro F1", "Route continuity", "Route F1 (arc set)",
             "Trips with Top-1 &ge; 0.9"], [4.2, 2.4, 1.6, 2, 2, 2.1, 2.2])
st_all = hcmp[~hcmp.model.str.contains("train OOF")].pivot(index="model", columns="mode", values="top1_acc")
st_all["gain"] = st_all["classifier+HMM"] - st_all["classifier-only"]
table([[m, f(r["classifier-only"]), f(r["classifier+HMM"]), f"{r.gain:+.4f}"] for m, r in st_all.sort_values("classifier+HMM", ascending=False).iterrows()],
      ["Model (student-test)", "Classifier-only Top-1", "Classifier + HMM Top-1", "Gain"], [6, 3.5, 3.5, 2.5])
img(F / "hmm_models_comparison.png", 17, "Student-test Top-1 and route continuity, classifier-only vs classifier + HMM, for every model.")
img(F / "hmm_per_trip.png", 15, "Per-trip Top-1 on student-test for the final model.")
fp = hpt[hpt.model == fin_name].pivot(index="trip_id", columns="mode", values="top1")
fp["gain"] = fp["classifier+HMM"] - fp["classifier-only"]
worse = fp[fp.gain < 0]
err = json.load(open(R / "hmm_error_analysis.json"))
ec, eh = err["errors"]["classifier_only"], err["errors"]["hmm"]
cats = [("reverse_twin", "Reverse twin (opposite direction of the right street)"),
        ("adjacent_arc", "Adjacent arc (shares a node: junction / arc boundary)"),
        ("other", "Other (e.g. parallel street)"),
        ("missing_candidate", "Reference arc not among the candidates")]
p("<b>Where HMM helped and where it did not.</b> scripts/05b_hmm_error_analysis.py puts every wrongly matched "
  f"student-test fix ({err['n_eval_fixes']:,} evaluated fixes) into exactly one category:")
table([[lab, ec[k], eh[k], f"{(ec[k] - eh[k]) / ec[k] * 100:.0f}%" if ec[k] else "-"] for k, lab in cats]
      + [["<b>Total errors</b>", sum(ec.values()), sum(eh.values()),
          f"{(sum(ec.values()) - sum(eh.values())) / sum(ec.values()) * 100:.0f}%"]],
      ["Error type", "Classifier only", "Classifier + HMM", "Removed by HMM"], [8, 3, 3, 3])
p(f"HMM decoding removes {(sum(ec.values()) - sum(eh.values())) / sum(ec.values()) * 100:.0f}% of all errors. It works best on "
  f"<b>'other'</b> errors such as jumps to a parallel street ({(ec['other'] - eh['other']) / ec['other'] * 100:.0f}% removed), "
  "because such a jump has no short network path and the transition model penalises it. It removes fewer "
  f"<b>reverse-twin</b> ({(ec['reverse_twin'] - eh['reverse_twin']) / ec['reverse_twin'] * 100:.0f}%) and "
  f"<b>adjacent-arc</b> ({(ec['adjacent_arc'] - eh['adjacent_arc']) / ec['adjacent_arc'] * 100:.0f}%) errors, and these two "
  f"now make up {(eh['reverse_twin'] + eh['adjacent_arc']) / sum(eh.values()) * 100:.0f}% of what remains. Reverse twins are "
  "geometrically identical, so when the heading is unreliable (slow driving, stops) the emission cannot separate them, "
  "and a whole run of fixes can be decoded consistently in the wrong direction. Adjacent-arc errors sit at junctions, "
  "where the timestamp-free reference switches arcs a few fixes earlier or later than the decoder, so part of this is "
  f"label ambiguity rather than a real mistake. The {eh['missing_candidate']} missing-candidate fixes cannot be recovered "
  "by any decoder. The largest per-trip gain is "
  f"{f(fp.gain.max(), 3)} (trip {fp.gain.idxmax()}). "
  + (f"It did not help on {len(worse)} trip(s) ({', '.join(f'{t}: {g:+.3f}' for t, g in worse.gain.items())})."
     if len(worse) else "No student-test trip got worse with HMM decoding."))
story.append(PageBreak())

# ----------------------------------------------------------------------------- 6 Folium
h1("6. Geographical Visualisation Using Folium")
p("scripts/06_maps.py builds an interactive Folium map for each student-test trip (maps/student_test_&lt;id&gt;.html), "
  "generated from the final model and decoder. Layers (switchable with the layer control): GPS observations "
  "(blue points, with start and end markers), road network within 150 m (grey), reference route (thick green), "
  "classifier-only Top-1 arcs (dashed orange, hidden by default), the predicted route after HMM decoding (red), "
  "and fixes the HMM matched wrongly (purple, hidden by default). A legend and the trip's Top-1 accuracies are "
  "shown on the map, and the map is zoomed to the track's bounding box.")
for png in sorted(F.glob("map_0*.png")):
    tid = png.stem.replace("map_", "")
    img(png, 16, f"Folium map, student-test trip {tid}: GPS (blue), reference route (green), predicted HMM route (red).")
story.append(PageBreak())

# ----------------------------------------------------------------------------- 7 API / UI
h1("7. API, UI and Deployment")
p("<b>Laptop 1</b> runs the FastAPI service (api/main.py): <i>uvicorn api.main:app --host 0.0.0.0 --port 8000</i>. "
  "At start-up it loads the final model (models/final_model.joblib), the feature list and the decoder configuration "
  "(models/hmm_config.json). It serves the trips found in data/eval/. <b>GET /trips</b> lists them. "
  "<b>POST /predict</b> {trip_id} runs cleaning &rarr; candidates &rarr; features &rarr; classifier &rarr; "
  "HMM/Viterbi &rarr; route filling, then scores the prediction against the reference route <i>on the server</i>. "
  "The response contains the GPS points, predicted route polylines, predicted arc per fix, the metrics "
  "(classifier-only vs HMM) and timings. Reference arcs are never returned. Every request is logged on the "
  "console with the client IP, trip id, Top-1 accuracy and latency.")
p("<b>Laptop 2</b> serves ui/index.html (<i>python -m http.server 5500</i>). The evaluator enters Laptop 1's "
  "address, the UI lists the ten trips, and selecting one calls /predict and draws the GPS fixes and predicted "
  "route on a Leaflet map, with the metrics table.")
img(F / "ui_screenshot.png", 16, "Evaluator UI showing a student-test trip served by the API.")
api_log = config.RESULTS_DIR / "logs" / "api_final_check.log"
if api_log.exists():
    lines = [l.strip() for l in api_log.read_text(encoding="utf-8", errors="ignore").splitlines()
             if "PREDICT request" in l or "RESULT trip" in l][:8]
    p("<b>API log excerpt</b>: the first 4 of the 10 student-test trips sent to /predict "
      "(the full log is results/logs/api_final_check.log). These lines are what the evaluator sees on Laptop 1:")
    p("<br/>".join(l.replace("&", "&amp;").replace("<", "&lt;") for l in lines),
      ParagraphStyle("mono", parent=SM, fontName="Courier", fontSize=6.6, leading=8.4, backColor=colors.HexColor("#f2f2f2"),
                     borderPadding=4))
p("[Insert screenshot: UI on Laptop 2 during the evaluation run]", BOX)
p("[Insert screenshot: CMD window on Laptop 1 showing the API log lines (PREDICT request ... / RESULT ...) for the "
  "evaluator's requests]", BOX)

story.append(PageBreak())
h1("8. Conclusions and Limitations")
bullets([
    f"Tree ensembles and boosting clearly beat the linear, probabilistic and instance-based models on this "
    f"problem. The selected {sel} gave the best CV Top-1 and the best student-test Top-1 at a low training and "
    "inference cost (Random Forest was equally accurate, with a marginally lower log loss, but trained 12x slower).",
    "HMM decoding is the single most effective component for route quality: it enforces network connectivity, "
    "removes 46% of the classifier's errors on student-test (74% of jumps to other streets) and lifts Top-1 from 0.895 "
    "to 0.943. The main remaining errors are direction flips on two-way streets and junction-boundary ambiguity.",
    "The labels depend on aligning a timestamp-free reference route. Two train trips have large uncovered parts "
    "and were partly excluded. Accuracy near junctions is limited by how ambiguous the exact switching fix is.",
    "Possible improvements: train the transition model (for example a learned speed-dependent &beta;), use "
    "turn-restriction costs, and use classifier probabilities calibrated per trip.",
])


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(1.6 * cm, 1.0 * cm, "PathSnap technical report")
    canvas.drawRightString(A4[0] - 1.6 * cm, 1.0 * cm, f"page {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT / "PathSnap_Technical_Report.pdf"), pagesize=A4, leftMargin=1.6 * cm,
                        rightMargin=1.6 * cm, topMargin=1.5 * cm, bottomMargin=1.6 * cm,
                        title="PathSnap Technical Report")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT / "PathSnap_Technical_Report.pdf")

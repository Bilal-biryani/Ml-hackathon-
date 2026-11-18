# PathSnap: map matching and route reconstruction (Team PT-040)

## Review 1 submission — Arfan (PES1UG24AM163)

Submitted by **Arfan Asgar (PES1UG24AM163)**. This submission includes the complete supplied Review 1 code, results, figures, maps, and available trained models. Arfan's contribution files are also preserved in `files/`, with the accompanying notes in `VIVA_AND_PUSH_Arfan.md`.

Team PT-040: Mohammad Bilal (PES1UG24AM162), Mohammed Sinan (PES1UG24AM165), Mohammed Arfan (PES1UG24AM163), Mirunjai (PES1UG24AM161)

GPS traces are noisy: a car driving on one road can appear to be on a parallel street or between buildings. PathSnap's goal is to find, for every GPS point, the **road segment (arc)** the vehicle was actually on.

## Review 1 status: data pipeline, EDA and model comparison
This review covers understanding the data, building the data pipeline, and training and comparing **7 models** (the rules ask for at least 5). Bayesian hyper-parameter tuning, HMM/Viterbi route reconstruction, the API and the UI come in Review 2.

| Stage | File | What it does |
|---|---|---|
| Configuration | `src/config.py` | Paths and constants (search radius, number of candidates, feature list) |
| Data loading and cleaning | `src/io.py` | Reads `.track`, `.nodes`, `.arcs`, `.route`; removes invalid or duplicate GPS points; sorts by time |
| Coordinates | `src/geo.py` | Converts lon/lat to metres (local projection); point-to-segment distance |
| Road network | `src/network.py` | Road graph, KD-tree for fast "which roads are near this point" queries, degrees, two-way arcs |
| Map helpers | `src/viz.py` | Folium map layers (GPS, road network, route, legend) |
| Labels | `src/labels.py` | Matches the reference route (no timestamps) to the GPS points: every point gets its true arc |
| Candidates and features | `src/features.py` | For every GPS point: nearby candidate arcs (within 50 m, at most 10) plus 20 features per candidate |
| Pipeline | `src/pipeline.py` | load → network → candidates → features (and labels for training) |
| Dataset builder | `scripts/02_build_dataset.py` | Runs the pipeline on all 80 train and 10 student-test trips |
| EDA | `scripts/01_eda.py` | Inventory, missing-value analysis, plots and Folium maps |
| Training data and metrics | `src/data.py`, `src/metrics.py` | Train/eval masks; macro F1, log loss, Top-1 accuracy, candidate miss rate, route continuity |
| Models | `src/models/` | `linear.py` (Logistic Regression, Naive Bayes), `neighbors_nn.py` (KNN, MLP), `trees.py` (Random Forest, Extra Trees), `boosting.py` (LightGBM), `__init__.py` (model list) |
| Model comparison | `scripts/03_train_compare.py` | 5-fold trip-grouped cross-validation of all models, then student-test evaluation and plots |

## Dataset
```
data/raw/train/<trip_id>/<trip_id>.track   lon lat time      (GPS, about 1 point per second)
                         <trip_id>.nodes   lon lat           (road-network nodes; row number = node id)
                         <trip_id>.arcs    from_node to_node (directed road segments; row number = arc id)
                         <trip_id>.route   arc_id            (reference route: ordered list of arcs)
data/raw/student_test/...                                     (same format, 10 trips)
```
The raw data is not in Git because of its size. Copy the provided folders into `data/raw/`.

## How to run
```bash
pip install -r requirements.txt
python scripts/02_build_dataset.py     # candidates + features + labels -> data/processed/  (~10 s)
python scripts/01_eda.py               # EDA tables, plots, Folium maps                      (~40 s)
python scripts/03_train_compare.py     # 7 models x 5 folds + student-test                   (~40 min)
python scripts/03_train_compare.py --plots-only   # only redraw the comparison figures
```

## How the problem is set up for machine learning
For every GPS point we take its candidate arcs, and each (point, arc) pair is one row with label 1 if the arc is the true arc. A model gives each candidate a probability, and the candidate with the highest probability is the prediction for that point.

- **Cross-validation:** 5-fold `GroupKFold` grouped by trip, so a trip is never split between training and validation.
- **Macro F1:** candidate-level F1 averaged over both classes, at threshold 0.5.
- **Top-1 accuracy:** the share of GPS points whose highest-scoring candidate is the true arc. **This is the main selection metric** because it measures map matching directly.

## Model comparison results
| Model | Family | CV macro F1 | CV Top-1 | Student-test Top-1 | Student-test macro F1 | Student-test log loss | Inference (ms/point) |
|---|---|---|---|---|---|---|---|
| LightGBM | Boosting | 0.9280 | 0.8802 | 0.8913 | 0.9322 | 0.0822 | 0.0216 |
| RandomForest | Bagging ensemble | 0.9290 | 0.8799 | 0.8852 | 0.9322 | 0.0821 | 0.0199 |
| MLP | Neural network | 0.9211 | 0.8747 | 0.8757 | 0.9223 | 0.0916 | 0.0031 |
| ExtraTrees | Randomised trees | 0.9247 | 0.8739 | 0.8764 | 0.9252 | 0.0890 | 0.0291 |
| LogisticRegression | Linear | 0.9002 | 0.8627 | 0.8610 | 0.9033 | 0.1314 | 0.0009 |
| KNN | Instance-based | 0.9083 | 0.8612 | 0.8617 | 0.9116 | 0.1236 | 0.5059 |
| GaussianNB | Probabilistic | 0.8393 | 0.8452 | 0.8384 | 0.8324 | 0.5710 | 0.0027 |

**Selected for Review 2: LightGBM.** It has the highest CV Top-1. Random Forest is practically tied, but LightGBM trains about 12× faster (9.7 s vs 122 s) and its model file is much smaller. The tree models beat the linear, probabilistic and instance-based models because they learn feature interactions (for example, heading only matters when the car is moving and a two-way twin arc is equally close). All numbers come from `results/model_comparison.csv`; the plots are in `figures/cmp_*.png`.

## Key data findings
- 80 train trips (208,417 GPS points) and 10 student-test trips (19,160 points). There are no missing values in any file, and timestamps always increase.
- 96.5% of time steps are exactly 1 s apart, but some trips have long gaps (up to 42 hours).
- Each trip's road network has between 649 and 307,392 arcs. 77% of candidate arcs have a reverse twin (two-way streets).
- The true arc is the **nearest** arc for only 56% of points, so distance alone is not enough. This is why heading and context features are needed.
- Two train trips (00000054, 00000033) have a reference route that covers only part of the GPS track. Points more than 50 m from the route are treated as unlabelled.
- With a 50 m radius and 10 candidates, the true arc is in the candidate set for 99.3% of reliably labelled points.

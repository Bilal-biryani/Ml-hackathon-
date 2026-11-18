# PathSnap — GPS Map Matching with Machine Learning

PathSnap is a GPS map-matching project that assigns noisy GPS observations to the most plausible directed road arcs in a trip-specific road network. The current repository implements the **data preparation, exploratory analysis, candidate generation, feature engineering, seven-model comparison, evaluation, saved models, and Folium-based analysis maps** for the first stage of the project.

The final project will extend this classifier pipeline with **Bayesian hyperparameter optimisation, HMM/Viterbi sequence decoding, route reconstruction, FastAPI deployment, and an evaluator-facing UI**.

> **Dataset policy:** the raw dataset is supplied by the college. It is treated as external, read-only source data. Do **not** edit, rename, regenerate, or commit the raw dataset files to GitHub.

---

## 1. Current project status

### Implemented

- Loading and cleaning of `.track`, `.nodes`, `.arcs`, and `.route` files
- Trip-wise road-network construction
- Local metric projection for geometric calculations
- KD-tree-backed road candidate generation
- Candidate-level feature engineering
- Reference-route alignment and binary candidate labels
- Group-aware cross-validation by trip
- Comparison of seven machine-learning models
- Student-test evaluation
- Saved trained models
- EDA figures and Folium maps
- Lightweight live training demonstration

### Still required for the final project

- Bayesian hyperparameter optimisation of the selected model
- HMM emission/transition formulation
- Viterbi decoding
- Full route reconstruction and route-level evaluation
- FastAPI backend for Laptop 1
- Evaluator UI for Laptop 2
- Final Folium prediction visualisation
- Final deployment instructions and technical report

---

## 2. Repository structure

```text
Ml-hackathon-/
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── config.py          # paths and shared constants
│   ├── io.py              # raw file readers
│   ├── geo.py             # local metric projection / geometry helpers
│   ├── network.py         # road-network representation and spatial index
│   ├── features.py        # candidate feature engineering
│   ├── labels.py          # reference-route alignment
│   ├── pipeline.py        # end-to-end trip preparation
│   ├── data.py            # processed modelling-table loading
│   ├── metrics.py         # candidate and observation-level metrics
│   ├── viz.py             # visualisation helpers
│   └── models/
│       ├── linear.py
│       ├── neighbors_nn.py
│       ├── trees.py
│       ├── boosting.py
│       └── common.py
│
├── scripts/
│   ├── 01_eda.py
│   ├── 02_build_dataset.py
│   ├── 03_train_compare.py
│   └── demo_live.py
│
├── models/                # committed trained model artifacts
├── results/               # committed experiment outputs
├── figures/               # EDA and model-comparison figures
├── maps/                  # generated Folium HTML maps
└── data/
    ├── raw/               # college-provided dataset; ignored by Git
    └── processed/         # generated modelling tables; ignored by Git
```

---

## 3. Dataset layout

The code expects the college-provided dataset to be placed under:

```text
data/raw/
├── train/
│   ├── <trip_id>/
│   │   ├── <trip_id>.track
│   │   ├── <trip_id>.nodes
│   │   ├── <trip_id>.arcs
│   │   └── <trip_id>.route
│   └── ...
└── student_test/
    ├── <trip_id>/
    │   ├── <trip_id>.track
    │   ├── <trip_id>.nodes
    │   ├── <trip_id>.arcs
    │   └── <trip_id>.route
    └── ...
```

The supplied student dataset contains:

- **80 training trips**
- **10 student-test trips**

The evaluator-test partition is not included in the student dataset and must not be used during training.

The repository `.gitignore` excludes both `data/raw/` and `data/processed/` so that college-provided and generated data are not committed.

---

## 4. Environment setup

Clone the repository and move into it:

```powershell
git clone https://github.com/Bilal-biryani/Ml-hackathon-.git
cd Ml-hackathon-
```

Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the pinned dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Main dependencies include NumPy, Pandas, SciPy, scikit-learn, LightGBM, Joblib, Matplotlib, and Folium.

---

## 5. Running the current pipeline

### Step 1 — Place the raw dataset

Copy the **unchanged college-provided files** into `data/raw/train/` and `data/raw/student_test/` using the folder structure above.

Do not preprocess or rewrite the source files manually.

### Step 2 — Build processed modelling tables

```powershell
python scripts/02_build_dataset.py
```

This creates generated files under `data/processed/`, including:

```text
train_cands.pkl
train_obs.pkl
student_test_cands.pkl
student_test_obs.pkl
```

It also writes the cleaning log to `results/cleaning_log.csv`.

### Step 3 — Run exploratory analysis

```powershell
python scripts/01_eda.py
```

**Current-revision note:** despite its `01_` filename, parts of this EDA script read the processed candidate tables produced by `02_build_dataset.py`. Therefore, in the current repository revision, run `02_build_dataset.py` before `01_eda.py`.

EDA outputs are written to:

```text
results/eda_*.csv
results/eda_summary.json
figures/eda_*.png
maps/eda_*.html
```

### Step 4 — Train and compare all models

```powershell
python scripts/03_train_compare.py
```

The training protocol uses **5-fold GroupKFold**, grouped by trip ID, so candidate rows from the same trip are never split across training and validation folds.

Outputs include:

```text
results/model_comparison.csv
results/cv_fold_scores.csv
results/per_trip_student_test.csv
results/training_histories.json
results/selected_model.json
figures/cmp_*.png
models/*.joblib
```

> **Important:** the current `PATHSNAP_QUICK=1` mode writes to the same output locations as the full experiment. Until the cleanup patch is applied, do not run quick mode if you want to preserve the committed official models/results.

### Optional — Live demonstration

After the processed tables exist:

```powershell
python scripts/demo_live.py
```

The demo performs a small grouped validation demonstration and does not intentionally write model/result artifacts.

---

## 6. Machine-learning formulation

For each GPS observation, the pipeline generates a small set of nearby directed road-arc candidates. Each candidate becomes one binary classification row:

```text
y = 1  -> candidate corresponds to the aligned reference arc
y = 0  -> candidate is an alternative road arc
```

The current feature set is defined centrally in `src/config.py` and includes:

- distance to candidate arc
- distance rank and relative distance features
- local candidate density
- heading agreement
- speed
- arc length and projected position on the arc
- distance to arc end
- neighbouring-fix distance context
- in-degree / out-degree
- two-way-road indicator

The classifier produces a score for every candidate. The highest-scoring arc for each GPS observation is the current classifier-only map-matching prediction.

---

## 7. Models compared

The repository compares seven distinct models:

1. Logistic Regression
2. Gaussian Naive Bayes
3. K-Nearest Neighbours
4. Random Forest
5. Extra Trees
6. LightGBM
7. Multi-Layer Perceptron (MLP)

The committed `results/model_comparison.csv` reports the following main results:

| Model | CV Macro F1 | CV Top-1 | Student Top-1 | Student Macro F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9002 | 0.8627 | 0.8610 | 0.9033 |
| GaussianNB | 0.8393 | 0.8452 | 0.8384 | 0.8324 |
| KNN | 0.9083 | 0.8612 | 0.8617 | 0.9116 |
| Random Forest | 0.9290 | 0.8799 | 0.8852 | 0.9322 |
| Extra Trees | 0.9247 | 0.8739 | 0.8764 | 0.9252 |
| **LightGBM** | **0.9280** | **0.8802** | **0.8913** | **0.9322** |
| MLP | 0.9211 | 0.8747 | 0.8757 | 0.9223 |

`results/selected_model.json` selects **LightGBM** using the repository's stated criterion:

> highest mean cross-validation observation-level Top-1 accuracy, with CV Macro F1 as the tie-breaker.

These values are the committed experiment results in the repository.

---

## 8. Evaluation metrics

The current code reports both candidate-level and observation-level metrics.

### Candidate-level

- Macro F1
- Macro precision
- Macro recall
- Binary candidate accuracy
- Log loss

### Observation / map-matching level

- Top-1 arc accuracy
- Top-1 accuracy given that the reference arc is present among candidates
- Candidate miss rate
- Predicted-arc continuity
- Arc-set overlap precision / recall / F1 over evaluated GPS fixes
- Per-trip evaluation

The current `route_f1` field should be interpreted as **arc-set overlap over evaluated GPS observations**, not as a full ordered reconstructed-route F1. Full route reconstruction will be evaluated after the HMM/Viterbi stage is implemented.

---

## 9. Current architecture

```text
College-provided trip files
        |
        v
Raw file loading / cleaning
        |
        v
Local metric projection
        |
        v
Trip road network + KD-tree spatial index
        |
        v
Nearby directed-arc candidate generation
        |
        v
Feature engineering + reference alignment
        |
        v
Candidate-level modelling tables
        |
        v
7-model GroupKFold comparison
        |
        v
Selected classifier: LightGBM
```

The planned final architecture extends this as:

```text
LightGBM candidate probabilities
        |
        v
Bayesian hyperparameter optimisation
        |
        v
Tuned emission scores
        |
        v
HMM transition model
        |
        v
Viterbi decoding
        |
        v
Road-route reconstruction
        |
        v
FastAPI backend (Laptop 1)
        |
       HTTP
        |
        v
Evaluator UI + Folium (Laptop 2)
```

---

## 10. Reproducibility and data-safety rules

- Never train on evaluator-test data.
- Keep training and student-test partitions separate.
- Keep all rows from a trip together during cross-validation.
- Do not modify the college-provided raw files.
- Do not commit `data/raw/` or `data/processed/`.
- Do not overwrite official experiment artifacts with reduced smoke-test runs.
- Record model configuration, random seed, metrics, and generated artifacts for each official experiment.
- Treat `.route` as ground truth used for training/evaluation only; inference must not depend on the route file.

---

## 11. Key generated artifacts already committed

### Models

```text
models/GaussianNB.joblib
models/LightGBM.joblib
models/LogisticRegression.joblib
models/MLP.joblib
```

### Results

```text
results/model_comparison.csv
results/cv_fold_scores.csv
results/cv_folds.json
results/per_trip_student_test.csv
results/selected_model.json
results/training_histories.json
results/confusion_matrices.json
```

### Figures

```text
figures/cmp_cv_macro_f1.png
figures/cmp_student_test.png
figures/cmp_continuity.png
figures/cmp_confusion_matrices.png
figures/cmp_logloss_time.png
figures/cmp_training_curves.png
figures/eda_*.png
```

### Maps

```text
maps/eda_trip_00000033_partial_route.html
maps/eda_trip_00000046.html
```

---

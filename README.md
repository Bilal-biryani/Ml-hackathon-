# PathSnap: map matching and route reconstruction

PathSnap matches noisy 1 Hz GPS tracks to a directed road network. It runs in three stages:

1. **Candidates and features:** for every GPS fix, the arcs within 50 m (widening to 400 m only if none are found), up to 10 per fix. Each candidate gets 20 geometric, kinematic and topological features.
2. **Candidate classifier:** a supervised model scores each candidate. Seven models were compared, and the best was tuned with Bayesian optimisation (Optuna TPE).
3. **HMM + Viterbi:** decodes the most likely sequence of arcs, using road-network shortest-path distances in the transition model.

A FastAPI backend (Laptop 1) runs the full pipeline. A browser UI (Laptop 2) lets the evaluator pick a trip and shows the predicted route with server-side accuracy.

## Repository layout

| Folder / file | Contents |
|---|---|
| `src/io.py` | Raw-file readers and track cleaning (`.track`, `.nodes`, `.arcs`, `.route`) |
| `src/geo.py` | Local metric projection and point-to-segment geometry |
| `src/network.py` | Road network per trip: KD-tree over arc geometry, routable graph, degrees, two-way arcs |
| `src/labels.py` | Monotone DP alignment of `.route` to GPS fixes, giving per-fix reference arcs |
| `src/features.py` | Candidate generation and feature engineering |
| `src/pipeline.py` | Shared trip preparation (same code in training and in the API) |
| `src/models/` | Model zoo, one file per family: `linear.py` (LogReg, GaussianNB), `neighbors_nn.py` (KNN, MLP), `trees.py` (RandomForest, ExtraTrees), `boosting.py` (LightGBM), `__init__.py` (`model_zoo`) |
| `src/metrics.py` | Macro F1, log loss, Top-1 accuracy, candidate miss rate, route continuity, route F1 |
| `src/hmm.py` | HMM emission and transition model, bounded Dijkstra, Viterbi, route filling |
| `src/inference.py` | `MapMatcher` (prediction only) and server-side `score_against_reference` |
| `src/viz.py` | Folium map builders |
| `scripts/01_eda.py` | EDA tables, plots and Folium maps |
| `scripts/02_build_dataset.py` | Candidate tables for train and student-test |
| `scripts/03_train_compare.py` | 5-fold GroupKFold comparison of 7 models plus student-test evaluation |
| `scripts/04_bayes_opt.py` | Optuna TPE search for the selected model; tuned vs untuned |
| `scripts/05_hmm_eval.py` | Decoder tuning; classifier-only vs classifier+HMM |
| `scripts/05b_hmm_error_analysis.py` | Error categories (reverse twin, adjacent arc, other, missing candidate) before and after HMM |
| `scripts/06_maps.py` | Folium maps of predictions (and PNG snapshots) |
| `scripts/07_build_report.py` | Builds `report/PathSnap_Technical_Report.pdf` from the result files |
| `api/main.py` | FastAPI backend; `api/client.py` is a command-line client |
| `start_api.bat`, `open_firewall.bat` | Laptop 1 launcher (prints the IP, starts the API) and optional firewall rule |
| `tests/` | Unit tests for the metrics (`python -m unittest discover tests`) |
| `ui/index.html` | Evaluator UI (Leaflet, single file) |
| `models/` | `final_model.joblib`, `final_config.json`, `hmm_config.json`, small comparison models |
| `results/` | CSV/JSON experiment results, Optuna trials, logs |
| `figures/`, `maps/` | Plots and interactive Folium HTML maps |

## Installation (both laptops)

Python 3.11 is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

Laptop 2 only needs a web browser to run the UI; Python is optional there and is only used to serve the page.

## Data

The raw data is not committed because of its size. Put it here:

```
data/raw/train/<trip_id>/<trip_id>.{track,nodes,arcs,route}           # 80 trips
data/raw/student_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}    # 10 trips
data/eval/<trip_id>/...                                               # evaluator trips, given on demo day
```

## Reproducing the experiments

```bash
python scripts/run_all.py
```

Or run the stages one by one: `02_build_dataset` → `01_eda` → `03_train_compare` → `04_bayes_opt` → `05_hmm_eval` → `05b_hmm_error_analysis` → `06_maps` → `07_build_report`. Seeds are fixed (`SEED = 42` in `src/config.py`), and the console output of each stage is in `results/logs/`.

## Two-laptop deployment

**Laptop 1 (model + API)**

1. Copy the ten evaluation trip folders into `data/eval/` (an extra wrapping folder such as `data/eval/evaluator_test/` is also found).
2. Start the server with `start_api.bat` (or `start_api.bat rehearsal` to serve the student-test trips). It runs:
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
3. The launcher prints the laptop's IP addresses. Allow Python through the Windows firewall if prompted (or run `open_firewall.bat` as administrator). Every request is logged in this window: client IP, trip, Top-1 accuracy and latency. These are the API logs to show the evaluator.

**Laptop 2 (UI)**

Run `ui/start_ui.bat <laptop-1-ip>`, or:

```bash
cd ui
python -m http.server 5500
```

and open `http://localhost:5500`. Open the page from `localhost`, not from an IP address, because Chrome and Edge block an IP-served page from calling another IP. Enter `http://<laptop-1-ip>:8000`, click **Connect**, then select any trip. The UI shows the nearby road network, the GPS fixes, the predicted route and the metrics computed on the server. Full step-by-step instructions are in `SETUP_GUIDE.md`.

**Testing the API from a terminal**

```bash
curl http://<laptop-1-ip>:8000/trips
curl -X POST http://<laptop-1-ip>:8000/predict -H "Content-Type: application/json" -d "{\"trip_id\": \"00000000\"}"
python api/client.py --api http://<laptop-1-ip>:8000 --trip 00000000
```

### API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | none | model name and params, features, HMM params |
| GET | `/trips` | none | trip ids and fix counts found in `PATHSNAP_TRIPS_DIR` |
| POST | `/predict` | `{"trip_id": "..."}` | GPS points, predicted route polylines, nearby road network, predicted arc per fix, metrics, timings |

The reference route is only read inside `score_against_reference` on the server. Only aggregate metrics are returned, never reference arcs.

## Leakage controls

- **Trip-level splits:** cross-validation uses `GroupKFold` grouped by trip, and the student-test trips are only used for development-stage comparison.
- **No route at prediction time:** `MapMatcher.predict` never opens `.route`. Features use only the track and the road network.
- **Training-data-only choices:** the Optuna search uses only the 80 training trips. HMM parameters are tuned on training trips using out-of-fold probabilities. No evaluator data is used for any choice.
- **No hardcoding:** there are no hardcoded trip ids or outputs. The API lists whatever trip folders are present.

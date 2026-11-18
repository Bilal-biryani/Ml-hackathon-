"""Central configuration: paths and pipeline constants shared by every stage."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
TRAIN_DIR = DATA_RAW / "train"
STUDENT_TEST_DIR = DATA_RAW / "student_test"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
MAPS_DIR = ROOT / "maps"

for _d in (PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR, MAPS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42

# --- road network -----------------------------------------------------------
NETWORK_MARGIN_M = 1000.0      # keep arcs within this distance of the track bounding box
DENSIFY_SPACING_M = 10.0       # sample spacing used to index arc geometry in the KD-tree

# --- candidate generation ---------------------------------------------------
CANDIDATE_RADIUS_M = 50.0      # first search radius around each GPS fix
CANDIDATE_RADIUS_STEPS = (50.0, 100.0, 200.0, 400.0)  # widened if nothing is found
MAX_CANDIDATES = 10            # keep the K closest arcs (twin directions count separately)

# --- feature engineering ----------------------------------------------------
HEADING_HALF_WINDOW = 2        # heading/speed from fix i-2 -> i+2
HEADING_MIN_DISP_M = 3.0       # below this displacement the heading is considered unreliable
CONTEXT_LAGS = (2, 5)          # distances of neighbouring fixes (i-lag, i+lag) to the candidate arc

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio", "n_cand", "cand_density_10m",
    "cos_heading", "abs_heading_diff", "heading_valid", "speed",
    "arc_len", "t_frac", "dist_to_end",
    "d_prev2", "d_next2", "d_prev5", "d_next5",
    "out_degree", "in_degree", "is_two_way",
]

"""Central configuration: paths and pipeline constants shared by every stage."""
import os
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


def _bounded_env_int(name: str, default: int, upper: int) -> int:
    """Read a positive integer environment variable and clamp it to a safe upper bound."""
    raw = os.environ.get(name)
    if raw is None:
        return max(1, min(default, upper))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return min(value, upper)


CPU_COUNT = os.cpu_count() or 1
N_JOBS = _bounded_env_int("PATHSNAP_N_JOBS", CPU_COUNT, CPU_COUNT)
# Candidate generation is memory-heavy, so use a smaller default than model fitting.
BUILD_JOBS = _bounded_env_int("PATHSNAP_BUILD_JOBS", min(4, CPU_COUNT), CPU_COUNT)

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

# --- HMM ----------------------------------------------------------------------
GAP_BREAK_S = 60.0             # time gap that resets the HMM chain (uniform transition)
UNREACHABLE_LOGP = -25.0       # log-transition score for candidate pairs with no network path

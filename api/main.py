"""PathSnap FastAPI backend (runs on Laptop 1).

Endpoints
  GET  /health            model + decoder information
  GET  /trips             evaluation trips available on this server
  POST /predict {trip_id} full pipeline: candidates -> classifier -> HMM/Viterbi -> server-side scoring

The reference route is read only inside score_against_reference(); only summary metrics leave the
server. Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
import logging
import os
import re
import socket
import sys
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.hmm import fill_route
from src.inference import MapMatcher, score_against_reference
from src.io import list_trips
from src.viz import nearby_arcs

MAX_GPS_POINTS = 6000
MAX_ROAD_SEGMENTS = 20000
TRIP_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def find_trips_dir(path):
    """Return the folder that directly contains the trip folders.

    The evaluator may hand over the trips inside an extra folder (for example eval/evaluator_test/<id>/),
    so if `path` holds no trip folders we look up to two levels down and take the folder with the most trips.
    """
    path = Path(path)
    if list_trips(path) or not path.exists():
        return path
    best, best_n = path, 0
    for sub in [p for p in path.rglob("*") if p.is_dir() and len(p.relative_to(path).parts) <= 2]:
        n = len(list_trips(sub))
        if n > best_n:
            best, best_n = sub, n
    return best


TRIPS_DIR = find_trips_dir(os.environ.get("PATHSNAP_TRIPS_DIR", ROOT / "data" / "eval"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
log = logging.getLogger("pathsnap")
STATE = {}


@asynccontextmanager
async def lifespan(app):
    t0 = time.time()
    STATE["matcher"] = MapMatcher()
    m = STATE["matcher"]
    log.info("Loaded model %s with %d features and HMM %s in %.2fs", m.cfg["model"], len(m.features),
             m.hmm_params.to_dict(), time.time() - t0)
    n_trips = len(list_trips(TRIPS_DIR))
    log.info("Serving trips from %s (%d trips)", TRIPS_DIR, n_trips)
    if n_trips == 0:
        log.warning("NO TRIPS FOUND in %s - copy the evaluation trip folders there and restart", TRIPS_DIR)
    for ip in _lan_ips():
        log.info("UI on Laptop 2 should use:  http://%s:8000", ip)
    yield


def _lan_ips():
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


app = FastAPI(title="PathSnap API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    log.info("%s %s from %s -> %d (%.0f ms)", request.method, request.url.path,
             request.client.host if request.client else "?", response.status_code, (time.time() - t0) * 1000)
    return response


class PredictRequest(BaseModel):
    trip_id: str


def _trip_dir(trip_id):
    if not TRIP_ID_RE.match(trip_id):
        raise HTTPException(400, "invalid trip_id")
    d = TRIPS_DIR / trip_id
    if d not in list_trips(TRIPS_DIR):
        raise HTTPException(404, f"trip {trip_id} not found")
    return d


@app.get("/health")
def health():
    m = STATE["matcher"]
    return {"status": "ok", "model": m.cfg["model"], "model_params": m.cfg["params"],
            "features": m.features, "hmm_params": m.hmm_params.to_dict(), "trips_dir": str(TRIPS_DIR)}


@app.get("/trips")
def trips():
    out = []
    for d in list_trips(TRIPS_DIR):
        track = next(d.glob("*.track"))
        with open(track) as f:
            n = sum(1 for line in f if line.strip())
        out.append({"trip_id": d.name, "n_fixes": n})
    return {"count": len(out), "trips": out}


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    d = _trip_dir(req.trip_id)
    client = request.client.host if request.client else "?"
    log.info("PREDICT request trip=%s from %s", req.trip_id, client)
    try:
        return _run_pipeline(req.trip_id, d)
    except HTTPException:
        raise
    except Exception as e:  # keep the server alive and tell the UI what went wrong
        log.error("FAILED trip=%s: %s\n%s", req.trip_id, e, traceback.format_exc())
        raise HTTPException(500, f"pipeline failed for trip {req.trip_id}: {type(e).__name__}: {e}")


def _road_segments(net, xy):
    """Road arcs near the track (map context only; taken from .nodes/.arcs, not from the reference route)."""
    arcs = nearby_arcs(net, xy, 150.0)
    pairs = np.unique(np.sort(net.arcs[arcs], axis=1), axis=0)[:MAX_ROAD_SEGMENTS] if len(arcs) else []
    nl = net.nodes_lonlat
    return [[[float(nl[u, 1]), float(nl[u, 0])], [float(nl[v, 1]), float(nl[v, 0])]] for u, v in pairs]


def _run_pipeline(trip_id, d):
    t0 = time.perf_counter()
    res = STATE["matcher"].predict(d)
    prep, pred_hmm, pred_clf = res["prep"], res["pred_hmm"], res["pred_clf"]
    t1 = time.perf_counter()
    segments, arc_seq = fill_route(prep.net, pred_hmm)
    nl = prep.net.nodes_lonlat
    route = [[[float(nl[n, 1]), float(nl[n, 0])] for n in seg] for seg in segments]
    step = max(1, len(prep.xy) // MAX_GPS_POINTS)
    gps = [[float(p[1]), float(p[0])] for p in prep.inp.track[::step, :2]]
    roads = _road_segments(prep.net, prep.xy)
    t2 = time.perf_counter()
    metrics = score_against_reference(d, prep, {"classifier_only": pred_clf, "hmm_viterbi": pred_hmm})
    total = time.perf_counter() - t0
    if metrics:
        log.info("RESULT trip=%s fixes=%d | Top-1 classifier-only=%.4f HMM=%.4f | continuity HMM=%.4f | "
                 "route F1 HMM=%.4f | %.2fs", trip_id, len(prep.xy), metrics["classifier_only"]["top1_accuracy"] or 0,
                 metrics["hmm_viterbi"]["top1_accuracy"] or 0, metrics["hmm_viterbi"]["route_continuity"],
                 metrics["hmm_viterbi"]["route_f1"], total)
    else:
        log.warning("RESULT trip=%s fixes=%d | no usable reference route on the server, metrics not computed | %.2fs",
                    trip_id, len(prep.xy), total)
    return {
        "trip_id": trip_id,
        "n_fixes": int(len(prep.xy)),
        "n_fixes_raw": int(prep.inp.raw_track_len),
        "n_candidate_rows": int(len(res["cands"])),
        "fixes_without_candidates": int((pred_hmm < 0).sum()),
        "gps": gps,
        "predicted_route": route,
        "road_network": roads,
        "predicted_arc_sequence": [int(a) for a in arc_seq],
        "predicted_arc_per_fix": [int(a) for a in pred_hmm],
        "metrics": metrics,
        "timings_s": {**{k: round(v, 3) for k, v in res["timings"].items()},
                      "route_build_s": round(t2 - t1, 3), "scoring_s": round(time.perf_counter() - t2, 3),
                      "total_s": round(total, 3)},
        "model": STATE["matcher"].cfg["model"],
    }

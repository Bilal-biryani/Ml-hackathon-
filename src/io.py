"""Raw-file readers and track cleaning.

The track/network readers never touch the .route file: the reference route is loaded by a
separate function that only training-label construction and server-side evaluation call.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TripInput:
    trip_id: str
    track: np.ndarray        # (n, 3) lon, lat, t  (cleaned)
    nodes: np.ndarray        # (m, 2) lon, lat
    arcs: np.ndarray         # (a, 2) from_node, to_node
    raw_track_len: int
    cleaning_log: dict


def _file(trip_dir: Path, ext: str) -> Path:
    trip_dir = Path(trip_dir)
    exact = trip_dir / f"{trip_dir.name}.{ext}"
    if exact.exists():
        return exact
    found = sorted(trip_dir.glob(f"*.{ext}"))
    if not found:
        raise FileNotFoundError(f"No .{ext} file in {trip_dir}")
    return found[0]


def read_track_raw(trip_dir) -> np.ndarray:
    return np.loadtxt(_file(trip_dir, "track"), dtype=float, ndmin=2)


def read_nodes(trip_dir) -> np.ndarray:
    return np.loadtxt(_file(trip_dir, "nodes"), dtype=float, ndmin=2)


def read_arcs(trip_dir) -> np.ndarray:
    return np.loadtxt(_file(trip_dir, "arcs"), dtype=np.int64, ndmin=2)


def read_route(trip_dir):
    """Reference arc sequence. Only for label construction / server-side scoring."""
    try:
        path = _file(trip_dir, "route")
    except FileNotFoundError:
        return None
    return np.loadtxt(path, dtype=np.int64, ndmin=1)


def clean_track(track: np.ndarray, nodes: np.ndarray):
    """Remove fixes that cannot be used and order the rest in time.

    Rules: non-finite values, coordinates outside valid lon/lat ranges, fixes lying far outside the
    road-network extent (> 0.05 deg ~ 5 km), and duplicate timestamps (first kept).
    """
    log = {"raw": int(len(track))}
    track = track[:, :3]
    ok = np.isfinite(track).all(axis=1)
    log["non_finite"] = int((~ok).sum())
    rng = (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90)
    log["out_of_range"] = int((ok & ~rng).sum())
    ok &= rng
    lo, hi = nodes.min(axis=0) - 0.05, nodes.max(axis=0) + 0.05
    inside = (track[:, 0] >= lo[0]) & (track[:, 0] <= hi[0]) & (track[:, 1] >= lo[1]) & (track[:, 1] <= hi[1])
    log["outside_network"] = int((ok & ~inside).sum())
    ok &= inside
    track = track[ok]
    order = np.argsort(track[:, 2], kind="stable")
    log["reordered"] = int((order != np.arange(len(order))).sum())
    track = track[order]
    keep = np.r_[True, np.diff(track[:, 2]) > 0]
    log["duplicate_time"] = int((~keep).sum())
    track = track[keep]
    log["kept"] = int(len(track))
    return track, log


def load_trip_input(trip_dir) -> TripInput:
    trip_dir = Path(trip_dir)
    raw = read_track_raw(trip_dir)
    nodes = read_nodes(trip_dir)
    arcs = read_arcs(trip_dir)
    track, log = clean_track(raw, nodes)
    return TripInput(trip_dir.name, track, nodes, arcs, len(raw), log)


def list_trips(part_dir):
    part_dir = Path(part_dir)
    if not part_dir.exists():
        return []
    return sorted(p for p in part_dir.iterdir() if p.is_dir() and any(p.glob("*.track")))

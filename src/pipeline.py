"""Shared trip preparation used by training scripts and the API (identical preprocessing)."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import candidate_table
from .io import TripInput, load_trip_input, read_route
from .labels import align_route
from .network import RoadNetwork


@dataclass
class PreparedTrip:
    inp: TripInput
    net: RoadNetwork
    xy: np.ndarray
    t: np.ndarray
    cands: pd.DataFrame


def prepare_trip(trip_dir) -> PreparedTrip:
    inp = load_trip_input(trip_dir)
    net = RoadNetwork(inp.nodes, inp.arcs, inp.track[:, :2])
    xy = net.to_xy(inp.track[:, :2])
    t = inp.track[:, 2]
    cands = candidate_table(net, xy, t, trip_id=inp.trip_id)
    return PreparedTrip(inp, net, xy, t, cands)


def reference_labels(trip_dir, prep: PreparedTrip):
    """Server-side / training-only: per-fix reference arc from the .route file."""
    route = read_route(trip_dir)
    if route is None:
        return None
    idx, lab, d = align_route(prep.net, prep.xy, route)
    return pd.DataFrame({"obs": np.arange(len(prep.xy)), "label_arc": lab, "route_idx": idx, "align_dist": d}), route

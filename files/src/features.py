"""Candidate generation + feature engineering (uses only the GPS track and the road network)."""
import numpy as np
import pandas as pd

from . import config


def gps_kinematics(xy, t, half=config.HEADING_HALF_WINDOW, min_disp=config.HEADING_MIN_DISP_M):
    """Heading unit vector, validity flag and speed from a centred window of fixes."""
    n = len(xy)
    i0 = np.clip(np.arange(n) - half, 0, n - 1)
    i1 = np.clip(np.arange(n) + half, 0, n - 1)
    disp = xy[i1] - xy[i0]
    dist = np.linalg.norm(disp, axis=1)
    dt = np.maximum(t[i1] - t[i0], 1e-9)
    valid = dist >= min_disp
    unit = disp / np.where(dist > 0, dist, 1.0)[:, None]
    unit[~valid] = 0.0
    speed = np.where(n > 1, dist / dt, 0.0)
    return unit, valid, speed


def candidate_table(net, xy, t, trip_id=""):
    """One row per (GPS fix, candidate arc) with the engineered features in config.FEATURES."""
    c = net.candidates(xy)
    obs, arc, dist, tt = c["obs"], c["arc"], c["dist"], c["t"]
    df = pd.DataFrame({"trip_id": trip_id, "obs": obs, "arc": arc, "dist": dist, "t_frac": tt,
                       "dist_rank": c["rank"]})
    if len(df) == 0:
        return df.assign(**{f: [] for f in config.FEATURES if f not in df})

    g = df.groupby("obs")["dist"]
    dmin = g.transform("min").to_numpy()
    df["n_cand"] = g.transform("size").to_numpy()
    df["dist_minus_min"] = dist - dmin
    df["dist_ratio"] = dist / (dmin + 1.0)
    df["cand_density_10m"] = (df["dist_minus_min"] <= 10.0).groupby(df["obs"]).transform("sum").to_numpy()

    unit, valid, speed = gps_kinematics(xy, t)
    au = net.arc_unit[arc]
    cosv = np.einsum("ij,ij->i", unit[obs], au)
    df["heading_valid"] = valid[obs].astype(np.int8)
    df["cos_heading"] = np.where(valid[obs], cosv, 0.0)
    df["abs_heading_diff"] = np.where(valid[obs], np.degrees(np.arccos(np.clip(cosv, -1, 1))), 90.0)
    df["speed"] = speed[obs]

    alen = net.arc_len[arc]
    df["arc_len"] = alen
    df["dist_to_end"] = np.minimum(tt, 1 - tt) * alen

    n = len(xy)
    for lag in config.CONTEXT_LAGS:
        dp, _ = net.distance_to_arcs(xy[np.clip(obs - lag, 0, n - 1)], arc)
        dn, _ = net.distance_to_arcs(xy[np.clip(obs + lag, 0, n - 1)], arc)
        df[f"d_prev{lag}"] = dp
        df[f"d_next{lag}"] = dn

    df["out_degree"] = net.out_degree[net.arcs[arc, 1]]
    df["in_degree"] = net.in_degree[net.arcs[arc, 0]]
    df["is_two_way"] = net.is_two_way[arc].astype(np.int8)
    return df

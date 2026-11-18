"""Exploratory data analysis on the raw files (train + student-test partitions).

Outputs: results/eda_inventory.csv, results/eda_missing_values.csv, results/eda_summary.json,
figures/eda_*.png, maps/eda_trip_<id>.html
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from src import config, viz
from src.geo import LocalProjection
from src.io import list_trips, read_arcs, read_nodes, read_route, read_track_raw
from src.network import RoadNetwork

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3})
FIG = config.FIGURES_DIR


def inventory():
    rows, miss = [], []
    for part, d in (("train", config.TRAIN_DIR), ("student_test", config.STUDENT_TEST_DIR)):
        for td in list_trips(d):
            tid = td.name
            tr, no, ar, ro = read_track_raw(td), read_nodes(td), read_arcs(td), read_route(td)
            for name, arr in (("track", tr), ("nodes", no), ("arcs", ar), ("route", ro)):
                a = np.asarray(arr, dtype=float).reshape(len(arr), -1)
                miss.append({"partition": part, "trip_id": tid, "file": name, "rows": len(a), "cols": a.shape[1],
                             "missing_values": int(np.isnan(a).sum()), "dtype": str(np.asarray(arr).dtype)})
            dt = np.diff(tr[:, 2])
            proj = LocalProjection(tr[:, 0].mean(), tr[:, 1].mean())
            xy = proj.forward(tr[:, 0], tr[:, 1])
            step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            n = len(no)
            g = csr_matrix((np.ones(len(ar)), (ar[:, 0], ar[:, 1])), shape=(n, n))
            nw, _ = connected_components(g, directed=True, connection="weak")
            ns, lab_s = connected_components(g, directed=True, connection="strong")
            deg = np.bincount(ar.ravel(), minlength=n)
            route_conn = np.mean(ar[ro[:-1], 1] == ar[ro[1:], 0]) if len(ro) > 1 else np.nan
            rows.append({
                "partition": part, "trip_id": tid, "n_fixes": len(tr), "duration_s": tr[-1, 2] - tr[0, 2],
                "dt_eq_1s": np.mean(dt == 1), "dt_max_s": dt.max(), "gaps_gt_10s": int((dt > 10).sum()),
                "missing_1hz_fixes": int(np.sum(np.maximum(dt - 1, 0))),
                "non_monotonic_t": int((dt <= 0).sum()),
                "path_len_km": step.sum() / 1000, "max_step_m": step.max(),
                "extent_x_km": np.ptp(xy[:, 0]) / 1000, "extent_y_km": np.ptp(xy[:, 1]) / 1000,
                "n_nodes": n, "n_arcs": len(ar), "isolated_nodes": int((deg == 0).sum()),
                "weak_components": nw, "strong_components": ns,
                "largest_scc_frac": np.bincount(lab_s).max() / n,
                "route_len": len(ro), "route_unique": len(np.unique(ro)), "route_connected_frac": route_conn,
                "kb_track": (td / f"{tid}.track").stat().st_size / 1024, "kb_nodes": (td / f"{tid}.nodes").stat().st_size / 1024,
                "kb_arcs": (td / f"{tid}.arcs").stat().st_size / 1024, "kb_route": (td / f"{tid}.route").stat().st_size / 1024,
            })
    inv = pd.DataFrame(rows)
    inv.to_csv(config.RESULTS_DIR / "eda_inventory.csv", index=False)
    mv = pd.DataFrame(miss)
    mv.groupby(["partition", "file"]).agg(trips=("trip_id", "nunique"), rows=("rows", "sum"), cols=("cols", "max"),
                                            missing_values=("missing_values", "sum")).reset_index() \
        .to_csv(config.RESULTS_DIR / "eda_missing_values.csv", index=False)
    return inv


def plots(inv):
    tr = inv[inv.partition == "train"]
    obs_path = config.PROCESSED_DIR / "train_obs.pkl"
    cands_path = config.PROCESSED_DIR / "train_cands.pkl"
    have_processed = obs_path.exists() and cands_path.exists()
    obs = pd.read_pickle(obs_path) if have_processed else None
    cands = pd.read_pickle(cands_path) if have_processed else None
    if not have_processed:
        print(
            "Processed candidate tables are not present. Running raw-data EDA only; "
            "candidate/feature diagnostics will be available after scripts/02_build_dataset.py."
        )

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(inv.n_fixes, bins=30, color="#3366cc"); ax[0].set_title("GPS fixes per trip (train+student-test)"); ax[0].set_xlabel("fixes")
    dts = np.concatenate([np.diff(read_track_raw(config.TRAIN_DIR / t)[:, 2]) for t in tr.trip_id])
    bins = [0.5, 1.5, 2.5, 5.5, 10.5, 30.5, 60.5, 300.5, 1e6]
    cnt, _ = np.histogram(dts, bins=bins)
    ax[1].bar(["1", "2", "3-5", "6-10", "11-30", "31-60", "61-300", ">300"], cnt, color="#dc3912"); ax[1].set_yscale("log")
    ax[1].set_title("Sampling interval (train)"); ax[1].set_xlabel("seconds between fixes")
    ax[2].scatter(tr.n_arcs, tr.n_nodes, s=12, color="#109618"); ax[2].set_xscale("log"); ax[2].set_yscale("log")
    ax[2].set_title("Road network size per trip (train)"); ax[2].set_xlabel("arcs"); ax[2].set_ylabel("nodes")
    fig.tight_layout(); fig.savefig(FIG / "eda_inventory.png"); plt.close(fig)

    # missing / gap analysis
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    s = inv.sort_values("missing_1hz_fixes", ascending=False).head(15)
    ax[0].barh(s.trip_id, s.missing_1hz_fixes / (s.duration_s + 1) * 100, color="#ff9900"); ax[0].invert_yaxis()
    ax[0].set_xlabel("% of expected 1 Hz fixes missing"); ax[0].set_title("Implied missing fixes (top 15 trips, both partitions)")
    if have_processed:
        far = obs.groupby("trip_id").align_dist.apply(lambda s: (s > 50).mean() * 100).sort_values(ascending=False).head(15)
        ax[1].barh(far.index, far.values, color="#990099"); ax[1].invert_yaxis()
        ax[1].set_xlabel("% fixes > 50 m from reference route"); ax[1].set_title("Reference-route coverage problems (train)")
    else:
        ax[1].axis("off")
        ax[1].text(0.5, 0.5, "Candidate diagnostics require\n02_build_dataset.py", ha="center", va="center")
    fig.tight_layout(); fig.savefig(FIG / "eda_missing_gaps.png"); plt.close(fig)

    # where the trips are: one point per trip (both partitions) -- the trips are spread worldwide
    locs = []
    for part, d in (("train", config.TRAIN_DIR), ("student_test", config.STUDENT_TEST_DIR)):
        for t in inv[inv.partition == part].trip_id:
            a = read_track_raw(d / t)
            locs.append((part, t, a[:, 0].mean(), a[:, 1].mean(), len(a)))
    locs = pd.DataFrame(locs, columns=["partition", "trip_id", "lon", "lat", "n_fixes"])
    locs.to_csv(config.RESULTS_DIR / "eda_trip_locations.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for part, col in (("train", "#3366cc"), ("student_test", "#dc3912")):
        s = locs[locs.partition == part]
        ax.scatter(s.lon, s.lat, s=18, color=col, label=f"{part} ({len(s)} trips)", alpha=0.8)
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 75); ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title("Trip locations (mean of each GPS track): the 90 trips are spread over 5 continents"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "eda_trip_locations.png"); plt.close(fig)

    # GPS trajectory shapes: 16 train trips, each in its own local metric frame
    sample = tr.sort_values("n_fixes").trip_id.to_numpy()[np.linspace(0, len(tr) - 1, 16).astype(int)]
    fig, axs = plt.subplots(4, 4, figsize=(13, 13))
    for ax, t in zip(axs.ravel(), sample):
        a = read_track_raw(config.TRAIN_DIR / t)
        p = LocalProjection(a[:, 0].mean(), a[:, 1].mean()).forward(a[:, 0], a[:, 1]) / 1000
        ax.plot(p[:, 0], p[:, 1], lw=0.8, color="#1f4fd1")
        ax.scatter(p[0, 0], p[0, 1], color="green", s=15, zorder=3); ax.scatter(p[-1, 0], p[-1, 1], color="red", s=15, zorder=3)
        ax.set_aspect("equal"); ax.set_title(f"{t}  ({len(a)} fixes)", fontsize=9); ax.tick_params(labelsize=7)
    fig.suptitle("GPS trajectories of 16 train trips (km, local projection; green = start, red = end)")
    fig.tight_layout(); fig.savefig(FIG / "eda_all_trajectories.png"); plt.close(fig)

    # road network + GPS + reference route for one trip
    tid = "00000046"
    td = config.TRAIN_DIR / tid
    no, ar, trk, ro = read_nodes(td), read_arcs(td), read_track_raw(td), read_route(td)
    net = RoadNetwork(no, ar, trk[:, :2])
    xy = net.to_xy(trk[:, :2])
    near = viz.nearby_arcs(net, xy, 400)
    fig, ax = plt.subplots(figsize=(9, 8))
    for u, v in ar[near]:
        ax.plot(net.node_xy[[u, v], 0], net.node_xy[[u, v], 1], color="#bbbbbb", lw=0.6)
    for a in ro:
        u, v = ar[a]
        ax.plot(net.node_xy[[u, v], 0], net.node_xy[[u, v], 1], color="green", lw=2.5)
    ax.scatter(xy[:, 0], xy[:, 1], s=3, color="#1f4fd1", zorder=3)
    ax.set_aspect("equal"); ax.set_xlabel("x (m, local projection)"); ax.set_ylabel("y (m)")
    ax.set_title(f"Train trip {tid}: road network (grey), reference route (green), GPS (blue)")
    fig.tight_layout(); fig.savefig(FIG / "eda_network_trip.png"); plt.close(fig)

    # zoomed view to show GPS noise vs road
    fig, ax = plt.subplots(figsize=(7, 6))
    c = xy[len(xy) // 2]
    win = 150
    for u, v in ar[near]:
        p = net.node_xy[[u, v]]
        ax.plot(p[:, 0], p[:, 1], color="#999999", lw=1)
    for a in ro:
        u, v = ar[a]
        ax.plot(net.node_xy[[u, v], 0], net.node_xy[[u, v], 1], color="green", lw=3)
    ax.scatter(xy[:, 0], xy[:, 1], s=10, color="#1f4fd1", zorder=3)
    ax.set_xlim(c[0] - win, c[0] + win); ax.set_ylim(c[1] - win, c[1] + win); ax.set_aspect("equal")
    ax.set_title(f"Zoom (300 m window) on train trip {tid}")
    fig.tight_layout(); fig.savefig(FIG / "eda_zoom_noise.png"); plt.close(fig)

    # label / candidate diagnostics (requires processed modelling tables)
    if have_processed:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].hist(np.clip(obs.align_dist, 0, 100), bins=50, color="#3366cc"); ax[0].axvline(50, color="r", ls="--")
        ax[0].set_title("Fix-to-reference-arc distance (train)"); ax[0].set_xlabel("m (clipped at 100)")
        ax[1].hist(obs.n_cand, bins=np.arange(0, 12) - 0.5, color="#dc3912"); ax[1].set_title("Candidates per fix (train)")
        pos_rank = cands[cands.y == 1].dist_rank.value_counts().sort_index()
        ax[2].bar(pos_rank.index, pos_rank.values / pos_rank.sum() * 100, color="#109618")
        ax[2].set_title("Distance rank of the true arc (train)"); ax[2].set_xlabel("rank (0 = nearest)"); ax[2].set_ylabel("%")
        fig.tight_layout(); fig.savefig(FIG / "eda_candidates.png"); plt.close(fig)

        # feature distributions true vs other candidates (reliable labels only)
        rel = cands.merge(obs[["trip_id", "obs", "align_dist"]], on=["trip_id", "obs"])
        rel = rel[rel.align_dist <= 50].sample(min(200000, len(rel)), random_state=config.SEED)
        feats = ["dist", "dist_minus_min", "abs_heading_diff", "d_next5", "speed", "dist_to_end"]
        fig, ax = plt.subplots(2, 3, figsize=(15, 8))
        for a, f in zip(ax.ravel(), feats):
            hi = np.percentile(rel[f], 99)
            for y, col, lab in ((0, "#999999", "other candidates"), (1, "#109618", "reference arc")):
                a.hist(np.clip(rel.loc[rel.y == y, f], None, hi), bins=50, alpha=0.6, density=True, color=col, label=lab)
            a.set_title(f); a.legend()
        fig.suptitle("Feature distributions: reference arc vs other candidates (train, reliable labels)")
        fig.tight_layout(); fig.savefig(FIG / "eda_feature_distributions.png"); plt.close(fig)

        corr = rel[config.FEATURES + ["y"]].corr()
        fig, ax = plt.subplots(figsize=(11, 9))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
        fig.colorbar(im); ax.set_title("Feature correlation (train sample)")
        fig.tight_layout(); fig.savefig(FIG / "eda_feature_corr.png"); plt.close(fig)


    # folium map for the EDA trip
    m = viz.base_map(trk)
    viz.add_network(m, net, xy)
    viz.add_route(m, net, ro, "Reference route (.route)", "#109618", weight=6)
    viz.add_gps(m, trk)
    viz.finish(m, config.MAPS_DIR / f"eda_trip_{tid}.html",
               [("GPS observations", "#1f4fd1"), ("Reference route", "#109618"), ("Road network", "#7f7f7f")],
               title=f"Train trip {tid}")

    # second map: a trip whose reference route does not cover the track
    tid2 = "00000033"
    td2 = config.TRAIN_DIR / tid2
    no2, ar2, trk2, ro2 = read_nodes(td2), read_arcs(td2), read_track_raw(td2), read_route(td2)
    net2 = RoadNetwork(no2, ar2, trk2[:, :2])
    m = viz.base_map(trk2)
    viz.add_network(m, net2, net2.to_xy(trk2[:, :2]))
    viz.add_route(m, net2, ro2, "Reference route (.route)", "#109618", weight=6)
    viz.add_gps(m, trk2)
    viz.finish(m, config.MAPS_DIR / f"eda_trip_{tid2}_partial_route.html",
               [("GPS observations", "#1f4fd1"), ("Reference route", "#109618"), ("Road network", "#7f7f7f")],
               title=f"Train trip {tid2}: partial reference route")

    # folium overview of all trip locations
    import folium
    m = folium.Map(location=[30, 10], zoom_start=2, control_scale=True)
    for part, col in (("train", "blue"), ("student_test", "red")):
        fg = folium.FeatureGroup(name=f"{part} trips")
        for r in locs[locs.partition == part].itertuples():
            folium.CircleMarker([r.lat, r.lon], radius=5, color=col, fill=True, fill_opacity=0.8,
                                tooltip=f"{part} trip {r.trip_id}: {r.n_fixes} fixes").add_to(fg)
        fg.add_to(m)
    viz.finish(m, config.MAPS_DIR / "eda_trip_locations.html", [("train trips", "blue"), ("student-test trips", "red")],
               title="Trip locations (both partitions)")
    region = np.select([locs.lon < -30, locs.lon > 60], ["Americas", "Asia/Oceania"], "Europe/Africa")

    summary = {
        "trips_by_region": pd.Series(region).value_counts().to_dict(),
        "max_trip_extent_km": float(max(inv.extent_x_km.max(), inv.extent_y_km.max())),
        "trips": inv.groupby("partition").size().to_dict(),
        "fixes": inv.groupby("partition").n_fixes.sum().to_dict(),
        "fixes_per_trip": inv.n_fixes.describe().round(1).to_dict(),
        "dt_eq_1s_overall": float(np.mean(dts == 1)),
        "max_gap_s": float(inv.dt_max_s.max()),
        "trips_with_gap_gt_60s": int((inv.dt_max_s > 60).sum()),
        "non_monotonic_timestamps": int(inv.non_monotonic_t.sum()),
        "nodes_per_trip": inv.n_nodes.describe().round(0).to_dict(),
        "arcs_per_trip": inv.n_arcs.describe().round(0).to_dict(),
        "isolated_nodes_total": int(inv.isolated_nodes.sum()),
        "weak_components_median": float(inv.weak_components.median()),
        "largest_scc_frac_median": float(inv.largest_scc_frac.median()),
        "route_connected_frac_mean": float(inv.route_connected_frac.mean()),
    }
    if have_processed:
        summary.update({
            "train_align_gt50_frac": float((obs.align_dist > 50).mean()),
            "train_candidate_miss_rate_all": float(1 - obs.label_in_cands.mean()),
            "train_candidate_miss_rate_reliable": float(1 - obs[obs.align_dist <= 50].label_in_cands.mean()),
            "true_arc_is_nearest_frac": float((cands[cands.y == 1].dist_rank == 0).mean()),
            "two_way_arc_frac": float(cands.is_two_way.mean()),
        })
    (config.RESULTS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    inv = inventory()
    plots(inv)

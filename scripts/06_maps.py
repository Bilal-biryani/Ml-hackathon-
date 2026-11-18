"""Folium maps of the final pipeline on the student-testing trips (+ PNG snapshots for the report).

Layers: GPS observations, nearby road network, reference route (where available),
classifier-only Top-1 arcs, HMM/Viterbi predicted route, mismatched fixes.
Outputs: maps/student_test_<id>.html, figures/map_<id>.png (via headless Edge/Chrome if found)
"""
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import folium
import numpy as np

from src import config, viz
from src.hmm import fill_route
from src.inference import MapMatcher
from src.io import list_trips, read_route
from src.pipeline import reference_labels

BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe", "msedge", "google-chrome", "chromium"]


def trip_map(matcher, trip_dir):
    res = matcher.predict(trip_dir)
    prep, ph, pc = res["prep"], res["pred_hmm"], res["pred_clf"]
    net, trk = prep.net, prep.inp.track
    m = viz.base_map(trk)
    viz.add_network(m, net, prep.xy)
    viz.add_gps(m, trk)               # drawn first so the routes stay visible on top
    route = read_route(trip_dir)
    if route is not None:
        viz.add_route(m, net, route, "Reference route (.route)", "#109618", weight=9)
    viz.add_route(m, net, pc, "Classifier-only Top-1 arcs", "#ff9900", weight=3, dash="6 6", show=False)
    segs, _ = fill_route(net, ph)
    fg = folium.FeatureGroup(name="Predicted route (classifier + HMM)")
    nl = net.nodes_lonlat
    folium.PolyLine([[[nl[n, 1], nl[n, 0]] for n in s] for s in segs], color="#d1361f", weight=4, opacity=0.9).add_to(fg)
    fg.add_to(m)
    acc = None
    ref = reference_labels(trip_dir, prep)
    if ref is not None:
        lab = ref[0]
        ev = lab.align_dist.to_numpy() <= 50
        wrong = np.nonzero(ev & (ph != lab.label_arc.to_numpy()))[0]
        acc = ((ph == lab.label_arc.to_numpy())[ev].mean(), (pc == lab.label_arc.to_numpy())[ev].mean())
        fw = folium.FeatureGroup(name=f"Mismatched fixes (HMM) [{len(wrong)}]", show=False)
        for i in wrong[:: max(1, len(wrong) // 1500)]:
            folium.CircleMarker([trk[i, 1], trk[i, 0]], radius=4, color="#7b1fa2", fill=True, weight=1,
                                tooltip=f"fix {i}").add_to(fw)
        fw.add_to(m)
    title = f"Student-test trip {prep.inp.trip_id}"
    if acc:
        title += f"<br>Top-1: HMM {acc[0]:.3f} | classifier {acc[1]:.3f}"
    path = config.MAPS_DIR / f"student_test_{prep.inp.trip_id}.html"
    viz.finish(m, path, [("GPS observations", "#1f4fd1"), ("Reference route", "#109618"),
                         ("Predicted route (HMM)", "#d1361f"), ("Classifier-only arcs", "#ff9900"),
                         ("Mismatched fixes", "#7b1fa2"), ("Road network", "#7f7f7f")], title=title)
    return path, acc


def snapshot(html, png, size="1400,900"):
    exe = next((b for b in BROWSERS if Path(b).exists() or shutil.which(b)), None)
    if exe is None:
        print("no headless browser found; skipping", png)
        return
    subprocess.run([exe, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={size}",
                    "--virtual-time-budget=15000", f"--screenshot={png}", Path(html).resolve().as_uri()],
                   check=False, timeout=120, capture_output=True)
    print("snapshot", png, Path(png).exists())


def main():
    matcher = MapMatcher()
    out = []
    for d in list_trips(config.STUDENT_TEST_DIR):
        p, acc = trip_map(matcher, d)
        out.append((d.name, p, acc))
        print(d.name, p.name, acc)
    # snapshots: best, median and worst HMM trip
    ranked = sorted(out, key=lambda r: r[2][0])
    for tid, p, _ in (ranked[0], ranked[len(ranked) // 2], ranked[-1]):
        snapshot(p, config.FIGURES_DIR / f"map_{tid}.png")
    eda = config.MAPS_DIR / "eda_trip_00000046.html"
    if eda.exists():
        snapshot(eda, config.FIGURES_DIR / "map_eda_00000046.png")


if __name__ == "__main__":
    main()

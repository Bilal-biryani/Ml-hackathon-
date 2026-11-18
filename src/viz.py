"""Folium map builders (GPS fixes, road network, reference and predicted routes)."""
import folium
import numpy as np


def arc_chains(net, arc_seq):
    """Turn an arc sequence into [lat, lon] polylines, breaking where arcs are not connected."""
    chains, cur, prev = [], [], None
    for a in arc_seq:
        if a is None or a < 0:
            continue
        a = int(a)
        if prev is not None and a == prev:
            continue
        u, v = net.arcs[a]
        if prev is not None and net.arcs[prev, 1] == u and cur:
            cur.append([net.nodes_lonlat[v, 1], net.nodes_lonlat[v, 0]])
        else:
            if len(cur) > 1:
                chains.append(cur)
            cur = [[net.nodes_lonlat[u, 1], net.nodes_lonlat[u, 0]], [net.nodes_lonlat[v, 1], net.nodes_lonlat[v, 0]]]
        prev = a
    if len(cur) > 1:
        chains.append(cur)
    return chains


def nearby_arcs(net, xy, buffer_m=150.0):
    lists = net.tree.query_ball_point(xy[:: max(1, len(xy) // 4000)], r=buffer_m)
    idx = np.unique(np.concatenate([np.asarray(x, dtype=np.int64) for x in lists if len(x)] or [np.zeros(0, np.int64)]))
    return np.unique(net.sample_arc[idx])


def base_map(track_lonlat):
    lat, lon = track_lonlat[:, 1], track_lonlat[:, 0]
    m = folium.Map(location=[float(lat.mean()), float(lon.mean())], zoom_start=14, tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]])
    return m


def add_network(m, net, xy, buffer_m=150.0, name="Road network (within 150 m)"):
    arcs = nearby_arcs(net, xy, buffer_m)
    lines = [[[net.nodes_lonlat[u, 1], net.nodes_lonlat[u, 0]], [net.nodes_lonlat[v, 1], net.nodes_lonlat[v, 0]]]
             for u, v in net.arcs[arcs]]
    fg = folium.FeatureGroup(name=name, show=True)
    if lines:
        folium.PolyLine(lines, color="#7f7f7f", weight=1.5, opacity=0.6).add_to(fg)
    fg.add_to(m)


def add_gps(m, track_lonlat, name="GPS observations", color="#1f4fd1", max_points=3000, show=True):
    step = max(1, len(track_lonlat) // max_points)
    fg = folium.FeatureGroup(name=name, show=show)
    for i in range(0, len(track_lonlat), step):
        lon, lat, t = track_lonlat[i, 0], track_lonlat[i, 1], track_lonlat[i, 2] if track_lonlat.shape[1] > 2 else i
        folium.CircleMarker([lat, lon], radius=2.5, color=color, fill=True, fill_opacity=0.9, weight=0,
                            tooltip=f"fix {i}  t={t:.0f}s").add_to(fg)
    fg.add_to(m)
    start, end = track_lonlat[0], track_lonlat[-1]
    folium.Marker([start[1], start[0]], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(fg)
    folium.Marker([end[1], end[0]], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(fg)


def add_route(m, net, arc_seq, name, color, weight=5, dash=None, show=True):
    fg = folium.FeatureGroup(name=name, show=show)
    chains = arc_chains(net, arc_seq)
    if chains:
        folium.PolyLine(chains, color=color, weight=weight, opacity=0.85, dash_array=dash, tooltip=name).add_to(fg)
    fg.add_to(m)


def add_legend(m, items, title="Legend"):
    rows = "".join(
        f'<div><span style="display:inline-block;width:18px;height:4px;background:{c};margin-right:6px;vertical-align:middle"></span>{n}</div>'
        for n, c in items)
    html = (f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 10px;'
            f'border:1px solid #999;border-radius:6px;font:12px sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.3)">'
            f'<b>{title}</b>{rows}</div>')
    m.get_root().html.add_child(folium.Element(html))


def finish(m, path, legend_items, title="Legend"):
    add_legend(m, legend_items, title)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(path))
    return path

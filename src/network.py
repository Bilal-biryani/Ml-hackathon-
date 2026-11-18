"""Per-trip road network: projection, spatial index over arc geometry and a routable graph."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from . import config
from .geo import LocalProjection, point_to_segment


class RoadNetwork:
    def __init__(self, nodes_lonlat, arcs, track_lonlat,
                 margin_m=config.NETWORK_MARGIN_M, spacing_m=config.DENSIFY_SPACING_M):
        self.nodes_lonlat = np.asarray(nodes_lonlat, dtype=float)
        self.arcs = np.asarray(arcs, dtype=np.int64)
        n_nodes = len(self.nodes_lonlat)
        track_lonlat = np.asarray(track_lonlat, dtype=float)
        self.proj = LocalProjection(track_lonlat[:, 0].mean(), track_lonlat[:, 1].mean())
        self.node_xy = self.proj.forward(self.nodes_lonlat[:, 0], self.nodes_lonlat[:, 1])

        a = self.node_xy[self.arcs[:, 0]]
        b = self.node_xy[self.arcs[:, 1]]
        self.arc_len = np.linalg.norm(b - a, axis=1)
        d = b - a
        self.arc_unit = d / np.where(self.arc_len > 0, self.arc_len, 1.0)[:, None]

        # topology (full network)
        self.out_degree = np.bincount(self.arcs[:, 0], minlength=n_nodes)
        self.in_degree = np.bincount(self.arcs[:, 1], minlength=n_nodes)
        key = self.arcs[:, 0] * n_nodes + self.arcs[:, 1]
        rkey = self.arcs[:, 1] * n_nodes + self.arcs[:, 0]
        order = np.argsort(key)
        pos = np.searchsorted(key[order], rkey)
        pos = np.clip(pos, 0, len(key) - 1)
        has = key[order][pos] == rkey
        self.reverse_arc = np.where(has, order[pos], -1)
        self.is_two_way = has

        # restrict to arcs near the trajectory
        txy = self.proj.forward(track_lonlat[:, 0], track_lonlat[:, 1])
        lo = txy.min(axis=0) - margin_m
        hi = txy.max(axis=0) + margin_m
        seg_lo = np.minimum(a, b)
        seg_hi = np.maximum(a, b)
        mask = (seg_hi >= lo).all(axis=1) & (seg_lo <= hi).all(axis=1)
        self.sub_arcs = np.nonzero(mask)[0]

        # routable graph over the restricted arcs (parallel arcs: keep the shortest)
        sa = self.sub_arcs[self.arcs[self.sub_arcs, 0] != self.arcs[self.sub_arcs, 1]]
        sub_nodes = np.unique(self.arcs[self.sub_arcs].ravel())
        self.node_local = np.full(n_nodes, -1, dtype=np.int64)
        self.node_local[sub_nodes] = np.arange(len(sub_nodes))
        self.sub_nodes = sub_nodes
        u = self.node_local[self.arcs[sa, 0]]
        v = self.node_local[self.arcs[sa, 1]]
        w = self.arc_len[sa] + 1e-3          # strictly positive so csgraph keeps every edge
        o = np.lexsort((w, v, u))
        u, v, w = u[o], v[o], w[o]
        first = np.r_[True, (u[1:] != u[:-1]) | (v[1:] != v[:-1])]
        self.graph = csr_matrix((w[first], (u[first], v[first])), shape=(len(sub_nodes),) * 2)

        # spatial index: densified points along every restricted arc
        self.spacing = spacing_m
        la = self.arc_len[self.sub_arcs]
        n = np.maximum(1, np.ceil(la / spacing_m)).astype(np.int64) + 1
        rep = np.repeat(np.arange(len(self.sub_arcs)), n)
        starts = np.cumsum(n) - n
        k = np.arange(n.sum()) - np.repeat(starts, n)
        t = k / np.repeat(n - 1, n)
        sa_ = a[self.sub_arcs]
        sb_ = b[self.sub_arcs]
        pts = sa_[rep] + (sb_ - sa_)[rep] * t[:, None]
        self.sample_arc = self.sub_arcs[rep]
        self.tree = cKDTree(pts)

    # ------------------------------------------------------------------ geometry
    def to_xy(self, lonlat):
        lonlat = np.asarray(lonlat, dtype=float)
        return self.proj.forward(lonlat[:, 0], lonlat[:, 1])

    def distance_to_arcs(self, pts_xy, arc_ids):
        """Element-wise distance (and position t) of points to arcs."""
        arc_ids = np.asarray(arc_ids, dtype=np.int64)
        return point_to_segment(pts_xy, self.node_xy[self.arcs[arc_ids, 0]], self.node_xy[self.arcs[arc_ids, 1]])

    def candidates(self, pts_xy, radius_steps=config.CANDIDATE_RADIUS_STEPS, k=config.MAX_CANDIDATES):
        """Candidate arcs per point. Returns dict of flat arrays sorted by (obs, dist).

        The radius is widened step by step only for points that have no arc in range.
        """
        pts_xy = np.asarray(pts_xy, dtype=float)
        n = len(pts_xy)
        todo = np.arange(n)
        out = []
        used_radius = np.full(n, np.nan)
        for r in radius_steps:
            if len(todo) == 0:
                break
            lists = self.tree.query_ball_point(pts_xy[todo], r=r + self.spacing / 2.0)
            lens = np.fromiter((len(x) for x in lists), dtype=np.int64, count=len(lists))
            if lens.sum() == 0:
                continue
            obs = np.repeat(todo, lens)
            arc = self.sample_arc[np.concatenate([np.asarray(x, dtype=np.int64) for x in lists if len(x)])]
            pair = np.unique(obs * len(self.arcs) + arc)
            obs, arc = pair // len(self.arcs), pair % len(self.arcs)
            d, t = self.distance_to_arcs(pts_xy[obs], arc)
            keep = d <= r
            obs, arc, d, t = obs[keep], arc[keep], d[keep], t[keep]
            found = np.unique(obs)
            used_radius[found] = r
            out.append((obs, arc, d, t))
            todo = np.setdiff1d(todo, found)
        if out:
            obs, arc, d, t = (np.concatenate(z) for z in zip(*out))
        else:
            obs = arc = np.zeros(0, dtype=np.int64)
            d = t = np.zeros(0)
        o = np.lexsort((arc, d, obs))
        obs, arc, d, t = obs[o], arc[o], d[o], t[o]
        start = np.r_[0, np.nonzero(np.diff(obs))[0] + 1] if len(obs) else np.zeros(0, dtype=np.int64)
        rank = np.arange(len(obs)) - np.repeat(start, np.diff(np.r_[start, len(obs)])) if len(obs) else np.zeros(0, dtype=np.int64)
        keep = rank < k
        return {"obs": obs[keep], "arc": arc[keep], "dist": d[keep], "t": t[keep],
                "rank": rank[keep], "radius": used_radius}

    # ------------------------------------------------------------------ routing
    def shortest_from(self, src_nodes, limit):
        """Network distance (m) from each global node in src_nodes to all subgraph nodes."""
        loc = self.node_local[np.asarray(src_nodes, dtype=np.int64)]
        res = np.full((len(loc), self.graph.shape[0]), np.inf)
        ok = loc >= 0
        if ok.any():
            res[ok] = dijkstra(self.graph, directed=True, indices=loc[ok], limit=limit)
        return res

    def path_nodes(self, src, dst, limit):
        """Global node ids along the shortest path src -> dst, or None if not reachable."""
        s, d = self.node_local[src], self.node_local[dst]
        if s < 0 or d < 0:
            return None
        if s == d:
            return [src]
        dist, pred = dijkstra(self.graph, directed=True, indices=s, limit=limit, return_predecessors=True)
        if not np.isfinite(dist[d]):
            return None
        path = [d]
        while path[-1] != s:
            path.append(pred[path[-1]])
        return [int(self.sub_nodes[i]) for i in reversed(path)]

    def connected(self, a, b):
        """True if arc b can directly follow arc a (same arc, or head of a == tail of b)."""
        a = np.asarray(a)
        b = np.asarray(b)
        return (a == b) | (self.arcs[a, 1] == self.arcs[b, 0])

    def arc_lonlat(self, arc_id):
        u, v = self.arcs[arc_id]
        return self.nodes_lonlat[[u, v]]

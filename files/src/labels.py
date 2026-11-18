"""Per-observation reference labels.

The .route file is an ordered list of arcs without timestamps, so it does not say which GPS fix
belongs to which arc. We align the two sequences with a monotone dynamic programme: every fix i is
assigned a route position j_i with j_0 <= j_1 <= ... <= j_{n-1} minimising the total
point-to-arc distance. The label of fix i is route[j_i].
"""
import numpy as np


def _dist_matrix(p, a, b):
    """Distances from points p (n,2) to segments a->b (r,2): returns (n, r)."""
    ab = b - a                                   # (r,2)
    denom = np.einsum("ij,ij->i", ab, ab)
    denom = np.where(denom > 0, denom, 1.0)
    ap = p[:, None, :] - a[None, :, :]           # (n,r,2)
    t = np.clip(np.einsum("nrj,rj->nr", ap, ab) / denom, 0.0, 1.0)
    proj = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(p[:, None, :] - proj, axis=2)


def align_route(net, pts_xy, route, chunk=512):
    """Return (route_index, label_arc, align_dist) per GPS fix."""
    route = np.asarray(route, dtype=np.int64)
    n, r = len(pts_xy), len(route)
    a = net.node_xy[net.arcs[route, 0]]
    b = net.node_xy[net.arcs[route, 1]]
    dp = np.empty((n, r), dtype=np.float64)
    for s in range(0, n, chunk):
        cost = _dist_matrix(pts_xy[s:s + chunk], a, b)
        for k in range(cost.shape[0]):
            i = s + k
            dp[i] = cost[k] if i == 0 else cost[k] + np.minimum.accumulate(dp[i - 1])
    idx = np.empty(n, dtype=np.int64)
    j = int(np.argmin(dp[-1]))
    for i in range(n - 1, -1, -1):
        idx[i] = j
        if i > 0:
            j = int(np.argmin(dp[i - 1, : j + 1]))
    lab = route[idx]
    d, _ = net.distance_to_arcs(pts_xy, lab)
    return idx, lab, d

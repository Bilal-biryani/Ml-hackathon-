"""HMM map matching decoded with Viterbi.

Hidden state at fix i: one of the candidate arcs of fix i (with its projected position t on the arc).
Emission:   log P(z_i | s) = w * log p_clf(s)                  (classifier probability)
Transition: log P(s_i | s_{i-1}) = -|d_route - d_gc| / beta      (Newson & Krumm style)
  d_gc    : straight-line distance between the two fixes
  d_route : network distance from the projected position on arc a to the projected position on arc b
            = (1 - t_a) len_a + shortest_path(head(a), tail(b)) + t_b len_b
            (same arc moving forward: (t_b - t_a) len_a; small backward jitter tolerated)
  No path within the search limit  -> UNREACHABLE log score (soft, so the chain never breaks)
  Time gap > gap_break_s            -> uniform transition (the chain restarts, emissions decide)
Fixes without candidates are skipped and receive no prediction.
"""
import heapq
from dataclasses import asdict, dataclass

import numpy as np

from . import config


@dataclass
class HMMParams:
    beta: float = 10.0
    emission_weight: float = 1.0
    unreachable_logp: float = config.UNREACHABLE_LOGP
    gap_break_s: float = config.GAP_BREAK_S
    backtrack_tol_m: float = 15.0

    def to_dict(self):
        return asdict(self)


class LocalRouter:
    """Bounded Dijkstra on the trip sub-graph (fast for the short hops between 1 Hz fixes)."""

    def __init__(self, net):
        g = net.graph.tocsr()
        self.indptr, self.indices, self.weights = g.indptr, g.indices, g.data
        self.node_local = net.node_local
        self._cache = {}

    def from_node(self, node_global, limit):
        key = int(node_global)
        hit = self._cache.get(key)
        if hit is not None and hit[0] >= limit:
            return hit[1]
        s = self.node_local[key]
        dist = {}
        if s >= 0:
            dist = {s: 0.0}
            heap = [(0.0, s)]
            done = set()
            while heap:
                d, u = heapq.heappop(heap)
                if u in done:
                    continue
                done.add(u)
                for k in range(self.indptr[u], self.indptr[u + 1]):
                    v = self.indices[k]
                    nd = d + self.weights[k]
                    if nd <= limit and nd < dist.get(v, np.inf):
                        dist[v] = nd
                        heapq.heappush(heap, (nd, v))
        if len(self._cache) > 512:
            self._cache.clear()
        self._cache[key] = (limit, dist)
        return dist

    def dist(self, src_global, dst_global, limit):
        d = self.from_node(src_global, limit)
        loc = self.node_local[int(dst_global)]
        return d.get(loc, np.inf) if loc >= 0 else np.inf


def build_steps(cands, n_obs):
    """Group candidate rows (already restricted to one trip) per fix, preserving order."""
    c = cands.sort_values(["obs", "dist_rank"])
    obs = c["obs"].to_numpy()
    arcs = c["arc"].to_numpy()
    tt = c["t_frac"].to_numpy()
    p = c["proba"].to_numpy()
    bounds = np.r_[0, np.nonzero(np.diff(obs))[0] + 1, len(obs)]
    steps = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b > a:
            steps.append((int(obs[a]), arcs[a:b], tt[a:b], p[a:b]))
    return steps


def transition_distances(net, xy, t, steps, gap_break_s=config.GAP_BREAK_S, backtrack_tol=15.0):
    """Per consecutive step pair: (d_route matrix, d_gc) or None if the time gap breaks the chain."""
    router = LocalRouter(net)
    out = [None]
    for (o0, A, ta, _), (o1, B, tb, _) in zip(steps[:-1], steps[1:]):
        dt = t[o1] - t[o0]
        if dt > gap_break_s:
            out.append(None)
            continue
        gc = float(np.linalg.norm(xy[o1] - xy[o0]))
        limit = max(2.0 * gc + 100.0, 200.0)
        la, lb = net.arc_len[A], net.arc_len[B]
        heads, tails = net.arcs[A, 1], net.arcs[B, 0]
        rd = np.empty((len(A), len(B)))
        for i in range(len(A)):
            dmap = router.from_node(heads[i], limit)
            base = (1.0 - ta[i]) * la[i]
            for j in range(len(B)):
                loc = net.node_local[tails[j]]
                sp = dmap.get(loc, np.inf) if loc >= 0 else np.inf
                if sp > limit:          # cached search may have used a larger limit
                    sp = np.inf
                rd[i, j] = base + sp + tb[j] * lb[j]
        same = A[:, None] == B[None, :]
        if same.any():
            fwd = (tb[None, :] - ta[:, None]) * la[:, None]
            rd = np.where(same & (fwd >= -backtrack_tol), np.abs(fwd), rd)
        out.append((rd, gc))
    return out


def viterbi(steps, trans, params: HMMParams):
    """Most likely candidate index per step."""
    w = params.emission_weight
    em = [w * np.log(np.clip(s[3], 1e-6, 1.0)) for s in steps]
    delta = em[0].copy()
    backs = []
    for k in range(1, len(steps)):
        tr = trans[k]
        if tr is None:
            T = np.zeros((len(steps[k - 1][1]), len(steps[k][1])))
        else:
            rd, gc = tr
            T = np.where(np.isfinite(rd), -np.abs(rd - gc) / params.beta, params.unreachable_logp)
        sc = delta[:, None] + T
        bp = np.argmax(sc, axis=0)
        delta = sc[bp, np.arange(sc.shape[1])] + em[k]
        delta -= delta.max()
        backs.append(bp)
    path = np.empty(len(steps), dtype=np.int64)
    path[-1] = int(np.argmax(delta)) if len(steps) else 0
    for k in range(len(steps) - 1, 0, -1):
        path[k - 1] = backs[k - 1][path[k]]
    return path


def decode(steps, trans, params, n_obs):
    """Per-fix predicted arc (-1 where the fix had no candidates)."""
    pred = np.full(n_obs, -1, dtype=np.int64)
    if not steps:
        return pred
    path = viterbi(steps, trans, params)
    for (o, arcs, _, _), k in zip(steps, path):
        pred[o] = arcs[k]
    return pred


def classifier_only(steps, n_obs):
    pred = np.full(n_obs, -1, dtype=np.int64)
    for o, arcs, _, p in steps:
        pred[o] = arcs[int(np.argmax(p))]
    return pred


def fill_route(net, pred_arcs, max_fill_m=3000.0):
    """Ordered node sequences (global ids) for drawing: consecutive distinct arcs, gaps filled with
    shortest paths where one exists within max_fill_m, otherwise a new segment is started."""
    seq = [int(a) for a in pred_arcs if a >= 0]
    dedup = [a for i, a in enumerate(seq) if i == 0 or a != seq[i - 1]]
    segments, cur = [], []
    for a in dedup:
        u, v = int(net.arcs[a, 0]), int(net.arcs[a, 1])
        if not cur:
            cur = [u, v]
            continue
        if cur[-1] == u:
            cur.append(v)
            continue
        path = net.path_nodes(cur[-1], u, max_fill_m)
        if path is not None:
            cur.extend(path[1:])
            cur.append(v)
        else:
            segments.append(cur)
            cur = [u, v]
    if cur:
        segments.append(cur)
    return segments, dedup

"""Geometry helpers.

The dataset is a single city (~8.6E, 49.9N). A local equirectangular projection centred on each
trip is accurate to well under 0.1 % over a few tens of kilometres, so distances in metres can be
computed with plain Euclidean geometry after projecting.
"""
import numpy as np

EARTH_RADIUS_M = 6371008.8


class LocalProjection:
    """Equirectangular projection around (lon0, lat0) giving metres."""

    def __init__(self, lon0: float, lat0: float):
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        self.kx = np.cos(np.deg2rad(self.lat0)) * EARTH_RADIUS_M * np.pi / 180.0
        self.ky = EARTH_RADIUS_M * np.pi / 180.0

    def forward(self, lon, lat):
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        return np.stack([(lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky], axis=-1)

    def inverse(self, xy):
        xy = np.asarray(xy, dtype=float)
        return np.stack([xy[..., 0] / self.kx + self.lon0, xy[..., 1] / self.ky + self.lat0], axis=-1)


def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.deg2rad, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def point_to_segment(p, a, b):
    """Element-wise projection of points p onto segments a->b (all arrays shape (n, 2)).

    Returns (distance, t) where t in [0, 1] is the clamped position along the segment.
    """
    p = np.asarray(p, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    t = np.einsum("ij,ij->i", p - a, ab) / np.where(denom > 0, denom, 1.0)
    t = np.clip(np.where(denom > 0, t, 0.0), 0.0, 1.0)
    proj = a + ab * t[:, None]
    return np.linalg.norm(p - proj, axis=1), t

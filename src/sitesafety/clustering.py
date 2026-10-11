"""Euclidean clustering of non-ground points (DBSCAN) and per-cluster shape features.

DBSCAN groups points that are density-connected: a point with at least `min_points`
neighbours within `eps` metres is a core point, and clusters are chains of core points plus
their neighbours. Everything else is noise (label -1). It needs no number of clusters and
handles arbitrary shapes, which suits LiDAR objects. Its weakness: one fixed `eps` for near
(dense) and far (sparse) objects, and objects that touch (a worker leaning on a wall) merge.
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "n_points",
    "height",          # top of the cluster above the ground plane (m)
    "bottom",          # lowest point above the ground plane (m)
    "z_extent",
    "length",          # footprint along its main horizontal axis (m)
    "width",           # footprint across it (m)
    "footprint_area",
    "aspect_hw",       # z_extent / length: tall and thin -> person-like
    "planarity",       # PCA shape descriptors from 3D eigenvalues l1 >= l2 >= l3
    "linearity",
    "range",           # horizontal distance to the sensor (m): density falls with range
    "density",         # points per m^3 of the bounding box
    "intensity_mean",
    "intensity_std",
]


def dbscan(xyz: np.ndarray, eps: float, min_points: int, backend: str = "sklearn") -> np.ndarray:
    """Cluster labels (-1 = noise) for an (N, 3) array.

    `open3d` is ~2x faster here; without it installed (e.g. in CI) we fall back to sklearn,
    which gives the same partition (DBSCAN is deterministic for core points).
    """
    if len(xyz) == 0:
        return np.zeros(0, dtype=np.int64)
    if backend == "open3d":
        try:
            import open3d as o3d
        except ImportError:
            return dbscan(xyz, eps, min_points, "sklearn")

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(xyz, np.float64)))
        return np.asarray(pcd.cluster_dbscan(eps, min_points, print_progress=False), np.int64)
    from sklearn.cluster import DBSCAN

    return DBSCAN(eps=eps, min_samples=min_points).fit_predict(xyz).astype(np.int64)


def cluster_features(points: np.ndarray, height: np.ndarray) -> np.ndarray:
    """Shape features of one cluster. `points` (N, 4) xyz+intensity, `height` above ground."""
    xyz = points[:, :3].astype(np.float64)
    xy = xyz[:, :2] - xyz[:, :2].mean(axis=0)
    # Main horizontal axis from the 2x2 covariance -> rotation-invariant length / width.
    if len(xy) >= 2:
        _, vecs = np.linalg.eigh(np.cov(xy.T) + 1e-9 * np.eye(2))
        proj = xy @ vecs
        width, length = np.ptp(proj, axis=0)
    else:
        length = width = 0.0
    z_ext = float(np.ptp(xyz[:, 2]))
    if len(xyz) >= 3:
        ev = np.sort(np.linalg.eigvalsh(np.cov(xyz.T) + 1e-9 * np.eye(3)))[::-1]
        l1, l2, l3 = np.maximum(ev, 1e-9)
        planarity, linearity = (l2 - l3) / l1, (l1 - l2) / l1
    else:
        planarity = linearity = 0.0
    centroid = xyz.mean(axis=0)
    volume = max(length, 0.05) * max(width, 0.05) * max(z_ext, 0.05)
    inten = points[:, 3] if points.shape[1] > 3 else np.zeros(len(points))
    return np.array(
        [
            len(points),
            float(height.max()),
            float(height.min()),
            z_ext,
            length,
            width,
            length * width,
            z_ext / max(length, 0.05),
            planarity,
            linearity,
            float(np.hypot(centroid[0], centroid[1])),
            len(points) / volume,
            float(inten.mean()),
            float(inten.std()),
        ],
        dtype=np.float64,
    )


def segment(points: np.ndarray, height: np.ndarray, cfg: dict, backend: str = "sklearn"):
    """DBSCAN + features. Returns (labels, list of cluster dicts)."""
    labels = dbscan(points[:, :3], cfg["eps"], cfg["min_points"], backend)
    clusters = []
    for k in range(labels.max() + 1 if len(labels) else 0):
        idx = np.flatnonzero(labels == k)
        if len(idx) < cfg["min_points"] or len(idx) > cfg["max_points"]:
            continue
        clusters.append(
            {
                "id": k,
                "idx": idx,
                "centroid": points[idx, :3].mean(axis=0),
                "features": cluster_features(points[idx], height[idx]),
            }
        )
    return labels, clusters

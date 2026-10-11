"""Point-cloud preprocessing: loading, cropping, voxel downsampling and ground removal.

Ground removal is a from-scratch RANSAC plane fit (with an Open3D `segment_plane` option for
comparison):

0. seed candidates: only points close to the lowest point of their 1 m x 1 m grid cell.
   Without this, a horizontal slab at ~1.2 m wins, because it cuts through every wall and car
   side and holds more points than the (sparse, voxel-thinned) road surface;
1. pick 3 random candidates, build the plane through them: n . p + d = 0 with |n| = 1;
2. count inliers, the points with |n . p + d| < threshold;
3. keep the plane with the most inliers, skipping planes tilted more than `max_tilt_deg`
   from horizontal (otherwise a large wall or bus side can win);
4. refit the plane to all inliers by least squares (SVD) for a less noisy estimate.

After the fit every point gets a signed height above the plane; points below
`keep_above` are treated as ground.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def load_sweep(path: Path | str) -> np.ndarray:
    """Argoverse 2 LiDAR sweep (.feather) -> float32 (N, 4) array [x, y, z, intensity].

    Coordinates are in the ego-vehicle frame (x forward, y left, z up, metres). The files store
    float16; we upcast before doing any arithmetic.
    """
    import pandas as pd

    df = pd.read_feather(path, columns=["x", "y", "z", "intensity"])
    return df.to_numpy(dtype=np.float32)


def crop(
    points: np.ndarray,
    range_m: float,
    z_min: float,
    z_max: float,
    self_box: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Keep points within a horizontal radius and a height window, minus the machine's own body."""
    r = np.hypot(points[:, 0], points[:, 1])
    keep = (r <= range_m) & (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    if self_box is not None:
        x0, x1, y0, y1 = self_box
        x, y = points[:, 0], points[:, 1]
        inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        keep &= ~inside
    return points[keep]


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    """Replace all points that fall into one voxel by their mean (all columns averaged)."""
    if voxel <= 0 or len(points) == 0:
        return points
    ijk = np.floor(points[:, :3] / voxel).astype(np.int64)
    ijk -= ijk.min(axis=0)
    dims = ijk.max(axis=0) + 1
    # One integer per voxel: 1-D np.unique is ~10x faster than np.unique(axis=0) on rows.
    keys = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    sums = np.zeros((len(counts), points.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, points)
    return (sums / counts[:, None]).astype(np.float32)


@dataclass
class Plane:
    normal: np.ndarray  # unit normal pointing up (n_z > 0)
    d: float

    def signed_distance(self, xyz: np.ndarray) -> np.ndarray:
        return xyz[:, :3] @ self.normal + self.d

    @property
    def tilt_deg(self) -> float:
        return float(np.degrees(np.arccos(np.clip(abs(self.normal[2]), -1.0, 1.0))))


def _plane_from_points(p: np.ndarray) -> Plane | None:
    n = np.cross(p[1] - p[0], p[2] - p[0])
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return None  # collinear sample
    n = n / norm
    if n[2] < 0:
        n = -n
    return Plane(n, float(-n @ p[0]))


def fit_plane_lstsq(xyz: np.ndarray) -> Plane:
    """Total-least-squares plane: normal = smallest singular vector of the centred points."""
    centroid = xyz.mean(axis=0)
    _, _, vt = np.linalg.svd(xyz - centroid, full_matrices=False)
    n = vt[-1]
    if n[2] < 0:
        n = -n
    return Plane(n, float(-n @ centroid))


def lowest_point_seeds(xyz: np.ndarray, cell: float = 1.0, band: float = 0.2) -> np.ndarray:
    """Mask of points within `band` metres of the lowest point in their x-y grid cell."""
    keys = np.floor(xyz[:, :2] / cell).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    cell_min = np.full(inverse.max() + 1, np.inf)
    np.minimum.at(cell_min, inverse, xyz[:, 2])
    return xyz[:, 2] <= cell_min[inverse] + band


def ransac_plane(
    xyz: np.ndarray,
    threshold: float = 0.15,
    iterations: int = 200,
    max_tilt_deg: float = 15.0,
    rng: np.random.Generator | None = None,
    refine: bool = True,
    seed_cell: float | None = 1.0,
) -> tuple[Plane, np.ndarray]:
    """Fit the dominant near-horizontal plane. Returns (plane, inlier mask over all points).

    `seed_cell=None` disables the lowest-point seeding (the naive variant, kept for ablation).
    """
    rng = rng or np.random.default_rng(0)
    xyz = np.asarray(xyz, dtype=np.float64)[:, :3]
    if len(xyz) < 3:
        raise ValueError("need at least 3 points")
    cand = xyz[lowest_point_seeds(xyz, seed_cell)] if seed_cell else xyz
    if len(cand) < 3:
        cand = xyz
    best, best_count = None, -1
    cos_max = np.cos(np.radians(max_tilt_deg))
    for _ in range(iterations):
        plane = _plane_from_points(cand[rng.choice(len(cand), 3, replace=False)])
        if plane is None or plane.normal[2] < cos_max:
            continue
        count = int((np.abs(plane.signed_distance(cand)) < threshold).sum())
        if count > best_count:
            best, best_count = plane, count
    if best is None:  # every sample was steep: fall back to a flat plane at the lowest decile
        best = Plane(np.array([0.0, 0.0, 1.0]), -float(np.percentile(xyz[:, 2], 10)))
    if refine:
        seed_in = np.abs(best.signed_distance(cand)) < threshold
        if seed_in.sum() >= 3:
            refined = fit_plane_lstsq(cand[seed_in])
            if refined.tilt_deg <= max_tilt_deg:
                best = refined
    return best, np.abs(best.signed_distance(xyz)) < threshold


def open3d_plane(xyz: np.ndarray, threshold: float, iterations: int) -> tuple[Plane, np.ndarray]:
    """Same job with Open3D's `segment_plane` (optional dependency)."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(xyz[:, :3], np.float64)))
    (a, b, c, d), idx = pcd.segment_plane(threshold, 3, iterations)
    n = np.array([a, b, c])
    scale = np.linalg.norm(n)
    n, d = n / scale, d / scale
    if n[2] < 0:
        n, d = -n, -d
    mask = np.zeros(len(xyz), dtype=bool)
    mask[np.asarray(idx)] = True
    return Plane(n, float(d)), mask


def fit_ground(points: np.ndarray, cfg: dict, rng: np.random.Generator | None = None) -> Plane:
    method = cfg.get("method", "ransac")
    if method == "open3d":
        return open3d_plane(points, cfg["threshold"], cfg["iterations"])[0]
    seed = None if method == "ransac_naive" else cfg.get("seed_cell", 1.0)
    return ransac_plane(
        points, cfg["threshold"], cfg["iterations"], cfg["max_tilt_deg"], rng, seed_cell=seed
    )[0]


def remove_ground(points: np.ndarray, cfg: dict, rng: np.random.Generator | None = None):
    """Return (kept points, their height above the ground plane, plane).

    Keeps points between `keep_above` and `max_height` metres above the plane: the upper cut
    drops tree canopies, awnings and building tops, which only slow clustering down and can
    glue a person to an overhang.
    """
    plane = fit_ground(points, cfg, rng)
    h = plane.signed_distance(points)
    keep = (h > cfg["keep_above"]) & (h < cfg.get("max_height", np.inf))
    return points[keep], h[keep], plane

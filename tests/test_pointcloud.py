"""Ground removal, voxelisation and clustering on synthetic point clouds."""

import numpy as np
import pytest

from sitesafety.clustering import FEATURE_NAMES, cluster_features, dbscan, segment
from sitesafety.pointcloud import (
    crop,
    fit_ground,
    fit_plane_lstsq,
    lowest_point_seeds,
    ransac_plane,
    remove_ground,
    voxel_downsample,
)


def make_scene(rng, slope=0.02, ground_z=-0.5):
    """Tilted ground + a dense wall + two 'people' columns."""
    g = rng.uniform(-20, 20, (6000, 2))
    ground = np.column_stack([g, ground_z + slope * g[:, 0] + rng.normal(0, 0.02, len(g))])
    wy = rng.uniform(-15, 15, 9000)
    wz = rng.uniform(ground_z, ground_z + 3, 9000)
    wall = np.column_stack([np.full_like(wy, 12.0) + rng.normal(0, 0.02, len(wy)), wy, wz])
    people = []
    for cx, cy in ((4.0, 3.0), (-6.0, -5.0)):
        n = 300
        z0 = ground_z + slope * cx
        people.append(np.column_stack([cx + rng.normal(0, 0.15, n), cy + rng.normal(0, 0.15, n),
                                       rng.uniform(z0 + 0.1, z0 + 1.75, n)]))
    pts = np.vstack([ground, wall, *people])
    return np.column_stack([pts, rng.uniform(0, 100, len(pts))]).astype(np.float32)


def test_lstsq_plane_recovers_known_plane():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-5, 5, (500, 2))
    z = 0.1 * xy[:, 0] - 0.05 * xy[:, 1] + 2.0
    plane = fit_plane_lstsq(np.column_stack([xy, z]))
    expected = np.array([-0.1, 0.05, 1.0]) / np.linalg.norm([-0.1, 0.05, 1.0])
    assert np.allclose(plane.normal, expected, atol=1e-6)
    assert plane.signed_distance(np.array([[0.0, 0.0, 2.0]]))[0] == pytest.approx(0, abs=1e-6)


def test_seeded_ransac_finds_ground_not_wall_band():
    rng = np.random.default_rng(1)
    pts = voxel_downsample(make_scene(rng), 0.1)
    plane, inliers = ransac_plane(pts, 0.15, 200, 15.0, np.random.default_rng(0))
    # Height of the plane at the origin should be the ground (-0.5), with the true slope.
    z0 = -plane.d / plane.normal[2]
    assert z0 == pytest.approx(-0.5, abs=0.05)
    assert plane.tilt_deg == pytest.approx(np.degrees(np.arctan(0.02)), abs=0.5)
    assert inliers.sum() > 1000


def test_lowest_point_seeds_exclude_wall():
    rng = np.random.default_rng(2)
    pts = make_scene(rng)
    seeds = lowest_point_seeds(pts[:, :3].astype(float))
    assert np.median(pts[seeds, 2]) < 0.0  # seeds are ground points
    assert seeds.mean() < 0.6


def test_remove_ground_keeps_people_drops_ground():
    rng = np.random.default_rng(3)
    pts = make_scene(rng)
    cfg = {"method": "ransac", "threshold": 0.15, "iterations": 200, "max_tilt_deg": 15,
           "keep_above": 0.15, "max_height": 2.5, "seed_cell": 1.0}
    kept, h, plane = remove_ground(pts, cfg, np.random.default_rng(0))
    assert (h > 0.15).all() and (h < 2.5).all()
    near_person = np.hypot(kept[:, 0] - 4.0, kept[:, 1] - 3.0) < 0.6
    assert near_person.sum() > 250
    assert len(kept) < len(pts) * 0.7


def test_naive_variant_is_selectable():
    rng = np.random.default_rng(4)
    pts = make_scene(rng)
    cfg = {"method": "ransac_naive", "threshold": 0.15, "iterations": 50, "max_tilt_deg": 15}
    plane = fit_ground(pts, cfg, np.random.default_rng(0))
    assert plane.normal[2] > 0.9


def test_voxel_downsample_merges_points_in_one_voxel():
    pts = np.array([[0.01, 0.01, 0.01, 10], [0.02, 0.03, 0.04, 20], [0.5, 0.5, 0.5, 30]],
                   dtype=np.float32)
    out = voxel_downsample(pts, 0.1)
    assert len(out) == 2
    assert sorted(out[:, 3].tolist()) == pytest.approx([15.0, 30.0])


def test_crop_range_height_and_self_box():
    pts = np.array([[1, 0, 0, 0], [30, 0, 0, 0], [5, 0, 5, 0], [0.5, 0.2, 0.3, 0]], np.float32)
    out = crop(pts, 25, -1, 3, self_box=(0.0, 0.8, -0.5, 0.5))
    assert out.tolist() == [[1, 0, 0, 0]]


def test_dbscan_separates_two_blobs_and_marks_noise():
    rng = np.random.default_rng(5)
    a = rng.normal([0, 0, 1], 0.1, (100, 3))
    b = rng.normal([5, 0, 1], 0.1, (100, 3))
    noise = np.array([[20.0, 20.0, 1.0]])
    labels = dbscan(np.vstack([a, b, noise]), eps=0.5, min_points=5)
    assert len(set(labels[:100])) == 1 and len(set(labels[100:200])) == 1
    assert labels[0] != labels[100] and labels[-1] == -1


def test_cluster_features_describe_a_person_column():
    rng = np.random.default_rng(6)
    n = 400
    pts = np.column_stack([rng.normal(0, 0.15, n), rng.normal(10, 0.12, n),
                           rng.uniform(0, 1.8, n), rng.uniform(0, 50, n)])
    f = dict(zip(FEATURE_NAMES, cluster_features(pts, pts[:, 2]), strict=True))
    assert f["height"] == pytest.approx(1.8, abs=0.05)
    assert f["length"] < 1.2 and f["width"] < 1.0
    assert f["range"] == pytest.approx(10, abs=0.1)
    assert f["linearity"] > 0.8  # tall and thin


def test_segment_filters_by_size():
    rng = np.random.default_rng(7)
    pts = np.vstack([rng.normal([0, 0, 1], 0.1, (50, 3)), rng.normal([4, 0, 1], 0.1, (3, 3))])
    pts = np.column_stack([pts, np.zeros(len(pts))])
    _, clusters = segment(pts, pts[:, 2], {"eps": 0.5, "min_points": 5, "max_points": 1000})
    assert len(clusters) == 1 and len(clusters[0]["idx"]) == 50

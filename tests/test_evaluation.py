"""Matching, metrics and cluster labelling used by the LiDAR evaluation."""

import numpy as np
import pytest

from sitesafety.av2 import GTBox
from sitesafety.evaluate_lidar import match, prf, split_gt, wilson
from sitesafety.lidar_detect import SweepResult, label_clusters, rule_scores

EVAL = {"person_categories": ["PEDESTRIAN"], "ignore_categories": ["BICYCLIST"],
        "min_interior_points": 5, "match_dist_m": 1.0}
CFG = {"eval": EVAL, "lidar": {"range_m": 25.0}}
RULE = {"height": [1.0, 2.1], "max_footprint": 1.3, "min_points": 8}


def box(cat, x, y, n=50):
    return GTBox(cat, np.array([x, y, 0.9]), np.array([0.6, 0.6, 1.8]), 0.0, n, "t")


def test_split_gt_care_dontcare_and_range():
    boxes = [box("PEDESTRIAN", 3, 0), box("PEDESTRIAN", 5, 0, n=2), box("BICYCLIST", 8, 0),
             box("PEDESTRIAN", 40, 0), box("REGULAR_VEHICLE", 2, 2)]
    care, dont = split_gt(boxes, CFG)
    assert care.tolist() == [[3, 0]]
    assert sorted(dont[:, 0].tolist()) == [5, 8]


def test_greedy_match_and_dont_care():
    pred = np.array([[0.1, 0.0], [10.0, 0.0], [20.2, 0.0], [30.0, 0.0]])
    gt = np.array([[0.0, 0.0], [10.5, 0.0], [50.0, 0.0]])
    dont = np.array([[20.0, 0.0]])
    tp, fp, hit = match(pred, gt, dont, 1.0)
    assert tp.tolist() == [True, True, False, False]
    assert fp.tolist() == [False, False, False, True]  # the one near a rider is ignored
    assert hit.tolist() == [True, True, False]


def test_match_is_one_to_one():
    tp, fp, hit = match(np.array([[0.0, 0.0], [0.2, 0.0]]), np.array([[0.1, 0.0]]),
                        np.zeros((0, 2)), 1.0)
    assert tp.sum() == 1 and fp.sum() == 1 and hit.all()


def test_prf():
    r = prf(8, 2, 8)
    assert r["precision"] == pytest.approx(0.8) and r["recall"] == pytest.approx(0.5)
    assert r["f1"] == pytest.approx(2 * 0.8 * 0.5 / 1.3, abs=1e-4)
    assert prf(0, 0, 0)["f1"] == 0.0


def test_label_clusters_by_point_majority():
    rng = np.random.default_rng(0)
    person = np.column_stack([rng.normal(3, 0.1, (60, 2)), rng.uniform(0.1, 1.7, 60)])
    rider = np.column_stack([rng.normal(8, 0.1, (60, 2)), rng.uniform(0.1, 1.7, 60)])
    wall = np.column_stack([np.full(60, 15.0), rng.uniform(-3, 3, 60), rng.uniform(0, 2, 60)])
    pts = np.vstack([person, rider, wall])
    clusters = [{"idx": np.arange(i * 60, (i + 1) * 60)} for i in range(3)]
    res = SweepResult(pts, pts[:, 2], None, clusters)
    labels = label_clusters(res, [box("PEDESTRIAN", 3, 3), box("BICYCLIST", 8, 8)], EVAL)
    assert labels.tolist() == [1, -1, 0]


def test_rule_scores():
    from sitesafety.clustering import FEATURE_NAMES

    f = np.zeros((2, len(FEATURE_NAMES)))
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    f[:, idx["n_points"]] = 50
    f[:, idx["length"]] = 0.6
    f[:, idx["width"]] = 0.5
    f[0, idx["height"]] = 1.75
    f[1, idx["height"]] = 3.5  # too tall: a pole or a truck side
    assert rule_scores(f, RULE).tolist() == [1.0, 0.0]


def test_wilson_interval():
    lo, hi = wilson(17, 18)
    assert lo < 17 / 18 < hi <= 1.0
    assert wilson(0, 0) == [0.0, 1.0]
    assert wilson(50, 100)[0] == pytest.approx(0.404, abs=1e-3)

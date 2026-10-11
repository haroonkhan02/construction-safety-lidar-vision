"""Boxes, projection, danger zones and the debouncer."""

import numpy as np
import pytest

from sitesafety.geometry import (
    box_corners_bev,
    points_in_box,
    project_to_image,
    quat_to_rotation,
    quat_to_yaw,
    rect_contains,
)
from sitesafety.zones import Debouncer, assign_zones, distance_to_machine, machine_box, occupancy

Z = {"machine_half_length": 2.5, "machine_half_width": 1.0, "machine_center_x": 1.4,
     "red_margin": 3.0, "amber_margin": 6.0, "debounce_frames": 2}


def yaw_quat(yaw):
    return np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)


def test_quaternion_yaw_and_rotation_agree():
    for yaw in (0.0, 0.7, -2.0, np.pi / 2):
        q = yaw_quat(yaw)
        assert quat_to_yaw(*q) == pytest.approx(yaw)
        R = quat_to_rotation(*q)
        assert R @ np.array([1.0, 0, 0]) == pytest.approx([np.cos(yaw), np.sin(yaw), 0])


def test_points_in_rotated_box():
    pts = np.array([[1.9, 0.0, 0.0], [0.0, 1.9, 0.0], [0.0, 0.0, 0.0]])
    # 4 x 1 box rotated by 90 degrees: long side now along y
    inside = points_in_box(pts, [0, 0, 0], [4, 1, 2], np.pi / 2)
    assert inside.tolist() == [False, True, True]
    corners = box_corners_bev(np.zeros(3), 4, 1, np.pi / 2)
    assert np.abs(corners[:, 1]).max() == pytest.approx(2.0)


def test_projection_of_point_straight_ahead_hits_principal_point():
    # Camera looking along ego +x: camera z = ego x, camera x = -ego y, camera y = -ego z.
    R = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)  # columns = camera axes in ego
    uv, front = project_to_image(np.array([[10.0, 0, 1.5], [-5.0, 0, 1.5]]), R,
                                 np.array([0, 0, 1.5]), 1000, 1000, 640, 480)
    assert front.tolist() == [True, False]
    assert uv[0] == pytest.approx([640, 480])
    uv2, _ = project_to_image(np.array([[10.0, 1.0, 1.5]]), R, np.array([0, 0, 1.5]),
                              1000, 1000, 640, 480)
    assert uv2[0, 0] < 640  # a point to the left lands left of centre


def test_zone_assignment_is_nested_and_red_has_priority():
    x0, x1, y0, y1 = machine_box(Z)
    xy = np.array([[x1 + 1.0, 0.0], [x1 + 4.0, 0.0], [x1 + 10.0, 0.0], [1.4, y1 + 2.9]])
    assert assign_zones(xy, Z) == ["red", "amber", None, "red"]
    assert occupancy(xy[2:3], Z) == {"red": False, "amber": False}
    assert occupancy(xy[:1], Z) == {"red": True, "amber": True}
    assert distance_to_machine(np.array([[x1 + 3.0, 0.0], [1.4, 0.0]]), Z).tolist() == \
        pytest.approx([3.0, 0.0])
    assert rect_contains(np.array([[0.0, 0.0]]), (-1, 1, -1, 1)).all()


def test_debouncer_needs_consecutive_frames():
    d = Debouncer(2)
    seq = [True, False, True, True, True, False]
    out = [d.update({"red": s, "amber": s})["red"] for s in seq]
    assert out == [False, False, False, True, True, False]

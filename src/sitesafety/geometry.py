"""Small 3D geometry helpers: quaternions, oriented boxes, rigid transforms, projection."""

from __future__ import annotations

import numpy as np


def quat_to_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Unit quaternion (scalar first) -> 3x3 rotation matrix."""
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_to_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    """Heading angle around +z (radians) of a quaternion."""
    return float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))


def points_in_box(
    points: np.ndarray,
    center: np.ndarray,
    size: np.ndarray,
    yaw: float,
    margin: float = 0.0,
) -> np.ndarray:
    """Boolean mask of points inside an oriented 3D box (yaw about z).

    `size` = (length along heading, width, height); `margin` grows every side.
    """
    d = points[:, :3] - np.asarray(center, dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    local_x = c * d[:, 0] + s * d[:, 1]
    local_y = -s * d[:, 0] + c * d[:, 1]
    half = np.asarray(size, dtype=np.float64) / 2 + margin
    return (
        (np.abs(local_x) <= half[0]) & (np.abs(local_y) <= half[1]) & (np.abs(d[:, 2]) <= half[2])
    )


def box_corners_bev(center: np.ndarray, length: float, width: float, yaw: float) -> np.ndarray:
    """4x2 footprint corners of an oriented box in the x-y plane."""
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    local = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * [length / 2, width / 2]
    return local @ rot.T + np.asarray(center)[:2]


def transform_points(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray):
    """Apply p' = R p + t to an (N, 3) array."""
    return points[:, :3] @ rotation.T + translation


def project_to_image(
    points_ego: np.ndarray,
    ego_R_cam: np.ndarray,
    ego_t_cam: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pinhole projection of ego-frame points into a camera (no lens distortion).

    The camera pose is given as ego_T_cam (camera -> ego). Returns (uv, in_front_mask).
    Camera axes follow the usual convention: z forward, x right, y down.
    """
    cam = (points_ego[:, :3] - ego_t_cam) @ ego_R_cam  # = R^T (p - t), row-vector form
    z = cam[:, 2]
    front = z > 0.1
    zs = np.where(front, z, 1.0)
    uv = np.stack([fx * cam[:, 0] / zs + cx, fy * cam[:, 1] / zs + cy], axis=1)
    return uv, front


def rect_contains(xy: np.ndarray, rect: tuple[float, float, float, float]) -> np.ndarray:
    """Axis-aligned rectangle (x_min, x_max, y_min, y_max) membership for (N, 2) points."""
    x0, x1, y0, y1 = rect
    xy = np.atleast_2d(xy)
    return (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)

"""Reading the downloaded Argoverse 2 logs: sweeps, ground-truth cuboids and camera calibration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import quat_to_rotation, quat_to_yaw


@dataclass
class GTBox:
    category: str
    center: np.ndarray  # (3,) ego frame
    size: np.ndarray  # (length, width, height)
    yaw: float
    num_points: int
    track: str


def load_manifest(root: Path) -> dict:
    return json.loads((root / "manifest.json").read_text())


def logs_with_role(root: Path, role: str) -> list[str]:
    return sorted(lid for lid, r in load_manifest(root)["logs"].items() if r == role)


def sweep_paths(root: Path, log_id: str, stride: int = 1) -> list[Path]:
    """Sorted sweep files; `stride` > 1 subsamples relative to the files on disk."""
    return sorted((root / "logs" / log_id / "lidar").glob("*.feather"))[::stride]


def eval_stride(root: Path, log_id: str) -> int:
    """The demo log is downloaded at 10 Hz; evaluate it at the same rate as the others."""
    m = load_manifest(root)
    return m["stride"] if log_id == m["demo_log"] else 1


class Annotations:
    """Cuboids of one log, indexed by LiDAR timestamp (annotations are at 10 Hz)."""

    def __init__(self, root: Path, log_id: str):
        import pandas as pd

        self.df = pd.read_feather(root / "annotations" / f"{log_id}.feather")
        self.groups = {t: g for t, g in self.df.groupby("timestamp_ns")}

    def boxes(self, timestamp_ns: int) -> list[GTBox]:
        g = self.groups.get(timestamp_ns)
        if g is None:
            return []
        out = []
        for r in g.itertuples(index=False):
            out.append(
                GTBox(
                    category=r.category,
                    center=np.array([r.tx_m, r.ty_m, r.tz_m]),
                    size=np.array([r.length_m, r.width_m, r.height_m]),
                    yaw=quat_to_yaw(r.qw, r.qx, r.qy, r.qz),
                    num_points=int(r.num_interior_pts),
                    track=r.track_uuid,
                )
            )
        return out


@dataclass
class Camera:
    name: str
    R: np.ndarray  # ego_R_cam
    t: np.ndarray  # ego_t_cam
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def load_camera(root: Path, log_id: str, name: str) -> Camera:
    import pandas as pd

    d = root / "logs" / log_id
    ext = pd.read_feather(d / "egovehicle_SE3_sensor.feather").set_index("sensor_name").loc[name]
    intr = pd.read_feather(d / "intrinsics.feather").set_index("sensor_name").loc[name]
    return Camera(
        name=name,
        R=quat_to_rotation(ext.qw, ext.qx, ext.qy, ext.qz),
        t=np.array([ext.tx_m, ext.ty_m, ext.tz_m]),
        fx=float(intr.fx_px),
        fy=float(intr.fy_px),
        cx=float(intr.cx_px),
        cy=float(intr.cy_px),
        width=int(intr.width_px),
        height=int(intr.height_px),
    )


def nearest_image(root: Path, log_id: str, timestamp_ns: int) -> Path | None:
    imgs = sorted((root / "logs" / log_id / "camera").glob("*.jpg"))
    if not imgs:
        return None
    stamps = np.array([int(p.stem) for p in imgs])
    return imgs[int(np.abs(stamps - timestamp_ns).argmin())]

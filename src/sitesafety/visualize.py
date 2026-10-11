"""Drawing: bird's-eye-view LiDAR frames, camera overlays and PPE compliance boxes."""

from __future__ import annotations

import cv2
import numpy as np

from .compliance import Worker
from .geometry import project_to_image
from .zones import assign_zones, machine_box, zone_rects

# BGR colours
RED, AMBER, GREEN = (40, 40, 230), (0, 170, 255), (80, 200, 80)
WHITE, GREY = (235, 235, 235), (110, 110, 110)
ZONE_COLOR = {"red": RED, "amber": AMBER, None: GREEN}


class BEV:
    """Top-down canvas: x forward is up, y left is left; `extent` metres around the sensor."""

    def __init__(self, size: int = 720, extent: float = 25.0):
        self.size, self.extent = size, extent
        self.scale = size / (2 * extent)

    def px(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        u = self.size / 2 - xy[:, 1] * self.scale
        v = self.size / 2 - xy[:, 0] * self.scale
        return np.stack([u, v], axis=1).round().astype(np.int32)

    def rect(self, img, r, color, thickness=2, fill_alpha=0.0):
        x0, x1, y0, y1 = r
        pts = self.px(np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0]]))
        if fill_alpha > 0:
            over = img.copy()
            cv2.fillPoly(over, [pts], color)
            cv2.addWeighted(over, fill_alpha, img, 1 - fill_alpha, 0, dst=img)
        cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)


def draw_bev(res, zcfg: dict, alerts: dict, gt_xy: np.ndarray | None = None, bev: BEV | None = None,
             title: str = "") -> np.ndarray:
    bev = bev or BEV()
    img = np.full((bev.size, bev.size, 3), 18, np.uint8)
    for r in (10, 20):  # range rings
        cv2.circle(img, (bev.size // 2, bev.size // 2), int(r * bev.scale), (45, 45, 45), 1,
                   cv2.LINE_AA)
    rects = zone_rects(zcfg)
    for name in ("amber", "red"):
        bev.rect(img, rects[name], ZONE_COLOR[name], 2, 0.22 if alerts.get(name) else 0.0)
    if res.cloud is not None and len(res.cloud):
        uv = bev.px(res.cloud[:, :2])
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < bev.size) & (uv[:, 1] >= 0) & (uv[:, 1] < bev.size)
        img[uv[ok, 1], uv[ok, 0]] = (70, 70, 70)
    hv = np.clip(res.height / 2.5, 0, 1)
    uv = bev.px(res.points[:, :2])
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < bev.size) & (uv[:, 1] >= 0) & (uv[:, 1] < bev.size)
    col = np.stack([200 * (1 - hv) + 55, 160 + 60 * hv, 120 * hv + 60], 1).astype(np.uint8)
    img[uv[ok, 1], uv[ok, 0]] = col[ok]
    people = [c for c, p in zip(res.clusters, res.person, strict=True) if p]
    zones = assign_zones(np.array([c["centroid"][:2] for c in people]).reshape(-1, 2), zcfg)
    for c, z in zip(people, zones, strict=True):
        q = bev.px(res.points[c["idx"], :2])
        img[q[:, 1].clip(0, bev.size - 1), q[:, 0].clip(0, bev.size - 1)] = ZONE_COLOR[z]
        cv2.circle(img, tuple(bev.px(c["centroid"][:2])[0]), 9, ZONE_COLOR[z], 2, cv2.LINE_AA)
    if gt_xy is not None:
        for p in bev.px(gt_xy):
            cv2.drawMarker(img, tuple(p), WHITE, cv2.MARKER_TILTED_CROSS, 7, 1, cv2.LINE_AA)
    bev.rect(img, machine_box(zcfg), WHITE, 2, 0.6)
    state = "RED ZONE INTRUSION" if alerts.get("red") else ("AMBER: person nearby"
                                                            if alerts.get("amber") else "clear")
    color = RED if alerts.get("red") else (AMBER if alerts.get("amber") else GREEN)
    cv2.rectangle(img, (0, 0), (bev.size, 34), (0, 0, 0), -1)
    cv2.putText(img, state, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    if title:
        cv2.putText(img, title, (bev.size - 260, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1,
                    cv2.LINE_AA)
    cv2.putText(img, "o person (LiDAR)   x ground truth", (bev.size - 300, bev.size - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, f"people: {len(people)}   {res.timings_ms.get('total', 0):.0f} ms",
                (10, bev.size - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
    return img


def _ground_z(plane, xy: np.ndarray) -> np.ndarray:
    n, d = plane.normal, plane.d
    return -(xy @ n[:2] + d) / n[2]


def draw_camera(img: np.ndarray, res, cam, zcfg: dict) -> np.ndarray:
    """Project the zone outlines (on the fitted ground plane) and person clusters into a camera."""
    out = img.copy()
    for name, r in zone_rects(zcfg).items():
        x0, x1, y0, y1 = r
        loop = np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0], [x0, y0]], float)
        dense = np.concatenate(
            [np.linspace(a, b, 40) for a, b in zip(loop[:-1], loop[1:], strict=True)]
        )
        xyz = np.column_stack([dense, _ground_z(res.plane, dense)])
        uv, front = project_to_image(xyz, cam.R, cam.t, cam.fx, cam.fy, cam.cx, cam.cy)
        for a, b, fa, fb in zip(uv[:-1], uv[1:], front[:-1], front[1:], strict=True):
            if fa and fb:
                cv2.line(out, tuple(a.astype(int)), tuple(b.astype(int)), ZONE_COLOR[name], 4,
                         cv2.LINE_AA)
    people = [c for c, p in zip(res.clusters, res.person, strict=True) if p]
    zones = assign_zones(np.array([c["centroid"][:2] for c in people]).reshape(-1, 2), zcfg)
    for c, z in zip(people, zones, strict=True):
        pts = res.points[c["idx"], :3]
        lo, hi = pts.min(0), pts.max(0)
        corners = np.array([[x, y, zz] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                            for zz in (lo[2] - 0.1, hi[2])])
        uv, front = project_to_image(corners, cam.R, cam.t, cam.fx, cam.fy, cam.cx, cam.cy)
        if front.all():
            (u0, v0), (u1, v1) = uv.min(0).astype(int), uv.max(0).astype(int)
            cv2.rectangle(out, (u0, v0), (u1, v1), ZONE_COLOR[z], 5)
    return out


def dashboard(bev_img: np.ndarray, cam_img: np.ndarray | None) -> np.ndarray:
    if cam_img is None:
        return bev_img
    h = bev_img.shape[0]
    cam = cv2.resize(cam_img, (int(cam_img.shape[1] * h / cam_img.shape[0]), h),
                     interpolation=cv2.INTER_AREA)
    return np.hstack([bev_img, cam])


def draw_workers(img: np.ndarray, workers: list[Worker], dets: np.ndarray | None = None):
    """Green = compliant, red = violation (labelled), grey = unknown PPE status."""
    out = img.copy()
    if dets is not None:
        for d in dets:
            if int(d[5]) != 5:  # PPE / other boxes: thin outline
                x0, y0, x1, y1 = d[:4].astype(int)
                cv2.rectangle(out, (x0, y0), (x1, y1), (200, 200, 200), 1)
    for w in workers:
        x0, y0, x1, y1 = w.box.astype(int)
        if w.violation:
            color, text = RED, "NO " + "/".join(v.upper() for v in w.violations)
        elif all(w.status[k] == "yes" for k in ("hardhat", "vest")):
            color, text = GREEN, "OK"
        else:
            color, text = GREY, "PPE ?"
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        cv2.putText(out, text, (x0, max(14, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                    cv2.LINE_AA)
    return out

"""Alert dashboard on the 10 Hz demo log (`cs-demo`).

For every sweep: run the LiDAR pipeline, debounce the zone occupancy, log alert start/stop
events to `alerts.jsonl`, and render a dashboard frame (bird's-eye view + front camera with the
zones and the people projected into the image). Writes an MP4, a small GIF for the README and the
single busiest frame as a PNG.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from . import av2
from .config import load_config
from .evaluate_lidar import split_gt
from .lidar_detect import ClusterClassifier, process_sweep
from .pointcloud import load_sweep
from .visualize import dashboard, draw_bev, draw_camera
from .zones import Debouncer, assign_zones, distance_to_machine, occupancy


def run(cfg: dict, out_dir: Path, kind: str = "gbt", max_frames: int | None = None,
        gif_path: Path | None = None) -> dict:
    root = Path(cfg["lidar"]["data"])
    man = av2.load_manifest(root)
    log_id, Z = man["demo_log"], cfg["zones"]
    clf = ClusterClassifier.load(kind, cfg["lidar"])
    cam = av2.load_camera(root, log_id, man["demo_camera"])
    ann = av2.Annotations(root, log_id)
    deb = Debouncer(Z["debounce_frames"])
    out_dir.mkdir(parents=True, exist_ok=True)
    writer, frames_small, events = None, [], []
    active = {"red": False, "amber": False}
    busiest, busiest_score = None, -1
    paths = av2.sweep_paths(root, log_id)[:max_frames]
    t0 = int(paths[0].stem)
    for p in paths:
        ts = int(p.stem)
        res = process_sweep(load_sweep(p), cfg, clf)
        xy = res.person_centroids[:, :2]
        alerts = deb.update(occupancy(xy, Z))
        for z in ("red", "amber"):
            if alerts[z] != active[z]:
                zones = assign_zones(xy, Z)
                events.append({
                    "t_s": round((ts - t0) / 1e9, 2), "timestamp_ns": ts, "zone": z,
                    "event": "start" if alerts[z] else "end",
                    "people_in_zone": int(sum((v == z) or (z == "amber" and v == "red")
                                              for v in zones)),
                    "closest_m": (round(float(distance_to_machine(xy, Z).min()), 2)
                                  if len(xy) else None),
                })
                active[z] = alerts[z]
        gt, _ = split_gt(ann.boxes(ts), cfg)
        bev = draw_bev(res, Z, alerts, gt_xy=gt, title=f"t = {(ts - t0) / 1e9:5.1f} s")
        img_path = av2.nearest_image(root, log_id, ts)
        cam_img = draw_camera(cv2.imread(str(img_path)), res, cam, Z) if img_path else None
        frame = dashboard(bev, cam_img)
        if writer is None:
            writer = cv2.VideoWriter(str(out_dir / "demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                     10, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
        score = 10 * int(alerts["red"]) + len(xy)
        if score > busiest_score:
            busiest, busiest_score = frame, score
        if paths.index(p) % 3 == 0:  # GIF at ~3 fps keeps the README asset small
            small = cv2.resize(frame, (frame.shape[1] * 300 // frame.shape[0], 300),
                               interpolation=cv2.INTER_AREA)
            frames_small.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    writer.release()
    cv2.imwrite(str(out_dir / "busiest_frame.png"), busiest)
    Path("reports/lidar").mkdir(parents=True, exist_ok=True)
    cv2.imwrite("reports/lidar/dashboard_frame.jpg", busiest, [cv2.IMWRITE_JPEG_QUALITY, 85])
    with open(out_dir / "alerts.jsonl", "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    if gif_path is not None and frames_small:
        from PIL import Image

        ims = [Image.fromarray(f).quantize(colors=96, method=Image.Quantize.MEDIANCUT)
               for f in frames_small]
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        ims[0].save(gif_path, save_all=True, append_images=ims[1:], duration=300, loop=0,
                    optimize=True)
    summary = {"log_id": log_id, "frames": len(paths), "classifier": kind,
               "alert_events": len(events), "out_dir": str(out_dir)}
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Render the LiDAR + camera alert dashboard.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--classifier", default="gbt", choices=["gbt", "rule"])
    ap.add_argument("--out", type=Path, default=Path("outputs/demo"))
    ap.add_argument("--gif", type=Path, default=Path("assets/demo.gif"))
    ap.add_argument("--max-frames", type=int, default=None)
    a = ap.parse_args()
    run(load_config(a.config, a.set), a.out, a.classifier, a.max_frames, a.gif)


if __name__ == "__main__":
    main()

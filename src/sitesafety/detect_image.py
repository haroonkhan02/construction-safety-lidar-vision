"""Run the PPE detector + compliance rules on images, a folder or a video (`cs-detect`)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .compliance import assign
from .config import load_config, pick_device
from .visualize import draw_workers

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def detect(model, img: np.ndarray, cfg: dict, device: str) -> np.ndarray:
    r = model.predict(img, conf=cfg["ppe"]["conf"], iou=cfg["ppe"]["iou"], device=device,
                      verbose=False)[0]
    if not len(r.boxes):
        return np.zeros((0, 6))
    return np.column_stack([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                            r.boxes.cls.cpu().numpy()])


def report(workers) -> dict:
    return {
        "workers": len(workers),
        "violations": sum(w.violation for w in workers),
        "details": [{"box": [round(float(v), 1) for v in w.box], "status": w.status}
                    for w in workers],
    }


def main() -> None:
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description="PPE compliance on images or a video.")
    ap.add_argument("source", type=Path, help="image, folder of images, or video file")
    ap.add_argument("--out", type=Path, default=Path("outputs/detect"))
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    device = pick_device(cfg["ppe"]["device"])
    model = YOLO(cfg["ppe"]["weights"])
    a.out.mkdir(parents=True, exist_ok=True)
    src = a.source
    if src.is_dir() or src.suffix.lower() in IMAGE_EXT:
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXT) if src.is_dir() \
            else [src]
        log = {}
        for f in files:
            img = cv2.imread(str(f))
            dets = detect(model, img, cfg, device)
            workers, _ = assign(dets, cfg["compliance"])
            cv2.imwrite(str(a.out / f.name), draw_workers(img, workers, dets))
            log[f.name] = report(workers)
        (a.out / "compliance.json").write_text(json.dumps(log, indent=2))
        print(f"{len(files)} images -> {a.out}  "
              f"({sum(v['violations'] for v in log.values())} violations)")
        return
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    writer = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets = detect(model, frame, cfg, device)
        workers, _ = assign(dets, cfg["compliance"])
        vis = draw_workers(frame, workers, dets)
        if writer is None:
            writer = cv2.VideoWriter(str(a.out / f"{src.stem}_ppe.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (vis.shape[1], vis.shape[0]))
        writer.write(vis)
    if writer is not None:
        writer.release()
    print(f"video -> {a.out / f'{src.stem}_ppe.mp4'}")


if __name__ == "__main__":
    main()

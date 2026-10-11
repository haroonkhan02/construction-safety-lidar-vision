"""Evaluate the PPE detector and the compliance rules on the test split (`cs-eval-ppe`).

1. Detection: COCO-style mAP@0.5 and mAP@0.5:0.95 per class, on all 82 test images and on the
   40 "clean" ones that share neither a source image nor a video with the training set.
2. Compliance: apply the same person <-> PPE rules to the ground-truth boxes and to the
   detections, match workers by IoU >= 0.5, and score
   * worker detection (recall / precision),
   * per-item status agreement (hardhat yes/no, vest yes/no) on matched workers,
   * violation flagging (does the worker break the hard-hat or vest rule?).
3. Speed: inference latency per image on the training device and on the CPU.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from .compliance import RULE_ITEMS, assign, match_workers, yolo_to_xyxy
from .config import load_config, pick_device
from .ppe_data import CLASS_NAMES, read_labels


def detection_metrics(model, data_yaml: Path, device: str, name: str, out_dir: Path) -> dict:
    m = model.val(data=str(data_yaml), split="test", device=device, plots=True, verbose=False,
                  project=str(out_dir.resolve()), name=name, exist_ok=True, batch=16)
    per_class = {CLASS_NAMES[c]: {"ap50": round(float(m.box.ap50[i]), 4),
                                  "ap50_95": round(float(m.box.ap[i]), 4)}
                 for i, c in enumerate(m.box.ap_class_index)}
    return {
        "mAP50": round(float(m.box.map50), 4),
        "mAP50_95": round(float(m.box.map), 4),
        "precision": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
        "per_class": per_class,
        "plots_dir": str(Path(m.save_dir)),
    }


def clean_yaml(root: Path, clean_names: list[str], out: Path) -> Path:
    lst = out.with_suffix(".txt")
    lst.write_text("\n".join(str((root / "test" / "images" / n).resolve()) for n in clean_names))
    base = yaml.safe_load((root.parent / "ppe.yaml").read_text())
    base["test"] = str(lst.resolve())
    out.write_text(yaml.safe_dump(base, sort_keys=False))
    return out


def compliance_eval(model, root: Path, cfg: dict, device: str) -> dict:
    ccfg, conf = cfg["compliance"], cfg["ppe"]["conf"]
    stats = {"gt_workers": 0, "pred_workers": 0, "matched": 0,
             "status": {k: {"agree": 0, "known_pairs": 0} for k in RULE_ITEMS},
             "violation": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
             "gt_orphan_no_ppe_boxes": 0, "gt_violation_workers": 0}
    examples = []
    for img_path in sorted((root / "test" / "images").glob("*.jpg")):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        gt_dets = yolo_to_xyxy(read_labels(root / "test" / "labels" / f"{img_path.stem}.txt"), w, h)
        r = model.predict(img, conf=conf, iou=cfg["ppe"]["iou"], device=device, verbose=False)[0]
        pred_dets = np.zeros((0, 6))
        if len(r.boxes):
            pred_dets = np.column_stack([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                         r.boxes.cls.cpu().numpy()])
        gt_w, gt_orph = assign(gt_dets, ccfg)
        pr_w, _ = assign(pred_dets, ccfg)
        stats["gt_workers"] += len(gt_w)
        stats["pred_workers"] += len(pr_w)
        stats["gt_orphan_no_ppe_boxes"] += gt_orph["NO-Hardhat"] + gt_orph["NO-Safety Vest"]
        stats["gt_violation_workers"] += sum(g.violation for g in gt_w)
        pairs = match_workers(pr_w, gt_w)
        stats["matched"] += len(pairs)
        for i, j in pairs:
            p, g = pr_w[i], gt_w[j]
            for k in RULE_ITEMS:
                if g.status[k] != "unknown":
                    stats["status"][k]["known_pairs"] += 1
                    stats["status"][k]["agree"] += int(p.status[k] == g.status[k])
        # Violation flagging over GT workers (missed workers count as missed violations) and
        # over unmatched predicted workers (a flagged phantom worker is a false alarm).
        pm = dict((j, i) for i, j in pairs)
        for j, g in enumerate(gt_w):
            flagged = j in pm and pr_w[pm[j]].violation
            key = ("tp" if flagged else "fn") if g.violation else ("fp" if flagged else "tn")
            stats["violation"][key] += 1
        matched_pred = {i for i, _ in pairs}
        stats["violation"]["fp"] += sum(p.violation for i, p in enumerate(pr_w)
                                        if i not in matched_pred)
        examples.append((img_path, pred_dets, len(gt_w), sum(g.violation for g in gt_w)))
    v = stats["violation"]
    stats["worker_recall"] = round(stats["matched"] / max(stats["gt_workers"], 1), 4)
    stats["worker_precision"] = round(stats["matched"] / max(stats["pred_workers"], 1), 4)
    for k in RULE_ITEMS:
        s = stats["status"][k]
        s["agreement"] = round(s["agree"] / max(s["known_pairs"], 1), 4)
    stats["violation_precision"] = round(v["tp"] / max(v["tp"] + v["fp"], 1), 4)
    stats["violation_recall"] = round(v["tp"] / max(v["tp"] + v["fn"], 1), 4)
    stats["conf"] = conf
    return stats, examples


def gallery(model, examples: list, cfg: dict, device: str, out: Path, n: int = 6) -> None:
    """Six test images (half with a ground-truth violation) with the predicted compliance."""
    from .compliance import assign
    from .visualize import draw_workers

    viol = [e for e in examples if e[3] > 0 and e[2] <= 6]
    ok = [e for e in examples if e[3] == 0 and 0 < e[2] <= 6]
    picks = viol[: n // 2] + ok[: n - min(len(viol), n // 2)]
    tiles = []
    for img_path, dets, n_gt, n_viol in picks[:n]:
        img = cv2.imread(str(img_path))
        workers, _ = assign(dets, cfg["compliance"])
        tile = cv2.resize(draw_workers(img, workers, dets), (420, 420))
        cv2.rectangle(tile, (0, 0), (420, 26), (0, 0, 0), -1)
        cv2.putText(tile, f"ground truth: {n_gt} workers, {n_viol} violating", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def latency(model, root: Path, device: str, n: int = 50) -> float:
    imgs = [cv2.imread(str(p)) for p in sorted((root / "test" / "images").glob("*.jpg"))[:n]]
    for im in imgs[:5]:
        model.predict(im, device=device, verbose=False)
    t = time.perf_counter()
    for im in imgs:
        model.predict(im, device=device, verbose=False)
    return round((time.perf_counter() - t) / len(imgs) * 1e3, 1)


def main() -> None:
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description="PPE detection + compliance on the test split.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--root", type=Path, default=Path("data/ppe/css-data"))
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    device = pick_device(cfg["ppe"]["device"])
    out = Path("reports/ppe")
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(cfg["ppe"]["weights"])
    audit = json.loads((out / "dataset_audit.json").read_text())
    res = {"weights": cfg["ppe"]["weights"], "device": device}
    res["test_all"] = detection_metrics(model, Path(cfg["ppe"]["data"]), device, "val_test_all",
                                        Path("outputs"))
    res["test_all"]["images"] = audit["images"]["test"]
    cy = clean_yaml(a.root, audit["test_clean_images"], a.root.parent / "ppe_test_clean.yaml")
    res["test_clean"] = detection_metrics(model, cy, device, "val_test_clean", Path("outputs"))
    res["test_clean"]["images"] = len(audit["test_clean_images"])
    comp, examples = compliance_eval(model, a.root, cfg, device)
    res["compliance"] = comp
    gallery(model, examples, cfg, device, out / "compliance_examples.jpg")
    res["latency_ms"] = {device: latency(model, a.root, device),
                         "cpu": latency(model, a.root, "cpu")}
    for f in ("confusion_matrix_normalized.png", "BoxPR_curve.png", "PR_curve.png"):
        src = Path(res["test_all"]["plots_dir"]) / f
        if src.exists():
            shutil.copy(src, out / f"test_{f}")
    (out / "results_test.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("latency_ms",)} | {
        "all": {k: res["test_all"][k] for k in ("mAP50", "mAP50_95")},
        "clean": {k: res["test_clean"][k] for k in ("mAP50", "mAP50_95")},
        "compliance": {k: comp[k] for k in ("worker_recall", "worker_precision",
                                            "violation_precision", "violation_recall")}}, indent=2))


if __name__ == "__main__":
    main()

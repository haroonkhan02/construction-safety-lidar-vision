"""Evaluate the LiDAR pipeline against Argoverse 2 cuboids (`cs-eval-lidar`).

Per sweep we compare predicted person clusters with ground-truth person cuboids:

* detection: greedy nearest-centre matching in the x-y plane within `match_dist_m`;
  predictions near a "don't care" box (riders, wheelchairs, people with < 5 LiDAR points)
  count neither as hits nor as false alarms;
* cluster recall: is there *any* cluster near the person (the ceiling for every classifier);
* zone alerts: per frame, is the red / amber zone occupied according to the predictions vs the
  ground truth, with and without the n-frame debouncer;
* ground plane: median |plane height - GT box bottom| over well-observed boxes.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import av2
from .config import load_config
from .lidar_detect import ClusterClassifier, process_sweep
from .pointcloud import load_sweep
from .zones import Debouncer, occupancy

BANDS = ((0, 10), (10, 20), (20, 25))


def split_gt(boxes: list[av2.GTBox], cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """(care person centres, don't-care centres) within range, as (N, 2) arrays."""
    E, rng = cfg["eval"], cfg["lidar"]["range_m"]
    care, dont = [], []
    for b in boxes:
        if np.hypot(*b.center[:2]) > rng:
            continue
        if b.category in E["person_categories"]:
            (care if b.num_points >= E["min_interior_points"] else dont).append(b.center[:2])
        elif b.category in E["ignore_categories"]:
            dont.append(b.center[:2])
    return np.array(care).reshape(-1, 2), np.array(dont).reshape(-1, 2)


def match(pred: np.ndarray, gt: np.ndarray, dont: np.ndarray, dist: float):
    """Greedy matching. Returns (tp_pred_mask, fp_pred_mask, gt_matched_mask)."""
    tp = np.zeros(len(pred), bool)
    gt_hit = np.zeros(len(gt), bool)
    if len(pred) and len(gt):
        d = np.linalg.norm(pred[:, None, :] - gt[None, :, :], axis=2)
        for flat in np.argsort(d, axis=None):
            i, j = np.unravel_index(flat, d.shape)
            if d[i, j] > dist:
                break
            if not tp[i] and not gt_hit[j]:
                tp[i], gt_hit[j] = True, True
    fp = ~tp
    if len(dont) and fp.any():
        near_dont = np.linalg.norm(pred[:, None, :] - dont[None, :, :], axis=2).min(1) <= dist
        fp &= ~near_dont
    return tp, fp, gt_hit


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(f, 4)}


def ground_error(plane, boxes: list[av2.GTBox], rng: float) -> float | None:
    sel = [b for b in boxes if b.num_points >= 20 and np.hypot(*b.center[:2]) <= rng]
    if not sel or abs(plane.normal[2]) < 1e-6:
        return None
    c = np.array([b.center for b in sel])
    bottoms = c[:, 2] - np.array([b.size[2] for b in sel]) / 2
    z_plane = -(c[:, :2] @ plane.normal[:2] + plane.d) / plane.normal[2]
    return float(np.median(np.abs(z_plane - bottoms)))


def evaluate(cfg: dict, role: str = "test", kinds=("rule", "gbt"), keep_scores=True) -> dict:
    root = Path(cfg["lidar"]["data"])
    E, Z = cfg["eval"], cfg["zones"]
    clfs = {k: ClusterClassifier.load(k, cfg["lidar"]) for k in kinds}
    det = {k: defaultdict(int) for k in kinds}
    band = {k: {b: defaultdict(int) for b in BANDS} for k in kinds}
    alerts = {k: {db: {z: defaultdict(int) for z in ("red", "amber")}
                  for db in ("raw", "debounced")} for k in kinds}
    cluster_hit = defaultdict(int)
    scores, labels = [], []  # every cluster's GBT score vs whether it matched a person
    ground_err, timings, n_sweeps, gt_total = [], defaultdict(list), 0, 0
    per_log = {}
    for lid in av2.logs_with_role(root, role):
        ann = av2.Annotations(root, lid)
        debs = {k: Debouncer(Z["debounce_frames"]) for k in kinds}
        log_gt = 0
        for p in av2.sweep_paths(root, lid, av2.eval_stride(root, lid)):
            boxes = ann.boxes(int(p.stem))
            # Passing the GBT here only makes the timing include classification.
            res = process_sweep(load_sweep(p), cfg, clfs.get("gbt"))
            n_sweeps += 1
            for name, v in res.timings_ms.items():
                timings[name].append(v)
            err = ground_error(res.plane, boxes, cfg["lidar"]["range_m"])
            if err is not None:
                ground_err.append(err)
            gt, dont = split_gt(boxes, cfg)
            gt_total += len(gt)
            log_gt += len(gt)
            cents = np.array([c["centroid"][:2] for c in res.clusters]).reshape(-1, 2)
            _, _, any_hit = match(cents, gt, dont, E["match_dist_m"])
            cluster_hit["hit"] += int(any_hit.sum())
            for b in BANDS:
                in_b = (np.hypot(*gt.T) >= b[0]) & (np.hypot(*gt.T) < b[1]) if len(gt) else []
                cluster_hit[f"hit_{b}"] += int(any_hit[in_b].sum()) if len(gt) else 0
                cluster_hit[f"gt_{b}"] += int(np.sum(in_b)) if len(gt) else 0
            gt_occ = occupancy(gt, Z)
            feats = np.array([c["features"] for c in res.clusters]).reshape(len(res.clusters), -1)
            for k, clf in clfs.items():
                s = clf.scores(feats) if len(feats) else np.zeros(0)
                is_p = s >= clf.threshold
                pred = cents[is_p]
                tp, fp, hit = match(pred, gt, dont, E["match_dist_m"])
                det[k]["tp"] += int(tp.sum())
                det[k]["fp"] += int(fp.sum())
                det[k]["fn"] += int((~hit).sum())
                pr = np.hypot(*pred.T) if len(pred) else np.zeros(0)
                gr = np.hypot(*gt.T) if len(gt) else np.zeros(0)
                for b in BANDS:
                    bp = (pr >= b[0]) & (pr < b[1])
                    bg = (gr >= b[0]) & (gr < b[1])
                    band[k][b]["tp"] += int((tp & bp).sum())
                    band[k][b]["fp"] += int((fp & bp).sum())
                    band[k][b]["fn"] += int((~hit & bg).sum())
                raw = occupancy(pred, Z)
                deb = debs[k].update(raw)
                for db, occ in (("raw", raw), ("debounced", deb)):
                    for z in ("red", "amber"):
                        cell = alerts[k][db][z]
                        if occ[z]:
                            cell["tp" if gt_occ[z] else "fp"] += 1
                        else:
                            cell["fn" if gt_occ[z] else "tn"] += 1
                if k == "gbt" and keep_scores and len(s):
                    # label each cluster: matched to a care person (1), don't-care (-1), else 0
                    tp_all, fp_all, _ = match(cents, gt, dont, E["match_dist_m"])
                    lab = np.where(tp_all, 1, np.where(fp_all, 0, -1))
                    scores.append(s)
                    labels.append(lab)
        per_log[lid] = {"sweeps": len(av2.sweep_paths(root, lid, av2.eval_stride(root, lid))),
                        "gt_persons": log_gt}
    out = {
        "role": role,
        "logs": per_log,
        "sweeps": n_sweeps,
        "gt_person_instances": gt_total,
        "config": {k: cfg["lidar"][k] for k in ("range_m", "voxel")} | {
            "ground": cfg["lidar"]["ground"], "cluster": cfg["lidar"]["cluster"],
            "match_dist_m": E["match_dist_m"], "min_interior_points": E["min_interior_points"],
            "zones": Z},
        "cluster_recall": round(cluster_hit["hit"] / max(gt_total, 1), 4),
        "cluster_recall_by_range": {
            f"{a}-{b}m": round(
                cluster_hit[f"hit_{(a, b)}"] / max(cluster_hit[f"gt_{(a, b)}"], 1), 4)
            for a, b in BANDS},
        "detection": {k: prf(**det[k]) for k in kinds},
        "detection_by_range": {k: {f"{a}-{b}m": prf(**band[k][(a, b)]) for a, b in BANDS}
                               for k in kinds},
        "alerts": {k: {db: {z: alert_metrics(alerts[k][db][z]) for z in ("red", "amber")}
                       for db in ("raw", "debounced")} for k in kinds},
        "ground_error_m": {"median": round(float(np.median(ground_err)), 3),
                           "p90": round(float(np.percentile(ground_err, 90)), 3),
                           "share_over_0.3m": round(float(np.mean(np.array(ground_err) > 0.3)), 4)},
        "timing_ms": {k: round(float(np.mean(v)), 1) for k, v in timings.items()},
        "thresholds": {k: clfs[k].threshold for k in kinds},
    }
    if scores:
        out["_scores"] = (np.concatenate(scores), np.concatenate(labels))
    return out


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """95 % Wilson score interval for a proportion k / n (sane for tiny n and k = n)."""
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(float(centre - half), 3), round(float(centre + half), 3)]


def alert_metrics(c: dict) -> dict:
    tp, fp, fn, tn = (int(c.get(k, 0)) for k in ("tp", "fp", "fn", "tn"))
    return {**prf(tp, fp, fn), "tn": tn, "frames_occupied": tp + fn,
            "precision_ci95": wilson(tp, tp + fp), "recall_ci95": wilson(tp, tp + fn)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate LiDAR person detection and zone alerts.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--role", default="test", choices=["test", "tune"])
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    res = evaluate(cfg, a.role)
    out = a.out or Path(f"reports/lidar/results_{a.role}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    sc = res.pop("_scores", None)
    if sc is not None:
        np.savez(out.with_suffix(".scores.npz"), scores=sc[0], labels=sc[1])
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("sweeps", "gt_person_instances", "cluster_recall",
                                          "detection", "ground_error_m", "timing_ms")}, indent=2))
    print(json.dumps(res["alerts"], indent=1)[:3000])


if __name__ == "__main__":
    main()

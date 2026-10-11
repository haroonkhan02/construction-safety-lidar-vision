"""LiDAR person detection: preprocess -> ground removal -> DBSCAN -> classify clusters.

Two cluster classifiers:

* `rule`  — a hand-written baseline: a person is 1.0–2.1 m tall with a footprint under 1.3 m.
* `gbt`   — gradient-boosted trees on 14 shape features, trained on clusters from the *tune*
  logs whose points mostly fall inside a ground-truth person cuboid. Its probability threshold
  is chosen on the tune logs (best F1), never on the test logs.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import av2
from .clustering import FEATURE_NAMES, segment
from .config import load_config
from .geometry import points_in_box
from .pointcloud import crop, load_sweep, remove_ground, voxel_downsample
from .zones import machine_box


@dataclass
class SweepResult:
    points: np.ndarray  # kept (non-ground) points
    height: np.ndarray  # their height above the ground plane
    plane: object
    clusters: list = field(default_factory=list)
    person: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    score: np.ndarray = field(default_factory=lambda: np.zeros(0))
    timings_ms: dict = field(default_factory=dict)
    n_raw: int = 0
    n_voxel: int = 0
    cloud: np.ndarray | None = None  # the cropped, voxelised cloud (ground included), for display

    @property
    def person_centroids(self) -> np.ndarray:
        cs = [c["centroid"] for c, p in zip(self.clusters, self.person, strict=True) if p]
        return np.array(cs).reshape(-1, 3)


def rule_scores(features: np.ndarray, rule: dict) -> np.ndarray:
    """1.0 for clusters that pass every person rule, else 0.0."""
    f = {n: features[:, i] for i, n in enumerate(FEATURE_NAMES)}
    lo, hi = rule["height"]
    ok = (
        (f["height"] >= lo)
        & (f["height"] <= hi)
        & (f["length"] <= rule["max_footprint"])
        & (f["width"] <= rule["max_footprint"])
        & (f["n_points"] >= rule["min_points"])
        & (f["bottom"] <= 0.6)  # must stand on (or near) the ground
    )
    return ok.astype(float)


class ClusterClassifier:
    def __init__(self, kind: str = "rule", cfg: dict | None = None, model=None, threshold=0.5):
        self.kind, self.cfg, self.model, self.threshold = kind, cfg or {}, model, threshold

    @classmethod
    def load(cls, kind: str, cfg: dict) -> ClusterClassifier:
        if kind == "rule":
            return cls("rule", cfg, threshold=0.5)
        import joblib

        blob = joblib.load(cfg["classifier"])
        return cls("gbt", cfg, blob["model"], blob["threshold"])

    def scores(self, features: np.ndarray) -> np.ndarray:
        if len(features) == 0:
            return np.zeros(0)
        if self.kind == "rule":
            return rule_scores(features, self.cfg["person_rule"])
        return self.model.predict_proba(features)[:, 1]


def process_sweep(
    raw: np.ndarray,
    cfg: dict,
    classifier: ClusterClassifier | None = None,
    backend: str = "open3d",
) -> SweepResult:
    """Full LiDAR pipeline on one sweep (ego/machine frame)."""
    L, Z = cfg["lidar"], cfg["zones"]
    t0 = time.perf_counter()
    pts = crop(raw, L["range_m"], L["z_min"], L["z_max"], machine_box(Z, margin=0.3))
    pts = voxel_downsample(pts, L["voxel"])
    t1 = time.perf_counter()
    kept, h, plane = remove_ground(pts, L["ground"])
    t2 = time.perf_counter()
    _, clusters = segment(kept, h, L["cluster"], backend)
    t3 = time.perf_counter()
    res = SweepResult(kept, h, plane, clusters, n_raw=len(raw), n_voxel=len(pts), cloud=pts)
    res.person = np.zeros(len(clusters), bool)
    res.score = np.zeros(len(clusters))
    if classifier is not None:
        feats = np.array([c["features"] for c in clusters]).reshape(-1, len(FEATURE_NAMES))
        res.score = classifier.scores(feats)
        res.person = res.score >= classifier.threshold
    t4 = time.perf_counter()
    res.timings_ms = {
        "crop_voxel": (t1 - t0) * 1e3,
        "ground": (t2 - t1) * 1e3,
        "cluster": (t3 - t2) * 1e3,
        "classify": (t4 - t3) * 1e3,
        "total": (t4 - t0) * 1e3,
    }
    return res


def label_clusters(res: SweepResult, boxes: list[av2.GTBox], eval_cfg: dict) -> np.ndarray:
    """Per cluster: 1 = person, 0 = background, -1 = ignore (rider, wheelchair, ...).

    A cluster takes the category of the GT cuboid that contains most of its points
    (cuboids grown by 0.2 m to absorb box-fitting noise).
    """
    labels = np.zeros(len(res.clusters), dtype=np.int64)
    persons = set(eval_cfg["person_categories"])
    ignore = set(eval_cfg["ignore_categories"])
    relevant = [b for b in boxes if b.category in persons | ignore]
    for i, c in enumerate(res.clusters):
        pts = res.points[c["idx"]]
        best, best_frac = None, 0.5
        for b in relevant:
            frac = points_in_box(pts, b.center, b.size, b.yaw, margin=0.2).mean()
            if frac > best_frac:
                best, best_frac = b, frac
        if best is not None:
            labels[i] = 1 if best.category in persons else -1
    return labels


def collect_cluster_dataset(cfg: dict, role: str, max_sweeps: int | None = None):
    """Features + labels of every cluster in the logs with this role."""
    root = Path(cfg["lidar"]["data"])
    X, y, groups = [], [], []
    for lid in av2.logs_with_role(root, role):
        ann = av2.Annotations(root, lid)
        for p in av2.sweep_paths(root, lid, av2.eval_stride(root, lid))[:max_sweeps]:
            res = process_sweep(load_sweep(p), cfg)
            if not res.clusters:
                continue
            lab = label_clusters(res, ann.boxes(int(p.stem)), cfg["eval"])
            X.append(np.array([c["features"] for c in res.clusters]))
            y.append(lab)
            groups += [lid] * len(lab)
    return np.vstack(X), np.concatenate(y), np.array(groups)


def train_classifier(cfg: dict) -> dict:
    """Fit the GBT cluster classifier on the tune logs; pick its threshold by grouped CV."""
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold, cross_val_predict

    X, y, groups = collect_cluster_dataset(cfg, "tune")
    keep = y >= 0
    X, y, groups = X[keep], y[keep], groups[keep]
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15, class_weight="balanced", random_state=0
    )
    # Out-of-fold probabilities with whole logs held out, so the threshold is not tuned on
    # clusters the model has memorised.
    oof = cross_val_predict(
        model, X, y, groups=groups, cv=GroupKFold(n_splits=len(set(groups))), method="predict_proba"
    )[:, 1]
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.1, 0.96, 0.05):
        pred = oof >= t
        tp = int((pred & (y == 1)).sum())
        f1 = 2 * tp / max(int(pred.sum()) + int((y == 1).sum()), 1)
        if f1 > best_f1:
            best_t, best_f1 = float(round(t, 2)), f1
    model.fit(X, y)
    out = Path(cfg["lidar"]["classifier"])
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": best_t, "features": FEATURE_NAMES}, out)
    summary = {
        "tune_clusters": int(len(y)),
        "tune_person_clusters": int(y.sum()),
        "tune_logs": sorted(set(groups.tolist())),
        "threshold": best_t,
        "oof_cluster_f1": round(best_f1, 4),
        "path": str(out),
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the LiDAR cluster classifier (tune logs).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    summary = train_classifier(load_config(a.config, a.set))
    Path("reports/lidar").mkdir(parents=True, exist_ok=True)
    Path("reports/lidar/classifier_train.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

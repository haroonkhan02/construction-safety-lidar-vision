"""Parameter selection and ablations on the *tune* logs only (`cs-ablate-lidar`).

For each DBSCAN `eps` (and each ground-removal method) we cache every sweep's clusters once,
then score person detection with leave-one-log-out cross-validation: the GBT classifier is
trained on 5 tune logs and tested on the 6th. The chosen setting is then evaluated once on the
test logs by `cs-eval-lidar`, so test results never influence a parameter.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from . import av2
from .config import load_config
from .evaluate_lidar import ground_error, match, prf, split_gt
from .lidar_detect import label_clusters, process_sweep, rule_scores
from .pointcloud import load_sweep


def cache_sweeps(cfg: dict, role: str = "tune") -> list[dict]:
    root = Path(cfg["lidar"]["data"])
    out = []
    for lid in av2.logs_with_role(root, role):
        ann = av2.Annotations(root, lid)
        for p in av2.sweep_paths(root, lid, av2.eval_stride(root, lid)):
            boxes = ann.boxes(int(p.stem))
            res = process_sweep(load_sweep(p), cfg)
            gt, dont = split_gt(boxes, cfg)
            out.append(
                {
                    "log": lid,
                    "feats": np.array([c["features"] for c in res.clusters]).reshape(
                        len(res.clusters), -1
                    ),
                    "cents": np.array([c["centroid"][:2] for c in res.clusters]).reshape(-1, 2),
                    "labels": label_clusters(res, boxes, cfg["eval"]),
                    "gt": gt,
                    "dont": dont,
                    "ground_err": ground_error(res.plane, boxes, cfg["lidar"]["range_m"]),
                    "ms": res.timings_ms["total"],
                }
            )
    return out


def score_detection(sweeps: list[dict], scores: list[np.ndarray], thr: float, dist: float) -> dict:
    tp = fp = fn = 0
    for s, sc in zip(sweeps, scores, strict=True):
        pred = s["cents"][sc >= thr] if len(sc) else np.zeros((0, 2))
        t, f, hit = match(pred, s["gt"], s["dont"], dist)
        tp, fp, fn = tp + int(t.sum()), fp + int(f.sum()), fn + int((~hit).sum())
    return prf(tp, fp, fn)


def lolo_gbt(sweeps: list[dict], dist: float) -> dict:
    """Leave-one-log-out GBT scores; threshold fixed at the value picked by training CV."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    logs = sorted({s["log"] for s in sweeps})
    scores: list[np.ndarray | None] = [None] * len(sweeps)
    for held in logs:
        X = np.vstack([s["feats"] for s in sweeps if s["log"] != held])
        y = np.concatenate([s["labels"] for s in sweeps if s["log"] != held])
        keep = y >= 0
        m = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, class_weight="balanced",
            random_state=0,
        ).fit(X[keep], y[keep])
        for i, s in enumerate(sweeps):
            if s["log"] == held:
                scores[i] = m.predict_proba(s["feats"])[:, 1] if len(s["feats"]) else np.zeros(0)
    best = max(
        (score_detection(sweeps, scores, t, dist) | {"threshold": round(float(t), 2)}
         for t in np.arange(0.3, 0.96, 0.05)),
        key=lambda r: r["f1"],
    )
    return best


def cluster_recall(sweeps: list[dict], dist: float) -> float:
    hit = total = 0
    for s in sweeps:
        _, _, h = match(s["cents"], s["gt"], s["dont"], dist)
        hit, total = hit + int(h.sum()), total + len(s["gt"])
    return round(hit / max(total, 1), 4)


def run(cfg: dict, eps_values: list[float], ground_methods: list[str]) -> dict:
    dist = cfg["eval"]["match_dist_m"]
    rows = []
    variants = [("eps", e, {"cluster": {"eps": e}}) for e in eps_values]
    variants += [("ground", g, {"ground": {"method": g}}) for g in ground_methods]
    for kind, value, patch in variants:
        c = copy.deepcopy(cfg)
        for sec, kv in patch.items():
            c["lidar"][sec].update(kv)
        sweeps = cache_sweeps(c, "tune")
        rule = [rule_scores(s["feats"], c["lidar"]["person_rule"]) if len(s["feats"]) else
                np.zeros(0) for s in sweeps]
        errs = [s["ground_err"] for s in sweeps if s["ground_err"] is not None]
        row = {
            "ablation": kind,
            "value": value,
            "cluster_recall": cluster_recall(sweeps, dist),
            "rule": score_detection(sweeps, rule, 0.5, dist),
            "gbt_lolo": lolo_gbt(sweeps, dist),
            "ground_err_median_m": round(float(np.median(errs)), 3),
            "ground_err_share_over_0.3m": round(float(np.mean(np.array(errs) > 0.3)), 4),
            "ms_per_sweep": round(float(np.mean([s["ms"] for s in sweeps])), 1),
        }
        print(json.dumps({k: row[k] for k in ("ablation", "value", "cluster_recall",
                                              "ground_err_median_m")}
                         | {"gbt_f1": row["gbt_lolo"]["f1"], "rule_f1": row["rule"]["f1"]}))
        rows.append(row)
    return {"role": "tune", "sweeps": len(sweeps), "rows": rows}


def to_markdown(res: dict) -> str:
    lines = [
        "| Ablation | Value | Cluster recall | Rule F1 | GBT F1 (LOLO) | Ground err (median m) "
        "| Ground err > 0.3 m | ms / sweep |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in res["rows"]:
        lines.append(
            f"| {r['ablation']} | {r['value']} | {r['cluster_recall']:.3f} | {r['rule']['f1']:.3f} "
            f"| {r['gbt_lolo']['f1']:.3f} | {r['ground_err_median_m']:.3f} "
            f"| {r['ground_err_share_over_0.3m']:.1%} | {r['ms_per_sweep']:.0f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="LiDAR parameter ablations on the tune logs.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--eps", nargs="*", type=float, default=[0.2, 0.25, 0.3, 0.4, 0.5, 0.7])
    ap.add_argument("--ground", nargs="*", default=["ransac_naive", "open3d"])
    a = ap.parse_args()
    res = run(load_config(a.config, a.set), a.eps, a.ground)
    out = Path("reports/lidar")
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablations_tune.json").write_text(json.dumps(res, indent=2))
    (out / "ablations_tune.md").write_text(to_markdown(res))
    print(to_markdown(res))


if __name__ == "__main__":
    main()

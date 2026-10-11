"""Per-worker PPE compliance from detector boxes.

The detector finds `Person` boxes and PPE boxes separately (Hardhat, NO-Hardhat, Safety Vest,
NO-Safety Vest, ...). A site rule needs *per-person* answers, so each PPE box is attached to
the person box that covers most of it, if

* at least `min_cover` of the PPE box lies inside the person box, and
* it sits where that item belongs: hard hats in the top `head_band` of the person box,
  vests with their centre inside `torso_band` (fractions of the person height).

Each worker then gets a status per item: "yes", "no" or "unknown" (item not visible, e.g. the
head is cut off). A violation is any "no". Masks are reported but not part of the site rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ppe_data import CLASS_NAMES

CID = {n: i for i, n in enumerate(CLASS_NAMES)}
ITEMS = {
    "hardhat": (CID["Hardhat"], CID["NO-Hardhat"], "head"),
    "vest": (CID["Safety Vest"], CID["NO-Safety Vest"], "torso"),
    "mask": (CID["Mask"], CID["NO-Mask"], "head"),
}
RULE_ITEMS = ("hardhat", "vest")


@dataclass
class Worker:
    box: np.ndarray  # xyxy
    conf: float = 1.0
    status: dict = field(default_factory=lambda: {k: "unknown" for k in ITEMS})
    evidence: dict = field(default_factory=dict)  # item -> (status, conf) of the chosen box

    @property
    def violation(self) -> bool:
        return any(self.status[k] == "no" for k in RULE_ITEMS)

    @property
    def violations(self) -> list[str]:
        return [k for k in RULE_ITEMS if self.status[k] == "no"]


def area(b: np.ndarray) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a: np.ndarray, b: np.ndarray) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def yolo_to_xyxy(labels: np.ndarray, width: int, height: int) -> np.ndarray:
    """(N, 5) [cls, cx, cy, w, h] normalised -> (N, 6) [x1, y1, x2, y2, conf=1, cls] pixels."""
    if len(labels) == 0:
        return np.zeros((0, 6))
    c, cx, cy, w, h = labels.T
    return np.stack(
        [(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height,
         np.ones_like(c), c], axis=1,
    )


def _placed_ok(ppe: np.ndarray, person: np.ndarray, region: str, cfg: dict) -> bool:
    ph = person[3] - person[1]
    if ph <= 0:
        return False
    rel_y = ((ppe[1] + ppe[3]) / 2 - person[1]) / ph
    if region == "head":
        return rel_y <= cfg["head_band"]
    lo, hi = cfg["torso_band"]
    return lo <= rel_y <= hi


def assign(dets: np.ndarray, cfg: dict) -> tuple[list[Worker], dict]:
    """dets: (N, 6) [x1, y1, x2, y2, conf, cls]. Returns workers and orphan PPE counts."""
    dets = np.asarray(dets, dtype=float).reshape(-1, 6)
    workers = [Worker(d[:4], float(d[4])) for d in dets if int(d[5]) == CID["Person"]]
    orphans = {"NO-Hardhat": 0, "NO-Safety Vest": 0, "Hardhat": 0, "Safety Vest": 0}
    for item, (pos_c, neg_c, region) in ITEMS.items():
        for d in dets[np.isin(dets[:, 5], [pos_c, neg_c])]:
            box, conf, is_pos = d[:4], float(d[4]), int(d[5]) == pos_c
            best, best_cover = None, cfg["min_cover"]
            for w in workers:
                cover = intersection(box, w.box) / max(area(box), 1e-9)
                if cover >= best_cover and _placed_ok(box, w.box, region, cfg):
                    best, best_cover = w, cover
            if best is None:
                name = CLASS_NAMES[int(d[5])]
                if name in orphans:
                    orphans[name] += 1
                continue
            status = "yes" if is_pos else "no"
            prev = best.evidence.get(item)
            if prev is None or conf > prev[1]:  # conflicting boxes: trust the confident one
                best.evidence[item] = (status, conf)
                best.status[item] = status
    return workers, orphans


def match_workers(pred: list[Worker], gt: list[Worker], thr: float = 0.5) -> list[tuple[int, int]]:
    """Greedy IoU matching of predicted to ground-truth workers."""
    pairs = sorted(
        ((iou(p.box, g.box), i, j) for i, p in enumerate(pred) for j, g in enumerate(gt)),
        reverse=True,
    )
    used_p, used_g, out = set(), set(), []
    for v, i, j in pairs:
        if v < thr:
            break
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            out.append((i, j))
    return out

"""Dataset downloaders.

* `cs-download-ppe`   — Roboflow "Construction Site Safety" (CC BY 4.0) from Kaggle, using
  the user's own `~/.kaggle/kaggle.json`.
* `cs-download-lidar` — a handful of Argoverse 2 Sensor logs (CC BY-NC-SA 4.0) from the public
  `argoverse` S3 bucket over plain HTTPS (no account needed). We first fetch the small
  `annotations.feather` of every validation log, rank logs by how many people stand close to
  the vehicle, and then download only the LiDAR sweeps we need.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

BUCKET = "https://argoverse.s3.amazonaws.com"
AV2_PREFIX = "datasets/av2/sensor"
KAGGLE_PPE = "snehilsanyal/construction-site-safety-image-dataset-roboflow"
PERSON_CATEGORIES = ("PEDESTRIAN", "OFFICIAL_SIGNALER")
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


# ----------------------------------------------------------------------------- PPE (Kaggle)
def download_ppe(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (out_dir / "css-data" / "train").exists():
        kaggle = shutil.which("kaggle") or str(Path(sys.executable).with_name("kaggle"))
        cmd = [kaggle, "datasets", "download", KAGGLE_PPE, "-p", str(out_dir), "--unzip"]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
    # The Kaggle upload also ships the uploader's own trained YOLOv8 run; we never use it.
    shutil.rmtree(out_dir / "results_yolov8n_100e", ignore_errors=True)
    return out_dir / "css-data"


# ----------------------------------------------------------------------------- Argoverse 2
def s3_list(prefix: str, delimiter: str | None = None) -> tuple[list[str], list[str]]:
    """Return (keys, common_prefixes) under `prefix`, following pagination."""
    keys, prefixes, token = [], [], None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if delimiter:
            params["delimiter"] = delimiter
        if token:
            params["continuation-token"] = token
        url = f"{BUCKET}/?{urllib.parse.urlencode(params)}"
        root = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
        keys += [e.text for e in root.iter(f"{S3_NS}Key")]
        prefixes += [e.text for e in root.iter(f"{S3_NS}Prefix") if e.text != prefix]
        nxt = root.find(f"{S3_NS}NextContinuationToken")
        if nxt is None:
            return keys, prefixes
        token = nxt.text


def fetch(key: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(f"{BUCKET}/{key}", timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.rename(dest)
    return dest


def rank_logs(ann_dir: Path, near_m: float, min_points: int) -> list[dict]:
    """Score each log by person boxes within `near_m` metres of the vehicle."""
    import pandas as pd

    rows = []
    for f in sorted(ann_dir.glob("*.feather")):
        a = pd.read_feather(f)
        p = a[a.category.isin(PERSON_CATEGORIES) & (a.num_interior_pts >= min_points)]
        dist = np.hypot(p.tx_m.to_numpy(), p.ty_m.to_numpy())
        rows.append(
            {
                "log_id": f.stem,
                "near_person_boxes": int((dist < near_m).sum()),
                "person_boxes": len(p),
                "sweeps": int(a.timestamp_ns.nunique()),
            }
        )
    return sorted(rows, key=lambda r: -r["near_person_boxes"])


def download_lidar(
    out_dir: Path,
    split: str = "val",
    n_logs: int = 12,
    stride: int = 5,
    near_m: float = 12.0,
    min_points: int = 5,
    demo_camera: str = "ring_front_center",
    workers: int = 16,
) -> dict:
    ann_dir = out_dir / "annotations"
    _, log_prefixes = s3_list(f"{AV2_PREFIX}/{split}/", delimiter="/")
    log_ids = [p.rstrip("/").split("/")[-1] for p in log_prefixes]
    print(f"{len(log_ids)} {split} logs; fetching annotations ...")
    with ThreadPoolExecutor(workers) as ex:
        list(
            ex.map(
                lambda lid: fetch(
                    f"{AV2_PREFIX}/{split}/{lid}/annotations.feather", ann_dir / f"{lid}.feather"
                ),
                log_ids,
            )
        )
    ranking = rank_logs(ann_dir, near_m, min_points)
    chosen = sorted(r["log_id"] for r in ranking[:n_logs])
    # Alternate the (alphabetically sorted) top logs between a tuning split and a test split,
    # so neither split gets all of the busiest scenes.
    roles = {lid: ("tune" if i % 2 == 0 else "test") for i, lid in enumerate(chosen)}
    test_logs = [r["log_id"] for r in ranking if roles.get(r["log_id"]) == "test"]
    demo_log = test_logs[0]

    jobs: list[tuple[str, Path]] = []
    for lid in chosen:
        base = f"{AV2_PREFIX}/{split}/{lid}"
        keys, _ = s3_list(f"{base}/sensors/lidar/")
        keys = sorted(keys)
        step = 1 if lid == demo_log else stride
        for k in keys[::step]:
            jobs.append((k, out_dir / "logs" / lid / "lidar" / Path(k).name))
        for name in ("egovehicle_SE3_sensor.feather", "intrinsics.feather"):
            jobs.append((f"{base}/calibration/{name}", out_dir / "logs" / lid / name))
        poses = out_dir / "logs" / lid / "poses.feather"
        jobs.append((f"{base}/city_SE3_egovehicle.feather", poses))
    cam_keys, _ = s3_list(f"{AV2_PREFIX}/{split}/{demo_log}/sensors/cameras/{demo_camera}/")
    for k in sorted(cam_keys)[::2]:
        jobs.append((k, out_dir / "logs" / demo_log / "camera" / Path(k).name))

    print(f"downloading {len(jobs)} files for {len(chosen)} logs (demo log {demo_log}) ...")
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda j: fetch(*j), jobs))

    manifest = {
        "source": f"{BUCKET}/{AV2_PREFIX}/{split}/",
        "license": "CC BY-NC-SA 4.0 (Argoverse 2 terms of use)",
        "split": split,
        "selection": {"near_m": near_m, "min_points": min_points, "n_logs": n_logs},
        "stride": stride,
        "demo_log": demo_log,
        "demo_camera": demo_camera,
        "logs": {lid: roles[lid] for lid in chosen},
        "ranking_top": ranking[: max(n_logs, 20)],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def ppe_main() -> None:
    ap = argparse.ArgumentParser(description="Download the Construction Site Safety PPE set.")
    ap.add_argument("--out", type=Path, default=Path("data/ppe"))
    print(download_ppe(ap.parse_args().out))


def lidar_main() -> None:
    ap = argparse.ArgumentParser(description="Download a few Argoverse 2 LiDAR logs.")
    ap.add_argument("--out", type=Path, default=Path("data/av2"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--logs", type=int, default=12)
    ap.add_argument("--stride", type=int, default=5, help="keep every n-th sweep (10 Hz)")
    a = ap.parse_args()
    m = download_lidar(a.out, a.split, a.logs, a.stride)
    print(json.dumps({k: m[k] for k in ("demo_log", "logs")}, indent=2))


if __name__ == "__main__":
    lidar_main()

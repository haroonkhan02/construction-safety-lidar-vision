"""Fine-tune a COCO-pretrained YOLO11 on the PPE dataset (`cs-train-ppe`)."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from .config import load_config, pick_device


def best_epoch(results_csv: Path) -> int:
    """Epoch with the highest validation mAP@0.5:0.95 (the checkpoint Ultralytics keeps)."""
    import pandas as pd

    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    return int(df.loc[df["metrics/mAP50-95(B)"].idxmax(), "epoch"])


def train(cfg: dict) -> Path:
    from ultralytics import YOLO

    p = cfg["ppe"]
    device = pick_device(p["device"])
    # A missing local path falls back to the bare name, which Ultralytics downloads.
    src = p["model"] if Path(p["model"]).exists() else Path(p["model"]).name
    model = YOLO(src)
    t0 = time.time()
    model.train(
        data=p["data"],
        epochs=p["epochs"],
        imgsz=p["imgsz"],
        batch=p["batch"],
        patience=p["patience"],
        seed=p["seed"],
        deterministic=False,
        workers=p["workers"],
        device=device,
        project=str(Path("models/ppe/runs").resolve()),
        name=p["run_name"],
        exist_ok=True,
        plots=True,
    )
    minutes = (time.time() - t0) / 60
    run_dir = Path(model.trainer.save_dir)
    out = Path(p["weights"])
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(run_dir / "weights" / "best.pt", out)
    summary = {
        "model": p["model"],
        "device": device,
        "epochs_requested": p["epochs"],
        "epochs_run": int(model.trainer.epoch) + 1,
        "best_epoch_by_val_map50_95": best_epoch(run_dir / "results.csv"),
        "minutes": round(minutes, 1),
        "run_dir": str(run_dir.resolve().relative_to(Path.cwd())),
        "weights": str(out),
    }
    Path("reports/ppe").mkdir(parents=True, exist_ok=True)
    Path("reports/ppe/train_summary.json").write_text(json.dumps(summary, indent=2))
    shutil.copy(run_dir / "results.csv", "reports/ppe/train_results.csv")
    print(json.dumps(summary, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune YOLO on the PPE dataset.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. ppe.epochs=5")
    a = ap.parse_args()
    train(load_config(a.config, a.set))


if __name__ == "__main__":
    main()

"""Report figures from the JSON results (`cs-plots`)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LIDAR = Path("reports/lidar")
PPE = Path("reports/ppe")


def lidar_figures() -> None:
    res = json.loads((LIDAR / "results_test.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    # 1) precision-recall of the GBT score over all test clusters
    sc_path = LIDAR / "results_test.scores.npz"
    ax = axes[0]
    if sc_path.exists():
        d = np.load(sc_path)
        s, lab = d["scores"], d["labels"]
        keep = lab >= 0
        s, lab = s[keep], lab[keep]
        n_gt = res["gt_person_instances"]
        thr = np.linspace(0.05, 0.99, 60)
        prec = [lab[s >= t].mean() if (s >= t).any() else np.nan for t in thr]
        rec = [lab[s >= t].sum() / n_gt for t in thr]
        ax.plot(rec, prec, color="#2b6cb0")
        t_op = res["thresholds"]["gbt"]
        op = res["detection"]["gbt"]
        ax.scatter([op["recall"]], [op["precision"]], color="#c53030", zorder=3,
                   label=f"GBT @ {t_op}")
        r = res["detection"]["rule"]
        ax.scatter([r["recall"]], [r["precision"]], color="#718096", marker="s", zorder=3,
                   label="rule baseline")
        ax.axvline(res["cluster_recall"], ls="--", color="grey", lw=1,
                   label=f"cluster recall ceiling {res['cluster_recall']:.2f}")
        ax.set(xlabel="recall (of GT people)", ylabel="precision", xlim=(0, 1), ylim=(0, 1.02),
               title="Person detection on test logs")
        ax.legend(fontsize=8, loc="lower left")
    # 2) recall by range band
    ax = axes[1]
    bands = list(res["detection_by_range"]["gbt"])
    x = np.arange(len(bands))
    for k, (name, col) in enumerate((("rule", "#a0aec0"), ("gbt", "#2b6cb0"))):
        ax.bar(x + (k - 0.5) * 0.3, [res["detection_by_range"][name][b]["recall"] for b in bands],
               0.3, label=f"{name} recall", color=col)
    ax.plot(x, [res["cluster_recall_by_range"][b] for b in bands], "k--o", ms=4,
            label="cluster recall")
    ax.set(xticks=x, xticklabels=bands, ylim=(0, 1), ylabel="recall",
           title="Recall by distance from the sensor")
    ax.legend(fontsize=8)
    # 3) zone alert precision / recall
    ax = axes[2]
    labels, P, R = [], [], []
    for k in ("rule", "gbt"):
        for db in ("raw", "debounced"):
            for z in ("red", "amber"):
                a = res["alerts"][k][db][z]
                labels.append(f"{k}\n{db}\n{z}")
                P.append(a["precision"])
                R.append(a["recall"])
    x = np.arange(len(labels))
    ax.bar(x - 0.2, P, 0.4, label="precision", color="#38a169")
    ax.bar(x + 0.2, R, 0.4, label="recall", color="#dd6b20")
    ax.set(xticks=x, ylim=(0, 1.05), title="Frame-level zone alerts (test logs)")
    ax.set_xticklabels(labels, fontsize=7)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(LIDAR / "lidar_results.png", dpi=130)
    plt.close(fig)

    abl = json.loads((LIDAR / "ablations_tune.json").read_text())["rows"]
    eps = [r for r in abl if r["ablation"] == "eps"]
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [r["value"] for r in eps]
    ax.plot(xs, [r["cluster_recall"] for r in eps], "k--o", label="cluster recall")
    ax.plot(xs, [r["gbt_lolo"]["f1"] for r in eps], "-o", color="#2b6cb0",
            label="GBT F1 (leave-one-log-out)")
    ax.plot(xs, [r["rule"]["f1"] for r in eps], "-s", color="#a0aec0", label="rule F1")
    ax.set(xlabel="DBSCAN eps (m)", ylabel="score", ylim=(0, 1),
           title="Clustering radius sweep (tune logs only)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(LIDAR / "eps_sweep.png", dpi=130)
    plt.close(fig)


def ppe_figures() -> None:
    res = json.loads((PPE / "results_test.json").read_text())
    names = list(res["test_all"]["per_class"])
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(names))
    for k, (split, col) in enumerate((("test_all", "#2b6cb0"), ("test_clean", "#dd6b20"))):
        vals = [res[split]["per_class"].get(n, {}).get("ap50", np.nan) for n in names]
        ax.bar(x + (k - 0.5) * 0.38, vals, 0.38, color=col,
               label=f"{split.replace('test_', '')} test ({res[split]['images']} images)")
    ax.set(xticks=x, ylim=(0, 1), ylabel="AP@0.5", title="PPE detector: AP@0.5 per class")
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PPE / "per_class_ap.png", dpi=130)
    plt.close(fig)

    csv = PPE / "train_results.csv"
    if csv.exists():
        import pandas as pd

        df = pd.read_csv(csv)
        df.columns = [c.strip() for c in df.columns]
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for c in ("train/box_loss", "train/cls_loss", "val/box_loss", "val/cls_loss"):
            axes[0].plot(df["epoch"], df[c], label=c)
        axes[0].set(xlabel="epoch", title="Losses")
        axes[0].legend(fontsize=8)
        axes[1].plot(df["epoch"], df["metrics/mAP50(B)"], label="val mAP@0.5")
        axes[1].plot(df["epoch"], df["metrics/mAP50-95(B)"], label="val mAP@0.5:0.95")
        axes[1].set(xlabel="epoch", ylim=(0, 1), title="Validation mAP")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(PPE / "training_curves.png", dpi=130)
        plt.close(fig)


def main() -> None:
    if (LIDAR / "results_test.json").exists():
        lidar_figures()
    if (PPE / "results_test.json").exists():
        ppe_figures()
    print("figures written to reports/")


if __name__ == "__main__":
    main()

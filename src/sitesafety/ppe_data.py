"""PPE dataset checks: class counts, the YOLO data YAML, and a cross-split duplicate audit.

The Roboflow export augments every *training* source image five times, and several source
images come from the same YouTube videos. Two checks catch leakage between splits:

1. identical source names (the part of the file name before `.rf.<hash>`), and
2. near-duplicate pixels: a 64-bit difference hash (dHash) per image; two images with a
   Hamming distance <= `max_hamming` are treated as the same scene.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

CLASS_NAMES = [
    "Hardhat",
    "Mask",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Person",
    "Safety Cone",
    "Safety Vest",
    "machinery",
    "vehicle",
]
SPLITS = ("train", "valid", "test")


def source_name(file_name: str) -> str:
    """'abc_jpg.rf.0123.jpg' -> 'abc_jpg' (the image Roboflow augmented)."""
    return file_name.split(".rf.")[0]


def video_group(file_name: str) -> str | None:
    """Frames cut from one video share a prefix: 'IMG_3103_mp4-17_jpg' -> 'IMG_3103_mp4'."""
    m = re.match(r"(.*?(?:mp4|mov|MP4|MOV)|youtube)[-_]\d+_jpg$", source_name(file_name))
    return m.group(1) if m else None


def clean_mask(names: list[str], train_names: list[str]) -> list[bool]:
    """True for evaluation images that share neither a source image nor a video with train."""
    train_src = {source_name(n) for n in train_names}
    train_vid = {video_group(n) for n in train_names} - {None}
    return [
        source_name(n) not in train_src and video_group(n) not in train_vid for n in names
    ]


def dhash(gray: np.ndarray, size: int = 8) -> int:
    """Difference hash: compare neighbouring pixels of a (size x size+1) thumbnail."""
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count() if hasattr(int, "bit_count") else bin(a ^ b).count("1")


def read_labels(path: Path) -> np.ndarray:
    """YOLO txt -> (N, 5) array [cls, cx, cy, w, h] in normalised coordinates."""
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    return np.loadtxt(path, ndmin=2, dtype=np.float32)


def write_data_yaml(root: Path, out: Path) -> Path:
    data = {
        "path": str(root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": dict(enumerate(CLASS_NAMES)),
    }
    out.write_text(yaml.safe_dump(data, sort_keys=False))
    return out


def near_duplicates(
    hashes_a: dict[str, int], hashes_b: dict[str, int], max_hamming: int
) -> list[tuple[str, str, int]]:
    pairs = []
    keys_b = list(hashes_b)
    arr_b = np.array([hashes_b[k] for k in keys_b], dtype=np.uint64)
    for ka, ha in hashes_a.items():
        x = np.bitwise_xor(arr_b, np.uint64(ha))
        d = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)
        j = int(d.argmin()) if len(d) else -1
        if j >= 0 and d[j] <= max_hamming:
            pairs.append((ka, keys_b[j], int(d[j])))
    return pairs


def audit(root: Path, max_hamming: int = 6) -> dict:
    stats: dict = {"images": {}, "sources": {}, "boxes": {}, "empty_images": {}}
    hashes: dict[str, dict[str, int]] = {}
    sources: dict[str, set[str]] = {}
    for split in SPLITS:
        imgs = sorted((root / split / "images").glob("*.jpg"))
        counts: Counter = Counter()
        empty = 0
        hashes[split] = {}
        for img in imgs:
            lab = read_labels(root / split / "labels" / (img.stem + ".txt"))
            empty += len(lab) == 0
            counts.update(int(c) for c in lab[:, 0])
            gray = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
            hashes[split][img.name] = dhash(gray)
        sources[split] = {source_name(p.name) for p in imgs}
        stats["images"][split] = len(imgs)
        stats["sources"][split] = len(sources[split])
        stats["empty_images"][split] = empty
        stats["boxes"][split] = {CLASS_NAMES[c]: counts[c] for c in range(len(CLASS_NAMES))}
    leaks = {}
    for a, b in (("valid", "train"), ("test", "train"), ("test", "valid")):
        same_name = sorted(sources[a] & sources[b])
        dup = near_duplicates(hashes[a], hashes[b], max_hamming)
        leaks[f"{a}_vs_{b}"] = {
            "same_source_name": len(same_name),
            "near_duplicate_images": len(dup),
            "examples": [list(p) for p in dup[:10]],
        }
    train_names = list(hashes["train"])
    for split in ("valid", "test"):
        names = sorted(hashes[split])
        mask = clean_mask(names, train_names)
        leaks[f"{split}_vs_train"]["shares_video_with_train"] = sum(
            video_group(n) in {video_group(t) for t in train_names} - {None} for n in names
        )
        leaks[f"{split}_vs_train"]["clean_images"] = int(sum(mask))
        stats[f"{split}_clean_images"] = [n for n, ok in zip(names, mask, strict=True) if ok]
    stats["leakage"] = leaks
    stats["dhash_max_hamming"] = max_hamming
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit the PPE dataset and write its YOLO YAML.")
    ap.add_argument("--root", type=Path, default=Path("data/ppe/css-data"))
    ap.add_argument("--out", type=Path, default=Path("reports/ppe/dataset_audit.json"))
    ap.add_argument("--max-hamming", type=int, default=6)
    a = ap.parse_args()
    write_data_yaml(a.root, a.root.parent / "ppe.yaml")
    stats = audit(a.root, a.max_hamming)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: stats[k] for k in ("images", "sources", "leakage")}, indent=2))
    print("test images with no source/video overlap with train:", len(stats["test_clean_images"]))


if __name__ == "__main__":
    main()

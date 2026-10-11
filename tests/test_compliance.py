"""Person <-> PPE association rules and the dataset audit helpers."""

import numpy as np
import pytest

from sitesafety.compliance import CID, assign, iou, match_workers, yolo_to_xyxy
from sitesafety.ppe_data import clean_mask, dhash, near_duplicates, source_name, video_group

CFG = {"min_cover": 0.6, "head_band": 0.45, "torso_band": [0.15, 0.85]}


def det(x0, y0, x1, y1, name, conf=0.9):
    return [x0, y0, x1, y1, conf, CID[name]]


def test_compliant_and_violating_workers():
    dets = np.array([
        det(0, 0, 100, 300, "Person"), det(30, 0, 70, 40, "Hardhat"),
        det(10, 90, 90, 200, "Safety Vest"),
        det(200, 0, 300, 300, "Person"), det(230, 5, 270, 45, "NO-Hardhat"),
        det(210, 90, 290, 200, "Safety Vest"),
    ])
    workers, orphans = assign(dets, CFG)
    assert [w.status["hardhat"] for w in workers] == ["yes", "no"]
    assert [w.violation for w in workers] == [False, True]
    assert workers[1].violations == ["hardhat"]
    assert orphans["NO-Hardhat"] == 0


def test_hardhat_on_the_belt_is_not_worn():
    # A hard hat box at hip height is not on the head -> status stays unknown.
    dets = np.array([det(0, 0, 100, 300, "Person"), det(30, 150, 70, 190, "Hardhat")])
    workers, orphans = assign(dets, CFG)
    assert workers[0].status["hardhat"] == "unknown"
    assert orphans["Hardhat"] == 1


def test_conflicting_boxes_keep_the_confident_one():
    dets = np.array([det(0, 0, 100, 300, "Person"), det(30, 0, 70, 40, "Hardhat", 0.4),
                     det(30, 0, 70, 40, "NO-Hardhat", 0.8)])
    workers, _ = assign(dets, CFG)
    assert workers[0].status["hardhat"] == "no"


def test_ppe_goes_to_the_person_that_covers_it():
    dets = np.array([det(0, 0, 100, 300, "Person"), det(80, 0, 180, 300, "Person"),
                     det(120, 5, 160, 45, "NO-Hardhat")])
    workers, _ = assign(dets, CFG)
    assert [w.status["hardhat"] for w in workers] == ["unknown", "no"]


def test_iou_and_matching():
    a, b = np.array([0, 0, 10, 10]), np.array([5, 0, 15, 10])
    assert iou(a, b) == pytest.approx(50 / 150)
    w = assign(np.array([det(0, 0, 10, 10, "Person"), det(50, 50, 60, 60, "Person")]), CFG)[0]
    g = assign(np.array([det(51, 50, 61, 60, "Person")]), CFG)[0]
    assert match_workers(w, g) == [(1, 0)]


def test_yolo_to_xyxy():
    out = yolo_to_xyxy(np.array([[5, 0.5, 0.5, 0.2, 0.4]]), 100, 200)
    assert out.tolist() == [[40.0, 60.0, 60.0, 140.0, 1.0, 5.0]]


def test_source_and_video_groups():
    assert source_name("IMG_3103_mp4-17_jpg.rf.abc.jpg") == "IMG_3103_mp4-17_jpg"
    assert video_group("IMG_3103_mp4-17_jpg.rf.abc.jpg") == "IMG_3103_mp4"
    assert video_group("youtube-260_jpg.rf.x.jpg") == "youtube"
    assert video_group("000005_jpg.rf.x.jpg") is None
    train = ["IMG_3103_mp4-2_jpg.rf.a.jpg", "photo_1_jpg.rf.b.jpg"]
    test = ["IMG_3103_mp4-9_jpg.rf.c.jpg", "photo_1_jpg.rf.d.jpg", "other_jpg.rf.e.jpg"]
    assert clean_mask(test, train) == [False, False, True]


def test_dhash_finds_near_duplicates():
    rng = np.random.default_rng(0)
    img = (rng.random((64, 64)) * 255).astype(np.uint8)
    shifted = np.clip(img.astype(int) + 3, 0, 255).astype(np.uint8)
    other = (rng.random((64, 64)) * 255).astype(np.uint8)
    ha, hb, hc = dhash(img), dhash(shifted), dhash(other)
    pairs = near_duplicates({"a": ha}, {"b": hb, "c": hc}, max_hamming=6)
    assert pairs and pairs[0][1] == "b"

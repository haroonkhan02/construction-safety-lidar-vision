"""Danger zones around the machine that carries the LiDAR, and debounced alerts.

The sensor platform is treated as a piece of heavy machinery (an excavator or dump truck).
Its body is a rectangle in the machine frame (x forward, y left). The red zone is the body grown
by `red_margin` metres (swing radius / blind spot), the amber zone by `amber_margin`. A zone is
"occupied" when a person's footprint centre lies inside it; red takes priority over amber.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .geometry import rect_contains

ZONE_ORDER = ("red", "amber")


def machine_box(z: dict, margin: float = 0.0) -> tuple[float, float, float, float]:
    cx, hl, hw = z["machine_center_x"], z["machine_half_length"], z["machine_half_width"]
    return (cx - hl - margin, cx + hl + margin, -hw - margin, hw + margin)


def zone_rects(z: dict) -> dict[str, tuple[float, float, float, float]]:
    return {"red": machine_box(z, z["red_margin"]), "amber": machine_box(z, z["amber_margin"])}


def assign_zones(xy: np.ndarray, z: dict) -> list[str | None]:
    """Most severe zone for each person position (None = outside every zone)."""
    rects = zone_rects(z)
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    out: list[str | None] = [None] * len(xy)
    for name in reversed(ZONE_ORDER):  # amber first, so red overwrites it
        for i in np.flatnonzero(rect_contains(xy, rects[name])):
            out[i] = name
    return out


def occupancy(xy: np.ndarray, z: dict) -> dict[str, bool]:
    """Is each zone occupied? Red people also count for amber (the zones are nested)."""
    zones = assign_zones(xy, z)
    return {"red": "red" in zones, "amber": any(v is not None for v in zones)}


def distance_to_machine(xy: np.ndarray, z: dict) -> np.ndarray:
    """Horizontal distance from each point to the machine body rectangle (0 inside)."""
    x0, x1, y0, y1 = machine_box(z)
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    dx = np.maximum.reduce([x0 - xy[:, 0], np.zeros(len(xy)), xy[:, 0] - x1])
    dy = np.maximum.reduce([y0 - xy[:, 1], np.zeros(len(xy)), xy[:, 1] - y1])
    return np.hypot(dx, dy)


class Debouncer:
    """Raise an alert only after `n` consecutive occupied frames; clear on the first empty one.

    One-frame false positives (a pole classified as a person for a single sweep) are the main
    nuisance-alarm source; requiring persistence trades a short delay for far fewer alarms.
    """

    def __init__(self, n: int):
        self.n = max(1, n)
        self.history: dict[str, deque] = {k: deque(maxlen=self.n) for k in ZONE_ORDER}

    def update(self, occupied: dict[str, bool]) -> dict[str, bool]:
        out = {}
        for k in ZONE_ORDER:
            self.history[k].append(bool(occupied.get(k, False)))
            h = self.history[k]
            out[k] = len(h) == self.n and all(h)
        return out

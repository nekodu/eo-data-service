"""Synthetic cloud-mask cube for tests and the CI smoke run. Never mixed with real data:
the cube carries attrs["synthetic"] = 1 and the API reports it."""

from __future__ import annotations

import numpy as np

from .cube import CLOUD, NODATA, frames_to_dataset


def synthetic_dataset(n_time: int = 24, ny: int = 96, nx: int = 128, seed: int = 0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:ny, 0:nx]
    n_blobs = 6
    cy = rng.uniform(0, ny, n_blobs)
    cx = rng.uniform(0, nx, n_blobs)
    r = rng.uniform(6, 16, n_blobs)
    vy, vx = 0.4, 1.2  # same "wind" for all blobs, so the past predicts the future
    t0 = np.datetime64("2026-01-01T00:00")
    times, frames = [], []
    for t in range(n_time):
        f = np.zeros((ny, nx), dtype=np.uint8)
        for i in range(n_blobs):
            d2 = (yy - (cy[i] + vy * t) % ny) ** 2 + (xx - (cx[i] + vx * t) % nx) ** 2
            f[d2 < (r[i] + rng.normal(0, 0.8)) ** 2] = CLOUD
        noise = rng.random((ny, nx)) < 0.01
        f[noise] ^= CLOUD
        f[:2, :] = NODATA
        frames.append(f)
        times.append(t0 + np.timedelta64(15 * t, "m"))
    attrs = {"source": "synthetic", "synthetic": 1, "step_minutes": 15}
    return frames_to_dataset(times, frames, attrs)

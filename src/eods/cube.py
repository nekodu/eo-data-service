"""Cloud-mask data cube stored as Zarr: one uint8 variable `cloud` over (time, y, x).

Values: 0 clear, 1 cloudy, 255 no data. y/x are pixel rows/columns of the
source grid inside the cropped window, so the cube stays in native geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

CLEAR, CLOUD, NODATA = 0, 1, 255
CHUNKS = {"time": 8, "y": 256, "x": 256}


@dataclass(frozen=True)
class Window:
    y0: int
    y1: int
    x0: int
    x1: int

    def __post_init__(self) -> None:
        if not (0 <= self.y0 < self.y1 and 0 <= self.x0 < self.x1):
            raise ValueError("window must satisfy 0 <= y0 < y1 and 0 <= x0 < x1")

    @property
    def cells(self) -> int:
        return (self.y1 - self.y0) * (self.x1 - self.x0)


def frames_to_dataset(
    times: list[np.datetime64], frames: list[np.ndarray], attrs: dict
) -> xr.Dataset:
    if len(times) != len(frames):
        raise ValueError("times and frames differ in length")
    stack = np.stack(frames).astype(np.uint8)
    ny, nx = stack.shape[1:]
    ds = xr.Dataset(
        {"cloud": (("time", "y", "x"), stack)},
        coords={
            "time": np.array(times, dtype="datetime64[ns]"),
            "y": np.arange(ny, dtype=np.int32),
            "x": np.arange(nx, dtype=np.int32),
        },
        attrs=attrs,
    )
    ds["cloud"].attrs.update(
        {
            "flag_values": [CLEAR, CLOUD],
            "flag_meanings": "clear cloudy",
            "_FillValue_note": "255 = no data",
        }
    )
    return ds


def write_cube(ds: xr.Dataset, path: Path, append: bool = False) -> None:
    ds = ds.sortby("time")
    if append and Path(path).exists():
        existing = xr.open_zarr(path)
        new_times = ~ds.time.isin(existing.time.values)
        ds = ds.sel(time=ds.time[new_times])
        if ds.sizes["time"] == 0:
            return
        if ds.time.min() <= existing.time.max():
            raise ValueError("appending would put time out of order; rebuild the cube instead")
        ds.to_zarr(path, mode="a", append_dim="time", consolidated=True)
    else:
        enc = {"cloud": {"chunks": tuple(min(CHUNKS[d], ds.sizes[d]) for d in ("time", "y", "x"))}}
        ds.to_zarr(path, mode="w", encoding=enc, consolidated=True)


def open_cube(path: Path) -> xr.Dataset:
    return xr.open_zarr(path, consolidated=True)


def describe(ds: xr.Dataset) -> dict:
    t = ds.time.values
    return {
        "variable": "cloud",
        "encoding": {"clear": CLEAR, "cloudy": CLOUD, "no_data": NODATA},
        "shape": {k: int(v) for k, v in ds.sizes.items()},
        "time_start": str(t.min()) if len(t) else None,
        "time_end": str(t.max()) if len(t) else None,
        "time_steps": int(len(t)),
        "chunks": {k: int(v[0]) for k, v in ds["cloud"].chunksizes.items()}
        if ds["cloud"].chunks
        else None,
        "attrs": {k: v for k, v in ds.attrs.items() if isinstance(v, str | int | float | bool)},
    }


def select(ds: xr.Dataset, window: Window, start=None, end=None) -> xr.DataArray:
    if window.y1 > ds.sizes["y"] or window.x1 > ds.sizes["x"]:
        raise ValueError(f"window outside cube ({ds.sizes['y']}x{ds.sizes['x']})")
    da = ds["cloud"].isel(y=slice(window.y0, window.y1), x=slice(window.x0, window.x1))
    if start is not None or end is not None:
        da = da.sel(time=slice(start, end))
    return da


def cloud_fraction(da: xr.DataArray) -> list[dict]:
    """Per time step: share of valid pixels that are cloudy, and share of pixels that are valid."""
    v = da.values
    valid = v != NODATA
    n_valid = valid.sum(axis=(1, 2))
    n_cloud = (v == CLOUD).sum(axis=(1, 2))
    total = v.shape[1] * v.shape[2]
    out = []
    for t, nv, nc in zip(da.time.values, n_valid, n_cloud, strict=True):
        out.append(
            {
                "time": str(t),
                "cloud_fraction": float(nc / nv) if nv else None,
                "valid_fraction": float(nv / total),
            }
        )
    return out

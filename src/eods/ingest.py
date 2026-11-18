"""Fetch MSG cloud-mask slots from the EUMETSAT Data Store and decode them into the cube.

Credentials come from EUMETSAT_CONSUMER_KEY / EUMETSAT_CONSUMER_SECRET and never leave
this process except to the Data Store token endpoint.
"""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .cube import CLEAR, CLOUD, NODATA, frames_to_dataset, write_cube

COLLECTION = "EO:EUM:DAT:MSG:CLM"
# CLM codes: 0 clear over water, 1 clear over land, 2 cloud, 3 not processed / off disk
CLM_TO_CUBE = np.full(256, NODATA, dtype=np.uint8)
CLM_TO_CUBE[[0, 1]] = CLEAR
CLM_TO_CUBE[2] = CLOUD

# lat/lon boxes; the pixel window is derived from the grid's own coordinates on first build
WINDOWS = {"europe": (35.0, 62.0, -12.0, 30.0)}


def _token():
    import eumdac

    key = os.environ.get("EUMETSAT_CONSUMER_KEY")
    secret = os.environ.get("EUMETSAT_CONSUMER_SECRET")
    if not key or not secret:
        raise SystemExit(
            "set EUMETSAT_CONSUMER_KEY and EUMETSAT_CONSUMER_SECRET (see .env.example)"
        )
    return eumdac.AccessToken((key, secret))


def fetch(start: str, end: str, raw: Path, every: int = 15) -> dict:
    import eumdac

    raw.mkdir(parents=True, exist_ok=True)
    store = eumdac.DataStore(_token())
    coll = store.get_collection(COLLECTION)
    t0, t1 = datetime.fromisoformat(start), datetime.fromisoformat(end)
    got, skipped = 0, 0
    for product in coll.search(dtstart=t0, dtend=t1):
        ts = product.sensing_end
        if (ts.hour * 60 + ts.minute) % every >= 5:  # keep one slot per `every` minutes
            continue
        target = raw / f"{product}.zip"
        if target.exists():
            skipped += 1
            continue
        tmp = target.with_suffix(".part")
        with product.open() as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst)
        tmp.replace(target)
        got += 1
    return {
        "collection": COLLECTION,
        "downloaded": got,
        "already_present": skipped,
        "dir": str(raw),
    }


def _read_grib(blob: bytes):
    import eccodes

    gid = eccodes.codes_new_from_message(blob)
    try:
        ni, nj = eccodes.codes_get(gid, "Ni"), eccodes.codes_get(gid, "Nj")
        vals = eccodes.codes_get_values(gid).reshape(nj, ni)
        meta = {
            k: eccodes.codes_get(gid, k)
            for k in ("gridType", "iScansNegatively", "jScansPositively", "dataDate", "dataTime")
        }
        return gid, vals, meta
    except Exception:
        eccodes.codes_release(gid)
        raise


def _grib_from_zip(path: Path) -> bytes:
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.lower().endswith((".grb", ".grb2", ".grib"))]
        if len(names) != 1:
            raise ValueError(f"{path.name}: expected one GRIB file, found {names}")
        return z.read(names[0])


def _slot_time(path: Path) -> np.datetime64:
    # product ids look like MSG4-SEVI-MSGCLMK-0100-0100-20261117121500.000000000Z-NA
    stamp = next(p for p in path.stem.split("-") if p[:8].isdigit() and len(p) >= 14)[:14]
    t = datetime.strptime(stamp, "%Y%m%d%H%M%S")
    t = t - timedelta(minutes=t.minute % 15, seconds=t.second)  # round to the nominal slot
    return np.datetime64(t, "m")


def pixel_window(first_zip: Path, box: tuple[float, float, float, float]) -> dict:
    import eccodes

    gid, vals, meta = _read_grib(_grib_from_zip(first_zip))
    try:
        lat = eccodes.codes_get_array(gid, "latitudes").reshape(vals.shape)
        lon = eccodes.codes_get_array(gid, "longitudes").reshape(vals.shape)
    finally:
        eccodes.codes_release(gid)
    lon = np.where(lon > 180, lon - 360, lon)
    la0, la1, lo0, lo1 = box
    inside = (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)
    rows, cols = np.flatnonzero(inside.any(axis=1)), np.flatnonzero(inside.any(axis=0))
    return {
        "y0": int(rows.min()),
        "y1": int(rows.max()) + 1,
        "x0": int(cols.min()),
        "x1": int(cols.max()) + 1,
        "lat_lon_box": list(box),
        "grid": meta,
    }


def build_cube(raw: Path, out: Path, window: str, append: bool = False) -> dict:
    import eccodes

    zips = sorted(raw.glob("*.zip"), key=_slot_time)
    if not zips:
        raise SystemExit(f"no downloaded slots in {raw}")
    win_file = raw / "window.json"
    if window in WINDOWS:
        if not win_file.exists():
            win_file.write_text(json.dumps(pixel_window(zips[0], WINDOWS[window]), indent=2))
        w = json.loads(win_file.read_text())
    else:
        y0, y1, x0, x1 = (int(v) for v in window.split(","))
        w = {"y0": y0, "y1": y1, "x0": x0, "x1": x1}
    if append and out.exists():
        from .cube import open_cube

        latest = open_cube(out).time.values.max()
        zips = [z for z in zips if _slot_time(z) > latest]
        if not zips:
            return {"slots": 0, "failed": [], "window": w, "out": str(out), "note": "up to date"}
    times, frames, bad = [], [], []
    for z in zips:
        try:
            gid, vals, _ = _read_grib(_grib_from_zip(z))
            eccodes.codes_release(gid)
        except Exception as e:  # noqa: BLE001 - a corrupt slot shouldn't stop the build
            bad.append(f"{z.name}: {e}")
            continue
        crop = vals[w["y0"] : w["y1"], w["x0"] : w["x1"]]
        codes = np.where(np.isfinite(crop), crop, 3).astype(np.int64).clip(0, 255)
        frames.append(CLM_TO_CUBE[codes])
        times.append(_slot_time(z))
    attrs = {
        "source": f"EUMETSAT Data Store {COLLECTION}",
        "synthetic": 0,
        "step_minutes": 15,
        "window": json.dumps(w),
    }
    write_cube(frames_to_dataset(times, frames, attrs), out, append=append)
    return {"slots": len(frames), "failed": bad, "window": w, "out": str(out)}

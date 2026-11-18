import argparse
import json
from pathlib import Path

import numpy as np


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="eods")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="write a synthetic cube (tests, CI)")
    s.add_argument("--out", type=Path, default=Path("data/cube.zarr"))
    s.add_argument("--steps", type=int, default=48)

    f = sub.add_parser("fetch", help="download cloud-mask slots from the EUMETSAT Data Store")
    f.add_argument("--start", required=True)
    f.add_argument("--end", required=True)
    f.add_argument("--every", type=int, default=15, help="minutes between slots to keep")
    f.add_argument("--raw", type=Path, default=Path("data/raw"))

    b = sub.add_parser("build", help="decode downloaded slots into the Zarr cube")
    b.add_argument("--raw", type=Path, default=Path("data/raw"))
    b.add_argument("--out", type=Path, default=Path("data/cube.zarr"))
    b.add_argument("--window", default="europe", help="named window or y0,y1,x0,x1")
    b.add_argument("--append", action="store_true")

    t = sub.add_parser("train", help="train the nowcast model and report against persistence")
    t.add_argument("--cube", type=Path, default=Path("data/cube.zarr"))
    t.add_argument("--model", type=Path, default=Path("models/nowcast.joblib"))
    t.add_argument("--history", type=int, default=4)
    t.add_argument("--horizon", type=int, default=4)

    c = sub.add_parser("compare", help="persistence vs gradient boosting vs PyTorch CNN")
    c.add_argument("--cube", type=Path, default=Path("data/cube.zarr"))
    c.add_argument("--out", type=Path, default=Path("models/cnn.pt"))
    c.add_argument("--steps", type=int, default=400)

    a = p.parse_args(argv)
    if a.cmd == "compare":
        from . import ml_torch
        from .cube import open_cube

        ds = open_cube(a.cube)
        net, _, result = ml_torch.compare(
            ds["cloud"].values,
            ds.time.values,
            synthetic=bool(ds.attrs.get("synthetic", 0)),
            tcfg=ml_torch.TorchConfig(steps=a.steps),
        )
        ml_torch.save(net, result, a.out)
        print(
            f"test {result.test_period[0]} .. {result.test_period[1]}, {result.n_pixels} pixels,"
            f" synthetic={result.synthetic}"
        )
        print(ml_torch.markdown(result))
        return
    if a.cmd == "synth":
        from .cube import write_cube
        from .synth import synthetic_dataset

        write_cube(synthetic_dataset(n_time=a.steps), a.out)
        print(f"synthetic cube -> {a.out}")
    elif a.cmd == "fetch":
        from .ingest import fetch

        print(json.dumps(fetch(a.start, a.end, a.raw, every=a.every), indent=2))
    elif a.cmd == "build":
        from .ingest import build_cube

        print(json.dumps(build_cube(a.raw, a.out, a.window, append=a.append), indent=2))
    elif a.cmd == "train":
        from . import ml
        from .cube import open_cube

        ds = open_cube(a.cube)
        cfg = ml.NowcastConfig(history=a.history, horizon=a.horizon)
        clf, report = ml.train(
            ds["cloud"].values, ds.time.values, cfg, synthetic=bool(ds.attrs.get("synthetic", 0))
        )
        ml.save(clf, report, a.model)
        out = {
            "model": report.model.__dict__,
            "persistence": report.persistence.__dict__,
            "test_period": report.test_period,
        }
        print(
            json.dumps(
                out, indent=2, default=lambda o: o if not isinstance(o, np.generic) else o.item()
            )
        )


if __name__ == "__main__":
    main()

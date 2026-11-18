import numpy as np
import pytest

from eods import ml
from eods.cube import CLOUD, NODATA, Window, cloud_fraction, open_cube, select, write_cube
from eods.synth import synthetic_dataset


def test_roundtrip_and_append(tmp_path):
    ds = synthetic_dataset(n_time=20)
    p = tmp_path / "c.zarr"
    write_cube(ds.isel(time=slice(0, 12)), p)
    write_cube(ds.isel(time=slice(8, 20)), p, append=True)  # overlap is skipped, not duplicated
    back = open_cube(p)
    assert back.sizes["time"] == 20
    np.testing.assert_array_equal(back["cloud"].values, ds["cloud"].values)


def test_append_out_of_order_refused(tmp_path):
    ds = synthetic_dataset(n_time=20)
    p = tmp_path / "c.zarr"
    write_cube(ds.isel(time=slice(10, 20)), p)
    with pytest.raises(ValueError):
        write_cube(ds.isel(time=slice(0, 5)), p, append=True)


def test_cloud_fraction_ignores_nodata():
    ds = synthetic_dataset(n_time=3)
    ds["cloud"][:] = CLOUD
    ds["cloud"][:, :2, :] = NODATA
    s = cloud_fraction(select(ds, Window(0, 10, 0, 10)))
    assert s[0]["cloud_fraction"] == 1.0
    assert s[0]["valid_fraction"] == pytest.approx(0.8)


def test_window_validation():
    with pytest.raises(ValueError):
        Window(5, 5, 0, 1)


def test_time_split_has_no_overlap():
    ds = synthetic_dataset(n_time=48)
    cfg = ml.NowcastConfig(samples_per_frame=300)
    _, rep = ml.train(ds["cloud"].values, ds.time.values, cfg, synthetic=True)
    assert np.datetime64(rep.train_period[1]) < np.datetime64(rep.test_period[0]) + np.timedelta64(
        15 * cfg.history, "m"
    )
    assert np.datetime64(rep.train_period[0]) < np.datetime64(rep.test_period[0])


def test_model_beats_persistence_on_moving_clouds():
    # synthetic clouds drift with a fixed wind, so the past carries information persistence ignores
    ds = synthetic_dataset(n_time=60, seed=3)
    _, rep = ml.train(
        ds["cloud"].values, ds.time.values, ml.NowcastConfig(samples_per_frame=1500), True
    )
    assert rep.model.csi > rep.persistence.csi


def test_predict_map_shape_and_nodata():
    ds = synthetic_dataset(n_time=40)
    clf, rep = ml.train(
        ds["cloud"].values, ds.time.values, ml.NowcastConfig(samples_per_frame=300), True
    )
    bundle = {"model": clf, "config": rep.config.__dict__}
    past = ds["cloud"].values[-4:]
    prob = ml.predict_map(bundle, past)
    assert prob.shape == past.shape[1:]
    assert np.isnan(prob[0, 0])  # top rows are no-data in the synthetic cube
    with pytest.raises(ValueError):
        ml.predict_map(bundle, past[-3:])

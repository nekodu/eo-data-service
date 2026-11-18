import pytest

torch = pytest.importorskip("torch")

from eods import ml, ml_torch  # noqa: E402
from eods.synth import synthetic_dataset  # noqa: E402


def test_net_keeps_spatial_shape():
    net = ml_torch.NowcastNet(history=4, width=8)
    out = net(torch.zeros(2, 5, 40, 56))
    assert out.shape == (2, 40, 56)


def test_compare_scores_all_models_on_same_pixels():
    ds = synthetic_dataset(n_time=48, seed=1)
    _, _, r = ml_torch.compare(
        ds["cloud"].values,
        ds.time.values,
        synthetic=True,
        cfg=ml.NowcastConfig(samples_per_frame=500),
        tcfg=ml_torch.TorchConfig(steps=60, width=16),
    )
    assert r.persistence.n == r.gradient_boosting.n == r.cnn.n == r.n_pixels > 0
    assert 0 <= r.cnn.brier <= 1
    assert "| cnn |" in ml_torch.markdown(r)


def test_cnn_learns_drift_better_than_persistence():
    ds = synthetic_dataset(n_time=72, seed=3)
    _, _, r = ml_torch.compare(
        ds["cloud"].values,
        ds.time.values,
        synthetic=True,
        cfg=ml.NowcastConfig(samples_per_frame=500),
        tcfg=ml_torch.TorchConfig(steps=300),
    )
    assert r.cnn.csi > r.persistence.csi


def test_save_writes_weights_only(tmp_path):
    net = ml_torch.NowcastNet(width=8)
    ds = synthetic_dataset(n_time=48)
    _, _, r = ml_torch.compare(
        ds["cloud"].values,
        ds.time.values,
        synthetic=True,
        cfg=ml.NowcastConfig(samples_per_frame=300),
        tcfg=ml_torch.TorchConfig(steps=5, width=8),
    )
    ml_torch.save(net, r, tmp_path / "cnn.pt")
    state = torch.load(tmp_path / "cnn.pt", weights_only=True)
    assert all(isinstance(v, torch.Tensor) for v in state.values())
    assert (tmp_path / "cnn.json").stat().st_size > 0

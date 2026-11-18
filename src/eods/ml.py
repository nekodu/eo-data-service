"""Cloud nowcasting baseline: will this pixel be cloudy `horizon` steps from now?

Features per pixel come from the last `history` frames only (own past values plus
neighbourhood cloud cover at two scales). The reference to beat is persistence,
i.e. "the cloud mask stays as it is now". Train/test is split by time, never by pixel,
because neighbouring pixels in the same frame are almost copies of each other.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
from scipy.ndimage import uniform_filter
from sklearn.ensemble import HistGradientBoostingClassifier

from .cube import CLOUD, NODATA

NEIGHBOURHOODS = (5, 21)


@dataclass
class NowcastConfig:
    history: int = 4
    horizon: int = 4
    samples_per_frame: int = 4000
    seed: int = 0


@dataclass
class Scores:
    n: int
    accuracy: float
    f1_cloud: float
    csi: float
    brier: float | None = None


@dataclass
class Report:
    config: NowcastConfig
    train_period: tuple[str, str]
    test_period: tuple[str, str]
    model: Scores
    persistence: Scores
    synthetic: bool
    notes: list[str] = field(default_factory=list)


def feature_names(history: int) -> list[str]:
    names = [f"cloud_t-{i}" for i in reversed(range(history))]
    for s in NEIGHBOURHOODS:
        names += [f"cover{s}_t-0", f"cover{s}_trend"]
    names += ["valid_share"]
    return names


def _frame_features(stack: np.ndarray) -> np.ndarray:
    """stack: (history, y, x) uint8 -> (y, x, n_features) float32."""
    h = stack.shape[0]
    valid = stack != NODATA
    cloud = (stack == CLOUD).astype(np.float32)
    feats = [cloud[i] for i in range(h)]
    for s in NEIGHBOURHOODS:
        # cloud cover among valid pixels around each pixel, now and its change over the history
        def cover(i: int, s: int = s) -> np.ndarray:
            num = uniform_filter(cloud[i] * valid[i], size=s, mode="nearest")
            den = uniform_filter(valid[i].astype(np.float32), size=s, mode="nearest")
            return np.divide(num, den, out=np.zeros_like(num), where=den > 0)

        now, first = cover(h - 1), cover(0)
        feats += [now, now - first]
    feats.append(valid.mean(axis=0).astype(np.float32))
    return np.stack(feats, axis=-1)


def build_samples(cube: np.ndarray, cfg: NowcastConfig, t_indices, rng=None, sample=True):
    """Return X, y, persistence prediction for target frames t+horizon, t in t_indices."""
    rng = rng or np.random.default_rng(cfg.seed)
    xs, ys, ps = [], [], []
    for t in t_indices:
        past = cube[t - cfg.history + 1 : t + 1]
        target = cube[t + cfg.horizon]
        f = _frame_features(past)
        ok = (target != NODATA) & (past[-1] != NODATA)
        idx = np.flatnonzero(ok)
        if sample and len(idx) > cfg.samples_per_frame:
            idx = rng.choice(idx, cfg.samples_per_frame, replace=False)
        xs.append(f.reshape(-1, f.shape[-1])[idx])
        ys.append((target.reshape(-1)[idx] == CLOUD).astype(np.int8))
        ps.append((past[-1].reshape(-1)[idx] == CLOUD).astype(np.int8))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ps)


def score(y: np.ndarray, pred: np.ndarray, prob: np.ndarray | None = None) -> Scores:
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    acc = float((pred == y).mean())
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    csi = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    brier = float(np.mean((prob - y) ** 2)) if prob is not None else None
    return Scores(n=int(len(y)), accuracy=acc, f1_cloud=f1, csi=csi, brier=brier)


def train(
    cube: np.ndarray, times: np.ndarray, cfg: NowcastConfig, synthetic: bool, test_share=0.25
):
    targets = np.arange(cfg.history - 1, cube.shape[0] - cfg.horizon)
    if len(targets) < 8:
        raise ValueError("not enough time steps for history + horizon + a test block")
    cut = int(len(targets) * (1 - test_share))
    # leave a gap of `horizon` frames so no test target is also a training input
    train_t, test_t = targets[: cut - cfg.horizon], targets[cut:]
    rng = np.random.default_rng(cfg.seed)
    X, y, _ = build_samples(cube, cfg, train_t, rng)
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=cfg.seed)
    clf.fit(X, y)
    Xt, yt, pt = build_samples(cube, cfg, test_t, rng)
    prob = clf.predict_proba(Xt)[:, 1]
    report = Report(
        config=cfg,
        train_period=(str(times[train_t[0]]), str(times[train_t[-1] + cfg.horizon])),
        test_period=(str(times[test_t[0]]), str(times[test_t[-1] + cfg.horizon])),
        model=score(yt, (prob >= 0.5).astype(np.int8), prob),
        persistence=score(yt, pt, pt.astype(np.float32)),
        synthetic=synthetic,
    )
    return clf, report


def save(clf, report: Report, model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": clf,
            "config": asdict(report.config),
            "features": feature_names(report.config.history),
        },
        model_path,
    )
    model_path.with_suffix(".json").write_text(json.dumps(asdict(report), indent=2))


def load(model_path: Path) -> dict:
    # joblib unpickles: only ever load a model file this service wrote itself
    return joblib.load(model_path)


def predict_map(bundle: dict, past: np.ndarray) -> np.ndarray:
    """past: (history, y, x) -> probability map (y, x); NaN where the latest frame has no data."""
    cfg = bundle["config"]
    if past.shape[0] != cfg["history"]:
        raise ValueError(f"need exactly {cfg['history']} frames")
    f = _frame_features(past)
    prob = bundle["model"].predict_proba(f.reshape(-1, f.shape[-1]))[:, 1].reshape(past.shape[1:])
    prob[past[-1] == NODATA] = np.nan
    return prob

"""Small fully convolutional nowcasting net in PyTorch, compared with persistence and the
gradient-boosting baseline on exactly the same test pixels.

Input per sample: the last `history` cloud masks (1 cloudy, 0 clear or no data) plus the
valid-data mask of the newest one. Output: cloud logit per pixel `horizon` slots ahead.
Dilated convolutions give a receptive field of about 31 pixels, enough to see a cloud
edge move several pixels per slot without pooling away the resolution.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import ml
from .cube import CLOUD, NODATA


class NowcastNet(nn.Module):
    def __init__(self, history: int = 4, width: int = 32):
        super().__init__()
        layers, ch = [], history + 1
        for dil in (1, 2, 4, 8):
            layers += [nn.Conv2d(ch, width, 3, padding=dil, dilation=dil), nn.ReLU(inplace=True)]
            ch = width
        layers.append(nn.Conv2d(width, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


@dataclass
class TorchConfig:
    width: int = 32
    steps: int = 400
    batch: int = 16
    crop: int = 64
    lr: float = 1e-3
    seed: int = 0


def _inputs(cube: np.ndarray, t: int, history: int) -> np.ndarray:
    past = cube[t - history + 1 : t + 1]
    x = (past == CLOUD).astype(np.float32)
    valid = (past[-1] != NODATA).astype(np.float32)[None]
    return np.concatenate([x, valid], axis=0)


def _target(cube: np.ndarray, t: int, cfg: ml.NowcastConfig):
    tgt = cube[t + cfg.horizon]
    y = (tgt == CLOUD).astype(np.float32)
    mask = ((tgt != NODATA) & (cube[t] != NODATA)).astype(np.float32)
    return y, mask


def train_net(cube: np.ndarray, cfg: ml.NowcastConfig, tcfg: TorchConfig, train_t) -> NowcastNet:
    torch.manual_seed(tcfg.seed)
    rng = np.random.default_rng(tcfg.seed)
    net = NowcastNet(cfg.history, tcfg.width)
    opt = torch.optim.Adam(net.parameters(), lr=tcfg.lr)
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    ny, nx = cube.shape[1:]
    c = min(tcfg.crop, ny, nx)
    net.train()
    for _ in range(tcfg.steps):
        xs, ys, ms = [], [], []
        for t in rng.choice(train_t, tcfg.batch):
            y0, x0 = rng.integers(0, ny - c + 1), rng.integers(0, nx - c + 1)
            x = _inputs(cube, t, cfg.history)[:, y0 : y0 + c, x0 : x0 + c]
            y, m = _target(cube, t, cfg)
            xs.append(x)
            ys.append(y[y0 : y0 + c, x0 : x0 + c])
            ms.append(m[y0 : y0 + c, x0 : x0 + c])
        x, y, m = (torch.from_numpy(np.stack(a)) for a in (xs, ys, ms))
        loss = (lossf(net(x), y) * m).sum() / m.sum().clamp(min=1)
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    return net


@torch.no_grad()
def predict_prob(net: NowcastNet, cube: np.ndarray, t: int, history: int) -> np.ndarray:
    x = torch.from_numpy(_inputs(cube, t, history)[None])
    return torch.sigmoid(net(x))[0].numpy()


@dataclass
class Comparison:
    test_period: tuple[str, str]
    n_pixels: int
    persistence: ml.Scores
    gradient_boosting: ml.Scores
    cnn: ml.Scores
    synthetic: bool


def compare(cube: np.ndarray, times: np.ndarray, synthetic: bool, cfg=None, tcfg=None):
    """Train both models on the same training block, score all three on every valid test pixel."""
    cfg = cfg or ml.NowcastConfig()
    tcfg = tcfg or TorchConfig()
    train_t, test_t = ml.split_targets(cube.shape[0], cfg)
    clf, _ = ml.train(cube, times, cfg, synthetic)
    net = train_net(cube, cfg, tcfg, train_t)

    ys, ps, gb, cn = [], [], [], []
    for t in test_t:
        y, m = _target(cube, t, cfg)
        keep = m.astype(bool)
        ys.append(y[keep].astype(np.int8))
        ps.append((cube[t] == CLOUD)[keep].astype(np.int8))
        bundle = {"model": clf, "config": asdict(cfg)}
        gb.append(np.nan_to_num(ml.predict_map(bundle, cube[t - cfg.history + 1 : t + 1]))[keep])
        cn.append(predict_prob(net, cube, t, cfg.history)[keep])
    y, p, g, c = (np.concatenate(a) for a in (ys, ps, gb, cn))
    result = Comparison(
        test_period=(str(times[test_t[0]]), str(times[test_t[-1] + cfg.horizon])),
        n_pixels=int(len(y)),
        persistence=ml.score(y, p, p.astype(np.float32)),
        gradient_boosting=ml.score(y, (g >= 0.5).astype(np.int8), g),
        cnn=ml.score(y, (c >= 0.5).astype(np.int8), c),
        synthetic=synthetic,
    )
    return net, clf, result


def save(net: NowcastNet, result: Comparison, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # state_dict only, so loading never unpickles arbitrary objects
    torch.save(net.state_dict(), path)
    path.with_suffix(".json").write_text(json.dumps(asdict(result), indent=2))


def markdown(result: Comparison) -> str:
    rows = ["| | accuracy | F1 (cloud) | CSI | Brier |", "|---|---|---|---|---|"]
    for name in ("persistence", "gradient_boosting", "cnn"):
        s: ml.Scores = getattr(result, name)
        cells = (s.accuracy, s.f1_cloud, s.csi, s.brier)
        rows.append(f"| {name.replace('_', ' ')} | " + " | ".join(f"{v:.3f}" for v in cells) + " |")
    return "\n".join(rows)

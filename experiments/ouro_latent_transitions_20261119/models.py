"""Unsupervised state compressors and natural-only transition estimators."""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.utils.extmath import randomized_svd
import torch
from torch import nn

SEED = 20261119
WIDTHS = (32, 64, 128)
ENCODERS = ("random", "pca", "structured", "auto")
METHODS = ("identity", "ridge", "rank8", "rank24", "mlp", "dictionary")


def tensor(x):
    return torch.as_tensor(x, dtype=torch.float32)


def numpy_prediction(module, x):
    with torch.inference_mode():
        return module(tensor(x)).numpy()


def frame(x, dim, random=False):
    if random:
        arr = np.random.default_rng(SEED).standard_normal((x.shape[1], dim), dtype=np.float32)
        return np.linalg.qr(arr, mode="reduced")[0]
    return randomized_svd(x, n_components=dim, n_iter=5, random_state=SEED)[2].T.copy()


class AutoEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.compress = nn.Sequential(nn.Linear(2*width, width), nn.GELU(), nn.Linear(width, width))
        self.expand = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2*width))
        for m in (self.compress, self.expand):
            nn.init.zeros_(m[-1].weight)
            nn.init.zeros_(m[-1].bias)

    def encode(self, x):
        return x[:, :self.width] + self.compress(x)

    def decode(self, z):
        return torch.cat((z, torch.zeros_like(z)), dim=1) + self.expand(z)


@dataclass
class Encoder:
    mean: np.ndarray
    basis: np.ndarray
    auto: AutoEncoder | None
    gain: float

    def encode(self, x):
        v = ((x-self.mean) @ self.basis) / self.gain
        return v if self.auto is None else numpy_prediction(self.auto.encode, v)

    def decode(self, z):
        v = z if self.auto is None else numpy_prediction(self.auto.decode, z)
        return (v*self.gain) @ self.basis.T + self.mean


def batches(n, steps, batch=128):
    g = torch.Generator().manual_seed(SEED)
    for _ in range(steps):
        for i in torch.randperm(n, generator=g).split(batch):
            yield i


def fit_encoder(states, kind, dim, raw_width):
    mean = states.mean(0)
    centered = states - mean
    auto = None
    if kind == "structured":
        basis = np.zeros((states.shape[1], dim), np.float32)
        offset = 0
        for r in range(5):
            local = dim//5 + (r < dim%5)
            cols = slice(r*raw_width, (r+1)*raw_width)
            basis[cols, offset:offset+local] = frame(centered[:, cols], local)
            offset += local
    else:
        basis = frame(centered, dim*2 if kind == "auto" else dim, kind == "random")
    proj = centered @ basis
    gain = float(np.sqrt(np.mean(proj[:, :dim]**2)))
    if kind == "auto":
        torch.manual_seed(SEED)
        auto = AutoEncoder(dim)
        optim = torch.optim.AdamW(auto.parameters(), lr=.001, weight_decay=.001)
        data = tensor(proj / gain)
        for ids in batches(len(data), 30):
            x = data[ids]
            z = auto.encode(x)
            loss = (auto.decode(z)-x).square().mean()
            std = (z.var(0, unbiased=False)+1e-5).sqrt()
            anchor = (x[:, :dim].var(0, unbiased=False)+1e-5).sqrt()
            loss = loss + .10*(std-anchor).square().mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
        auto.eval()
    return Encoder(mean, basis, auto, gain)


@dataclass
class Affine:
    center: np.ndarray
    bias: np.ndarray
    weight: np.ndarray
    train: np.ndarray | None = None

    def __call__(self, x):
        x = x - self.center
        if self.train is not None:
            x = x @ self.train.T
        return x @ self.weight + self.bias


def affine(x, y, penalty=1.):
    center, bias = x.mean(0), y.mean(0)
    xc, yc = x-center, y-bias
    dual = len(x) < x.shape[1]
    gram = xc @ xc.T if dual else xc.T @ xc
    gram.flat[::len(gram)+1] += penalty
    coef = cho_solve(cho_factor(gram, lower=True), yc if dual else xc.T @ yc)
    return Affine(center, bias, coef, xc if dual else None)


class Residual(nn.Module):
    def __init__(self, dim, kind):
        super().__init__()
        self.kind = kind
        self.base = nn.Linear(dim, dim)
        if kind == "mlp":
            self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
            nn.init.zeros_(self.net[-1].bias)
            nn.init.zeros_(self.net[-1].weight)
        else:
            self.gate = nn.Linear(dim, 4)
            self.down = nn.Linear(dim, 4*8)
            self.up = nn.Parameter(torch.randn(4, 8, dim)*.01)

    def forward(self, x):
        if self.kind == "mlp":
            correction = self.net(x)
        else:
            weights = torch.softmax(self.gate(x), dim=-1)
            expert = torch.einsum("bei,eid->bed", self.down(x).reshape(-1, 4, 8), self.up)
            correction = torch.einsum("be,bed->bd", weights, expert)
        return x + self.base(x) + correction


@dataclass
class Transition:
    kind: str
    predict_delta: Affine | None = None
    rank: np.ndarray | None = None
    residual: Residual | None = None

    def __call__(self, x):
        if self.kind == "identity":
            return x
        if self.residual is not None:
            return numpy_prediction(self.residual, x)
        d = self.predict_delta(x)
        if self.rank is not None:
            d = (d @ self.rank) @ self.rank.T
        return x + d


def fit_transition(x, y, kind, penalty):
    if kind == "identity":
        return Transition(kind)
    ridge = affine(x, y-x, penalty)
    if kind == "ridge":
        return Transition(kind, ridge)
    if kind.startswith("rank"):
        natural_delta = ridge(x)-ridge.bias
        basis = frame(natural_delta, min(x.shape[1], int(kind[4:])))
        return Transition(kind, ridge, basis)
    torch.manual_seed(SEED)
    net = Residual(x.shape[1], kind)
    with torch.no_grad():
        net.base.weight.copy_(tensor(ridge.weight.T))
        net.base.bias.copy_(tensor(ridge.bias-ridge.center @ ridge.weight))
    opt = torch.optim.AdamW(net.parameters(), lr=.0007, weight_decay=.003)
    a,b = tensor(x),tensor(y)
    for ids in batches(len(x), 45):
        loss = (net(a[ids])-b[ids]).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return Transition(kind, residual=net.eval())


@dataclass
class System:
    name: str
    encoder: Encoder | None = None
    transition: Transition | None = None
    direct: Affine | None = None

    def __call__(self, x, context):
        if self.name == "context":
            return self.direct(context)
        if self.direct is not None:
            return x + self.direct(x)
        if self.encoder is None:
            return x
        return self.encoder.decode(self.transition(self.encoder.encode(x)))

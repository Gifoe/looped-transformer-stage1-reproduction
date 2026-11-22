"""Minimal signed dose laws for a single train-derived activation direction."""
from pathlib import Path

import torch
from torch import nn

from minimal import Minimal, WIDTH, rms

CAP = 0.45


def prior_axis(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    output = payload["state"]["out.weight"].float().squeeze(1)
    assert output.shape == (WIDTH,)
    return (output / rms(output)).to(device)


class SharedController(nn.Module):
    kind = "linear"

    def __init__(self, form, fixed_dose=0.45):
        super().__init__()
        if form not in {"fixed", "hard", "smooth"}:
            raise ValueError(f"Unknown scalar form {form}")
        self.form = form
        self.register_buffer("direction", torch.zeros(WIDTH))
        if form == "fixed":
            self.register_buffer("fixed_dose", torch.tensor(fixed_dose))
        else:
            self.coefficients = nn.Parameter(torch.zeros(WIDTH))

    def amplitude(self, h):
        if self.form == "fixed":
            return torch.ones_like(h[..., :1]) * self.fixed_dose
        raw = ((h / rms(h)) * self.coefficients).sum(-1, keepdim=True) / (WIDTH ** 0.5)
        if self.form == "hard":
            return raw.clamp(-CAP, CAP)
        return CAP * torch.tanh(raw)

    def delta(self, h, previous=None):
        return self.amplitude(h) * rms(h) * self.direction


def new_controller(form, init_path, fixed_dose=0.45):
    initial = torch.load(init_path, map_location="cpu", weights_only=True)
    if form == "rank1":
        net = Minimal("linear", 1)
        net.load_state_dict(initial["state"])
        return net
    net = SharedController(form, fixed_dose)
    vector = initial["state"]["out.weight"].float().squeeze(1)
    with torch.no_grad():
        net.direction.copy_(vector / rms(vector))
        if form != "fixed":
            old = initial["state"]["encode.weight"].float().squeeze(0)
            weight = old * rms(vector) * (WIDTH ** 0.5)
            if form == "smooth":
                # Match the slope of the hard law at its origin.
                weight = weight / CAP
            net.coefficients.copy_(weight)
    return net


def save_controller(path, net, meta):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(dict(
        form=net.form if isinstance(net, SharedController) else "rank1",
        state={name: data.detach().cpu() for name, data in net.state_dict().items()},
        meta=meta,
    ), tmp)
    tmp.replace(path)


def load_controller(path, device="cuda"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["form"] == "rank1":
        net = Minimal("linear", 1)
    else:
        net = SharedController(saved["form"])
    net.load_state_dict(saved["state"])
    return net.to(device).eval()


def controller_size(net):
    learned = sum(p.numel() for p in net.parameters() if p.requires_grad)
    storage = sum(t.numel() for t in net.state_dict().values())
    return dict(trainable_parameters=learned, stored_scalars=storage)

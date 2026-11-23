"""Apply rank-one feedback to selected causal predictor positions."""
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class Schedule:
    first: int = 0
    stop: int | None = None
    cap: float = 0.45
    after_cap: float | None = None
    min_dose: float = 0.0
    headroom_ratio: float = 0.0
    checkpoint: str = "t1"

    def record(self):
        return asdict(self)


ORIGINAL = Schedule()


class ScheduledFeedback:
    kind = "linear"

    def __init__(self, net, policy, group, positions):
        self.net = net
        self.policy = policy
        # pack_generation emits the positions of each example contiguously.
        # Derive the first predictor of every example without any generated labels.
        # The group vector starts at zero and contains each active example.
        prompt_end = torch.full((positions.shape[0],), positions.shape[0] + 1, device=positions.device,
                                dtype=positions.dtype)
        prompt_end.fill_(torch.iinfo(positions.dtype).max)
        prompt_end.scatter_reduce_(0, group, positions, reduce="amin", include_self=True)
        self.step = (positions - prompt_end[group]).unsqueeze(-1)

    def delta(self, state, previous=None):
        p = self.policy
        delta = self.net.delta(state, previous)
        eps = 1e-12
        ratio = (delta.square().mean(-1, keepdim=True) + eps).sqrt() / (
            state.square().mean(-1, keepdim=True) + eps).sqrt()
        cap = p.cap
        if p.after_cap is not None:
            cap = torch.where(self.step > 0, p.after_cap, p.cap)
        scale = (cap / ratio).clamp(max=1.)
        active = (self.step >= p.first) & (ratio >= p.min_dose)
        if p.stop is not None:
            active = active & (self.step < p.stop)
        if p.headroom_ratio > 0:
            active = active & (ratio <= p.headroom_ratio)
        return delta * scale * active


def for_generation(net, schedule, packed):
    if schedule == ORIGINAL or schedule == Schedule(checkpoint="t2"):
        return net
    return ScheduledFeedback(net, schedule, packed[1], packed[2])

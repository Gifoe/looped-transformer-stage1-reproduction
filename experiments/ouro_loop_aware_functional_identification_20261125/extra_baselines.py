"""Run historical Rank-8 and LoopCD on the four new official populations."""

from functools import partial
from unittest.mock import patch

import evaluate_core as core
import minimal
import torch


RANK8 = "513a964b07f6a70d79fdbf2a36effa85fc757194cb89fc84993a2323da2abeb8"
LOOPCD = ("fixed", "adaptive")


def rank8_controller(path, device):
    if minimal.sha(path) != RANK8:
        raise ValueError("Rank-8 checkpoint differs from the historical freeze")
    payload = torch.load(path, weights_only=True, map_location="cpu")
    model = minimal.Minimal(**payload["arch"])
    model.load_state_dict(payload["state"], strict=True)
    if model.kind != "linear" or model.bound != .45 or sum(p.numel() for p in model.parameters()) != 32768:
        raise ValueError("Unexpected historical Rank-8 architecture")
    return model.to(device).requires_grad_(False).eval()


class ContrastedReadout:
    def __init__(self, model, variant, generation):
        self.model = model
        self.variant = variant
        self.generation = generation

    def parameters(self):
        return self.model.parameters()

    def lm_head(self, states):
        early, final = (self.model.lm_head(state).float() for state in states.unbind(dim=1))
        if self.variant == "fixed":
            weight = .3 if self.generation else .5
        else:
            top = (final.topk(2, dim=-1).values - final.logsumexp(dim=-1, keepdim=True)).exp()
            weight = 1 - (top[:, :1] - top[:, 1:])
            if self.generation:
                weight = .5 * weight
        return final + weight * (final - early)


def native_states(model, packed, depth, net):
    if depth != 4 or net is not None:
        raise ValueError("LoopCD requires Native Ouro at K4")
    tokens, sample, position, _ = packed
    trunk = model.model.model
    trunk.total_ut_steps = depth
    _, states, _ = trunk(input_ids=tokens, use_cache=False)
    if len(states) != depth:
        raise ValueError("Unexpected native recurrence depth")
    return torch.stack((states[0][sample, position], states[-1][sample, position]), dim=1), None


def contrast(evaluator, variant, generation, archive, model, tokenizer, net, *args):
    with patch.object(minimal, "forward", native_states):
        return evaluator(archive, ContrastedReadout(model, variant, generation), tokenizer, net, *args)


def run(a):
    if a.arm == "rank8":
        with patch.object(core, "controller", rank8_controller):
            core.run(a)
    elif a.arm in LOOPCD:
        candidates = partial(contrast, core.candidates, a.arm, False)
        generate = partial(contrast, core.generate, a.arm, True)
        with patch.object(core, "candidates", candidates), patch.object(core, "generate", generate):
            core.run(a)
    else:
        core.run(a)

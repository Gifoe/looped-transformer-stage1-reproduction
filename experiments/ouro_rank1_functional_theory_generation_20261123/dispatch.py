"""Explicit scoring/generation entry points for the frozen Rank-One package."""
import json

from controller import load_controller
from generation import Arm, model_next
from minimal import DEPTHS, sha
from schedule import ORIGINAL, Schedule, for_generation
from train import evaluate

VALIDATED_GENERATION_DEPTHS = {"option_letter_6": DEPTHS, "numeric_solution_256": (4,), "option_letter_8": (4,)}


class RankOnePackage:
    def __init__(self, control, proposal, experimental_weight=None, acceptance=None):
        weights = control / "results" / "weights"
        requested = json.loads(proposal.read_text(encoding="utf-8"))
        self.policies = {k: ORIGINAL for k in DEPTHS}
        paths = {"t1": weights / "rank1_t1.pt", "t2": weights / "rank1_t2.pt"}
        if experimental_weight is not None:
            paths["train"] = experimental_weight
        accepted = json.loads(acceptance.read_text(encoding="utf-8")) if acceptance else {}
        if accepted.get("generation_deployment_accepted", False):
            if accepted["proposal_sha256"] != sha(proposal):
                raise ValueError("Acceptance receipt does not cover the selected proposal")
            self.policies = {k: Schedule(**requested["policies"][requested["selected"][f"k{k}"]])
                             for k in DEPTHS}
        names = {"t1", *(policy.checkpoint for policy in self.policies.values())}
        self.checkpoints = {}
        for key in names:
            if sha(paths[key]) != requested["weights_sha256"][key]:
                raise ValueError(f"Mismatched checkpoint digest for {key}")
            self.checkpoints[key] = load_controller(paths[key])

    def score_candidates(self, old, model, tokenizer, raw, depths=DEPTHS):
        return evaluate(old, model, tokenizer, raw, self.checkpoints["t1"], 1.0, depths)

    def greedy_next(self, model, depth, packed, last, *, protocol=None):
        policy = self.policies[depth] if depth in VALIDATED_GENERATION_DEPTHS.get(protocol, ()) else ORIGINAL
        net = for_generation(self.checkpoints[policy.checkpoint], policy, packed)
        return model_next(model, None, Arm("minimal", net, 1.0), depth, packed, last)

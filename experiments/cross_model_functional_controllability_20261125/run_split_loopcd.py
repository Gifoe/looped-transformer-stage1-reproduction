"""Run a predeclared subset of the frozen LoopCD protocol."""
from pathlib import Path
from types import SimpleNamespace
import argparse

import evaluate_loopcd as evaluation


class Tasks(dict):
    def __init__(self, registry, selected):
        super().__init__(registry)
        self.selected = selected

    def items(self):
        return ((key, self[key]) for key in self.selected)


def main():
    parser = argparse.ArgumentParser()
    for name in ("models", "model", "t1", "reference", "paper", "transfer", "stage2", "followup", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--scope", choices=("main_fixed", "generation_adaptive"), required=True)
    args = parser.parse_args()
    if args.scope == "generation_adaptive":
        evaluation.TASKS = Tasks(evaluation.TASKS, ("mmlu_pro",))
    base_generation = evaluation.generation

    def generate(*parameters):
        arms = evaluation.ARMS
        evaluation.ARMS = ("fixed",) if args.scope == "main_fixed" else ("adaptive",)
        try:
            return base_generation(*parameters)
        finally:
            evaluation.ARMS = arms

    evaluation.generation = generate
    evaluation.run(args)


if __name__ == "__main__":
    main()

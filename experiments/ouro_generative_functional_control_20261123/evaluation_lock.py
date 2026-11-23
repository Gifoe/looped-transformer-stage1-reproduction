"""Fingerprint the freeze-time decoder, scoring rules and shared Rank-1 hook."""

from pathlib import Path

import generation
import minimal


def code_hashes():
    directory = Path(__file__).resolve().parent
    scripts = ("evaluate_suite.py", "evaluate_generation.py", "evaluation_lock.py", "task_data.py")
    hashes = {name: minimal.sha(directory / name) for name in scripts}
    hashes["historical/generation.py"] = minimal.sha(generation.__file__)
    hashes["historical/minimal.py"] = minimal.sha(minimal.__file__)
    return hashes

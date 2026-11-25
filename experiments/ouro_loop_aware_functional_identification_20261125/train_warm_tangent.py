"""Train the matched T1-warm arm with forecast-derived source weights."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import minimal
import train_identification as shared


def main(a):
    plan = {
        "forecast_sha256": minimal.sha(a.forecast / "FORECAST.json"),
        "trainer_sha256": minimal.sha(Path(shared.__file__)),
        "selector_sha256": minimal.sha(Path(__file__)),
        "initialization_sha256": minimal.sha(a.initialize),
        "across_source_signal": "loop_factor_influence_warm_t1",
        "within_source_signal": "unchanged warm_loop question weights",
    }
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / "TANGENT_LOCK.json"
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError("T1 tangent fit lock differs")
    else:
        minimal.dump(path, plan)
    original = shared.selection

    def weights(rows, probes, forecast):
        # Within-source rank and optimizer schedule are unchanged.
        revised = deepcopy(forecast)
        for record in revised["records"]:
            scores = record["predicted"]
            scores["loop_and_absolute"] = scores["loop_factor_influence_warm_t1"]
        return original(rows, probes, revised)

    with patch.object(shared, "selection", weights):
        shared.fit(a)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "previous", "model", "data", "moments", "forecast", "out", "initialize"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=261125137)
    parser.set_defaults(arm="loop", include=None, limit=None, max_work=None)
    main(parser.parse_args())

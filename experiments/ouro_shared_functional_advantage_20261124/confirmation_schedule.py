"""Plan the fixed confirmation order and estimate it using saved development costs."""

import argparse
from collections import defaultdict
import json
from math import ceil
from pathlib import Path
from statistics import median


def schedule(plan):
    core = set(plan["transfer_arms"]) | {"historical_retention", "ranking", "generation"}
    primary = [("k4", arm) for arm in plan["run_order"] if arm in core]
    transfer = [("transfer", arm) for arm in plan["transfer_arms"]]
    extra = [("k4", arm) for arm in plan["run_order"] if arm not in core]
    return [*primary, *transfer, *extra]


def costs(development, ledger):
    reports = defaultdict(dict)
    totals = defaultdict(float)
    for path in development.glob("*.json"):
        result = json.loads(path.read_text())
        plan = result["plan"]
        arm = plan["arm"]
        reports[plan["dataset"], plan["mode"], plan["depth"]][arm] = result
        totals[arm] += result["cost"]["seconds"]
    elapsed, uncovered = 0, []
    for row in ledger.read_text().splitlines():
        columns = row.split("\t")
        if columns[0] != "END":
            continue
        label, seconds, status = columns[1], int(columns[4]), int(columns[5])
        elapsed += seconds
        if label.startswith("dev_") and status == 0:
            arm = label.removeprefix("dev_")
            if arm in totals:
                uncovered.append(max(0, seconds - totals[arm]))
    return reports, median(uncovered), elapsed


def forecast(development, ledger, data):
    from evaluate_controller import jobs
    from freeze_eval import ADVANTAGE, BASELINES
    from task_data import load

    reports, overhead, spent = costs(development, ledger)
    proposals = (*ADVANTAGE, "teacher_resume", "on_policy_resume", "anchored_policy")
    all_arms = (*BASELINES, *proposals)
    cases = list(jobs(data, "confirmation", (2, 3, 4, 5, 6), 10000, 96))
    counts = {path: len(load(Path(path))) for path in {case["path"] for case in cases}}
    estimates = {}
    ranges = [("scaled_by_batch", 0), ("scaled_by_items", 1)]
    for selected in proposals:
        primary = list(dict.fromkeys((selected, "native", "historical_rank", "historical_retention", "historical_mixed", *all_arms)))
        transfer = [selected, "native", "historical_rank", "historical_mixed", "mixed"]
        order = schedule(dict(run_order=primary, transfer_arms=transfer))
        summaries = {name: dict(k4=0.0, transfer=0.0) for name, _ in ranges}
        cases_count = 0
        for stage, arm in order:
            for case in cases:
                depth = case["depth"]
                if (depth == 4) != (stage == "k4"):
                    continue
                cases_count += 1
                ds, mode, batch = (case[key] for key in ("dataset", "mode", "batch"))
                sample_limit = (96 if depth == 4 else 48) if mode == "gsm_process" else case["limit"]
                n = min(counts[case["path"]], sample_limit)
                proxy_depth = min(4, depth)
                observed = reports[ds, mode, proxy_depth]
                if arm in observed:
                    source = observed[arm]
                    sec, items = source["cost"]["seconds"], source["plan"]["n"]
                else:
                    sec = median(row["cost"]["seconds"] for row in observed.values())
                    items = median(row["plan"]["n"] for row in observed.values())
                multipliers = (ceil(n / batch) / ceil(items / batch), n / items)
                for name, index in ranges:
                    summaries[name][stage] += sec * multipliers[index] * depth / proxy_depth
        estimates[selected] = dict(conditions=cases_count, model_loads=len(order), decoder_seconds=summaries)
    remaining_dev = sum(median(row["cost"]["seconds"] for row in reports["gsm8k", "gsm_process", k].values())
                        for k in (3, 4))
    return dict(gpu_used_s=spent, gpu_remaining_s=18000-spent, measured_median_overhead_s=overhead,
                two_missing_development_decoder_s=remaining_dev, estimates=estimates,
                scope="Forecast from completed development timings. No official accuracy or timing was measured.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--development", type=Path, default=Path("results/development"))
    parser.add_argument("--ledger", type=Path, default=Path("GPU_LEDGER.tsv"))
    parser.add_argument("--data", type=Path, default=Path("data/confirmation"))
    args = parser.parse_args()
    if args.protocol:
        for stage, arm in schedule(json.loads(args.protocol.read_text())):
            print(f"{stage}\t{arm}")
    else:
        print(json.dumps(forecast(args.development, args.ledger, args.data), indent=2))


if __name__ == "__main__":
    main()

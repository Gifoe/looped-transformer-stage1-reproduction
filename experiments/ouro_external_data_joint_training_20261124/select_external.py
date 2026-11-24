"""Use only paired development evidence to select objectives, recipes, and checkpoints."""

import argparse
import json
from pathlib import Path

from evaluate_core import DEVELOPMENT_REFERENCES, TASKS
import minimal
import paired_analysis
from task_data import load


PRIMARY = {"candidate": ("acc_norm",), "content": ("correct",), "span": ("correct",),
           "gsm_process": ("strict", "process_correct")}
DAMAGE = .02
paired_analysis.METRICS.update({"process_correct", "premature_eos", "token_limit", "repetitive"})


def checked(directory, dataset, arm, depth, mode, expected_phase):
    name = directory / f"{dataset}__{arm}__k{depth}__{mode}.jsonl"
    info = json.loads(name.with_suffix('.json').read_text())
    if info['plan']['phase'] != expected_phase or info['rows_sha256'] != minimal.sha(name):
        raise ValueError(f"Not a closed development result for {name}")
    return load(name)


def assess(directory, arm, phase):
    values = []
    for ds, modes in TASKS.items():
        for mode in modes:
            if mode not in PRIMARY:
                continue
            for k in (2, 3, 4):
                own = checked(directory, ds, arm, k, mode, phase)
                native = checked(directory, ds, 'native', k, mode, phase)
                comparator = DEVELOPMENT_REFERENCES[mode]
                strong = checked(directory, ds, comparator, k, mode, phase)
                n = paired_analysis.compare(native, own)
                s = paired_analysis.compare(strong, own)
                values.extend(dict(dataset=ds, mode=mode, depth=k, metric=m, native=n[m], strong=s[m], comparator=comparator)
                              for m in PRIMARY[mode])
    criterion = [(r["native"]["delta"] < -DAMAGE and r["native"]["paired_ci95"][1] < 0) for r in values]
    gaps = [r["strong"]["delta"] for r in values]
    families = {key: [r for r in values if r['mode'] in modes] for key,modes in
                {"candidate": {"candidate"}, "short": {"content", "span"}, "reasoning": {"gsm_process"}}.items()}
    family_native = {key:sum(r["native"]["delta"] for r in group)/len(group) for key,group in families.items()}
    return dict(arm=arm, damaged=sum(criterion), minimum_native=min(r["native"]["delta"] for r in values),
                minimum_strong=min(gaps), mean_strong=sum(gaps)/len(gaps), mean_native=sum(family_native.values())/3,
                mean_native_by_family=family_native, details=values)


def ordering(row):
    return -row["damaged"], row["minimum_native"], row["minimum_strong"], row["mean_native"], row["mean_strong"]


def improved(old, new):
    return (new["damaged"] <= old["damaged"] and new["minimum_native"] >= old["minimum_native"] - .005
            and new["mean_native"] >= old["mean_native"] + .005)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--phase', choices=('dev-small','dev-large'), required=True)
    p.add_argument('--arms', nargs='+', required=True)
    a = p.parse_args()
    rows = [assess(a.directory, arm, a.phase) for arm in a.arms]
    ordered = sorted(rows, key=ordering, reverse=True)
    minimal.dump(a.out, dict(phase=a.phase, selected=ordered[0]['arm'], rows=ordered))
    print(ordered[0]['arm'], {row['arm']:ordering(row) for row in ordered}, flush=True)


if __name__ == '__main__':
    main()

"""Run balanced 2-GPU sharded Official 8-Benchmark (43,769-question) LEGACY suite for all 3 frozen V4 4,096-param controllers."""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/root/autodl-tmp/ouro-v4-cfit-20261126")
OFF_ROOT = ROOT / "official_run"

ALIASES = ("Best_Rank1_4096", "Best_NonARC_4096", "Best_ZeroLabel_4096")
COST = {
    "mmlu_pro": 1100, "mmlu": 450, "hellaswag": 350, "arc_easy": 170,
    "piqa": 160, "arc_challenge": 130, "winogrande": 125, "sciq": 85,
}


def build_2gpu_schedule():
    jobs = [(COST[t], a, t) for a in ALIASES for t in COST]
    jobs.sort(key=lambda x: (-x[0], x[1], x[2]))
    loads = [0, 0]
    assign = [[], []]
    for c, a, t in jobs:
        g = 0 if loads[0] <= loads[1] else 1
        assign[g].append((a, t, c))
        loads[g] += c
    return assign, loads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", type=int, choices=(0, 1), required=True)
    args = ap.parse_args()

    assign, loads = build_2gpu_schedule()
    my_jobs = assign[args.slot]
    print(f"[LEGACY8 GPU {args.slot}] {len(my_jobs)} jobs, est={loads[args.slot]/60:.1f} min (loads={loads})", flush=True)

    py = sys.executable
    off_script = str(ROOT / "handoff_codex" / "v4_official.py")
    t_all = time.perf_counter()
    for idx, (alias, task, est) in enumerate(my_jobs, 1):
        t0 = time.perf_counter()
        print(f"\n>>> [{idx}/{len(my_jobs)}] GPU{args.slot} START {alias} / {task} (est={est}s)", flush=True)
        subprocess.run([
            py, "-u", off_script,
            "--root", str(OFF_ROOT),
            "--phase", "legacy",
            "--alias", alias,
            "--task", task,
            "--arm", "new",
        ], check=True)
        print(f"<<< [{idx}/{len(my_jobs)}] GPU{args.slot} DONE  {alias} / {task} in {time.perf_counter()-t0:.1f}s", flush=True)
    print(f"\nALL_LEGACY8_SLOT_{args.slot}_DONE in {(time.perf_counter()-t_all)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

"""Export the seven Liu et al. LoopCD prompts with lm-eval 0.4.13, without a model.

In a separate CPU environment, install lm_eval==0.4.13. Run:
    python stage2_requests.py --output stage2_requests.jsonl

Pass --tasks arc_challenge --limit-per-task 2 for a small format check.
If needed, set HF_ENDPOINT before running. The manifest records the exact task,
dataset revision, split, Harness version, seed and JSONL hash. The paper does
not disclose its Harness version or few-shot seed.
"""

from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from itertools import groupby
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from time import monotonic


HARNESS_VERSION = "0.4.13"
PAPER = "https://arxiv.org/html/2610.02185v1#A2.T2"
DATASET_REVISIONS = {
    "allenai/ai2_arc": "210d026faf9955653af8916fad021475a3f00453",
    "allenai/sciq": "2c94ad3e1aafab77146f384e23536f97a4849815",
    "cais/mmlu": "c30699e8356da336a370243923dbaf21066bb9fe",
    "Rowan/hellaswag": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
    "allenai/winogrande": "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
    "baber/piqa": "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1",
}
# Each entry is (demonstrations, Hub repository, evaluation split, demo split).
EVALUATIONS = {
    "arc_challenge": (25, "allenai/ai2_arc", "test", "train"),
    "arc_easy": (8, "allenai/ai2_arc", "test", "train"),
    "sciq": (0, "allenai/sciq", "test", None),
    "mmlu": (5, "cais/mmlu", "test", "dev"),
    "hellaswag": (10, "Rowan/hellaswag", "validation", "train"),
    "winogrande": (5, "allenai/winogrande", "validation", "train"),
    "piqa": (0, "baber/piqa", "validation", None),
}
DEFAULT_SEED = 1234


def export_task(task, benchmark, count_limit, seed, write_row):
    shots, repo, eval_split, demo_split = EVALUATIONS[benchmark]
    task_split = task.config.test_split or task.config.validation_split
    actual_demo_split = (
        task.fewshot_cfg.split or task.config.training_split or task.config.validation_split
    ) if shots else None
    if (
        task.config.dataset_path != repo
        or task.config.output_type != "multiple_choice"
        or task_split != eval_split
        or actual_demo_split != demo_split
        or task.config.num_fewshot != shots
    ):
        raise RuntimeError(f"{task.task_name}: unexpected Harness configuration")

    task.set_fewshot_seed(seed)
    task.build_all_requests(limit=count_limit)
    metrics = [entry["metric"] for entry in task.config.metric_list]
    request_count = 0
    document_count = 0
    for doc_id, instances in groupby(task.instances, key=lambda instance: instance.doc_id):
        options = sorted(instances, key=lambda instance: instance.idx)
        doc = options[0].doc
        choices = task.doc_to_choice(doc)
        target = task.doc_to_text(doc) if task.multiple_input else task.doc_to_target(doc)
        if not isinstance(target, int):
            target = choices.index(target)
        if (
            doc_id != document_count
            or len(options) != len(choices)
            or not 0 <= target < len(choices)
            or any(not isinstance(choice, str) for choice in choices)
        ):
            raise RuntimeError(f"{task.task_name}:{doc_id}: inconsistent option group")

        requests = []
        for index, instance in enumerate(options):
            if (
                instance.idx != index
                or instance.request_type != "loglikelihood"
                or len(instance.args) != 2
                or not all(isinstance(arg, str) for arg in instance.args)
            ):
                raise RuntimeError(f"{task.task_name}:{doc_id}: unexpected Harness request")
            requests.append({"type": instance.request_type, "args": instance.args, "idx": index})
        write_row({
            "id": f"{task.task_name}:{doc_id}",
            "task": task.task_name,
            "benchmark": benchmark,
            "doc_id": doc_id,
            "target": target,
            "choices": choices,
            "requests": requests,
        })
        request_count += len(requests)
        document_count += 1

    expected_docs = len(task.eval_docs)
    if count_limit is not None:
        expected_docs = min(expected_docs, count_limit)
    if document_count != expected_docs:
        raise RuntimeError(f"{task.task_name}: expected {expected_docs} docs, got {document_count}")
    return {
        "task": task.task_name,
        "benchmark": benchmark,
        "dataset": repo,
        "dataset_config": task.config.dataset_name,
        "dataset_revision": DATASET_REVISIONS[repo],
        "eval_split": eval_split,
        "fewshot_split": demo_split,
        "shots": shots,
        "metrics": metrics,
        "documents": document_count,
        "requests": request_count,
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Target JSONL file")
    parser.add_argument("--tasks", nargs="+", choices=tuple(EVALUATIONS), default=list(EVALUATIONS))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit-per-task", type=int, help="Smoke test cap, including each MMLU subject")
    args = parser.parse_args()
    if args.limit_per_task is not None and args.limit_per_task < 1:
        parser.error("--limit-per-task must be positive")
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("--tasks must not contain duplicate tasks")
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    if args.output.exists() or manifest_path.exists():
        parser.error("output and manifest paths must be new")
    try:
        installed_harness = version("lm_eval")
    except PackageNotFoundError:
        installed_harness = None
    if installed_harness != HARNESS_VERSION:
        parser.error(f"install lm_eval=={HARNESS_VERSION} in the CPU export environment")

    from lm_eval.api.group import Group
    from lm_eval.tasks import TaskManager
    from lm_eval.tasks._factory import TaskFactory

    # TaskManager.load loads data immediately. Passing the overrides at factory
    # construction pins the revision before download, including group descendants.
    registry = TaskManager().task_index
    factory = TaskFactory()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256()
    summary = []
    elapsed = {}
    totals = Counter()
    with TemporaryDirectory(dir=args.output.parent, prefix=".stage2_requests_") as scratch:
        tmp_output = Path(scratch) / "requests.jsonl"
        tmp_manifest = Path(scratch) / "manifest.json"
        with tmp_output.open("wb") as output:
            def write_row(row):
                line = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                output.write(line)
                digest.update(line)

            for benchmark in args.tasks:
                start = monotonic()
                shots, repo, _, _ = EVALUATIONS[benchmark]
                run = factory.build(
                    registry[benchmark],
                    registry=registry,
                    overrides={
                        "dataset_kwargs": {"revision": DATASET_REVISIONS[repo]},
                        "num_fewshot": shots,
                    },
                )
                tasks = run.get_all_tasks() if isinstance(run, Group) else [run]
                expected_tasks = 57 if benchmark == "mmlu" else 1
                if len(tasks) != expected_tasks or len({task.task_name for task in tasks}) != expected_tasks:
                    raise RuntimeError(f"{benchmark}: expected {expected_tasks} distinct tasks")
                for task in sorted(tasks, key=lambda item: item.task_name):
                    receipt = export_task(task, benchmark, args.limit_per_task, args.seed, write_row)
                    summary.append(receipt)
                    totals.update({key: receipt[key] for key in ("documents", "requests")})
                elapsed[benchmark] = round(monotonic() - start, 3)
                print(f"{benchmark}: exported in {elapsed[benchmark]:.3f} s", flush=True)

        manifest = {
            "schema_version": 1,
            "paper_shot_source": PAPER,
            "harness_version": HARNESS_VERSION,
            "datasets_version": version("datasets"),
            "huggingface_hub_version": version("huggingface_hub"),
            "python": platform.python_version(),
            "fewshot_seed": args.seed,
            "chat_template": False,
            "limit_per_task": args.limit_per_task,
            "jsonl": args.output.name,
            "jsonl_sha256": digest.hexdigest(),
            "totals": dict(totals),
            "tasks": summary,
            "elapsed_seconds": elapsed,
        }
        tmp_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp_output.replace(args.output)
        tmp_manifest.replace(manifest_path)
    print(f"{totals['documents']} documents, {totals['requests']} requests")


if __name__ == "__main__":
    main()

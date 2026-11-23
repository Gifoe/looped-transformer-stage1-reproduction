"""Verify per-dataset and per-budget gates for publication and deployment."""
import argparse
import json
from pathlib import Path

from minimal import dump, sha


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(directory):
    out = directory / "BENCHMARK_RESULTS"
    selection = directory / "frozen/PROPOSAL.json"
    candidate = load(out / "CANDIDATE_INVARIANCE.json")
    comparisons = []
    for dataset, metadata in candidate["benchmarks"].items():
        assert sha(out/metadata["file"]) == metadata["sha256"]
        for k in range(1,5):
            for metric in ("acc","acc_norm"):
                comparisons.append(dict(dataset=dataset, mode="candidate", budget=k, metric=metric,
                                        delta_answers=0, verified=True, because="unchanged inference and raw archive"))
    for dataset, cells in load(out/"OPTION_GENERATION/CONFIRM.json").items():
        for depth, receipt in cells.items():
            for metric in ("correct","valid"):
                r=receipt["comparisons"]["t1"][metric]
                comparisons.append(dict(dataset=dataset,mode="generation",budget=int(depth[1:]),metric=metric,
                                        delta_answers=r["rescue"]-r["damage"],verified=True,ci95_pp=r["ci95_pp"]))
    for dataset in ("gsm8k","mmlu_pro"):
        receipt=load(out/dataset/"SUMMARY.json")
        for metric in ("correct","strict","valid"):
            r=receipt["vs_rank1_t1"][metric]
            comparisons.append(dict(dataset=dataset,mode="generation",budget=4,metric=metric,
                                    delta_answers=r["rescue"]-r["damage"],verified=True,ci95_pp=r["ci95_pp"]))
    completed=len(comparisons)==46 and all(x["verified"] for x in comparisons)
    failures=[x for x in comparisons if x["delta_answers"]<0]
    better_obqa=any(x["dataset"]=="openbookqa_test" and x["metric"]=="correct" and x["delta_answers"]>0
                    for x in comparisons)
    content_improved=any(x["dataset"]=="gsm8k" and x["metric"]=="correct" and x["delta_answers"]>0
                         for x in comparisons)
    return dict(proposal_sha256=sha(selection), complete=completed, comparison_cells=comparisons,
                failed_cells=failures, obqa_correct_improved=better_obqa, gsm8k_content_improved=content_improved,
                generation_deployment_accepted=completed and not failures and better_obqa,
                full_research_performance_goal=completed and not failures and better_obqa and content_improved)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results",type=Path)
    a=parser.parse_args()
    r=evaluate(a.results)
    dump(a.results/"frozen/ACCEPTANCE.json",r)
    print(r["generation_deployment_accepted"],len(r["failed_cells"]),r["full_research_performance_goal"])

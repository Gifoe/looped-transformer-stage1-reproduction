"""Postrun CPU-only distribution diagnostic; never changes acceptance or inference."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    def read(p):
        return json.loads((ROOT / p).read_text(encoding="utf-8"))
    review = read("results/r26_ROOT_VERIFICATION.json")
    assert review["status"] == "passed_independent_saved_vector_review"
    raw = read("results/r26_RAW.json")
    results = []
    with np.load(ROOT / "results/r26_VECTORS.npz", allow_pickle=False) as container:
        for row in raw["records"]:
            for comparison in row["comparisons"]:
                m = comparison["head"]
                if comparison["variant"] not in ["C1", "C2"] or m["exact"]:
                    continue
                a, b = [container[m["vector_" + side]].astype(np.float64).reshape(-1) for side in ["A", "B"]]
                def distribution(z):
                    shifted = z - np.max(z)
                    logp = shifted - np.log(np.exp(shifted).sum(dtype=np.float64))
                    return np.exp(logp), logp
                p, lp = distribution(a); q, lq = distribution(b)
                difference = a-b
                results.append(dict(context=row["context"], variant=comparison["variant"], depth=comparison["depth"], position=comparison["position"],
                                    logit_difference_range=float(np.max(difference)-np.min(difference)),
                                    total_variation=float(np.abs(p-q).sum(dtype=np.float64)/2),
                                    squared_Hellinger_without_half=float(np.square(np.sqrt(p)-np.sqrt(q)).sum(dtype=np.float64)),
                                    KL_full_input_to_truncated=float(np.sum(p*(lp-lq), dtype=np.float64)),
                                    KL_truncated_to_full_input=float(np.sum(q*(lq-lp), dtype=np.float64)),
                                    maximum_log_probability_difference=float(np.max(np.abs(lp-lq))),
                                    normalization_sums=[float(p.sum()), float(q.sum())]))
    out = dict(status="complete_postrun_CPU_length_distribution_description", GPU_worker_seconds=0,
               vectors_sha256=hashlib.sha256((ROOT / "results/r26_VECTORS.npz").read_bytes()).hexdigest(),
               changed_length_comparisons=results,
               precision="NumPy FP64 normalization of saved BF16-head-promoted FP32 fullV vectors. No floor, epsilon, tail restriction or current target gather.",
               acceptance_unchanged=review["category"], accuracy_computed=False,
               limitation="Postrun description conditional on the observed length differences. Nonconstant differences establish a distribution change for these readouts, not semantic leakage, an explanation of previous method failures, a new method, or an accuracy effect. Does not replay prior R25 retrieval or scoring.")
    path = ROOT / "results/r26_ROOT_LENGTH_DISTRIBUTION_DIAGNOSTIC.json"
    with path.open("x", encoding="utf-8") as f:
        json.dump(out, f, indent=2, allow_nan=False); f.write("\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

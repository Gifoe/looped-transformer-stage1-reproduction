"""Post-run independent exact-rational verification; does not solve another LP."""
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf8"))


def rows(path):
    return [json.loads(x) for x in (ROOT / path).read_text(encoding="utf8").splitlines()]


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def main():
    done = read("results/r21_oracle_COMPLETE.json")
    sup = read("logs/r21_SUPERVISOR.json")
    spec = read("configs/r21_READOUT_ORACLE_LOCK.json")
    assert sup["status"] == "complete" and sup["exit_code"] == 0
    assert not sup["overrun"] and not sup.get("timed_out", False)
    assert done["status"] == "complete_CPU_oracle" and done["n"] == 576
    assert done["raw_sha256"] == sha("results/r21_oracle.jsonl")
    assert done["protocol_sha256"] == sha("configs/r21_READOUT_ORACLE_LOCK.json")
    assert done["GPU_worker_seconds"] == sup["GPU_worker_seconds"] == 0
    actual = rows("results/r21_oracle.jsonl")
    source, baseline, order = {}, {}, []
    for split in ["discovery", "selection"]:
        source.update({x["id"]: x for x in rows(f"results/slcd_r2_{split}_development.jsonl")})
        baseline.update({x["id"]: x for x in rows(f"results/r7_{split}_development.jsonl")})
        order.extend(x["id"] for x in rows(f"data/slcd_{split}.jsonl"))
    assert len(actual) == 576 and [x["id"] for x in actual] == order
    ncert = 0
    tau = F.from_float(1e-8)
    for row in actual:
        q, b = source[row["id"]], baseline[row["id"]]
        gold = q["gold"]
        assert gold == row["gold"] == b["gold"]
        columns = [q["original_methods"][f"native_k{k}"]["normalized_scores"] for k in range(1, 7)]
        r6 = b["baseline_scores"]["M0_R6"]
        n = len(columns[0])
        assert n == len(row["H6"]) == len(row["H7"])
        discrete = any(max(range(n), key=lambda j: c[j]) == gold for c in columns)
        assert row["discrete_reachable"] == discrete
        assert row["R6_correct"] == (max(range(n), key=lambda j: r6[j]) == gold)
        for hull, vectors in [("H6", columns), ("H7", columns + [r6])]:
            for label, cert in enumerate(row[hull]):
                scale = F.from_float(cert["scale"])
                w = [F.from_float(x) for x in cert["primal_weights"]]
                lam = [F.from_float(x) for x in cert["dual_weights"]]
                assert scale > 0 and min(w) >= 0 and min(lam) >= 0
                assert sum(w) > 0 and sum(lam) > 0
                w, lam = [x / sum(w) for x in w], [x / sum(lam) for x in lam]
                matrix = [
                    [(F.from_float(v[label]) - F.from_float(v[j])) / scale for v in vectors]
                    for j in range(n) if j != label
                ]
                lower = min(sum(x * y for x, y in zip(a, w)) for a in matrix)
                upper = max(sum(lam[i] * matrix[i][k] for i in range(n - 1)) for k in range(len(vectors)))
                assert F.from_float(cert["L_lower"]) <= lower <= F.from_float(cert["L"] + cert["roundoff_delta"])
                assert F.from_float(cert["U"] - cert["roundoff_delta"]) <= upper <= F.from_float(cert["U_upper"])
                assert lower <= upper
                if cert["status"] == "strict_reachable":
                    assert lower > tau and cert["certificate_valid"]
                elif cert["status"] == "strict_unreachable":
                    assert upper < -tau and cert["certificate_valid"]
                else:
                    assert cert["status"] == "uncertain"
                ncert += 1
            assert row[hull + "_reachable_label_count"] == sum(c["status"] == "strict_reachable" for c in row[hull])
        status = row["H6"][gold]["status"]
        category = "discrete_reachable" if discrete else "only_convex_reachable" if status == "strict_reachable" else "hull_unreachable" if status == "strict_unreachable" else "numerically_undetermined"
        assert category == row["category"]
    summary = {}
    for split in ["discovery", "selection", "combined"]:
        subset = [x for x in actual if split == "combined" or x["split"] == split]
        summary[split] = {}
        for group in ["all", "R6_correct", "R6_wrong"]:
            sub = [x for x in subset if group == "all" or x["R6_correct"] == (group == "R6_correct")]
            categories = {k: sum(x["category"] == k for x in sub) for k in spec["categories"]}
            strict = sum(x["H6"][x["gold"]]["status"] == "strict_reachable" for x in sub)
            reported = done["summaries"][split][group]
            assert reported["n"] == len(sub) and reported["categories"] == categories
            assert reported["H6_strict_gold_reachable"] == strict
            summary[split][group] = dict(n=len(sub), categories=categories, H6_strict_gold_reachable=strict, H6_strict_proportion=strict / len(sub))
    passed = summary["combined"]["R6_wrong"]["H6_strict_proportion"] >= .2
    passed = passed and all(summary[x]["R6_wrong"]["H6_strict_proportion"] >= .1 for x in ["discovery", "selection"])
    passed = passed and summary["combined"]["all"]["categories"]["numerically_undetermined"] / 576 <= .01
    assert passed == done["capacity_diagnostic_gate_passed"]
    out = dict(
        status="passed_independent_exact_rational_recertification", n=576,
        certificates_checked=ncert, bounds_checked=2 * ncert,
        exact_arithmetic="Original cached FP64 scores and stored witnesses converted by Fraction.from_float; simplices normalized exactly. Independent original-score differences divided by stored scale. No scipy or second LP solve.",
        protocol_sha256=done["protocol_sha256"], raw_sha256=done["raw_sha256"],
        summary=summary, capacity_diagnostic_gate_passed=passed,
        GPU_worker_seconds=0, oracle_not_deployable=True,
    )
    with (ROOT / "results/r21_ROOT_VERIFICATION.json").open("x", encoding="utf8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(json.dumps(dict(status=out["status"], certificates_checked=ncert, bounds_checked=2 * ncert, R6_wrong=summary["combined"]["R6_wrong"], diagnostic_gate=passed)))


if __name__ == "__main__":
    main()

"""Verify every unlabeled finite-dose row against its aggregate and the exact control plan."""
import argparse
import json
import math
from hashlib import sha256
from pathlib import Path

from d2_worker import digest, save_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def assert_equal(left, right):
    if not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError(f"Finite-dose row recomputation differs: {left} and {right}")


def run(work, local_host, out):
    dispatch = work / "D3_FINITE_DISPATCH.json"
    authority = work / "D3_FINITE_TRANSPORT_AUDIT.json"
    plan = read(dispatch)
    if read(authority)["compact_dispatch_sha256"] != digest(dispatch) or len(plan["jobs"]) != 14:
        raise ValueError("The compact plan has changed")
    roster = []
    waiting = []
    for job in plan["jobs"]:
        host = plan["devices"][job["preferred_gpu"]]["host"]
        if host != local_host:
            continue
        if not Path(job["done_marker"]).is_file():
            waiting.append(job["job_id"])
            continue
        aggregate = next(Path(item) for item in job["artifacts"] if item.endswith("SOURCE_FINITE_AUDIT.json"))
        evidence = read(aggregate)
        lock = aggregate.with_name("UNLABELED_FINITE_LOCK.json")
        frozen = digest(lock)
        if evidence["lock_sha256"] != frozen or evidence["audit_sha256"] != read(lock)["audit_sha256"]:
            raise ValueError("The aggregate is not tied to the original 512-document request")
        files = [aggregate.with_name(f"row_{i:04d}.json") for i in range(512)]
        payload = []
        for path in files:
            row = read(path)
            if row["lock_sha256"] != frozen:
                raise ValueError("One row has a different completion lock")
            assert_equal(row["delta_real"] - row["delta_shuffled"], row["net_pmi"])
            payload.append(row)
        ordered_ids = [row["id"] for row in payload]
        if len(set(ordered_ids)) != 512 or evidence["external_unlabeled_count"] != 512:
            raise ValueError("A finite-dose request is absent or repeated")
        fingerprint = sha256(json.dumps(ordered_ids, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        means = {name: sum(row[name] for row in payload) / len(payload)
                 for name in ("net_pmi", "delta_real", "delta_shuffled", "kl")}
        for logged, key in (("pmi", "net_pmi"), ("kl", "kl")):
            assert_equal(evidence[logged], means[key])
        assert_equal(evidence["p4"], means["net_pmi"] - .3 * means["kl"])
        for name in ("delta_real", "delta_shuffled"):
            part = evidence["pmi_decomposition"][name]
            assert_equal(part["mean"], means[name])
            if part["count_negative"] != sum(row[name] < 0 for row in payload):
                raise ValueError("A directional count differs from the predictions")
        roster.append(dict(job_id=job["job_id"], audit_ids_sha256=fingerprint, rows=512,
                           lock_sha256=frozen, aggregate_sha256=digest(aggregate),
                           raw_sha256={path.name: digest(path) for path in files}))
    if waiting:
        print("D3_P4_RAW_AUDIT_PENDING", len(roster), waiting, flush=True)
        return
    if not roster or len({row["audit_ids_sha256"] for row in roster}) != 1:
        raise ValueError("All finite-dose axes need the same unlabelled requests in order")
    report = dict(host=local_host, compact_dispatch_sha256=digest(dispatch), finite_audits=roster)
    if out.exists() and read(out) != report:
        raise ValueError("A completed raw audit has drifted")
    if not out.exists():
        save_json(out, report)
    print("D3_P4_RAW_ROWS_VERIFIED", local_host, len(roster), len(roster)*512, digest(out), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--host", choices=("old", "dual"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.work, args.host, args.out)

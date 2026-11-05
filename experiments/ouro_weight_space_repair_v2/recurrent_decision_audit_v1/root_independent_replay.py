"""Independent CPU-only replay from frozen raw records; no scorer imports or model.

This file writes only root_independent_replay.json beside itself. Whole-question
bootstrap sampling keeps all rotations and both endpoint predictions together.
"""
import collections
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time

import numpy as np
from scipy.stats import binomtest

OUT = Path(__file__).resolve().parent
BASE = OUT.parent


def read(path):
    return json.loads((BASE / path).read_text(encoding="utf8"))


def lines(path):
    return [json.loads(x) for x in (BASE / path).read_text(encoding="utf8").splitlines() if x.strip()]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def paired(a, b, gold):
    c1, c4 = a == gold, b == gold
    rescues = ~c1 & c4
    damages = c1 & ~c4
    d = c4.astype(int) - c1.astype(int)
    rng = np.random.default_rng(2021)
    boot = np.concatenate([d[rng.integers(0, len(d), (100, len(d)))].mean(1) for _ in range(200)])
    r, s = int(rescues.sum()), int(damages.sum())
    return dict(k1_correct=int(c1.sum()), k4_correct=int(c4.sum()), rescue=r, damage=s,
                correct_correct=int((c1 & c4).sum()), wrong_wrong=int((~c1 & ~c4).sum()),
                wrong_wrong_changed=int((~c1 & ~c4 & (a != b)).sum()),
                prediction_unchanged=int((a == b).sum()), net=int(d.sum()),
                rescue_ids=[ids[i] for i in np.flatnonzero(rescues)],
                damage_ids=[ids[i] for i in np.flatnonzero(damages)],
                exact_mcnemar_p=float(binomtest(min(r, s), r+s, .5).pvalue) if r+s else 1.,
                paired_bootstrap_95_delta_pp=(np.quantile(boot, [.025, .975])*100).tolist(),
                bootstrap_seed=2021, bootstrap_replicates=20000)


started = time.perf_counter()
authority = read("configs/r11_PROTOCOL.json")
public = json.loads((OUT / "root_base_commit_files.json").read_text(encoding="utf8"))
public_checks = []
for entry in public["entries"]:
    if entry.get("absent_as_direct_blob"):
        continue
    raw = (BASE / entry["path"]).read_bytes()
    git_sha = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    assert len(raw) == entry["bytes"] and git_sha == entry["git_blob_sha1"], entry["path"]
    public_checks.append(dict(path=entry["path"], sha256=sha(raw), public_git_blob_sha1=git_sha))

index = read("artifacts/r11_RECORDS_INDEX.json")
contents_raw = (BASE / "artifacts/r11_RECORDS_CONTENTS.json").read_bytes()
assert sha(contents_raw) == index["contents_sha256"]
archive = bytearray()
for part in index["parts"]:
    raw = (BASE / part["path"]).read_bytes()
    assert len(raw) == part["bytes"] and sha(raw) == part["sha256"]
    archive.extend(raw)
assert len(archive) == index["archive_bytes"] and sha(archive) == index["archive_sha256"]
manifest = {x["path"]: x for x in json.loads(contents_raw)["files"]}
with tarfile.open(fileobj=io.BytesIO(archive), mode="r:xz") as tar:
    assert len(tar.getmembers()) == len(manifest)
    assert {m.name for m in tar.getmembers()} == set(manifest)
    for member in tar.getmembers():
        assert member.isfile() and not member.issym() and not member.islnk()
        entry = manifest[member.name]
        raw = tar.extractfile(member).read()
        assert sha(raw) == entry["sha256"] and len(raw) == entry["bytes"]
        assert (BASE / member.name).read_bytes() == raw, member.name

raw_scores = (BASE / "results/r11_test_selected.jsonl").read_bytes()
complete = read("results/r11_test_selected_COMPLETE.json")
assert sha(raw_scores) == complete["score_file_sha256"]
records = lines("results/r11_test_selected.jsonl")
items = {x["id"]: x for x in lines("data/slcd_test.jsonl")}
assert sha((BASE / "data/slcd_test.jsonl").read_bytes()) == authority["dataset_sha256"]["test"]
assert len(records) == len(items) == len({r["id"] for r in records}) == 1172
ids = [x["id"] for x in records]
pred = {k: [] for k in ("joint1_only", "joint4_only", "cyclic_joint1_only", "cyclic_joint4_only")}
stability = {1: [], 4: []}
choice_counts = collections.Counter()
readout_counts = collections.Counter()
tie_counts = collections.Counter()
for r in records:
    item = items[r["id"]]
    assert r["gold"] == item["gold"] and r["prompt_sha256"] == item["prompt_sha256"]
    assert r["actual_auxiliary_depth"] == 4 and r["robust_references"] is True
    n = len(item["choices"])
    choice_counts[n] += 1
    assert len(r["cyclic_rotations"]) == n
    for depth in (1, 4):
        matrices, selected = [], []
        for shift, rotation in enumerate(r["cyclic_rotations"]):
            order = [(j+shift) % n for j in range(n)]
            assert rotation["order"] == order
            mapped = np.asarray(rotation["sum_scores"][f"native_k{depth}"], dtype=float)
            assert mapped.shape == (n,) and np.isfinite(mapped).all()
            displayed = mapped[order]
            content_prediction = order[int(displayed.argmax())]
            selected.append(content_prediction)
            matrices.append(mapped)
            tie_counts[f"k{depth}_rotation_top_ties"] += int((displayed == displayed.max()).sum() > 1)
            tie_counts[f"k{depth}_mapped_vs_displayed_argmax_disagreements"] += int(content_prediction != int(mapped.argmax()))
            for name in rotation["sum_scores"]:
                readout_counts[name] += 1
        matrix = np.stack(matrices)
        counts = collections.Counter(selected)
        raw_variance = float(matrix.var(axis=0, ddof=0).mean())
        centered = matrix - matrix.mean(axis=1, keepdims=True)
        pair_agreement = sum(c*(c-1) for c in counts.values())/(n*(n-1))
        mean_scores = matrix.mean(axis=0)
        single = r["auxiliary_sum_scores"][f"native_k{depth}"]
        assert single == matrices[0].tolist()
        for name, scores in [(f"joint{depth}_only", np.asarray(single)), (f"cyclic_joint{depth}_only", mean_scores)]:
            assert np.array_equal(scores, r["joint_baseline_scores"][name]), (r["id"], name)
            p = int(scores.argmax())
            assert p == r["predictions"][name] and bool(p == r["gold"]) == r["correct"][name]
            pred[name].append(p)
        stability[depth].append(dict(id=r["id"], unanimous=len(counts)==1,
            pairwise_agreement=pair_agreement, raw_score_variance=raw_variance,
            centered_score_variance=float(centered.var(axis=0, ddof=0).mean()),
            native_correct=pred[f"joint{depth}_only"][-1] == r["gold"],
            cyclic_correct=pred[f"cyclic_joint{depth}_only"][-1] == r["gold"]))

gold = np.asarray([r["gold"] for r in records])
transitions = {}
for prefix in ("joint", "cyclic_joint"):
    transitions[prefix] = paired(np.asarray(pred[prefix+"1_only"]), np.asarray(pred[prefix+"4_only"]), gold)
assert transitions["joint"]["net"] == 387
assert transitions["cyclic_joint"]["net"] == 379
aggregate = {}
for depth, rows in stability.items():
    aggregate[str(depth)] = dict(
        unanimous_questions=sum(r["unanimous"] for r in rows),
        mean_pairwise_agreement=float(np.mean([r["pairwise_agreement"] for r in rows])),
        mean_raw_score_variance=float(np.mean([r["raw_score_variance"] for r in rows])),
        mean_centered_score_variance=float(np.mean([r["centered_score_variance"] for r in rows])),
        strata={str(u): dict(n=sum(r["unanimous"]==u for r in rows),
            native_correct=sum(r["unanimous"]==u and r["native_correct"] for r in rows),
            cyclic_correct=sum(r["unanimous"]==u and r["cyclic_correct"] for r in rows)) for u in (False, True)})

result = dict(status="passed_independent_raw_record_replay", n=1172,
    source_base_commit=public["base_commit"], public_source_and_input_checks=public_checks,
    complete_archive_regular_members_verified=len(manifest),
    raw_score_file_sha256=sha(raw_scores), transitions=transitions,
    per_depth_stability=aggregate, choice_count_distribution=dict(choice_counts),
    tie_counts=dict(tie_counts),
    readout_names_in_actual_cyclic_records=sorted(readout_counts),
    k2_k3_joint_scores_present=False,
    limitations="K1/K4 endpoint comparison only. Rotations correlated within question. Historical exposed test, descriptive only. No causal or unseen-loop timing inference.",
    model_or_gpu_used=False, seconds=time.perf_counter()-started)
(OUT / "root_independent_replay.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf8")
print(json.dumps({k: v for k, v in result.items() if k not in ("transitions", "public_source_and_input_checks")}, indent=2))
print(json.dumps({k: {a:b for a,b in v.items() if not a.endswith("_ids")} for k,v in transitions.items()}, indent=2))

"""Read existing audited metadata; export label-free inputs and independent E5 summaries.

No model execution or mutation of any existing experiment. Run on the source host.
"""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import time
import zipfile

COUNTS = {"train": 1119, "validation": 299, "test": 1172}
COVERAGE_HASHES = {
    "train": "44b6bafcdcf7066859e56e4c83bfb77ec9d7c22f78c4c6952b6315b41e52f16f",
    "validation": "df994533519ce919ade07fa0cfde356a78136c01fcf0832efb0fa238db9247d8",
    "test": "df164143c5921971c54a314ef560b3e3cc8566ff13636a931690efe53943d5e1",
}
ZIP_HASH = "2b19267f48fa42f9cfdc7447cd505f021eb85fd00ef102bd636664b96b24548e"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_lines(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def group(c1, c2):
    return {(False, True): "rescue", (True, False): "damage",
            (True, True): "stable_correct", (False, False): "stable_wrong"}[(c1, c2)]


def summarize(rows):
    n = len(rows)
    correct = [sum(r["correct"][u] for r in rows) for u in range(4)]
    result = {"n": n, "correct": correct,
              "accuracy": [x/n for x in correct],
              "top_ties": [sum(r["top_ties"][u] for r in rows) for u in range(4)],
              "trajectories": dict(sorted(collections.Counter("".join(str(int(x)) for x in r["correct"]) for r in rows).items())),
              "first_correct_loop": dict(sorted(collections.Counter(str(next((u+1 for u, c in enumerate(r["correct"]) if c), "never")) for r in rows).items())),
              "transitions": []}
    for u in range(3):
        rescue = [r for r in rows if not r["correct"][u] and r["correct"][u+1]]
        damage = [r for r in rows if r["correct"][u] and not r["correct"][u+1]]
        deltas = {m: [r["metrics"][u+1][m]-r["metrics"][u][m] for r in rows]
                  for m in ("meanwrong_gap", "k1_fixedwrong_gap", "dynamic_margin")}
        result["transitions"].append({
            "from_loop": u+1, "to_loop": u+2,
            "rescue": len(rescue), "damage": len(damage), "net": len(rescue)-len(damage),
            "wrong_opportunities": n-correct[u], "correct_at_risk": correct[u],
            "conditional_rescue_rate": len(rescue)/(n-correct[u]) if n>correct[u] else None,
            "conditional_damage_rate": len(damage)/correct[u] if correct[u] else None,
            "prediction_changes": sum(r["predictions"][u] != r["predictions"][u+1] for r in rows),
            "mean_metric_deltas": {m: statistics.mean(v) for m,v in deltas.items()},
            "median_metric_deltas": {m: statistics.median(v) for m,v in deltas.items()},
            "rescue_ids": [r["id"] for r in rescue], "damage_ids": [r["id"] for r in damage]})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    root, out = args.root.resolve(), args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("Refuse to replace an existing export namespace")
    out.mkdir(parents=True, exist_ok=True)
    zpath = root/"publication/cleanup_receipts/metadata_preservation_20261112.zip"
    assert sha(zpath.read_bytes()) == ZIP_HASH, "preserved metadata archive identity"
    demo_path = root/"repo/experiments/ouro_weight_space_repair_v2/data/slcd_fixed_demonstrations.json"
    demos = json.loads(demo_path.read_text())
    demo_ids = set(demos["ids"])
    assert len(demo_ids) == 25
    all_inputs, all_labels, audits, source_stats = [], [], [], collections.Counter()
    with zipfile.ZipFile(zpath) as z:
        for split, count in COUNTS.items():
            run = "resume-test-20261111" if split == "test" else "run-20261107T053919Z-6d01138e"
            coverage_path = root/"orchestration_runs"/run/("coverage_"+split+".json")
            cov_bytes = coverage_path.read_bytes()
            assert sha(cov_bytes) == COVERAGE_HASHES[split], ("coverage identity", split)
            coverage = json.loads(cov_bytes)
            assert len(coverage["questions"]) == count
            seen = set()
            for q in sorted(coverage["questions"], key=lambda r:r["id"]):
                ident = q["id"]
                assert ident not in seen and q["status"] == "complete" and not q["errors"]
                seen.add(ident)
                rel = "orchestration_runs/run-20261107T053919Z-6d01138e/shards/"+split+"/"+sha(ident.encode())+".json"
                p = root/rel
                raw = p.read_bytes() if p.exists() else z.read(rel)
                where = "original_file" if p.exists() else "preservation_zip"
                source_stats[where] += 1
                assert sha(raw) == q["metadata_sha256"], ("metadata identity", ident)
                m = json.loads(raw)
                assert m["id"] == ident and m["split"] == split
                records = sorted(m["records"], key=lambda r:r["candidate"])
                assert [r["candidate"] for r in records] == list(range(len(records)))
                ids = records[0]["encoding"]["input_ids"]
                rs = records[0]["encoding"]["predecessor_positions"]
                ih = sha(json.dumps(ids).encode())
                assert len(rs) == 1 and 0 <= rs[0] < len(ids)
                for r in records:
                    e = r["encoding"]
                    assert e["input_ids"] == ids and e["predecessor_positions"] == rs
                    assert r["input_ids_sha256"] == ih and len(e["target_ids"]) == 1
                    assert len(r["scores"]) == 4 and not e.get("truncated", False)
                targets = [r["encoding"]["target_ids"][0] for r in records]
                assert len(set(targets)) == len(targets)
                scores = [[r["scores"][u] for r in records] for u in range(4)]
                pred = [max(range(len(s)), key=lambda j:s[j]) for s in scores]
                gold = m["gold_index"]
                assert 0 <= gold < len(targets)
                if isinstance(m.get("predictions"), list):
                    assert pred == m["predictions"], ("native argmax", ident)
                wrong = [j for j in range(len(targets)) if j != gold]
                rival = max(wrong, key=lambda j:scores[0][j])
                correct = [j == gold for j in pred]
                metrics = [{"meanwrong_gap":s[gold]-statistics.mean(s[j] for j in wrong),
                            "k1_fixedwrong_gap":s[gold]-s[rival],
                            "dynamic_margin":s[gold]-max(s[j] for j in wrong)} for s in scores]
                inp = {"id":ident,"split":split,"ids":ids,"R":rs[0],"targets":targets,
                       "input_ids_sha256":ih,"metadata_sha256":q["metadata_sha256"],
                       "historical_tensor_sha256":q["tensor_sha256"],"metadata_source":where}
                label = {"id":ident,"split":split,"gold":gold,"historical_scores":scores,
                         "historical_predictions":pred,"is_demonstration":ident in demo_ids,
                         "historical_runtime":m["runtime"],"metadata_sha256":q["metadata_sha256"]}
                audit = {"id":ident,"split":split,"gold":gold,"scores":scores,"predictions":pred,
                         "correct":correct,"top_ties":[s.count(max(s)) > 1 for s in scores],
                         "metrics":metrics,"k1_fixed_wrong":rival,"k1_k2_group":group(*correct[:2]),
                         "is_demonstration":ident in demo_ids,"metadata_sha256":q["metadata_sha256"]}
                all_inputs.append(inp); all_labels.append(label); audits.append(audit)
    assert len(all_inputs) == 2590
    train = [r for r in audits if r["split"] == "train" and not r["is_demonstration"]]
    assert len(train) == 1094
    selection = []
    for name in ("rescue", "damage", "stable_correct", "stable_wrong"):
        pool = [r for r in train if r["k1_k2_group"] == name]
        pool.sort(key=lambda r:sha(("ouro-e4-v1|"+r["id"]).encode()))
        assert len(pool) >= 16
        selection += [{"id":r["id"],"historical_group":name,
                       "selection_hash":sha(("ouro-e4-v1|"+r["id"]).encode())} for r in pool[:16]]
    selection.sort(key=lambda r:r["selection_hash"])
    selected_ids = [r["id"] for r in selection]
    for i, r in enumerate(selection):
        r["unrelated_donor_id"] = selected_ids[(i+1) % len(selection)]
    val_ids = sorted(r["id"] for r in all_inputs if r["split"] == "validation")
    val_map = {ident:val_ids[(i+1) % len(val_ids)] for i,ident in enumerate(val_ids)}
    write_lines(out/"all_inputs.jsonl", all_inputs)
    write_lines(out/"all_labels.jsonl", all_labels)
    write_lines(out/"train64.inputs.jsonl", [r for r in all_inputs if r["id"] in set(selected_ids) and r["split"] == "train"])
    write_lines(out/"validation.inputs.jsonl", [r for r in all_inputs if r["split"] == "validation"])
    write_lines(out/"historical_per_question_audit.jsonl", audits)
    write_json(out/"train64_selection.json", {"status":"FROZEN_BEFORE_NEW_INTERVENTIONS","rule":"16 per historical K1/K2 group, SHA256(ouro-e4-v1|ID)","ids":selected_ids,"questions":selection,"is_population_estimate":False})
    write_json(out/"validation_donor_map.json", val_map)
    summaries = {split:summarize([r for r in audits if r["split"] == split]) for split in COUNTS}
    summaries["train_non_demonstration"] = summarize(train)
    write_json(out/"historical_e5_summary.json", {"status":"HISTORICAL_NATIVE_DESCRIPTIVE_ONLY","precision":"Offline FP64 arithmetic on saved FP32 log probabilities; no new model forward", "splits":summaries})
    manifest = {p.name:{"bytes":p.stat().st_size,"sha256":sha(p.read_bytes())} for p in sorted(out.iterdir()) if p.is_file()}
    write_json(out/"EXPORT_MANIFEST.json", {"count":2590,"coverage_sha256":COVERAGE_HASHES,"zip_sha256":ZIP_HASH,
               "demo_manifest_sha256":sha(demo_path.read_bytes()),"source_counts":dict(source_stats),
               "files":manifest,"builder_sha256":sha(Path(__file__).read_bytes()),"seconds":time.time()-started})
    import tarfile
    with (out/"portable_inputs.tar.gz").open("wb") as f:
        with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w|") as tar:
                for p in sorted(out.iterdir()):
                    if p.is_file() and p.name != "portable_inputs.tar.gz":
                        tar.add(p, arcname=p.name)
    print(json.dumps({"out":str(out),"count":len(all_inputs),"source_counts":dict(source_stats),
                      "portable_bytes":(out/"portable_inputs.tar.gz").stat().st_size,
                      "portable_sha256":sha((out/"portable_inputs.tar.gz").read_bytes()),
                      "native_correct":{k:v["correct"] for k,v in summaries.items()},"seconds":time.time()-started}), flush=True)


if __name__ == "__main__":
    main()

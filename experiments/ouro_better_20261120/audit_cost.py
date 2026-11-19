"""Recompute the untouched standalone deployment measurements."""
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent/"iteration1"
POST=HERE/"postgate"
DEPTHS=(1,2,3,4,6)


def load(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source=POST/"STANDALONE_COST_ROWS.jsonl"
    rows=[json.loads(x) for x in source.read_text().splitlines()]
    result=load(POST/"STANDALONE_COST.json")
    manifest=load(POST/"POSTGATE_COST_MANIFEST.json")
    assert len(rows)==result["executions"]==4736
    assert (POST/"UNCONTENDED_GPU_START.txt").read_text().strip()=="pid"
    ids=set(manifest["ids"])
    expected={f"{a}/native_k{k}" for a in ("base","binding","ce","distill","dropout") for k in DEPTHS}
    expected|={f"base/loopcd_{v}_k{k}" for v in ("fixed","adaptive","hidden") for k in DEPTHS[1:]}
    assert len(expected)==37 and len(ids)==64
    grouped=defaultdict(list)
    token_lengths={}
    divergence=Counter()
    predictions={}
    for arm in ("base","binding","ce","distill","dropout"):
        f=load(HERE/"dev"/f"predictions.{arm}.json")
        predictions[arm]={q["id"]:q["original"] for q in f}
    for r in rows:
        arm,method,k=r["arm"],r["method"],r["depth"]
        identifier=r["id"]
        name=f"{arm}/{method}"
        assert identifier in ids and name in expected and r["repeat"] in (0,1)
        assert int(method[-1])==k and r["cache_layers"]==r["physical_layers_per_token"]==24*k
        assert r["vocabulary_projections_per_token"]==(2 if "fixed" in method or "adaptive" in method else 1)
        token_lengths.setdefault(identifier,r["prompt_tokens"])
        assert token_lengths[identifier]==r["prompt_tokens"]
        for col in ("prefill_ms","forced_decode_ms","equal_work_total_ms","natural_total_ms"):
            assert math.isfinite(r[col]) and r[col]>0
        assert math.isclose(r["prefill_ms"]+r["forced_decode_ms"],r["equal_work_total_ms"])
        assert r["incremental_allocated_peak_bytes"]>0 and len(r["natural_generated_ids"])<=12
        divergence[name]+=predictions[arm][identifier][method]["token_id"]!=r["first_token"]
        grouped[name].append(r)
    assert set(grouped)==expected
    fields=("prefill_ms","forced_decode_ms","equal_work_total_ms","natural_total_ms",
            "incremental_allocated_peak_bytes")
    for name,values in grouped.items():
        assert len(values)==128 and {(x["id"],x["repeat"]) for x in values}=={(q,j) for q in ids for j in (0,1)}
        for key in fields:
            z=[r[key] for r in values]
            assert math.isclose(np.median(z),result["costs"][name][key]["median"])
            assert math.isclose(np.quantile(z,.95),result["costs"][name][key]["p95"])
    for k in DEPTHS:
        by_arm={a:defaultdict(list) for a in ("base","binding")}
        for a in by_arm:
            for r in grouped[f"{a}/native_k{k}"]:
                by_arm[a][r["id"]].append(r["equal_work_total_ms"])
        qids=list(by_arm["base"])
        quotients=np.array([np.median(by_arm["binding"][q])/np.median(by_arm["base"][q]) for q in qids])
        sampled=np.random.default_rng(101).integers(0,len(qids),(9999,len(qids)))
        bound=float(np.quantile(np.median(quotients[sampled],axis=1),.95))
        ref=result["binding_vs_base_same_budget"][str(k)]
        assert math.isclose(np.median(quotients),ref["median_prompt_paired_ratio"])
        assert math.isclose(bound,ref["upper_one_sided_95"])
    audit={"method_protocol_depth_cells":len(grouped),"execution_rows":len(rows),
           "common_prompt_ids":len(ids),"repetitions":2,"status":"PASS",
           "first_token_disagreements_cached_batch1_vs_no_cache_batch_by_cell":dict(divergence),
           "raw_sha256":sha(source),"summary_sha256":sha(POST/"STANDALONE_COST.json")}
    (POST/"STANDALONE_COST_AUDIT.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(audit["status"],audit["execution_rows"],"batch_diff",sum(divergence.values()),flush=True)


if __name__=="__main__":
    main()


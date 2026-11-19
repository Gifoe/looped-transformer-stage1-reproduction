"""Independently reconstruct development counts and verify the matched training ledger."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
RESULTS=HERE/"iteration1"
DEPTHS=(1,2,3,4,6)
ARMS=("binding","ce","distill","dropout")


def read(p):
    return json.loads(p.read_bytes())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def independent_parser(t, order):
    if re.fullmatch(r"[ \t]*[A-Ea-e][.)]?[ \t]*",t) is None:
        return None
    loc=ord(t.strip().rstrip(".)").upper())-ord("A")
    return order[loc] if loc<len(order) else None


def main():
    archive=RESULTS/"dev/COMPLETE.json"
    r=read(archive)
    split=read(HERE/"SPLIT_MANIFEST.json")
    source=read(HERE/"EXECUTION_FREEZE.json")
    assert r["execution_freeze_sha256"]==sha(HERE/"EXECUTION_FREEZE.json")
    assert source["split_sha256"]==sha(HERE/"SPLIT_MANIFEST.json")
    for code,dig in source["code_hashes"].items():
        assert sha(HERE/code)==dig
    fit={a["id"] for a in split["cohorts"]["fit"]}
    assert set(read(RESULTS/"frozen_teacher_k6_candidates.json"))==fit
    schedule=read(RESULTS/"TRAIN_EXECUTION.json")["item_order"]
    assert len(schedule)==len(fit)==512 and set(schedule)==fit
    resources={}
    dropout_count=0
    for name in ARMS:
        row=read(RESULTS/f"train.{name}.json")
        assert sha(RESULTS/f"adapter.{name}.safetensors")==row["checkpoint_sha256"]==r["adapter_hashes"][name]
        keys=("train_items","updates","prompt_views","padded_input_tokens",
              "full_physical_layer_calls_per_prompt","trainable_parameters")
        resources[name]=tuple(row[k] for k in keys)
        assert tuple(x["update"] for x in row["training_history"])==tuple(range(1,129))
        for opt in row["training_history"]:
            if name=="dropout":
                assert len(opt["gates"])==1
                batch=opt["gates"][0]
                assert len(batch)==6 and all(len(per_loop)==8 for per_loop in batch)
                assert {v for per_loop in batch for v in per_loop}<={0,2}
                dropout_count+=sum(v==2 for per_loop in batch for v in per_loop)
            else:
                assert opt["gates"]==[]
    assert len(set(resources.values()))==1
    size=set(resources.values()).pop()
    assert size[:3]==(512,128,1024) and size[4:]==(144,1572864)
    expected=[x["id"] for x in split["cohorts"]["dev"]]
    tables={}
    policies_ref={}
    for arm in ("base",*ARMS):
        path=RESULTS/"dev"/f"predictions.{arm}.json"
        assert r["predictions_sha256"][path.name]==sha(path)
        rows=read(path)
        assert [x["id"] for x in rows]==expected and len(rows)==128
        recomputed=defaultdict(Counter)
        for q in rows:
            orig, perm=q["original_order"],q["permuted_order"]
            assert len(orig)==len(perm) and set(orig)==set(perm) and orig!=perm
            same=(orig,perm,q["gold"],q["original_prompt_digest"],q["permuted_prompt_digest"])
            if arm=="base":
                policies_ref[q["id"]]=same
            else:
                assert policies_ref[q["id"]]==same
            for name in q["original"]:
                o,p=(independent_parser(q[k][name]["decoded_subtoken"],order)
                     for k,order in zip(("original","permuted"),(orig,perm)))
                assert (o,p)==(q["original"][name]["exact"],q["permuted"][name]["exact"])
                a,b=(q[k][name]["restricted"] for k in ("original","permuted"))
                gold=q["gold"]
                add={"original_correct":o==gold,"permuted_correct":p==gold,"original_invalid":o is None,
                     "permuted_invalid":p is None,"original_restricted_correct":a==gold,"permuted_restricted_correct":b==gold,
                     "same_content":o is not None and o==p,"same_content_restricted":a==b,
                     "both_exact_correct":o==gold and p==gold,
                     "both_restricted_correct":a==gold and b==gold}
                for k,value in add.items():
                    recomputed[name][k]+=value
        for method, counts in recomputed.items():
            for k,v in counts.items():
                assert r["methods"][arm][method][k]==v
        tables[arm]=recomputed
    greatest=[]
    exact=[]; mean_ce=0; agreement=0
    for k in DEPTHS:
        n=f"native_k{k}"
        inds=[x for x in tables["base"] if x.endswith(f"_k{k}")]
        greatest.append(max(tables["base"][x]["original_correct"] for x in inds))
        chosen=r["same_budget_strength"][str(k)]
        binding=tables["binding"][n]
        ordinary=tables["ce"][n]
        chosen_base=tables["base"][chosen]
        assert chosen_base["original_correct"]==greatest[-1]
        exact.append(binding["original_correct"]-chosen_base["original_correct"])
        mean_ce+=binding["original_correct"]-ordinary["original_correct"]
        agreement+=binding["same_content"]-chosen_base["same_content"]
    expected_pass=(min(exact)>0 and mean_ce>0 and agreement>0)
    assert expected_pass==r["passed"] and exact==r["screen"]["exact_gain_by_depth"]
    return {"status":"PASS","n_dev":len(expected),"fit_resources":resources,
            "dropout_applied":dropout_count,"dropout_sample_count":128*6*8,
            "independent_recomputed_gates":{"absolute_net_by_depth":exact,
                  "mean_gain_vs_ce":mean_ce/5,"mean_same_content_gain":agreement/5,
                  "method_passed":expected_pass},
            "held_out_data_files_read":False}


if __name__=="__main__":
    audit=main()
    (RESULTS/"ITERATION1_AUDIT.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(audit["independent_recomputed_gates"],flush=True)


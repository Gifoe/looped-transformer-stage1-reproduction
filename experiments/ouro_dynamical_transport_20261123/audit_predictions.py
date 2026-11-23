"""Verify intact receipts and test the sealed predictions on fresh evaluation rows."""

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import minimal
from archive_audit import dump, read, sha
from state_horizon import score_arrays


ENDPOINTS=("E1_FRESH_arc","E2_FRESH_arc","HORIZON_arc","E3_FRESH_arc",
           "E1_FRESH_csqa","E2_FRESH_csqa","HORIZON_csqa","E3_FRESH_csqa",
           "E2_FRESH_QASC","E2_FRESH_K2ONLY","CONTEXT_arc","CONTEXT_csqa",
           "E2_MIXED_arc","E2_MIXED_csqa","E3_QASC","PATH_arc","PATH_csqa","PATH_qasc")


def stable_boot(values, seed=41092):
    values=np.asarray(values,dtype=float)
    rng=np.random.default_rng(seed)
    draws=rng.integers(len(values),size=(5000,len(values)))
    return dict(mean=float(values.mean()),ci95=np.quantile(values[draws].mean(1),[.025,.975]).tolist())


def util(row,arm):
    return score_arrays(row["scores"][arm],row["gold"],row["characters"])["utility"]


def raw_checked(folder,file,ids,forecast_time,archived):
    detail=read(file)
    rows=file.with_suffix(".jsonl")
    assert (archived or file.stat().st_mtime>=forecast_time) and sha(rows)==detail["rows_sha256"], file
    output=minimal.load_jsonl(rows)
    have={r["id"] for r in output}
    assert len(have)==len(output) and have<=ids, folder
    return output,detail


def phase_times(root,forecast_time):
    phases=[]
    active=None
    charged=0
    for line in (root/"GPU_LEDGER.tsv").read_text(encoding="utf-8").splitlines():
        tag,phase,time,*rest=line.split("\t")
        time=int(time)
        if tag=="START":
            assert active is None and int(rest[0])==18000-charged
            if phase in ENDPOINTS:
                assert time>=int(forecast_time), phase
            active=(phase,time)
        else:
            end,seconds,status=map(int,rest)
            assert active==(phase,time) and seconds==end-time>=0
            charged+=seconds
            phases.append(dict(phase=phase,seconds=seconds,code=status))
            active=None
    success={p["phase"] for p in phases if p["code"]==0}
    assert active is None and set(ENDPOINTS)<=success and charged<=18000
    return dict(gpu_seconds=charged,successful_stages=sorted(success),trace=phases)


def prediction_stats(rows,direct,cal):
    actual=np.array([util(r,"early")-util(r,"late") for r in rows])
    data={k:np.array([r["diagnostics"][k] for r in rows]) for k in ("linear_defect","finite_gradient_defect")}
    approx=dict(**data,**{f"scaled_{k}":v*cal[k] for k,v in data.items()},discovery_mean=np.full(len(rows),direct))
    results={}
    for k,v in approx.items():
        rm=abs(actual-v)
        rho=float(spearmanr(actual,v).statistic) if v.std()>1e-10 else None
        norm=abs(actual-data["linear_defect"])
        results[k]=dict(mae=float(rm.mean()),mae_vs_jvp=stable_boot(rm-norm),spearman=rho,
                        sign_accuracy=float(np.mean(np.sign(actual)==np.sign(v))),
                        n_negative=int((actual<0).sum()),
                        negative_recall=float(np.mean(v[actual<0]<0)) if np.any(actual<0) else None)
    return dict(point_predictions=results,actual=stable_boot(actual),n=len(rows))


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root",type=Path)
    a=cli.parse_args()
    result=a.root/"results"
    lock_file=result/"FORECAST_LOCK.json"
    lock=read(lock_file)
    mixed=read(result/"MIXED_LOCK.json")
    assert mixed["parent_forecast"]==sha(lock_file)
    path_lock_file=result/"PATH_LOCK.json"
    path_lock=read(path_lock_file)
    assert path_lock["script_sha256"]==sha(a.root/"path_integration.py")
    t=lock_file.stat().st_mtime
    chronology=result/"FORECAST_RECEIPT.json"
    archived=chronology.exists()
    if archived:
        frozen=read(chronology)
        assert frozen["locked_sha256"]==sha(lock_file)
        t=frozen["server_lock_mtime"]
    else:
        # This receipt carries the original server timestamp across Git checkouts.
        dump(chronology,dict(locked_sha256=sha(lock_file),server_lock_mtime=t,
                             source="stat of original lock file on GPU server"))
    confirmation_log=next(x for x in (a.root/"run_confirmation.log",a.root/"logs/run_confirmation.log") if x.is_file())
    assert f"LOCKED {sha(lock_file)}" in confirmation_log.read_text(encoding="utf-8").splitlines()[0]
    out={}
    for tag,by_cell in lock["discovery_sha256"].items():
        path=result/tag
        assert sha(path/"MANIFEST.json")==by_cell["manifest_sha256"]
        assert all(sha(path/f"{k}.json")==v for k,v in by_cell.items() if k!="manifest_sha256")
    data=a.root/"data"
    ids={d:{r["id"] for r in minimal.load_jsonl(data/f"{d}_confirm.jsonl")} for d in ("arc","csqa")}
    assert all(sha(data/f"{d}_confirm.jsonl")==lock["data_sha256"][d] and len(ids[d])==96 for d in ids)
    assert not(ids["arc"]&ids["csqa"])
    new_ids={d:{r["id"] for r in minimal.load_jsonl(data/"path"/f"{d}_confirm.jsonl")} for d in ids}
    assert all(len(new_ids[d])==48 and not(new_ids[d] & (ids["arc"]|ids["csqa"])) for d in ids)
    for stage in ENDPOINTS:
        loc=result/stage
        domain="csqa" if stage.endswith("_csqa") else "arc"
        dataset=ids[domain]
        if stage.startswith("E1"):
            f=read(loc/"SUMMARY.json")
            assert f["sha256_forecast"]==sha(lock_file) and f["n"]==len(dataset)
            for k,row in f["results"].items():
                raw=loc/f"h{k}_projected_rows.json"
                assert sha(raw)==row["projected_rows_sha256"] and set(read(raw))==dataset
            out[stage]=f["results"]
        elif stage.startswith("E2"):
            source={"E2_FRESH_QASC":"ARC_QASC_RECHECK","E2_FRESH_K2ONLY":"ARC_K2ONLY"}.get(stage,
                   "ARC_T1" if domain=="arc" else "CSQA_T1")
            manifest=read(loc/"MANIFEST.json")
            expected_weights=[mixed["controller_sha256"]] if stage.startswith("E2_MIXED") else lock["weights_sha256"]
            if stage.startswith("E2_MIXED"):
                assert set(manifest["paired_cells"])==set(mixed["conditions"])
            assert manifest["controller_sha256"] in expected_weights and manifest["data_sha256"]==lock["data_sha256"][domain]
            out[stage]={}
            for key in manifest["paired_cells"]:
                k,K=key.split(":")
                code=f"k{k}_K{K}"
                rows,rec=raw_checked(loc,loc/f"{code}.json",dataset,t,archived)
                assert len(rows)==manifest["n"]
                assert rec["transport"]["relative_adjoint_discrepancy"]["mean"]<.002
                out[stage][code]=dict(summary=rec,validation=prediction_stats(rows,lock["discovery_values"][source][code],
                                                                            lock["regression"]["zero_intercept"]))
        elif stage.startswith("E3"):
            man=read(loc/"MANIFEST.json")
            assert man["forecast_sha256"]==sha(lock_file) and set(man["paths"])=={"attention","value","query_key","mlp","attention_mlp"}
            out[stage]={}
            for code in [f"k{k}_K{j}" for k,j in (s.split(":") for s in man["cells"])]:
                rows,rec=raw_checked(loc,loc/f"{code}.json",dataset,t,archived)
                assert len(rows)==man["n"] and rec["diagnostics"][0]["native_early_replay_exact"]
                out[stage][code]=dict(results=rec["results"],
                    qk_minus_v=stable_boot([util(r,"restore_query_key")-util(r,"restore_value") for r in rows]),
                    combined_minus_additive_proxy=stable_boot([util(r,"restore_attention_mlp")+util(r,"early")-
                                                               util(r,"restore_attention")-util(r,"restore_mlp") for r in rows]),
                    mean_sign_flip_rate=float(np.mean([x["constant_sign_flip_fraction"] for x in rec["diagnostics"]])),
                    shuffled_cross_question_rate=float(np.mean([x["shuffled_cross_question_fraction"] for x in rec["diagnostics"]])))
        elif stage.startswith("HORIZON"):
            out[stage]={}
            for k in (1,2):
                rows,rec=raw_checked(loc,loc/f"h{k}.json",dataset,t,archived)
                assert len(rows)==72 and rec["forecast_sha256"]==sha(lock_file)
                out[stage][str(k)]=rec
        elif stage.startswith("PATH"):
            tag=stage.removeprefix("PATH_")
            f=read(loc/"SUMMARY.json")
            raw=loc/"PREDICTIONS.jsonl"
            assert sha(raw)==f["rows_sha256"] and sha(path_lock_file)==f["lock_sha256"]
            assert sha(data/"path"/f"{domain}_confirm.jsonl")==path_lock["phases"][tag]["data_sha256"]
            rows=minimal.load_jsonl(raw)
            assert len(rows)==f["n"]==48 and {r["id"] for r in rows}==new_ids[domain]
            out[stage]=f
        else:
            rec=read(loc/"SUMMARY.json")
            raw=loc/"PREDICTIONS.jsonl"
            assert sha(raw)==rec["rows_sha256"] and rec["forecast_sha256"]==sha(lock_file)
            rows=minimal.load_jsonl(raw)
            assert len(rows)==rec["n_matched_prefix"] and all(r["id"] in dataset and r["context"]["donor_id"] in dataset
                                                                       and r["id"]!=r["context"]["donor_id"] for r in rows)
            out[stage]=rec
    receipt=dict(forecast_sha256=sha(lock_file),cohorts_sha256=sha(data/"COHORTS.json"),results=out,
                 gpu=phase_times(a.root,t))
    dump(result/"DYNAMICAL_AUDIT.json",receipt)
    print("PASS",len(out),"independent phase receipts,",receipt["gpu"]["gpu_seconds"],"GPU seconds",flush=True)


if __name__=="__main__":
    main()

"""Out-of-sample coupling test using directions frozen on archived Train."""

import argparse
from pathlib import Path

import numpy as np
import torch

import minimal
from archive_audit import dump, read, sha, weights
from e1_coupling import collect, permute
from functional_operator import batches


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("archive","previous","model","t1","transport","data","forecast-lock","out"):
        p.add_argument(f"--{name}",required=True,type=Path)
    p.add_argument("--domain",choices=("arc","csqa"),required=True)
    a=p.parse_args()
    locked=read(a.forecast_lock)
    if locked["data_sha256"][a.domain]!=sha(a.data) or sha(a.t1) not in locked["weights_sha256"]:
        raise ValueError("Holdout corpus or rank-one checkpoint differs")
    archive,model,tok,rt=minimal.runtime(a.archive,a.previous,a.model)
    rows=minimal.load_jsonl(a.data)
    info,work=batches(archive,tok,rows,4)
    a.out.mkdir(parents=True,exist_ok=True)
    values={}
    for k in (1,2,3):
        moment=a.transport/"results"/"moments"/a.domain/f"h{k}_m1.pt"
        source=torch.load(moment,weights_only=True,map_location="cpu")
        axes={"archived_top":(source["u"][:,0],source["w"][:,0]),"t1":weights(a.t1)}
        matrix, raw=collect(model,archive,info,work,k,1,list(axes.values()))
        full, qavg, null, position, position_null=permute(raw,181820+k)
        options={}
        for idx,axis in enumerate(axes):
            d=position_null[:,idx]
            p_hi=(1+np.count_nonzero(d>=position[idx]))/(1+len(d))
            p_lo=(1+np.count_nonzero(d<=position[idx]))/(1+len(d))
            options[axis]=dict(full=float(full[idx]),question_mean=float(qavg[idx]),role=float(position[idx]),
                               role_null=float(d.mean()),delta_role=float(position[idx]-d.mean()),
                               null95=np.quantile(d,[.025,.975]).tolist(),p_two_sided=float(min(1.,2*min(p_hi,p_lo))),
                               position_permutation=d.tolist())
        evidence={info[i]["id"]:[(role,g.tolist(),h.tolist()) for role,g,h in entries] for i,entries in raw.items()}
        dump(a.out/f"h{k}_projected_rows.json",evidence)
        values[str(k)]=dict(probes=options,train_axes_matrix_sha=sha(moment),native_matrix_frobenius=float(matrix.norm()),
                            projected_rows_sha256=sha(a.out/f"h{k}_projected_rows.json"))
        print(a.domain,k,{x:(y["delta_role"],y["p_two_sided"]) for x,y in options.items()},flush=True)
    dump(a.out/"SUMMARY.json",dict(n=len(rows),results=values,sha256_forecast=sha(a.forecast_lock),runtime=rt))


if __name__=="__main__":
    main()

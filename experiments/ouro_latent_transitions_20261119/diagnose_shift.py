"""Measure where natural paired differences and physical patch inputs diverge."""
import argparse
from pathlib import Path

import numpy as np

from fit_study import Data, Scale, TASKS, export, grouped_rows, load_lines
from summarize_transition import summary


def run(past, runs, out):
    all_rows = []
    categories = []
    for task in TASKS:
        data = Data(past, runs, task, 2048)
        scale = Scale(data.x[data.data_mask("fit")])
        source = runs/task
        raw = np.load(source/"causal_full.npy", mmap_mode="r")
        rows = load_lines(source/"causal.jsonl")
        state = scale.forward(raw).reshape(-1,2,2,5,2048)
        disp = state[:,1]-state[:,0]
        receivers = [data.ids[r["id"],r["world"]] for r in rows]
        donors = [data.ids[r["id"],r["donor"]] for r in rows]
        native = (scale.forward(data.x[donors])-scale.forward(data.x[receivers])).reshape(-1,2,5,2048)
        for split in ("confirm", "transfer"):
            eligible = np.array([r["arm"] == "semantic" and r["split"] == split for r in rows])
            actual = np.array(rows,dtype=object)[eligible]
            for kind, arr in (("natural_world_difference",native),("physical_patch",disp)):
                v = arr[eligible]
                energies = np.sum(v*v, axis=-1)
                for k in (0,1):
                    denominator = np.sum(energies[:,k], axis=-1)
                    for role in range(5):
                        all_rows.extend(grouped_rows(actual,{
                            "energy_share":energies[:,k,role]/np.maximum(denominator,1e-12),
                            "rms":np.sqrt(energies[:,k,role]/2048)
                        },{"task":task,"split":split,"input":kind,"loop":k+1,"role":role}))
            for i in np.flatnonzero(data.data_mask(split)):
                r = data.captured[i]
                gold = r["gold"]
                a,b = map(np.array,r["scores"])
                ok1 = a[gold] > np.max(np.delete(a,gold))
                ok2 = b[gold] > np.max(np.delete(b,gold))
                tracks = np.array(r["scores"])
                margins = tracks[:,gold]-np.max(np.delete(tracks,gold,axis=1),axis=1)
                category = { (False,True):"correction", (True,False):"regression",
                             (True,True):"stay_correct", (False,False):"stay_wrong"}[ok1,ok2]
                categories.append({"task":task,"split":split,"category":category,"id":r["id"],
                    "world":r["world"],"k1_margin":margins[0],"k2_margin":margins[1]})
    export(out/"shift_by_group.csv",all_rows)
    export(out/"shift_summary.csv",summary(all_rows,("task","split","input","loop","role"),
                 ("energy_share","rms")))
    export(out/"natural_accuracy_paths.csv",categories)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    for arg in ("past","runs","output"):
        p.add_argument(arg,type=Path)
    a=p.parse_args()
    run(a.past,a.runs,a.output)

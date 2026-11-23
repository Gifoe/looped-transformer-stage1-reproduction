"""Bind the previously published single mixed checkpoint to the dynamics audit."""

import argparse
from pathlib import Path

from archive_audit import dump, read, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-study",type=Path,required=True)
    p.add_argument("--core-forecast",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    r=read(a.parent_study/"results/forecast_round2/SINGLE_CONTROLLER.json")
    candidate=sha(a.parent_study/r["checkpoint"])
    if candidate!=r["checkpoint_sha256"]:
        raise ValueError("Historical mixed weight differs")
    new=dict(parent_forecast=sha(a.core_forecast),controller_sha256=candidate,conditions=["2:4","4:6"],
             predictions=["JVP-late cosine < .3",
                          "Orthogonal task projection exceeds the along-axis term at each tested loop and domain"])
    if a.out.exists() and read(a.out)!=new:
        raise ValueError("Existing extension lock differs")
    dump(a.out,new)


if __name__=="__main__":
    main()

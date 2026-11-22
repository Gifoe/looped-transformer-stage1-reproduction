"""Compare the two loops' own replayed MLP interaction at the same dose and position."""

import argparse
from pathlib import Path

import torch

from csc_bank import KINDS, load_bank, metadata
from csc_data import repair
from csc_oracle import replay, stat, summary
from native import Forward, load, save, setup


@torch.inference_mode()
def run(model, root):
    assert load(root / "ORACLE_SUMMARY.json")["gate"]["full"]
    m,tok = setup(model)
    f = Forward(m)
    rows = load(root / "NATIVE.json")["rows"]
    cases = [dict(metadata(tok,r,k),repair=k=="joint" and repair(r)) for r in rows for k in KINDS
             if r["form"]=="borrow" and r["split"] in ("test","shift")]
    bank = load_bank(root)
    remap = {}
    for r in cases:
        for p in r["prefixes"]:
            key = r["id"],r["kind"],p
            from_bank = bank[key]
            updates = from_bank["updates"].clone()
            early = from_bank["formed_k1"][1].float()
            late = updates[4]
            updates[0] = early * (late.norm()/early.norm().clamp_min(1e-9))
            updates[1] = early
            remap[key] = dict(base=from_bank["base"],updates=updates)
    arms = ("identity","after_a","after_a_geom","after_m","formed_m","formed_m_geom")
    reverse = ("identity","formed_m","formed_m_geom")
    # The underlying replay helper selects its cached vectors by slot. In this
    # experimental bank, slots 0 and 1 denote the renormalized and raw K1 MLP terms.
    raw = replay(f,tok,cases,remap,arms,root / "FORMATION_RAW.json",rollback=reverse)
    rename = {"after_a":"formed_k1_equal_norm","after_a_geom":"formed_k1_equal_norm_geom",
              "after_m":"formed_k1_natural","formed_m":"formed_k2","formed_m_geom":"formed_k2_geom"}
    translated = [dict(r,pred={rename.get(a,a):v for a,v in r["pred"].items()}) for r in raw]
    reversal, paired = {}, {}
    for split in ("test","shift"):
        q = [r for r in raw if r["split"]==split and r["repair"]]
        reversal[split] = dict(n=len(q),**{rename.get(k,k):sum(r["rollback"][k]["number"]==r["foil"] for r in q)
                                         for k in reverse})
        eligible = [r for r in translated if r["split"]==split]
        paired[split] = dict(all=stat(eligible,"formed_k2","formed_k1_equal_norm"),
                             exact_repair=stat([r for r in eligible if r["repair"]],"formed_k2","formed_k1_equal_norm"))
    save(root / "FORMATION_SUMMARY.json",dict(metrics=summary(translated),rollback=reversal,
                                             versus_equal_norm_k1=paired,raw_arm_names=rename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model",type=Path)
    parser.add_argument("root",type=Path)
    args = parser.parse_args()
    run(args.model,args.root)

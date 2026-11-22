"""Export reusable update coordinates and a layer-wise norm audit from the archived bank."""

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import median

import torch

from csc_bank import LAYER, load_bank
from csc_data import repair
from native import load, save, sha


def formed_geometry(v):
    a,b = v["formed_k1"][1].float(),v["updates"][4].float()
    an,bn = float(a.norm()),float(b.norm())
    return an,bn,an/max(bn,1e-9),float(a@b)/max(an*bn,1e-9)


def geometry_summary(v):
    fields = ("k1_norm","k2_norm","k1_to_k2","cosine")
    return dict(zip(fields,(median(x) for x in zip(*v,strict=True)),strict=True))


def export(root):
    torch.set_num_threads(4)
    lock = load(root / "ORACLE_LOCK.json")
    basis = torch.load(root / "ORACLE_BASIS.pt",map_location="cpu",weights_only=True)["basis"]
    bank = load_bank(root)
    original = {a["id"]:a for a in load(root / "NATIVE.json")["rows"]}
    keys, updates = [], []
    profiles = defaultdict(list)
    formation = defaultdict(list)
    shared_prefix = {}
    examples = defaultdict(list)
    for (rid,kind,prefix),v in bank.items():
        row = original[rid]
        if kind=="joint" and repair(row) and all(tuple(w[3]["tokens"][:len(prefix)])==prefix for w in row["native"]):
            previous = shared_prefix.get(rid)
            if previous is None or len(prefix)>previous[0]:
                shared_prefix[rid] = len(prefix),formed_geometry(v)
        if prefix:
            continue
        update = v["updates"][lock["slot"]].float()
        keys.append((rid,kind))
        updates.append(update)
        if kind=="joint" and repair(row):
            group = row["form"],row["split"]
            profiles[group].append(v["profile"])
            formation[group].append(formed_geometry(v))
            if row["split"] in ("test","shift") and len(examples[group])<4:
                examples[group].append(len(keys)-1)
    all_updates = torch.stack(updates)
    ids = sorted(i for group in examples.values() for i in group)
    torch.save(dict(keys=keys,coordinates=all_updates@basis.T,norm=all_updates.norm(dim=-1),
                    sample_full_updates=all_updates[ids],sample_keys=[keys[i] for i in ids],
                    layer=LAYER+1,definition=lock["update"]),root / "ORACLE_COORDS.pt")
    report = {}
    active = defaultdict(list)
    for rid,value in shared_prefix.items():
        r = original[rid]
        active[r["form"],r["split"]].append(value)
    for key, traces in profiles.items():
        layer_data = {}
        for site in traces[0]:
            a = [x[site] for x in traces]
            layer_data[site] = dict(phase_a=median(t["phase_change"][0] for t in a),
                                   phase_m=median(t["phase_change"][1] for t in a),
                                   mixed_a=[median(t["native_mixed"][k][0] for t in a) for k in (0,1)],
                                   mixed_m=[median(t["native_mixed"][k][1] for t in a) for k in (0,1)])
        pre_digit = active[key]
        report["/".join(key)] = dict(n=len(traces),layers=layer_data,replay_formed_mlp=geometry_summary(formation[key]),
                                    shared_whitespace=dict(with_whitespace=sum(size>0 for size,_ in pre_digit),
                                                           **geometry_summary([v for _,v in pre_digit])))
    save(root / "ORACLE_ATLAS.json",dict(groups=report,bank_entries=len(bank),
                                        max_replay_identity_error=max(v["identity_error"] for v in bank.values()),
                                        coordinates_sha256=sha(root / "ORACLE_COORDS.pt")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    export(parser.parse_args().root)

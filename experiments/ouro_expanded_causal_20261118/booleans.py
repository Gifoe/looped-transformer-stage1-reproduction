"""Fresh Boolean rule tasks for separating logical task competence from source routing."""
import argparse
from pathlib import Path

from transformers import AutoTokenizer

from collect import aligned, dump_line, digest, load_lines, write_json
from factory import Text, rng_for, COLORS, CITIES, NAMES

PREMISES=("Both lights must be on for the alarm to ring. Otherwise the alarm is silent.",
          "The alarm rings when at least one light is on. Otherwise it is silent.",
          "The alarm rings exactly when one light is on and the other is off. Otherwise it is silent.")
SIZES={"dev":30,"confirm":90,"C":90}


def row(split,i,tok,seen):
    family=i%3
    for attempt in range(100):
        rng=rng_for(f"boolean-v1/{split}/{i}/{attempt}")
        place=rng.choice(CITIES)
        employee=rng.choice(NAMES)
        keys=rng.sample(("cup","bell","book","drum","hat","lamp","coin","star","flag"),2)
        labels=rng.sample(keys,2)
        lights=rng.sample(COLORS,4)
        second=(1,0,rng.randrange(2))[family]
        first=rng.randrange(2)
        state=("off","on")
        order=rng.sample(range(2),2)
        emitted=rng.sample(("ringing","silent"),2)
        sig=(family,place,employee,tuple(lights[:2]),tuple(keys),second,first)
        if sig in seen:
            continue
        worlds={}
        for n,a,b,u in (("r",0,0,0),("x",1,0,0),("y",0,1,0),("d",1,1,0),("n",0,0,1),("m",1,1,1)):
            t=Text()
            t.add(f"Use the alarm rules at the {place} station.\n","early_query")
            t.add(f"An unrelated poster is {lights[2+u]}.\n","noise")
            for j in order:
                t.add(f"The {lights[j]} light is ")
                t.add(state[second if j else first^a], "signal" if not j else None)
                t.add(".\n")
            t.add(PREMISES[family]+"\n","bridge")
            end=len(t.value)
            t.add(f"When the alarm is ringing, {employee} takes the {keys[b]}.\n","map")
            t.add(f"When the alarm is silent, {employee} takes the {keys[1^b]}.\n","map")
            t.add(f"Which item does {employee} take?","query")
            p=Text()
            p.value=t.value[:end]+"What state is the alarm in?"
            p.choose(emitted)
            t.choose(labels)
            bit=(first^a)
            ring=(bit and second, bit or second, bit^second)[family]
            worlds[n]={"prompt":t.value,"roles":t.regions,"prefix_chars":end,
                       "probe":p.value,"gold":labels.index(keys[(1-int(ring))^b]),
                       "mid":emitted.index("ringing" if ring else "silent")}
        g={"id":f"logic/boolean_{split}/{i:03}","task":"logic","split":split,"family":family,
           "mode":"lexical","source":"generated Boolean rule","options":labels,
           "intermediate_options":emitted,"meta":{"signature":sig},"worlds":worlds}
        assert worlds["r"]["gold"]==worlds["d"]["gold"]!=worlds["x"]["gold"]
        if aligned(tok,g):
            seen.add(sig)
            return g
    raise ValueError(f"No aligned Boolean realization for {split}/{i}")


def prepare(out,m):
    out.mkdir(exist_ok=True,parents=True)
    tok=AutoTokenizer.from_pretrained(m,trust_remote_code=True,local_files_only=True)
    seen=set()
    for split,count in SIZES.items():
        with (out/f"{split}_input.jsonl").open("x",encoding="utf-8") as f:
            for i in range(count):
                dump_line(f,row(split,i,tok,seen))
    (out/"inputs.jsonl").write_bytes((out/"dev_input.jsonl").read_bytes())


def freeze(out):
    cal=load_lines(out/"calibration"/"rows.jsonl")
    rates={}
    for i in range(3):
        targets=[r for r in cal if r["family"]==i]
        counts={g: [r for r in targets if r["id"]==g] for g in {r["id"] for r in targets}}
        rates[str(i)]=sum(all(q["pred"][3]==q["gold"] for q in v) for v in counts.values())/len(counts)
    files=[out/f"{s}_input.jsonl" for s in SIZES]
    path=out/"inputs.jsonl"
    path.write_bytes(b"".join(f.read_bytes() for f in files))
    write_json(out/"frozen_protocol.json",{"input_sha256":digest(path),
        "intervention_sha256":digest(Path(__file__).with_name("intervene.py")),
        "calibration_sha256":digest(out/"calibration"/"rows.jsonl"),
        "joint_K4_dev_by_family":rates,"natural_competence_gate":.60,
        "inputs":{f.name:digest(f) for f in files}})
    print("Boolean joint K4 development",rates,flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("action",choices=("prepare","freeze"))
    cli.add_argument("--study",type=Path,required=True)
    cli.add_argument("--model",type=Path)
    args=cli.parse_args()
    if args.action=="prepare":
        prepare(args.study,args.model)
    else:
        freeze(args.study)

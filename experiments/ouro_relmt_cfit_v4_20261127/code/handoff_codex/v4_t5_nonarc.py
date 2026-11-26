"""Screen the planned strictly non-ARC sources against archived ARC text-only references."""
import argparse
from pathlib import Path

from stage0_data import gram_keys, items, question_fields, text_fields
from v4_common import save_json, sha256_file
from v4_t5_locked import NONARC, V3, original, read


def run(root):
    out=root/"handoff_codex/T5_NONARC_AUDIT.json"
    if out.exists():
        return
    env=read(V3/"stage0_env/ENV_MANIFEST.json")
    rows=[
        V3/"stage6_d45/anchors/ARC_EASY_TRAIN_TEXT.jsonl",
        V3/"stage5_d6/anchors/huginn_arc_train/train.parquet",
        V3/"stage0_env/UNIFIED_EXT_DEV/arc_easy_val.jsonl",
        Path("/root/ouro-functional-transferability-20261123/data/arc_challenge_validation.jsonl"),
    ]
    rows += [Path(d["path"]) for d in env["input_inventory"]
             if d["name"] in {"arc_challenge","arc_easy"} and d["role"]=="blind_test"]
    exact,grams=set(),set()
    for path in rows:
        for row in items(path):
            exact.update(question_fields(row))
            grams.update(key for text in text_fields(row) for key in gram_keys(text))
    exclusions={}
    for source in NONARC:
        pool=original.resolve_pool_path(source)
        omitted=[]
        total=0
        for row in items(pool):
            total+=1
            why="exact" if exact.intersection(question_fields(row)) else None
            if why is None and any(key in grams for text in text_fields(row) for key in gram_keys(text)):
                why="8gram"
            if why: omitted.append({"id":row["sample_id"],"rule":why})
        if not total>len(omitted):
            raise ValueError("The strict non-ARC filter emptied a declared pool")
        exclusions[source]={"n":total,"excluded":omitted,"pool_sha256":sha256_file(pool)}
    save_json(out,{"method":"ARC text only; exact normalized stem or eight-word shingle","references":{
        str(path):sha256_file(path) for path in rows},"sources":exclusions,"experiment_test_labels_used":False})
    print("V4_NONARC_AUDIT_FROZEN",sha256_file(out),{s:len(v["excluded"]) for s,v in exclusions.items()},flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    run(p.parse_args().root)

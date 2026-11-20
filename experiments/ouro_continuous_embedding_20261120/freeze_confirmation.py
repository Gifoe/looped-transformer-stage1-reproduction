"""Lock all methods and label-aligned ARC intervention prompts before testing."""
import argparse
import hashlib
from pathlib import Path

from transformers import AutoTokenizer

from embedding_run import arc_edits, read_json, write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("results",type=Path)
    a=p.parse_args()
    root=a.results
    manifest=read_json(root/"MANIFEST.json")
    tokenizer=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True)
    arc=arc_edits(tokenizer,manifest)
    assert len(arc)==6
    assert not {g[0]["item"] for g in arc}&{w["item"] for k in ("dev","test") for w in manifest["arc"][k]}
    for original,*donors in arc:
        for donor in donors:
            assert donor["arc_slot_to_content"][donor["y"]]==original["y"]
            assert sorted(donor["arc_slot_to_content"])==list(range(4))
    write_json(root/"ARC_CAUSAL_MANIFEST.json",arc)
    code=("dynamics_embedding.py","embedding_run.py","embedding_experiments.py")
    source=Path(__file__).resolve().parent
    files={p:source/p for p in code}
    files.update({f"results/{p}":root/p for p in ("ARC_CAUSAL_MANIFEST.json","MANIFEST.json","DEVELOPMENT.json")})
    sha={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in files.items()}
    write_json(root/"CONFIRMATION_FREEZE.json",dict(status="FROZEN_BEFORE_HOLDOUT",sha256=sha,
          labeled_arc_ids=[p[0]["item"] for p in arc],gate=2e-5,allow_tf32=False))


if __name__=="__main__":
    main()

"""Separate option-generation correctness changes associated with formatting from choice changes."""
import argparse
from collections import Counter
from pathlib import Path

from minimal import dump, load_jsonl


def run(control, outputs):
    frozen = control / "results/GENERATION_RESULTS"
    result={}
    for split in ("arc_validation","openbookqa_test"):
        depths={}
        for depth in range(1,5):
            base=load_jsonl(frozen/f"{split}__rank1_t1__k{depth}.jsonl")
            revised=load_jsonl(outputs/f"{split}__locked__k{depth}.jsonl")
            old={r["id"]:r for r in base}
            assert len(old)==len(revised) and {r["id"] for r in revised}==old.keys()
            correct=sum(r["correct"] for r in base)
            valid=sum(r["valid"] for r in base)
            record=Counter(n=len(base),old_correct=correct,old_valid=valid,
                           new_correct=sum(r["correct"] for r in revised),new_valid=sum(r["valid"] for r in revised))
            earliest=Counter()
            for row in revised:
                p=old[row["id"]]
                assert p["prompt"]==row["prompt"] and p["gold"]==row["gold"]
                b=p["valid"] and row["valid"]
                if b:
                    record["both_valid"]+=1
                    record["both_valid_old_correct"]+=p["correct"]
                    record["both_valid_new_correct"]+=row["correct"]
                if row["correct"] and not p["correct"]:
                    record["valid_choice_rescue" if p["valid"] else "invalid_to_correct"]+=1
                if p["correct"] and not row["correct"]:
                    record["valid_choice_damage" if row["valid"] else "correct_to_invalid"]+=1
                if p["tokens"]!=row["tokens"]:
                    x,y=p["tokens"],row["tokens"]
                    i=next((j for j,(a,b) in enumerate(zip(x,y)) if a!=b),min(len(x),len(y)))
                    earliest[i]+=1
            depths[f"k{depth}"]=dict(record,first_different_token_index=dict(earliest))
        result[split]=depths
    dump(outputs/"FORMAT_DECOMPOSITION.json",result)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control",type=Path)
    parser.add_argument("outputs",type=Path)
    a=parser.parse_args()
    run(a.control,a.outputs)

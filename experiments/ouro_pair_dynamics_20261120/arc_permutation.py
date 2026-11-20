"""Native option-order diagnostic, with choice scores reindexed by content."""
import argparse
from pathlib import Path
import re

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from causal_worlds import masked, sha
from dynamics_embedding import read_json, write_json
from pair_fields import PairOps, load_runtime, timed
from predictor import batches


def transpose(tokenizer,item):
    txt = tokenizer.decode(item["ids"], clean_up_tokenization_spaces=False)
    options = list(re.finditer(r"(?m)^([A-D])\) (.+)$",txt))
    if [m.group(1) for m in options] != list("ABCD"):
        return None
    indices = [1, 2, 3, 0]
    parts, origin = [], 0
    for next_slot, m in enumerate(options):
        a,b = m.span(2)
        parts.extend((txt[origin:a],options[indices[next_slot]].group(2)))
        origin=b
    swapped = "".join(parts)+txt[origin:]
    ids = tokenizer.encode(swapped,add_special_tokens=False)
    ref = tokenizer.encode(txt,add_special_tokens=False)
    if ref != item["ids"] or not ids or ids[-1]!=item["ids"][-1]:
        return None
    return dict(item,ids=ids,read=len(ids)-1,slot_to_content=indices)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    a = p.parse_args()
    src=a.out/"ARC_PREDICTIONS.json"
    assert sha(src)==read_json(a.out/"ARC_PREDICTION_SHA.json")["sha256"]
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model,tokenizer,_=load_runtime(a.model)
    model.requires_grad_(False).float()
    initial=read_json(a.out/"MANIFEST.json")["arc_confirmation"]
    rows, out, failures, timing=[], [], [], []
    for w in initial:
        changed=transpose(tokenizer,w)
        if changed is not None:
            rows.append(changed)
        else:
            failures.append(w["item"])
    with sdpa_kernel(SDPBackend.MATH):
        for pack in batches(rows,8):
            op=PairOps(model,[masked(w) for w in pack])
            h=op.start
            data=[dict(item=w["item"],ids=w["ids"],slot_to_content=w["slot_to_content"],scores={}) for w in pack]
            for k in (1,2,3,4):
                if k!=1:
                    with timed(op,f"arc_permutation/K{k}",timing):
                        h=op.native_suffix(h)
                for w, z in zip(data,op.scores(h).tolist(),strict=True):
                    w["scores"][str(k)]=[z[w["slot_to_content"].index(content)] for content in range(4)]
            out.extend(data)
    write_json(a.out/"ARC_PERMUTATION.json",dict(rows=out,omitted=failures,
               scope="exploratory native diagnostic, fixed four-way cyclic permutation",timing=timing,arc_prediction_sha256=sha(src)))


if __name__=="__main__":
    main()


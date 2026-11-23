"""Normalize primitive metadata on a verified, locally trained checkpoint."""
import argparse
import json
from pathlib import Path

import torch
from torch.torch_version import TorchVersion

from minimal import dump, sha


def run(root):
    path=root/"rank1_generation.pt"
    receipt=root/"TRAIN_RECEIPT.json"
    info=json.loads(receipt.read_text())
    old=sha(path)
    assert old==info["sha256"] and "previous_serialization_sha256" not in info
    with torch.serialization.safe_globals([TorchVersion]):
        checkpoint=torch.load(path,map_location="cpu",weights_only=True)
    assert checkpoint["form"]=="rank1"
    expected={"encode.weight":(1,2048),"out.weight":(2048,1)}
    assert {name:tuple(tensor.shape) for name,tensor in checkpoint["state"].items()}==expected
    checkpoint["meta"]=json.loads(json.dumps(checkpoint["meta"]))
    temporary=path.with_suffix(".tmp")
    torch.save(checkpoint,temporary)
    check=torch.load(temporary,map_location="cpu",weights_only=True)
    assert all(torch.equal(v,check["state"][k]) for k,v in checkpoint["state"].items())
    temporary.replace(path)
    info["previous_serialization_sha256"]=old
    info["sha256"]=sha(path)
    dump(receipt,info)
    print("Serialized primitive metadata; both trained tensors verified unchanged",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("fit_directory",type=Path)
    run(p.parse_args().fit_directory)

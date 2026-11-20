"""Descriptive full-width witness replay after the primary predictions are frozen."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from causal_worlds import masked, sha
from dynamics_embedding import read_json, write_json
from pair_fields import PairOps, load_runtime, pair, timed
from predictor import edit


def capture(op, initial, method):
    path = {}
    state = initial
    for k in (2, 3, 4):
        for l in range(24):
            state = pair(op, l, state, method, steps=2)
            path[f"K{k}/L{l+1}"] = state.detach().cpu().numpy()
        path[f"K{k}/pre_norm"] = state.detach().cpu().numpy()
        state = op.norm(state) if method == "native" else op.norm_flow(state)
        path[f"K{k}/norm"] = state.detach().cpu().numpy()
    return path


def compare(p, n, start):
    def norm(z):
        return np.linalg.norm(z, axis=-1)
    initial = n - start
    update = p - start
    delta = (p[3] - p[1]) - (p[2] - p[0])
    exact = (n[3] - n[1]) - (n[2] - n[0])
    return dict(relative_token_l2=(norm(p-n)/np.maximum(norm(n),1e-12)).tolist(),
                update_ratio=(norm(update)/np.maximum(norm(initial),1e-12)).tolist(),
                update_cos=((update*initial).sum(-1)/np.maximum(norm(update)*norm(initial),1e-12)).tolist(),
                mixed_cos=((delta*exact).sum(-1)/np.maximum(norm(delta)*norm(exact),1e-12)).tolist(),
                mixed_relative_l2=(norm(delta-exact)/np.maximum(norm(exact),1e-12)).tolist())


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    man = read_json(a.out/"MANIFEST.json")
    phase = read_json(a.out/"FREE_METHOD.json")["phase"]
    pred = a.out/f"PREDICTIONS_{phase}.json"
    assert sha(pred) == read_json(a.out/f"PREDICTION_SHA_{phase}.json")["sha256"]
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, _, _ = load_runtime(a.model)
    model.requires_grad_(False).float()
    rows, timing = [], []
    with sdpa_kernel(SDPBackend.MATH):
        for family, groups in man["groups"][phase].items():
            g = groups[0]
            op = PairOps(model, [masked(w) for w in g["worlds"]])
            start = edit(op, [g])
            h = start.cpu().numpy()
            paths = {}
            for method in ("full", "prefix_anderson", "native"):
                with timed(op, f"states/{family}/{method}", timing):
                    paths[method] = capture(op,start,method)
            # Keep FP32 numerical comparisons. Shard float16 full-width witnesses per transition.
            for k in (2,3,4):
                keys = [f"K{k}/{v}" for v in ("L1","L6","L12","L18","L24","pre_norm","norm")]
                for name, values in paths.items():
                    np.savez_compressed(a.out/f"STATE_{family}_K{k}_{name}.npz",
                        initial=h.astype("float16"), ids=op.ids.cpu().numpy(), keys=keys,
                        states=np.stack([values[x] for x in keys]).astype("float16"))
            rows.append(dict(group=g["group"],family=family,source=g["source"],rule=g["rule"],
                roles=g["worlds"][0]["roles"],read=g["worlds"][0]["read"],
                methods={name:{k:compare(z,paths["native"][k],h) for k,z in records.items()}
                         for name,records in paths.items() if name!="native"}))
    write_json(a.out/"STATE_AUDIT.json",dict(scope="first validation group per family, batch of four lanes; descriptive replay",
              anchor_sha256=sha(pred),timing=timing,rows=rows))


if __name__ == "__main__":
    main()


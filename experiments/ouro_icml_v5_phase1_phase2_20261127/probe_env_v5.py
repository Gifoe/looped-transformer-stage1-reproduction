"""Quick probe on cqa1: model layers, Ouro config, cached models/corpora, and SIGMA_H.pt contents."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/ouro-v4-cfit-20261126")
import torch
import v4_common

print("=== /dev/shm/ouro-cross-model-20261125 ===")
p_cross = Path("/dev/shm/ouro-cross-model-20261125")
if p_cross.exists():
    for x in sorted(p_cross.iterdir()):
        print(" ", x.name)

print("\n=== HF caches ===")
for d in [Path("/root/.cache/huggingface/hub"), Path("/root/.cache/huggingface/datasets"), Path("/root/autodl-pub")]:
    if d.exists():
        print(f"DIR {d}:", [x.name for x in sorted(d.iterdir())[:20]])

print("\n=== fit_unlabeled.jsonl sample ===")
rows = v4_common.load_jsonl(v4_common.UNLABELED_FIT)
print(f"fit_unlabeled count={len(rows)}, keys={list(rows[0].keys())}")
for i in range(3):
    print(f"  [{i}] len={len(rows[i]['text'])} | {rows[i]['text'][:180]!r}")

print("\n=== SIGMA_H.pt keys ===")
sd = torch.load("/root/autodl-tmp/ouro-v4-cfit-20261126/SIGMA_H.pt", map_location="cpu", weights_only=False)
print({k: (tuple(v.shape) if hasattr(v, "shape") else v) for k, v in sd.items()})

print("\n=== Load Ouro-1.4B Base structure ===")
cfg, arch, model, tok = v4_common.load_base_runtime()
c = model.config
print("config:", c.hidden_size, c.intermediate_size, c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads)
print("layer0:", model.model.layers[0])
print("model.model children:", [n for n, _ in model.model.named_children()])
print("embed_tokens:", model.model.embed_tokens)

# Quick centered vs uncentered Sigma_h spectrum check RIGHT HERE (free 1-second check!)
S = sd["sigma_unlab"].double()
mu = sd["mu_unlab"].double()
S_c = S - torch.outer(mu, mu)
for label, M in [("Uncentered E[hh^T]", S), ("Centered Cov(h)", S_c)]:
    ev = torch.linalg.eigvalsh((M + M.T) * 0.5).flip(0).clamp_min(0.0)
    sr = float(ev.sum() / ev[0].clamp_min(1e-12))
    print(f"{label}: trace={ev.sum():.2f}, lam1={ev[0]:.2f}, top1={ev[0]/ev.sum()*100:.2f}%, top16={ev[:16].sum()/ev.sum()*100:.2f}%, stable_rank={sr:.3f}")
mu_norm2 = float(mu @ mu)
print(f"||mu||^2 / trace(E[hh^T]) = {mu_norm2 / float(S.trace()) * 100:.2f}%")

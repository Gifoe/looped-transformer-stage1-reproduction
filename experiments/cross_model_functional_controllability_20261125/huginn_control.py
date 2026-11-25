"""Train a frozen Huginn-0125 model with a pre-recurrence Rank-1 write."""

import argparse
from contextlib import contextmanager
from hashlib import file_digest
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
REPOSITORY = "tomg-group-umd/huginn-0125"
WIDTH = 5280
CAP = .45
DEPTHS = (8, 16, 32)
SEED = 261125137


def sha256(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


class RankOne(nn.Module):
    def __init__(self):
        super().__init__()
        self.read = nn.Linear(WIDTH, 1, bias=False)
        self.write = nn.Linear(1, WIDTH, bias=False)
        nn.init.zeros_(self.write.weight)

    def forward(self, h):
        proposed = self.write(self.read(h))
        rms = h.square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
        update_rms = proposed.square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
        return proposed * (CAP * rms / update_rms).clamp(max=1)


def load_model(path):
    transformers.utils.import_utils._torchvision_available = False
    lock = json.loads((path / "MODEL_LOCK.json").read_text(encoding="utf-8"))
    if (lock["revision"], lock["repository"]) != (REVISION, REPOSITORY):
        raise ValueError("Unexpected Huginn release")
    for name, digest in lock["files"].items():
        if sha256(path / name) != digest:
            raise ValueError(f"Changed Huginn model artifact {name}")
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, local_files_only=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map={"": 0},
    ).requires_grad_(False).eval()
    c = model.config
    if (c.n_embd, c.n_layers_in_prelude, c.n_layers_in_recurrent_block, c.n_layers_in_coda) != (5280, 2, 4, 2):
        raise ValueError("Unexpected Huginn geometry")
    if c.injection_type != "linear" or c.state_init != "like-init" or c.test_time_noise != 0:
        raise ValueError("Unexpected native injection or initialization")
    if model.gradient_checkpointing:
        raise ValueError("The frozen final-iteration forward must retain its active write")
    return model, tok


@contextmanager
def receiver(model, depth, index, controller):
    original = model.core_block_forward
    calls = []

    def run(x, embedded, freqs, mask, cache, block_idx, current_step):
        step = int(current_step)
        calls.append(step)
        if controller is not None and step == depth - 1:
            rows, positions = index
            h = x[rows, positions].float()
            update = (h + controller(h)).to(x.dtype)
            x = x.index_put(index, update)
        return original(x, embedded, freqs, mask, cache, block_idx, current_step)

    model.core_block_forward = run
    try:
        yield calls
    finally:
        del model.core_block_forward


def forward(model, tokens, predictors, head_index, depth, seed, controller=None, cache=None, cache_position=None):
    index = tokens.device.index
    if index is None:
        raise ValueError("A fixed CUDA device is required")
    training = controller is not None and torch.is_grad_enabled()
    if training and cache is not None:
        raise ValueError("Training uses the uncached native suffix")
    steps = (depth - 1, 1) if training else (depth, 0)
    details = dict(return_logits=head_index is not None, return_latents=False,
                   return_head=head_index is None, return_stats=False)

    def select_head(_, args):
        return (args[0][:0, 0] if head_index is None else args[0][head_index],)

    head = model.lm_head.register_forward_pre_hook(select_head)
    try:
        with torch.random.fork_rng(devices=[index]):
            torch.cuda.manual_seed(seed)
            with receiver(model, depth, predictors, controller) as seen:
                out = model(input_ids=tokens, num_steps=steps, output_details=details,
                            use_cache=cache is not None, past_key_values=cache, cache_position=cache_position)
        if seen != list(range(depth)):
            raise RuntimeError(f"Unexpected actual recurrence trace for K{depth}")
        return out.hidden_states[predictors] if head_index is None else out.logits
    finally:
        head.remove()


def learn(a):
    from arc_features import archive as load_archive
    from train import partitions, ranking

    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, tok = load_model(a.model)
    archive = load_archive(a.archive)
    rows = archive.read_data(a.corpus, "train")
    if len(rows) != 1119:
        raise ValueError("Huginn training must cover the full original ARC-Challenge Train population")
    items, metadata = partitions(archive, tok, rows)
    order = np.random.default_rng(SEED).permutation(len(rows)).tolist()
    for start in range(0, len(order), 64):
        order[start:start + 64] = sorted(order[start:start + 64], key=lambda q: max(len(x[0]) for x in items[q]))
    net = RankOne().cuda()
    if sum(p.numel() for p in net.parameters()) != 10560:
        raise ValueError("Unexpected Rank-1 parameter count")
    opt = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.03)
    a.out.mkdir(parents=True, exist_ok=True)
    work = a.out / "TRAIN_STATE.pt"
    lock = dict(revision=REVISION, source_sha256=sha256(a.corpus / "train.parquet"),
                depth_schedule=list(DEPTHS), cap=CAP, seed=SEED, batch=4, objective="historic_t1_raw_and_normalized")
    log, elapsed, peak = [], 0., 0
    if work.exists():
        prior = torch.load(work, weights_only=True, map_location="cpu")
        if prior["lock"] != lock:
            raise ValueError("Existing Huginn training state has a different frozen specification")
        net.load_state_dict(prior["controller"])
        opt.load_state_dict(prior["optimizer"])
        log, elapsed, peak = prior["log"], prior["seconds"], prior["peak_allocated_bytes"]
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for i in range(len(log), 280):
        selected = order[4*i:4*i+4]
        ids, rows, pos, targets = archive.pack([member for j in selected for member in items[j]])
        where = (rows, pos)
        depth = DEPTHS[i % len(DEPTHS)]
        logits = forward(model, ids, where, where, depth, SEED + i, net)
        token_scores = -F.cross_entropy(logits, targets, reduction="none")
        loss, _ = ranking(token_scores, rows, metadata, selected)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
        if not torch.isfinite(norm):
            raise RuntimeError(f"Nonfinite Huginn controller gradient at update {i + 1}")
        opt.step()
        log.append(dict(update=i+1, depth=depth, loss=float(loss.detach()), gradient_norm=float(norm.detach())))
        if (i + 1) % 10 == 0:
            temp = work.with_suffix(".partial")
            torch.save(dict(lock=lock, controller=net.state_dict(), optimizer=opt.state_dict(), log=log,
                            seconds=elapsed+time.perf_counter()-start,
                            peak_allocated_bytes=max(peak, torch.cuda.max_memory_allocated())), temp)
            temp.replace(work)
            print("HUGINN FIT", i + 1, depth, round(log[-1]["loss"], 5), flush=True)
    output = a.out / "rank1_huginn.pt"
    temp = output.with_suffix(".partial")
    torch.save(dict(state={key: value.detach().cpu() for key, value in net.state_dict().items()}, meta=lock), temp)
    temp.replace(output)
    report = dict(lock=lock, steps=len(log), examples=len(order), trainable_params=10560, log=log,
                  checkpoint_sha256=sha256(output), seconds=elapsed+time.perf_counter()-start,
                  peak_allocated_bytes=max(peak, torch.cuda.max_memory_allocated()))
    (a.out / "FIT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "archive", "corpus", "out"):
        p.add_argument(f"--{name}", required=True, type=Path)
    learn(p.parse_args())

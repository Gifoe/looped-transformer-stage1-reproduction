"""Train a shared 4,096-parameter Rank-1 with exact question and view schedules."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import torch

from evaluate_generation import generate
import minimal
from task_data import encode_target, load
from train_generative import next_logits, normalized_nll, pack_entries, rank_loss


DEPTHS = (2, 3, 4)
QUOTAS_PER_STEP = (2, 2, 4)
SETS = {"A": ("obqa", 0.), "B": ("obqa", .25), "C": ("scienceqa", None), "D": ("scienceqa", None)}


def source_order(rows, seed, epoch, source, width):
    rng = np.random.default_rng([seed, epoch, sum((i + 1) * ord(c) for i, c in enumerate(source))])
    index = rng.permutation(len(rows))
    result = []
    for start in range(0, len(index), 40 * width):
        window = [rows[j] for j in index[start:start+40*width]]
        window.sort(key=lambda r: next(iter(r['views'].values()))['input_tokens'])
        batches = [window[i:i+width] for i in range(0, len(window), width)]
        result.extend(batches[j] for j in rng.permutation(len(batches)))
    return result


def epoch_schedule(data, recipe, seed, epoch):
    sources = ("csqa", SETS[recipe][0], "gsm8k")
    batches = [source_order(data[src], seed, epoch, src, n) for src, n in zip(sources, QUOTAS_PER_STEP, strict=True)]
    for step, (csqa, science, gsm) in enumerate(zip(*batches, strict=True)):
        depth = DEPTHS[((epoch - 1) * len(batches[0]) + step) % len(DEPTHS)]
        science_view = (epoch + step) % 2
        math_view = (epoch + step) % 4
        selected = [(r, "short") for r in csqa]
        selected.extend((r, "explanation" if recipe == "D" and i == science_view else "short") for i, r in enumerate(science))
        selected.extend((r, "final" if i == math_view else "process") for i, r in enumerate(gsm))
        yield depth, selected


def pools(data):
    info = json.loads((data / "MANIFEST.json").read_text())
    records = {}
    ids = set()
    for source, spec in info["sources"].items():
        path = data / f"{source}_train.jsonl"
        if minimal.sha(path) != spec["splits"]["train"]["sha256"]:
            raise ValueError(f"Manifest does not match {path}")
        records[source] = load(path)
        for row in records[source]:
            if row["sample_id"] in ids:
                raise ValueError(f"Repeated question {row['sample_id']}")
            ids.add(row["sample_id"])
    return records


def digest_factors(net):
    payload = sha256()
    for p in net.parameters():
        payload.update(p.detach().cpu().numpy().tobytes())
    return payload.hexdigest()


def snapshot(path, obj):
    tmp = path.with_suffix(".part")
    torch.save(obj, tmp)
    tmp.replace(path)


def features(tok, rows):
    result = {}
    for group in rows.values():
        for item in group:
            for key, row in item["views"].items():
                prefix, full, size = encode_target(tok, row)
                if size != row["response_tokens"] or full[-1] != tok.eos_token_id or tok.eos_token_id in full[:-1]:
                    raise ValueError(f"Bad EOS or response mask at {item['sample_id']}/{key}")
                result[item["sample_id"], key] = prefix, full, size
    return result


@torch.enable_grad()
def train(a, archive, model, tok, rows, runtime):
    a.out.mkdir(parents=True, exist_ok=True)
    minimal.dump(a.out / "RUNTIME.json", runtime)
    model.eval().requires_grad_(False)
    torch.manual_seed(a.seed)
    net = minimal.Minimal("linear", 1, .45).cuda()
    params = list(net.parameters())
    if any(p.dtype != torch.float32 for p in params) or sum(p.numel() for p in params) != 4096 or any(p.requires_grad for p in model.parameters()):
        raise ValueError("The run must fit exactly 4,096 FP32 parameters with a frozen Ouro backbone")
    initial = digest_factors(net)
    start_epoch, start_step = 1, 0
    checkpoint = a.out / "resume.pt"
    optimizer = torch.optim.AdamW(params, lr=a.lr, weight_decay=.03)
    rule = dict(recipe=a.recipe, rank_weight=a.rank_weight, seed=a.seed, initial_factors_sha256=initial, lr=a.lr,
                dataset_sha256=minimal.sha(a.data / "MANIFEST.json"), code_sha256=minimal.sha(Path(__file__)),
                model_sha256=runtime["model_sha256"], depths=list(DEPTHS), per_step=list(QUOTAS_PER_STEP),
                epochs=a.epochs, microbatch=a.microbatch, smoke_updates=a.smoke_updates, max_work=a.max_work)
    lock = a.out / "TRAIN_LOCK.json"
    if lock.exists() and json.loads(lock.read_text()) != rule:
        raise ValueError(f"Conflicting run lock at {lock}")
    minimal.dump(lock, rule)
    history, seconds = [], 0.
    if checkpoint.exists():
        old = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if old["lock"] != minimal.sha(lock):
            raise ValueError("Cannot restore an optimizer against a changed run lock")
        net.load_state_dict(old["net"])
        optimizer.load_state_dict(old["optimizer"])
        start_epoch, start_step, history, seconds = old["epoch"], old["step"], old["history"], old["seconds"]
    encoded = features(tok, rows)
    completed_work = sum(row['padded_token_loops'] for row in history)
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    def save_state(epoch, step):
        snapshot(checkpoint, dict(net=net.state_dict(), optimizer=optimizer.state_dict(), lock=minimal.sha(lock),
                                  epoch=epoch, step=step, history=history, seconds=seconds + time.perf_counter() - start))

    for epoch in range(start_epoch, a.through_epoch + 1):
        plan = list(epoch_schedule(rows, a.recipe, a.seed, epoch))
        if a.smoke_updates:
            plan = plan[:a.smoke_updates]
        off = start_step if epoch == start_epoch else 0
        for step, (depth, items) in enumerate(plan[off:], start=off):
            if a.max_work and completed_work >= a.max_work:
                break
            optimizer.zero_grad(set_to_none=True)
            generated, works, target_tokens = 0., 0, 0
            for i in range(0, len(items), a.microbatch):
                chosen = items[i:i+a.microbatch]
                x = pack_entries(archive, [encoded[r["sample_id"], tag] for r, tag in chosen])
                if x[-1].numel() != sum(r["views"][tag]["response_tokens"] for r, tag in chosen):
                    raise ValueError("Packed mask includes extra or missing answer tokens")
                loss = normalized_nll(next_logits(model, x, depth, net), x, len(chosen)) * (len(chosen) / len(items))
                loss.backward()
                generated += float(loss.detach())
                works += x[0].numel() * depth
                target_tokens += x[-1].numel()
            ranked = None
            if a.rank_weight:
                candidates = [r["views"][tag]["candidate"] for r, tag in items if "candidate" in r["views"][tag]]
                ranked, cost = rank_loss(archive, model, tok, net, candidates, depth)
                (a.rank_weight * ranked).backward()
                works += cost
            grad = float(torch.nn.utils.clip_grad_norm_(params, 1.))
            if not np.isfinite(grad) or grad <= 0:
                raise RuntimeError(f"No finite controller gradient at {epoch}/{step}")
            optimizer.step()
            completed_work += works
            trace = dict(epoch=epoch, step=step+1, depth=depth, nll=generated, grad=grad,
                         ranking=float(ranked.detach()) if ranked is not None else None,
                         target_tokens=target_tokens, padded_token_loops=works,
                         views=[(r["dataset"], r["sample_id"], tag, r["views"][tag]["task_type"],
                                 r["views"][tag]["response_tokens"]) for r, tag in items],
                         cumulative_seconds=seconds + time.perf_counter() - start)
            history.append(trace)
            if step == 0 or (step+1) % 50 == 0:
                print(epoch, step+1, depth, round(generated, 4), target_tokens, round(trace["cumulative_seconds"],1), flush=True)
            if (step+1) % 50 == 0:
                save_state(epoch, step+1)
        torch.cuda.synchronize()
        folder = a.out / f"epoch_{epoch:02d}"
        folder.mkdir(exist_ok=True)
        p = folder / "controller.pt"
        minimal.save_net(p, net.eval(), dict(epoch=epoch, parameters=4096, recipe=a.recipe, seed=a.seed))
        view_counts = Counter((r["depth"], source, kind) for r in history for source, _, _, kind, _ in r["views"])
        exposure = {f"K{k}/{source}/{view}": count for (k, source, view), count in sorted(view_counts.items())}
        minimal.dump(folder/"TRAIN.json", dict(checkpoint_sha256=minimal.sha(p), parameters=4096, optimizer_updates=len(history),
                     exposed_views=exposure, target_tokens=sum(x["target_tokens"] for x in history),
                     padded_token_loops=sum(x["padded_token_loops"] for x in history), seconds=seconds+time.perf_counter()-start,
                     peak_cuda_bytes=torch.cuda.max_memory_allocated(), logs=history))
        save_state(epoch+1, 0)
        print("CHECKPOINT", epoch, len(history), p, flush=True)
        if a.max_work and completed_work >= a.max_work:
            minimal.dump(a.out/'WORK_STOP.json',dict(epoch=epoch, updates=len(history), target=a.max_work,
                         actual=completed_work, overshoot=completed_work-a.max_work, checkpoint=str(p), sha256=minimal.sha(p)))
            break
    if a.smoke_updates:
        sample = rows["gsm8k"][0]["views"]["final"]
        with torch.inference_mode():
            generated, time_receipt = generate(archive, model, tok, net, [sample], "gsm_final", 2, 1, 8)
        minimal.dump(a.out / "GENERATION_SMOKE.json", dict(rows=generated, cost=time_receipt))
    print("DONE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "previous", "model", "data", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--recipe", required=True, choices=SETS)
    parser.add_argument("--rank-weight", type=float, choices=(0., .25))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--through-epoch", type=int, default=3)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--smoke-updates", type=int)
    parser.add_argument("--max-work",type=int,help="Stop after first completed optimizer step reaching padded token×loop cap")
    a = parser.parse_args()
    a.rank_weight = SETS[a.recipe][1] if a.recipe in ("A", "B") else a.rank_weight
    if a.rank_weight is None or not 1 <= a.through_epoch <= a.epochs:
        raise ValueError("Freeze a rank objective and validate the epoch limits")
    rows = pools(a.data)
    archive, model, tok, runtime = minimal.runtime(a.archive, a.previous, a.model)
    train(a, archive, model, tok, rows, runtime)


if __name__ == "__main__":
    main()

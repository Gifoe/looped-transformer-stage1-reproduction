"""Audit loaded weights and profile fixed-shape CUDA microbenchmarks."""
import argparse
import hashlib
from pathlib import Path

import torch

from study import NativeBatch, dump, load_runtime
from study import twohop


def model_digest(model):
    checksum = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        checksum.update(name.encode("utf-8") + b"\0")
        checksum.update(str((param.dtype, tuple(param.shape))).encode("ascii") + b"\0")
        checksum.update(param.detach().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return checksum.hexdigest()


def measure(fn):
    fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    finish = torch.cuda.Event(enable_timing=True)
    millis = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        start.record()
        fn()
        finish.record()
        finish.synchronize()
        millis.append(start.elapsed_time(finish))
    return dict(cuda_event_ms=millis, median_ms=sorted(millis)[1],
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    args = p.parse_args()
    model, tok, runtime = load_runtime(args.model)
    before = model_digest(model)
    for attempt in range(200):
        try:
            worlds = twohop(tok, "development", 0, attempt)
        except ValueError:
            continue
        break
    else:
        raise RuntimeError("No valid fixed timing group")
    batch = NativeBatch(model, worlds[:4], trace=True, parity=True)

    def native():
        model.model.total_ut_steps = 2
        model(input_ids=batch.ids, use_cache=False, exit_at_step=1)

    entries = {"native_K2_full_model_and_head": measure(native)}
    for m in (1, 2, 4, 8, 16, 32):
        entries[f"A{m}_conditioned_on_K1"] = measure(lambda: batch.euler(m))
        entries[f"B{m}_conditioned_on_K1"] = measure(lambda: batch.branch_path(m))
    after = model_digest(model)
    assert after == before
    dump(args.out / "GPU_TIMING.json", dict(runtime=runtime, shape=list(batch.ids.shape),
         weights_before=before, weights_after=after, weight_parity="PASS",
         scope="A/B CUDA events cover the constructed K1-to-end interval without output head. Native CUDA events cover both loops and dense head. Peaks include resident weights and the captured test batch.",
         measurement=entries))


if __name__ == "__main__":
    main()

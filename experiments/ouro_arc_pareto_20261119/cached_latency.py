"""Run the frozen latency protocol with the supported HF growing cache."""
import argparse
import json
from pathlib import Path
from diagnostics import latency
from run_arc import dump, load_runtime, read_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    model, tokenizer, _ = load_runtime(args.model)
    from transformers import DynamicCache

    def supply_cache(_, positional, kwargs):
        if kwargs.get("use_cache") and kwargs.get("past_key_values") is None:
            cache = DynamicCache()
            slots = model.model.total_ut_steps * len(model.model.layers)
            cache.append_new_layers(slots - 1)
            assert len(cache.layers) == slots
            kwargs["past_key_values"] = cache
        return positional, kwargs

    # The checkpoint cache assigns removed writable cache attributes in HF
    # 4.55. Ouro natively accepts an existing growable transformers cache.
    handle = model.model.register_forward_pre_hook(supply_cache, with_kwargs=True)
    try:
        latency(model, tokenizer, read_data(args.data, "validation"), args.out)
    finally:
        handle.remove()
    path = args.out / "latency.json"
    records = json.loads(path.read_text())
    records["protocol"] = (
        "Ouro native recurrent KV path with explicit Transformers DynamicCache; batch 1; "
        "fresh cache per prompt; one forced token after prefill; synchronized CUDA wall time"
    )
    records["cache_note"] = "Pinned Ouro default UniversalTransformerCache fails construction with Transformers 4.55.4"
    dump(path, records)


if __name__ == "__main__":
    main()

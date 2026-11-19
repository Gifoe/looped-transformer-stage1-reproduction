"""Audit six real recurrent passes and run standalone native cache timing."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import pyarrow.parquet as pq
import torch
from transformers import DynamicCache
from transformers.masking_utils import create_causal_mask

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ouro_arc_pareto_20261119"))
from run_arc import load_runtime

DEPTHS = (1, 2, 3, 4, 6)


def digest(x):
    return hashlib.sha256(x.encode()).hexdigest()


def rendered(row):
    choices = "".join(f"\n{chr(65+i)}. {c}" for i,c in enumerate(row["choices"]["text"]))
    return f'Question: {row["question"]}' + choices + "\nAnswer:"


def set_depth(model, k):
    model.model.total_ut_steps = k
    model.config.total_ut_steps = k


def unrolled(model, ids, K):
    core = model.model
    h = core.embed_tokens(ids)
    positions = torch.arange(ids.shape[1], device=ids.device)
    position_ids = positions[None, :]
    mask = create_causal_mask(config=model.config, input_embeds=h, attention_mask=None,
                              cache_position=positions, past_key_values=None, position_ids=position_ids)
    rope = core.rotary_emb(h, position_ids)
    output = {}
    for loop in range(K):
        for physical_layer in core.layers:
            assert physical_layer.attention_type == "full_attention"
            h = physical_layer(h, attention_mask=mask, position_ids=position_ids, past_key_value=None,
                               use_cache=False, cache_position=positions, position_embeddings=rope,
                               current_ut=loop)
        h = core.norm(h)
        output[loop+1] = h
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("train", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    model, tokenizer, env = load_runtime(args.model)
    rows = pq.read_table(args.train, columns=["id","question","choices"]).to_pylist()
    selected = sorted(rows, key=lambda x: digest("ouro-better-v1:native-parity:"+x["id"]))[:6]
    audited = []
    for row in selected:
        prompt = rendered(row)
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        x = torch.tensor([token_ids], device="cuda")
        set_depth(model, 6)
        stack, hooks = [], []
        for l, layer in enumerate(model.model.layers):
            def step(_, positional, kw, layer_index=l):
                stack.append((kw["current_ut"],layer_index))
            hooks.append(layer.register_forward_pre_hook(step, with_kwargs=True))
        try:
            _, native, _ = model.model(input_ids=x,use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        expected = [(k,l) for k in range(6) for l in range(24)]
        assert stack == expected
        reference = unrolled(model, x, 6)
        standalone = {}
        for k in DEPTHS:
            assert torch.equal(native[k-1],reference[k]), (row["id"],k)
            expected_logits = model.lm_head(native[k-1][:,-1:]).float()
            set_depth(model,k)
            z = model(input_ids=x, use_cache=False, exit_at_step=k-1, logits_to_keep=1).logits.float()
            assert torch.equal(z,expected_logits), (row["id"],k)
            cache = DynamicCache()
            cache.append_new_layers(24*k-1)
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = model(input_ids=x, past_key_values=cache, use_cache=True, exit_at_step=k-1, logits_to_keep=1)
            torch.cuda.synchronize()
            prefill_ms = 1000*(time.perf_counter()-start)
            assert out.past_key_values is cache and len(cache.layers)==24*k
            assert all(layer.get_seq_length()==len(token_ids) for layer in cache.layers)
            prefill_difference=float((z-out.logits.float()).abs().max())
            prefill_agreement=bool(z.argmax(-1).item()==out.logits.argmax(-1).item())
            forced = torch.tensor([[int(z.argmax(-1).item())]], device="cuda")
            torch.cuda.synchronize()
            start = time.perf_counter()
            dec = model(input_ids=forced, past_key_values=cache, use_cache=True,
                        exit_at_step=k-1, logits_to_keep=1).logits
            torch.cuda.synchronize()
            decode_ms = 1000*(time.perf_counter()-start)
            n = model(input_ids=torch.cat((x,forced),1), use_cache=False,
                      exit_at_step=k-1, logits_to_keep=1).logits
            assert all(layer.get_seq_length()==len(token_ids)+1 for layer in cache.layers)
            standalone[k] = {
                "cache_layers":len(cache.layers), "prefill_ms":prefill_ms, "decode_ms":decode_ms,
                "cached_vs_uncached_prefill_max_abs_logit":prefill_difference,
                "cached_vs_uncached_prefill_top1_equal":prefill_agreement,
                "cache_vs_full_extended_max_abs_logit":float((dec.float()-n.float()).abs().max()),
                "cache_vs_full_extended_top1_equal": bool(dec.argmax(-1).item()==n.argmax(-1).item()),
            }
        audited.append({"id":row["id"],"prompt_sha256":digest(prompt),"prompt_tokens":len(token_ids),
                        "exact_native_layer_calls":len(stack),"depths":standalone})
    summary = {"status":"PASS","outcomes":audited,"runtime":env,
               "scope":"Unlabeled ARC Train native technical parity, six fixed prompts. Timing diagnostic only."}
    args.out.write_text(json.dumps(summary,indent=2)+"\n")
    print("PASS",len(audited),len(DEPTHS),sum(u["exact_native_layer_calls"] for u in audited),flush=True)


if __name__ == "__main__":
    main()


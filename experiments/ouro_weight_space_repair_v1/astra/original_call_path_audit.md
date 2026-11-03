# Exact original-call parity audit

Status: source review by Astra; the root agent subsequently restored the exact legacy call and executed the comparison on the first3 fixed discovery questions. All original-versus-current score/token/hidden errors and independent-depth errors were zero (`results/original_call_parity/COMPLETE.json`). Astra did not execute GPU jobs or change the scorer. Earlier explicit-mask studies are superseded engineering records.

**Finding about the superseded scorer:** its explicit all-one attention mask changed HF 4.55.4 attention dispatch on PyTorch >=2.6. The accepted scorer now omits this mask on the uncached path.

The old call supplies only `input_ids` and `use_cache=False`. Pinned Ouro internally constructs zero-based consecutive positions. HF mask preprocessing then derives a packed-sequence index tensor whenever the attention mask and cache are absent. Even ordinary consecutive positions produce a tensor, not `None`. For PyTorch >=2.6, that tensor disables skipping mask construction. The all-one-mask call bypasses this branch and can return no materialized mask. Both encode ordinary causal attention mathematically, but they take different numerical paths. [HF 4.55.4 masking source](https://raw.githubusercontent.com/huggingface/transformers/v4.55.4/src/transformers/masking_utils.py)

HF's SDPA integration also uses the presence of an attention mask to choose between explicit repeated key/value heads and `enable_gqa=True`; it derives the SDPA causal flag from mask absence. Actual kernel selection remains runtime-dependent, so the source identifies differing dispatch inputs rather than proving a particular GPU kernel executed. [HF 4.55.4 SDPA source](https://raw.githubusercontent.com/huggingface/transformers/v4.55.4/src/transformers/integrations/sdpa_attention.py)

The pinned local Ouro config has default RoPE, no sliding attention and no attention dropout. Explicit zero-based `position_ids`/`cache_position` equal the values generated internally for an uncached full sequence. RoPE itself is therefore not the identified distinction. HFLM numerical tolerance does not prove equivalence to the old call; it is a separate comparison.

## Minimal runtime comparison

`scripts/check_original_call_parity.py` defaults to the first3 fixed discovery questions (the actual acceptance run); optional low-margin engineering sampling exists but was not used. It evaluates all answer candidates through:

1. The exact old `model.model(input_ids=x, use_cache=False)` call at Kmax6 and original slice-based BF16 head/FP32 log-softmax.
2. The same direct readout with explicit metadata, isolating the mask/position change.
3. The current scorer, to expose any additional scoring differences.

It captures only first-attention mask/position metadata, continuation states, scores, predictions and margins. It repeats the original call on one candidate per question to measure repeat stability, saves actual installed masking/SDPA source hashes, and never reads validation/test examples. It does not adjust acceptance tolerances. Run under the normal independent-budget supervisor.

## Consequence for mechanism conclusions

If numerical differences are material relative to the small discovery margins, or any relevant answer/transition classification flips, reset accepted discovery and causal studies to the exact original no-mask call. Preserve earlier results as separately labeled engineering artifacts and keep their charged compute. Do not combine their selected failures or intervention rankings with a newly changed kernel as if the protocol were continuous.

Even if this tiny check has no flips, it cannot establish cohort-wide classification invariance. Source-confirmed different dispatch already rules out the phrase “identical original path.” Given the task's explicit reuse requirement and the unusually small number of harmful transitions, returning to the exact call is preferable to inventing a looser numerical tolerance. Freeze the corrective decision before comparing intervention outcomes under the revised path.

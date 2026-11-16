# Execution and numerical audit

All work and scientific analysis were performed by Sol. ASTRA_FINAL_DECISION.md retains the requested filename; it is not Astra-authored. No subagent, backbone training, LoRA, new Gate, additional parameter grid, layer/head search or Test access was used.

## Source and protocol

Reference 7857ee2e4a7413d52fb1fc8f2d641e10056a406c; completed CATC source/results 17c957cbca8b3a883ff7e4ca5d3b97200c3aa9ff. The fixed Ouro checkpoint, actual imported modeling source, BF16 SDPA, Torch/CUDA/Transformers versions and GPU identity are recorded. Original model parameters require no gradients and remain unchanged. Full 25-shot inputs and original target IDs/order are reused by hash; Q excludes final R. Only K2 R at pre-o_proj L10-16 is directly written. Original projection/normalization/residual/MLP and natural K3/K4 execute. All 96 physical-layer/pass events and non-R equality are asserted.

Fresh three-case zero-coefficient execution matches all four original 49152-dimensional outputs bitwise. All four nonzero fixed settings also match original K2/K4 heads exactly. Original and symmetric decomposition errors are recorded on 21 technical sites. These are algebraic decompositions, not unique accuracy mechanisms. FP64 probabilities are reconstructed from actual post-RoPE Q/K and full causal normalization; original fused SDPA remains untouched.

Actual-dose matching uses deterministic outcome-free scalar search after BF16 writeback, 32 bracket and 32 bisection iterations, choosing the closest feasible candidate under the 10% cap. Every site's target, realized norm, error, amplitude and cap is retained; unmatched cases are not excluded. The selected structured method is the original sequential feedback variant: later K2 sites consume upstream modified states.

## Measured serial deployment

Three original technical questions, one warmup plus three measured repetitions each, four full recurrent passes and one full-shaped K4 head. Every full-vocabulary K4 endpoint matches the corresponding collector.

| Policy | Median seconds | Relative to native | Peak allocated GiB |
|---|---:|---:|---:|
| native | 0.18621 | 1.000 | 3.082 |
| structured | 0.19285 | 1.036 | 3.082 |
| matched_random | 0.21795 | 1.170 | 3.082 |
| matched_scaling | 0.21746 | 1.168 | 3.082 |

Capture times and matched deployment times are separate. Old native capture uses four output heads while modified capture uses two; their capture timers are not matched inference overhead. Experimental random/scaling controls use companion structured norm targets; reported control timings exclude acquiring those targets. Peak allocated memory is allocator-measured, not total system/device occupancy.

## Budget and reuse

{
  "start_utc": "2026-11-16T14:45:08.895668+00:00",
  "science_completed_utc": "2026-11-16T15:05:24.768140+00:00",
  "science_wall_envelope_seconds": 1215.872472,
  "hard_budget_seconds": 10800,
  "within_budget": true,
  "new_control_body_forwards": 1024,
  "new_validation_body_forwards": 299,
  "numerical_parity_body_forwards": 15,
  "control_benchmark_body_forwards": 6,
  "matched_profile_body_forwards": 60,
  "note": "Every body forward executes four recurrent passes. Matched deployment profile includes12referenceforwards+48warmup/measuredforwards. Reused full Train/native captures are separately identified; old GPU hours are not counted as new work."
}

The 30/70/50/30 minute stage slots are ceilings and priority guidance; reuse avoided spending them on duplicate forwards. All 256 development, 512 collected control questions, 326 remaining confirmation and 299 exposed Validation questions are complete. One identical development prompt is excluded from confirmation inference, leaving 511. Original512 remains reported. The input-only repair was retrospective, after initial results were read. No coefficient or method was reselected. Dataset exposure and prior mechanism/E6/H2 negative evidence are disclosed in the saved prior reports. CPU replay verifies scalar arithmetic and statistics; it does not repeat GPU forwards.

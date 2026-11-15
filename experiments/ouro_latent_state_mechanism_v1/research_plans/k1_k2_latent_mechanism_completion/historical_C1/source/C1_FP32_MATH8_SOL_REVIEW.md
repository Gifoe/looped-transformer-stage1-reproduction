# C1 FP32 math eight-case independent Sol review

2026-11-14. Independent read-only CPU inspection of all eight completed math-backend metadata records and complete archived tensors, with corresponding BF16 references and UX/UA/UXA treatment tensors. No GPU, model forward, fit, Test read, frozen-source edit or publication occurred. CPU process used `CUDA_VISIBLE_DEVICES=`. Audit implementation is `C1_FP32_MATH8_SOL_REVIEW.py`; structured record is `C1_FP32_MATH8_SOL_REVIEW.json`.

**Result: tensor/provenance/branch checks pass, but small directional and interaction effects are numerically sensitive.** All six K2 endpoint predictions remain unchanged across these eight BF16-versus-FP32 cases. This is a selected eight-case combined dtype/backend sensitivity check, not independent population replication or an isolated precision effect.

## Checks against actual records and tensors

- All eight case IDs exactly match the existing fixed source/sample freeze. Each FP32 metadata hash and both compressed/raw tensor hashes were verified. Each BF16 reference JSON hash and input-token SHA bridge matches the FP32 receipt; associated BF16 tensor archives were independently loaded and hash checked.
- Each of N/P/U/UX/UA/UXA per case records exactly 48 `aten::_scaled_dot_product_attention_math` calls, without efficient/flash dispatch. Identity records deterministic algorithms on, CUBLAS workspace `:4096:8` and TF32 off. Successful frozen producer retains strict actual body/head/first-layer count and weight-version assertions. Checkpoint/source identities are unchanged.
- All saved full-vocabulary endpoint tensor byte hashes match metadata. Offered logits, first-offered predictions and complete exact top-tie sets independently match tensor values. FP32 endpoint tensors are FP32, finite and CPU-readable; fresh deltas are FP64 differences of that case's FP32 N and P actual attention references, not copied BF16 deltas.
- At all seven upstream sites for all eight cases and U/UX/UA/UXA forwards, saved exact proposal equals `before.double()-d`, stored output equals the FP32 cast, and cast residual equals the actual stored output minus exact proposal. All site records match U across the three branch restorations.
- UX/UA/UXA retain U's actual x, a, Q, K, V and O tensors exactly. Independent sums reconstruct U=(xU+aU), UX=(xN+aU), UA=(xU+aN), UXA=(xN+aN); actual postadd, MLP input and shared skip match these sums. Actual post-MLP equals selected y plus captured b. UXA full-vocabulary endpoint equals N bitwise on every case.
- Strict donor consistency remains enforced: the frozen successful producer asserts each P/U/UX/UA/UXA forward's K1 Q region against the N donor before intervention. Only the N `Q9_K1_Q_FP32` is archived; per-forward pre-assert K1 Q tensors are not separately exported. This review therefore does not claim an independent raw-tensor cross-forward donor comparison. Four-repeat native math diagnosis separately showed exact repeat stability on the first fixed case; it is not an all-eight repeated-native experiment.

GPU-produced full-vocabulary FP32 log-softmax scores were preserved as the primary endpoint scores. Independent CPU log-softmax recomputation differs by at most 2.09808349609375e-5 (FP32 math endpoints) and 2.1457672119140625e-5 (BF16 endpoints), reflecting a different reduction implementation. Those observations are recorded, not used to overwrite production scores, relax existing identities or assert CPU/GPU score bit equality. Raw offered logits and their ties match exactly.

## Prediction and tie changes

N, P, U, UX, UA and UXA: zero prediction changes in eight cases each; neither runtime has an exact top tie at these endpoints. K1: `Mercury_7040740` changes BF16 prediction 1 with top set [1,2] to FP32 prediction 2 with top set [2]. Both are incorrect. The remaining seven K1 predictions do not change. No FP32 top ties occur in any of the seven endpoints across eight cases. Thus this check finds no K2 discrete-accuracy change; it does not establish precision-invariant continuous causal magnitudes.

## Frozen production-score contrast directions

Values below are meanwrong gap contrasts computed using exact rational arithmetic on the stored FP32 production-score values, converted to decimal only for display. Each cell is **BF16 → FP32 math**. Restoration/loss labels follow the unchanged order: X restore UX−U; X loss N−UA; A restore UA−U; A loss N−UX; interaction N−UX−UA+U. No condition, sample or contrast was selected from these results.

| Fixed ID | N−U | UX−U | N−UA | UA−U | N−UX | Interaction |
|---|---:|---:|---:|---:|---:|---:|
| Mercury_7040740 | 2.1666666 → 2.2853122 | 2.0833334 → 2.0801441 | 2.0624999 → 2.1564692 | 0.10416673 → 0.12884297 | 0.083333244 → 0.20516811 | -0.020833482 → 0.076325143 |
| Mercury_7140035 | 2.0833333 → 2.1005837 | 1.6458332 → 1.6734162 | 1.6249999 → 1.630429 | 0.45833337 → 0.47015472 | 0.43750011 → 0.42716753 | -0.020833259 → -0.042987193 |
| Mercury_405467 | 1.5833333 → 1.5723565 | 1.0416666 → 1.0334143 | 0.83333318 → 0.80565568 | 0.75000009 → 0.76670086 | 0.54166669 → 0.53894225 | -0.2083334 → -0.22775862 |
| Mercury_SC_401004 | 3.0833334 → 3.0513912 | 3.0833333 → 3.0923118 | 3.0416667 → 2.9660487 | 0.041666642 → 0.085342422 | 4.4237822e-08 → -0.040920623 | -0.041666598 → -0.12626305 |
| MCAS_1999_4_12 | 0.20833311 → 0.22170712 | 0.16666639 → 0.19093133 | 0.1666665 → 0.18645592 | 0.041666609 → 0.0352512 | 0.041666715 → 0.030775788 | 1.0617077e-07 → -0.0044754116 |
| VASoL_2007_5_39 | 0.20833338 → 0.063103348 | 3.4769376e-08 → -0.051156238 | 1.4901161e-08 → -0.060684914 | 0.20833337 → 0.12378826 | 0.20833335 → 0.11425959 | -1.9868215e-08 → -0.0095286767 |
| Mercury_7219643 | 3.2499999 → 3.3668941 | 3.2499997 → 3.4230748 | 3.2916667 → 3.3122804 | -0.041666726 → 0.054613675 | 2.2631139e-07 → -0.056180681 | 0.041666953 → -0.11079436 |
| Mercury_402062 | 0.04166674 → -0.054068311 | -0.041666852 → -0.1404896 | 2.3345153e-07 → -0.14263857 | 0.041666506 → 0.088570261 | 0.083333592 → 0.086421291 | 0.041667085 → -0.0021489697 |

Sign changes across the eight records are respectively 1/8, 1/8, 2/8, 1/8, 2/8 and 4/8. These counts include BF16 exact-zero or near-zero production-score effects; they must not be summarized as six robust nonzero causal reversals. In particular `Mercury_402062` N−U changes +0.0416667 to −0.0540683, while `Mercury_7219643` A restoration changes −0.0416667 to +0.0546137. The four large native-rescue cases retain clearly positive total upstream loss and large X restoration/loss; weak A effects and interaction signs are less stable. This descriptive statement concerns the fixed eight cases only and supplies no population confidence interval or new branch support decision.

## Limits and disposition

The original FP32 efficient-attention attempt failed its strict donor guard and remains a preserved technical failure, not a scientific result. Stable math/deterministic execution changes backend and several reproducibility settings along with dtype, so it cannot isolate BF16 quantization from attention-backend effects. Eight native-stratified Train cases cannot validate the full Train/Val population, norm/unrelated specificity controls, or deployable behavior; the math subset includes only the prescribed fresh references and four-corner endpoints. BF16 full-cohort evidence retains its separate existing validity and statistical gates. This review neither upgrades any scientific claim nor permits a new Experiment A/B run; subsequent design and execution require the resumed authorized directive.

Audit source SHA256: `487b94323e52609a065ce427a4e6c4c22abad801a31081e3b76926cd297bb0ea`.
Structured review SHA256: `ae7e976d2a6a9943dcc2441bf8489f0f5962b217e7d39713d73b00a935b0f35f`.

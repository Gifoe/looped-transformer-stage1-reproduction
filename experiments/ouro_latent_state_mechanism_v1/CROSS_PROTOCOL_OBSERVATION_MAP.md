# Historical K1/K4 endpoint observation map

These are observations from a previously exposed test set, not a new benchmark run or causal interpretation. Joint is the original-order score; cyclic pooled averages mapped-back scores over rotations and includes that original view. No raw score magnitudes are compared across these views.

## Agreement and counterexamples

| Endpoint | Same prediction, both correct | Same prediction, both wrong | Different, joint only correct | Different, cyclic only correct | Different, both wrong |
|---|---:|---:|---:|---:|---:|
| K1 | 518 | 287 | 116 | 139 | 112 |
| K4 | 1001 | 106 | 20 | 35 | 10 |

Of 367 questions where K1 predictions differ, 335 agree at K4: 288 agree on the correct answer and 47 on a wrong answer. The reverse occurs on 33 questions: 21 start jointly wrong and 12 jointly correct; at K4, 11 are joint-only correct, 17 cyclic-only correct, and five both wrong with different answers. The remaining 32 disagree at both endpoints and 772 agree at both. These are cross-readout endpoints, not a permutation-invariance or label-binding test.

The correctness transition cross-table has 301 common rescues and 15 common damages. Ten questions have opposite rescue/damage transitions (joint rescue/cyclic damage: 2; joint damage/cyclic rescue: 8). Both views change predicted option on 363 questions: 340 end on the same option, comprising 301 jointly correct and 39 jointly wrong; the other 23 end on different options. The 39, 33, 10, 15 and 23 case sets and all pairwise intersection counts are saved under `case_ids` and `case_intersections` in the JSON summary, with all IDs and complete candidate scores in the JSONL table. Sets overlap: all 15 common damages are included in the 39 jointly wrong same-final cases; all 10 rescue/damage conflicts are among the 23 different-final cases.

## Within-view ranking changes

First-index tie ordering is retained. Joint has 643 questions with unchanged top prediction from K1 to K4; 474 nevertheless change full candidate order. Of these 643, 586 remain correct and 57 remain wrong; the gold candidate's rank improves on 18/57 stable-wrong questions. For cyclic pooled, 413/696 unchanged-top questions change full order, and gold rank improves on 27/71 stable-wrong questions. A changed total order can reflect a tie being resolved, so the JSON also includes strict pair inversions, pairs with exact ties, and the denominator of pairs strict at both endpoints.

For the stable-wrong stratum, joint has 47 strict inversions across 314 eligible candidate pairs, with 25 pairs tied at an endpoint; cyclic pooled has 59/417 and six such tied pairs. These are pair-weighted descriptive ratios, not per-question averages. A near-tie diagnostic flags `0 < |score difference| <= 1e-6` within the same view at either endpoint; zero flagged pairs occur in both stable-wrong strata. Full option-count strata, per-question counts and all candidate vectors are in the machine-readable artifacts.

## Leave-original-out sensitivity

After verifying shared request provenance and the exact arithmetic identity for all records, the non-original rotation pool has 629/1172 correct at K1 and 1033/1172 at K4. It predicts differently from the original-including cyclic pool on 167 questions at K1 and 27 at K4. These are descriptive properties of the same previously exposed items and the same shared demonstrations; do not call them independent validation or infer why the model changes.

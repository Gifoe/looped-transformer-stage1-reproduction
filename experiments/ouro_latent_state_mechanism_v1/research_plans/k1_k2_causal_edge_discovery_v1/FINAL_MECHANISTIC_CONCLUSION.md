# Final mechanistic conclusion

**No reproduced Level1 semantic edge transport was established by this fixed study. Level2 and Level3 remain unexecuted and unresolved.**

All implementation, execution, numerical verification and analysis were performed by Sol. No Astra agent, training, optimizer, LoRA, new Gate, head selection, additional site search or ARC Test was used.

Development: all9fixed paths/types yield5/128crossed predictions, equal to the native5/128; no candidate passes the preregistered semantic-specificity gate. Actual BF16 perturbations are nonzero on every development transfer; failure is not a zero-write implementation artifact. Native development K1/K2 accuracy is67/128 and104/128; actual textual counterfactual K1/K2 is45/128 and92/128. These capability baselines limit how strongly a failed intervention can be interpreted.

Fresh fixed-grid results:

| Path/type | Transfer cross | Native cross | Same-variable | Unrelated | Equal-dose | Strict cross | Strict novel | A gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| SM_L6_V | 36/256 | 34 | 34 | 35 | 37 | 33 | 0 | False |
| SM_L6_K | 35/256 | 34 | 34 | 36 | 37 | 31 | 0 | False |
| SM_L6_KV | 36/256 | 34 | 34 | 36 | 36 | 31 | 0 | False |
| SM_L10_V | 36/256 | 34 | 34 | 35 | 36 | 31 | 0 | False |
| SM_L10_K | 35/256 | 34 | 34 | 37 | 37 | 32 | 0 | False |
| SM_L10_KV | 35/256 | 34 | 34 | 37 | 37 | 33 | 1 | False |
| SR_L14_V | 40/256 | 34 | 34 | 39 | 38 | 35 | 1 | False |
| SR_L14_K | 39/256 | 34 | 34 | 34 | 37 | 32 | 0 | False |
| SR_L14_KV | 43/256 | 34 | 34 | 40 | 36 | 39 | 4 | False |

Fresh grid passes: []. These do not trigger a new selection after the development gate failed. Full ordinary, strict and novel counts, outcome categories and cluster intervals remain visible.

Level1: the tested source-local messages are insufficient to establish a reproducible semantically specific crossed-answer effect beyond the fixed controls. Level2: M integration and subsequent M-to-R routing were not tested by block/restore because A did not justify a candidate. Level3: the K1/K2 intervention interaction was not run because B lacks evidence. The native K1-to-K2 competence gain does not by itself identify the missing edge or imply first-hop information was absent from all K1 states.

The result does not refute the earlier broader boundary substitution, whose exact current-native parity and development reference are preserved. It also does not establish a unique distributed algorithm, a unique composition site, absence of any circuit, or donor-answer copying. Alternatives include a broader/distributed first-hop code, redundant routes, insufficient magnitude at the selected rows, or downstream integration not addressed by these three sites. These remain hypotheses; no new operator/site search is automatically authorized.


## Positive observations and their limits

On fresh256units, native crossed predictions are34. L14 S-to-R jointKV gives43 (ordinary difference+3.516pp; descriptive cluster95%[+0.391,+6.641]pp), with39unique patched maxima and4strictlynovel crossings across4differentbasepairs. Its unrelated control gives40 andequal-dosecontrol36. The ordinary paired95% differences against thesecontrols includezero; every one-sidedBonferroni36lowerbound fails, and the fixed5ppminimum gain is not reached. All256dosecontrols for this candidate matchwithin1%. These positive cases are retained as bounded observations, not a validated circuit or authorization for B/C.

FreshnativeK1/K2correctness is110/256 and218/256; actualtextcounterfactualK1/K2 is136/256 and210/256. Thus the family has substantial taskcompetence, but it is not perfect. The originalsame-device development localizedboundary substitution yields17/128crossedanswers vs5/128native, whereas each testedattentionedge yields5/128. The pairedoperatorcomparison is retained separately and does not pool oldheldout512withfreshdata.

# Intermediate-variable exchange

Patch only the donor's matched causal source span at the true normalized K1->K2 boundary; retain receiver's other K1 states and R. Then execute complete native K2. This is a semantic cross-world prediction, not a donor-answer target. Source cannot have read donor mappings or answer options.

| Condition | Crossed /512 | Original | Donor | Other | Cross margin (95% cluster interval) |
|---|---:|---:|---:|---:|---|
| K1 | 137 | 217 | 92 | 66 | -0.4349 [-0.4978, -0.3741] |
| native | 15 | 446 | 30 | 21 | -1.6469 [-1.7181, -1.5767] |
| text_edit_K1 | 222 | 137 | 86 | 67 | -0.0403 [-0.0963, 0.0168] |
| text_edit_K2 | 448 | 16 | 17 | 31 | 1.1482 [1.0577, 1.2363] |
| source_boundary | 86 | 325 | 62 | 39 | -0.8102 [-0.8717, -0.7505] |
| same_variable | 15 | 446 | 30 | 21 | -1.6469 [-1.7181, -1.5767] |
| unrelated | 14 | 229 | 53 | 216 | -1.5757 [-1.6450, -1.5073] |
| dose | 21 | 434 | 35 | 22 | -1.5067 [-1.5742, -1.4414] |
| readout_copy | 13 | 445 | 32 | 22 | -1.6334 [-1.7058, -1.5632] |
| descendant_edit | 441 | 19 | 21 | 31 | 1.1149 [1.0255, 1.2019] |


Four preregistered heldout cross-margin specificity contrasts (source minus control; Bonferroni98.75% base-pair intervals):

- same_variable: 0.8367 [0.7820, 0.8901]
- unrelated: 0.7655 [0.6588, 0.8719]
- dose: 0.6965 [0.6433, 0.7508]
- readout_copy: 0.8232 [0.7683, 0.8792]

The frozen strong abstraction gate is {'stratified_majority_and_margin': False, 'all4_control_specificity': True, 'PASS': False}. All four specificity comparisons pass; the majority-crossed/positive-margin criterion across templates, directions and labels fails. There is reproducible, semantically directed causal influence. Of the86 crossed outputs,68 differ from BOTH actual original and donor native predictions; the corresponding strictly novel counts are0same-variable,11unrelated,7dose and2readout-copy. The86 total includes18 cases where one native input already predicted the crossed label, so86 is not the number of genuinely new crossed outputs. The localized source replacement is not a reliable substitute for the entire first-hop variable across this dataset. Examples and failures are both retained in analysis/B_EXAMPLES.json and all unit records.

The one preregistered alternative replaces receiver B[source_start:R] with the receiver's textual-firsthop-counterfactual states, keeping original R. It gives 441/512 crossed answers, compared with 448/512 for the full text edit. This is a distributed-descendant compatibility diagnostic. Those downstream K1 states can already encode relations or an answer, so this result cannot locate a single intermediate variable or prove K2 alone computes composition. No alternative site sweep was run.

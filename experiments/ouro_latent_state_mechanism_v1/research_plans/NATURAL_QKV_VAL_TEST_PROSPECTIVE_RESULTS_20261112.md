# QKV validation/test prospective results — preliminary

Recorded UTC: 2026-11-12T04:04:33.180917+00:00

After the complete train analysis, QKV train-informed prospective checks were frozen before val/test effects (SHA256 a5d1fe62e91af1b6759a54edd55105996967009d1d5e39869fb38bdb24423376). These are within-project checks after earlier ARC exposure, not independent blind validation. The standalone CUDA logit replay passed; source/endpoint CPU audits were linked per question to the replay ledger (link SHA256 2b52a23348db90f0a42a629d60bbddeb5443e5c36d79c326aa2354538de28952). Historical natural full-tensor hashes were reused from earlier complete coverage, not rehashed this time; CUDA replay uses frozen collector check code in a separate process, and injection provenance relies on the frozen collector. Independent scientific review is pending.

| split | N | correct F11/F21/F12/F22 | mean D=F12−F21 | KV effects mean, Q1 / Q2 | Q effects mean, KV1 / KV2 | interaction mean |
|---|---:|---|---:|---|---|---:|
| validation | 299 | 150/172/202/225 | 0.381759 | 0.497352 / 0.474150 | 0.115594 / 0.092391 | −0.023202 |
| test | 1172 | 626/678/818/882 | 0.424364 | 0.481891 / 0.461911 | 0.057527 / 0.037547 | −0.019980 |

P1 D>0, P2 correct(F12)>correct(F21), and P3 positive KV means and positive net rescue on both edges hold for full val/test and natural-no-top-tie subsets. Validation KV edge net rescue 52/53, Q edge 22/23; test KV 192/204, Q 52/64. Val no-tie N287 D=0.392639 and F21/F12 correct 167/197; test no-tie N1107 D=0.435411 and F21/F12 correct 654/785. Q mean and Q net rescue on both backgrounds are positive, while mean final interaction is negative. Complete distributions, exact BF16 Fraction signs and counterexamples remain in the machine-readable split summaries and rows.

Internal SDPA and post-local-norm a interaction vectors are nonzero in all val/test samples when saved BF16 values are converted to FP64, while final candidate-gap Fraction interaction is exactly zero in 11 validation and 51 test samples. These are different levels, not interchangeable notions of interaction.

Joint KV replacement has a larger mean effect under this fixed K1 receiver than query replacement, and query replacement also has a measurable effect. Joint KV does not distinguish K from V. The experiment cannot establish semantic source, query direction versus scale, unique natural mechanism, or natural sufficiency: Q2/KV2 required upstream K2 computation and the receiver states are mixed. a_R Wγ projection is a readout coordinate, not a direct final-logit contribution.

Validation summary SHA f090781a88ed664176896dba6d1edb4719d9e590e8fc6534c66b28397e82095d; test summary SHA b4eb0284b85f6ae80c893cfcad0c99296ad96a160ad452197f38e1c32f14f337. Original evidence and frozen plans are unmodified.

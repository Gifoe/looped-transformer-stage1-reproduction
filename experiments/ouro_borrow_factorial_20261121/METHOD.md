# Independent low-field states, target digits, and causal interactions

Write A = 100h_A + a and B = 100h_B + b. The receiving low difference x=b-a is positive. The A edit increases a by d and the B edit decreases b by 100-d. Fix the high digits and choose x above the larger of the two single displacements. For s and t in {0,1} the true full answer is

```text
z(s,t) = 100(h_B-h_A) + x - s*d - t*(100-d).
```

The lower part before reduction modulo 100 is nonnegative except at s=t=1. The joint output has the original lower two digits and a hundreds digit reduced by one. The whole integer z is additive in these changes. A nonlinear digit boundary or a nonzero residual-state finite difference alone is insufficient to infer a symbolic borrow variable.

Let H00, H10, H01 be the actual states after the first full pass for the base and each single-edit prompt. Let P_A and P_B denote the two disjoint masks. The forward hybrid is

```text
X(s,t) = (1-P_A-P_B) H00 + P_A H(s,0) + P_B H(0,t).
```

The second pass runs on X with normal layer computations. Each first-pass readout remains the receiving prompt's original vector. The B field is obtained with A held at the original receiving value. It may encode that earlier context. The two donor prompts individually specify different complete answers from the desired joint answer. The joint textual input is used independently as an accuracy benchmark and as the receiving input in the separate reverse experiment.

The L14 natural-correction test holds the same problem and every recipient context key and value fixed. Let a_k be the actual normalized readout attention weights of one receiving pass, and a_o the weights from the opposite pass's Q on the receiving K. The exact partial value readout shift for an operand field S is sum_{j in S}(a_o-a_k)_j V_j. Two field shifts add before the natural output projection. Nonlinear RMS normalization and later shared computation can change how those shifts matter for the answer. Coefficients from a partial path are algebraic, so a single changed part need not have an attention row summing to one. Norm and angle controls act on the resulting AV activation rather than changing the underlying symbolic prompt.

The separate factorial readout test evaluates r(s,t) at input to L17. The joint interaction vector is i=r11-r10-r01+r00. Removing i by intervention keeps the base and two single shifts. For Gaussian control z preserve both its norm and projections onto r00 and r11-r00, the latter independently checked. The output after the edited pivot uses the same joint field graft and full autoregressive decoding. Failure to generate the exact original no-borrow integer after removing i is kept separate from generic accuracy loss. The interpretations apply to the tested directions and poses of model state. [Interchange intervention methodology](https://arxiv.org/abs/2106.02997) · [Arithmetic circuits in other trained Transformers](https://arxiv.org/abs/2402.02619)


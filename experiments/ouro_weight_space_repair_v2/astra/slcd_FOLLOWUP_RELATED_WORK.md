# Follow-up mechanisms and prior art

This is a bounded primary-source check, not a proof that no equivalent method exists. Accuracy is the main objective; renaming known operations does not make them novel.

- [Adaptive Projected Guidance, ICLR2025](https://arxiv.org/abs/2410.02416) already decomposes diffusion guidance into parallel/orthogonal components. R2 instead uses the same LM's late-loop update as a reference; projection itself is not new.
- [Attention-space Contrastive Guidance](https://arxiv.org/abs/2601.13707) already suppresses text-prior-aligned components in training-free LVLM contrastive guidance. Its attention paths differ from the recurrent output-logit path here.
- [Distill to Detect](https://arxiv.org/abs/2607.01208) presents Fisher-weighted projection of a logit-distribution shift in a distillation framework. R2 does no distillation or parameter fitting, but cannot claim first use of Fisher logit projection.
- [BitResEdit](https://arxiv.org/abs/2606.13558) uses training-free contrastive log-odds updates inside a Bernoulli-KL trust region for visual autoregressive editing. R4 uses categorical full-vocabulary KL along a recurrent contrast, solved by bounded bisection; KL-controlled contrast guidance is not new.
- [Trust Region Policy Optimization](https://proceedings.mlr.press/v37/schulman15.html) is an earlier trust-region framework for policy training. Its policy-improvement guarantees do not apply to these fixed LM readouts.
- [Asymptotic Probability Decoding](https://aclanthology.org/2024.emnlp-main.484/) studies nonlinear extrapolation as an alternative to linear contrastive decoding. It uses different model sizes, not a single loop trajectory; generic probability extrapolation is not new.
- [Surface Form Competition / PMI_DC](https://aclanthology.org/2021.emnlp-main.564/) corrects answer ranking using task-domain prior likelihood. R5's shortAnswer: prior belongs to this family, estimating text/format preferences rather than25-shot demo bias.
- [Calibrate Before Use](https://proceedings.mlr.press/v139/zhao21c.html) estimates context bias using demonstrations and content-free test input. Keeping demos is a different calibration intervention; R5 has not implemented or evaluated it.

R2 coefficient: c=clip(Cov_p4(d,u)/Var_p4(u),-1,1), d=z4-z1, u=z5-z4. This minimizes the bounded local residual variance. Clipping means the update need not be exactly orthogonal. The matched-norm control matches only the token-local quadratic Fisher norm, not fullKL or candidate-score displacement.

For original A1 vs token-level A2, let b=z4+.5(z4-z1), u=z5-z4 and A(z)=logsumexp(z). Their exact character-normalized score difference is sum_t[A(b-u)-A(b)+A(z5)-A(z4)]/L. It is generally candidate-dependent through each continuation's history; there is no universal sign. This algebra does not by itself prove that normalization caused accuracy changes.

R4 Fisher-RMS enforces a local quadratic energy budget, not an actualKL upper bound. The exact-KL variant returns the feasible lower endpoint after16 bisection steps on[0,1], so weight precision is finite. Both are fixed, training-free readouts; neither certifies calibrated answers.

R5 A1-based rules are prior-corrected ranking scores. Because A1 is not a normalized probability model, these should not all be called strict mutual information. Cache keys never use gold, answer frequency or corpus statistics. No test answers may be enumerated before final method lock.

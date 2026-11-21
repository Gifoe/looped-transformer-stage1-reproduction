# Operator-conditioned counterfactual arithmetic

Write H1(x) for the exact normalized state after the first shared pass. The output of the first pass is directly read at the final prefix position. Let P select token positions from the first changed numeral token up to the token before B's first digit. In the rule-conditioned assay the state passed to the second loop is

```text
H1_hybrid(A,B,r;A*) = (I-P) H1(A,B,r) + P H1(A*,B*,r*)
```

The source P H1(A*,B*,r*) is invariant under changing its own B* and r* within the verified token alignment. Causal masking excludes B* and r* at those positions. The source is shared across both recipient rules. The actual initial readout H1 outside P is held fixed at every greedy step. These identities are tested with the pinned frozen model on the used prompt batches.

Let d=A*-A. A correctly completed addition under intervention has output shift +d. A correctly completed reverse subtraction under the same graft has shift -d. Conditional on both originals and both counterfactuals being correctly completed, the paired difference of shifts is 2d. An operator-independent additive offset has paired difference zero. Native rule-specific candidate updating and new computation are both consistent with the primary result. The pathway intervention tests which projected value vectors carry the effect. At early A positions the second-loop K and V traces are also independent of the later rule under the same causal ordering. Conditional computation may arise within the later queries, other contextual states, MLP responses, or the readout.

The behavioral hundreds-transfer calculation follows directly by writing A=100h_A+l_A and B=100h_B+l_B. In addition the output hundreds is h_A+h_B+c for c=(l_A+l_B)>=100. In reverse subtraction it is h_B-h_A-b for b=l_B<l_A. Using the correct remaining low two digits while omitting the transfer predicts -100 and +100 respectively. The same-answer holdout keeps exact answer and foil tokenization fixed while changing whether the transfer is required. This is a behavioral signature. The native-rollback cohort measures whether the natural same-prompt V evolution causally matters for the particular previously wrong complete integer. It is evaluated separately from generic damaged accuracy and from rule-switch transplantation.

Interchange interventions are established methodology. The experiments here use the published Ouro shared-pass architecture and a frozen released checkpoint. Circuit studies of independently trained arithmetic networks are relevant comparisons with their own scope. [Interchange interventions](https://arxiv.org/abs/2106.02997) · [Ouro](https://arxiv.org/abs/2510.25741) · [Arithmetic circuit study](https://arxiv.org/abs/2402.02619)


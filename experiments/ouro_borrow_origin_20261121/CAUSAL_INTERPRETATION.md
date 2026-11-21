# What the source and restoration experiments can identify

Let the native layer map be \(F(h)=h+A(h)+M(h+A(h))\), including both branches' native post-normalization. The joint corner input is \(h_{11}\), the additive reference is \(s=h_{10}+h_{01}-h_{00}\), and \(\Delta f=f_{11}-f_{10}-f_{01}+f_{00}\). For the output pivot, the exact decomposition is

\[
\Delta F=\underbrace{F(h_{11})-F(s)}_P+\underbrace{A(s)-A(h_{10})-A(h_{01})+A(h_{00})}_{N_A}
+\underbrace{M(s+A(s))-M(h_{10}+A(h_{10}))-M(h_{01}+A(h_{01}))+M(h_{00}+A(h_{00}))}_{N_M}.
\]

Here each \(h\) is the full preceding sequence, and the expression retains only the designated readout token after physical block replay. \(P\) can contain nonlinear transformations of the incoming interaction. \(N_M\) can include influence from attention in its input. A late residual edit in the \(N_M\) direction does not identify individual gates or rule out attention as a causal source.

With only the joint K1 prefix edited, the three reference corners remain untouched. The K2 reference and both generated terms are necessarily static. In this arm a \(P\) restoration at the K2 layer output algebraically restores the original joint-corner site. It is a localization check, and is insufficient to attribute an exclusive causal mechanism to propagation. The symmetric four-world operation is necessary for a test that permits both candidate terms to move. The signed coefficients equally distribute subtraction of the K1 mixed difference across the physical writes. This creates four counterfactual states that do not arise during ordinary inference.

The primary source effect compares the same native response across four-world source edits and dose-matched geometry. Factorial removal compares exact components at their native magnitudes. An additional controlled test fixes their common dose and adjusts geometry at that dose. In a removal arm, one output can recur because of a readout threshold. The native arithmetic controls, the physical source edit and the within-question rescue beyond matched geometry jointly constrain that explanation. Even with specific rescue, a complete internal rule implementation would require natural input-to-gate tracing beyond this intervention.

This branch uses a single frozen checkpoint with shared parameters and a fixed prefix. The estimates apply to exact early +100 repairs under the stated prompting protocol. They do not define accuracy or useful inference compute on other benchmarks, models or depths.

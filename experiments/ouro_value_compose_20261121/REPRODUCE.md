# Reproduce K2 first-operand value composition

Use the pinned checkpoint and runtime versions from results/VERIFICATION.json. Keep the preceding published experiment directories on PYTHONPATH as shown in env.sh. Copy the archived DECK.json to a new output directory and run value_deck.py followed by value_test.py with the local model and output directories. Use report.py to independently recompute the tables, checksums and figure. Shell entrypoints record source locks and the GPU job outcomes.

For every generated prefix \(p\) and context background \(c\), capture unedited K2 attention \(A_{\ell,c,p}\) and its values. The intervention then evaluates each physical head with
\[
o_\ell=A_{\ell,c,p}\bigl(V_{\ell,\text{current}}[A\leftarrow V_{\ell,\text{donor}}(A),\,B\leftarrow V_{\ell,c,p}(B)]\bigr).
\]
The normal output projection, residual and MLP follow. The donor contributes only the first operand's three V positions. The second operand uses the unedited receiving-background V at all layers. Earlier queries are causally unaffected. At each edited head and token, the shuffled displacement preserves the norm and radial projection of the genuine donor displacement.

Use the same continuation tokens on each receiver and natural donor computation, and regenerate every token under the selected intervention until the existing termination condition. Donor B changes in both auxiliary prompts. The two donor A fields are identical, and that prefix's causal V invariance is directly checked. The target A–B mixture differs from the receiver answer, the old foil and both donors' answers. None of those numbers are supplied as an output constraint.

On a locked pre-digit diagnostic subset, compare a no-change frozen-attention replay to its unedited output, check the exact preserved B values, and verify that opposed versus same-borrow contexts do not change the previous A positions beyond numerical tolerance. Report fixed-weight causal effects separately from what would happen if attention routes adapted after each value change.

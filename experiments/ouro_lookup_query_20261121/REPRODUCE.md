# Reproduce bidirectional Q transfer on lookup tables

Use the public experiment modules from env.sh, the exact pinned model and the runtime versions in results/VERIFICATION.json. Put the frozen DECK.json in the chosen fresh output directory or regenerate it against the inherited manifest. Run query_deck.py and then query_test.py with the model and output directories. report.py independently checks table permutations, developmental selection, all generated integers and summary artifacts.

At the locked physical layer, the receiving attention is
\[
\operatorname{softmax}(Q^\mathrm{source}_{\mathrm{last}} K_\mathrm{recv}^{\top}/\sqrt{d})V_\mathrm{recv}
\]
at the current final query, with native masking, projection and continuation. The other loop's Q comes from the same emitted answer prefix. On table remapping the donor is always the original source table, and the receiving K/V come from the recipient table. An unchanged-Q replay checks the native prediction. The position control moves matched query displacement to the fixed final table semicolon, and leaves the current final query native. Its perturbation preserves norm and radial fraction relative to the receiving local query.

Physical layer selection uses only development cases and the native head-averaged attention mass toward the true table field minus the initially emitted wrong table field. Offsets include the field's key token and all three value digits. This diagnostic does not define exact generation success. Reordered values retain the actual donor number in a different table row. Crossed layouts move the requested key's row independently of the value swap.

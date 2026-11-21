# K1 operand attention and K2 borrowing

Use the pinned checkpoint and software versions in results/VERIFICATION.json. Start from the public experiment directories in env.sh and put the archived DECK.json in the fresh results directory. The pinned checkpoint content, the deck, model code, source locks and output receipts are independently checked before publishing.

Run edge_deck.py with the model and results paths. Then run edge.py with the same arguments. The latter first selects the head from development prompts without reading behavioral outputs. Alternatively seal that selection with --selection-only before running the native benchmark. No test results enter the selection. report.py checks the inherited history, every sample and donor, the fixed effect estimates, and the generated plot. See the shell entrypoints and results hashes.

For the recipient attention row \(a\), a key-set \(S\) and dropped fractions \(\lambda_j\) on those keys, the intervention is

\[
a'_j=\frac{(1-\lambda_j\,\mathbf{1}_{j\in S})a_j}{1-\sum_{i\in S}\lambda_i a_i}.
\]

All masks, values, the output projection and post-normalization use the native block. In the real source arm \(\lambda_j=1\) for the two first-operand low digits at the last second-operand digit query. The unrelated-edge control chooses the closest covering pair of earlier non-operand keys and partially deletes its probability to match the exact mass where possible. If no pair covers the requested mass it drops the maximum available. Geometry is an independent control at the resultant normalized write boundary. The inherited K2 propagation/generated decomposition is performed from the actual four native prefixes under each earlier intervention. A single late pivot rescue cannot recover a diffuse influence that is still present at other K2 token positions.

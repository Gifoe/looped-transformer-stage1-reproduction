# R11 selected-only path static review

Existing Astra reviewer inspected only scripts, not empirical rows or GPU output. It identified a certain pre-launch parity-schema error: R10 stored target hashes for only the original n choices, while the new path stores n-squared hashes across cyclic shifts. Before the full candidate implementation lock, this was corrected to compare the first n hashes, all cyclic request hashes, all target counts and common complete LL readouts. The receipt explicitly limits historical token-hash coverage.

The remaining gates, selected-only outcomes, exact score replay, inherited budget and public-final-lock requirement had no additional definite runtime blocker in the static review. This is not independent numerical acceptance: the fresh first2 BF16 parity run remains mandatory before confirmation.

No frozen mathematical definition was changed in this source review.

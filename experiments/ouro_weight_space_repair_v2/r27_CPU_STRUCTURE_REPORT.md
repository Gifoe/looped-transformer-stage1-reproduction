# R27 CPU structure snapshot

The single prescribed 64-nonce pool and unchanged lookup template passed tokenizer-only structural admission. Both tables selected the first eligible index pair [0,3]; row pool indices are [0,3,1,2] and [1,2,0,3]. All four A/B query pairs have matching input shape within their table (1177 tokens for table1 and 1174 for table2). Leading-space source labels and expected continuations are single tokens, source offsets/IDs agree, and A/B differ only at the two prescribed swapped label positions.

Root independently reconstructed every pool entry, prompt byte, token ID, offset, source prefix and structural comparison using the pinned Rust tokenizer. No model, real activation, ARC gold/prediction/score or task accuracy was accessed. Generation used the one finite pool only, with no template/label/pool retry.

This is structural admission only, not model functional evidence or ARC improvement. No GPU lock, model authorization, Phase A/B job or automatic follow-up is created. Runtime hook identity/cleanup, actual source/query independence, complete cost, source freezing and explicit START still require separate review. Fifteen GPU ledgers remain unchanged at 18874.135196831077 / 21600 seconds, with 2725.8648031689227 seconds remaining. Private tokenizer and transport files are excluded.

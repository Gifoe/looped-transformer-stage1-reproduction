# R27 CPU-only structure authorization before input generation

Bind proposal v2 SHA256 `5f2b13c148c58ce38a34196a6cccd04e0204ea2056b64bd98401fd36a1791df7`. This authorizes only deterministic input generation, tokenizer structure checks and synthetic tensor checks. It does not authorize importing/loading a model, reading ARC outcome rows, extracting real activations, a GPU job, a controller, a deployment method or another benchmark. Use the existing experiment directory and parents only.

Keep the original 25-demonstration prefix exactly as stored in the R25 memory manifest. Construct each current question with this exact ASCII template, where rows have no trailing whitespace:

    Use the lookup table below. Each code maps to exactly one label.
    {key0} -> {label0}
    {key1} -> {label1}
    {key2} -> {label2}
    {key3} -> {label3}
    For code {query}, what is its label? Reply with only A, B, C, or D.

Append this question through the original `Question: ` / newline / `Answer:` scaffold. No expected label is appended to a forward input. Labels A/B/C/D occur once each in row order in version A. Version B swaps rows 0 and 1 in table 1, and rows 2 and 3 in table 2. The two queries are those swapped row keys. Expected labels are diagnostic metadata derived from the table, never ARC labels.

Use exactly one NumPy default_rng(2021). Generate table 1's 32 candidates and then table 2's 32 candidates, each `zx` plus six independent characters from `abcdefghijklmnopqrstuvwxyz0123456789`. Any duplicate among the 64 is no-admission; do not redraw. For token-length matching, use the literal suffix `For code {nonce}, what is its label? Reply with only A, B, C, or D.` followed by newline and `Answer:`. Pick the lexicographically first index pair i<j with equal tokenizer length in that suffix. Other keys are the earliest two unused candidates. Put the query pair in rows 0/1 of table 1 and rows 2/3 of table 2. No pair means no-admission. No expanded pool, label change or template retry follows a failure.

With the pinned tokenizer, audit full A/B prompts and all four queries, not only isolated fragments. Each source label field includes the single space after the literal ` ->` and the label letter, excluding the following newline. It must contain exactly one fully contained token. Each expected continuation consisting of one leading space and the label must also be one token under original encode_pair. Verify that swapped source token IDs match those label-token meanings, all A/B input shapes agree, the two queries for a table have the same full input shape, source positions are identical within a table, and no differences outside the two swapped label positions occur between A/B for the same query. Source labels must precede the query, remain outside the query and all old demonstrations, and consist of four distinct positions. Record offsets, IDs, input hashes, the entire finite pools and exact chosen indices.

No current ARC question, gold, predicted answer or score enters generation or selection. Tokenizer-only structural selection is disclosed and is not an unconditional random sample. CPU structural admission is not model task completion or functional evidence. Any model proposal still needs R26-aware source invariance, identity-hook, cleanup, complete cost and freeze review; same-shape CPU checks do not replace those runtime checks.

Keep the proposal snapshot and this pre-generation contract unchanged. Implement only a CPU generator and audit with outputs in new r27_ files. Preserve failure records; no input revision after viewing model behavior is authorized. The original 21600-second GPU cap and current balance do not change.

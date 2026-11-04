# R26 independent closure and next-action review

The one authorized job completed normally in 55.08359357668087 supervised GPU-worker seconds. Root independently decoded the raw BF16 bits and saved FP32 full-vocabulary vectors, verified every input variant and all 126 fixed comparison pairs, recomputed all exactness/mismatch/maximum-difference statistics, and checked the fifteenth ledger. All 269 parameter hashes also match the previous R25 run. No model was run locally for this review.

The frozen category is length_bitwise_sensitive: repeated inputs pass 42/42 pairs, same-shape future-content changes pass 42/42, and true truncations pass 36/42. The six nonexact pairs are N0's two cuts at native4, and the first key predecessor at native1 through native4. This is a finite counterexample to unrestricted cross-length bitwise equivalence. It is not evidence of semantic future leakage or a causal attribution to a particular kernel or precision operation.

Root's separate postrun FP64 distribution description shows nonconstant differences. Native4 total variation is approximately 0.008011 and 0.001914 for the N0 cuts, and 0.017677 at the first key cut. These are distribution effects for the saved readouts; no target, gold, argmax, previous retrieval or accuracy was evaluated. They cannot explain R25's negative result without further evidence. The original acceptance rule was not changed.

Accept Astra's new postrun decision. Preserve all historical scores and implementations. Do not silently canonicalize inputs, replace caches, change masks/backends/dtypes, or rerun a closed method. A future canonical-prefix primitive would need a separate execution contract, matched baseline/late-term recomputation and a complete budget. No such model job is authorized here.

Fifteen ledgers now total 18874.135196831077/21600 seconds, leaving 2725.8648031689227. The broader ARC objective is unachieved and remains active. No other benchmark is authorized.

Authorize Sol to archive and publish the R26 postrun scripts, reports, original vectors and receipts using a lease from fa0bf55b46bb292394a5976f99c7d1e5f84bf7c8. Prefer one ordinary binary tar.xz archive of the measured roughly 3.2 MB, uploaded with base64 transport, plus a contents manifest and release index. Do not turn it into many small parts. If the connector demonstrably rejects that payload, at most two ordered lossless parts with explicit hashes are acceptable. Exclude all PRIVATE files, upstream source, model weights and transport coordination. Preserve every old public blob and every frozen source/input byte. Root must independently verify public readback, archive members and server state before delivery is called complete.

The separate R27 authorization permits only bounded CPU structure preparation under its fixed proposal snapshot. No synthetic model, activation patch, additional GPU job, new ARC score or benchmark follows automatically from this closure.

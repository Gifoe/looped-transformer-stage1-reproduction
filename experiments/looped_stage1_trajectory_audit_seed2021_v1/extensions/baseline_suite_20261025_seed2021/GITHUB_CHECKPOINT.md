# Incomplete experiment checkpoint — 2026-10-26

This upload preserves current code and measured results. It is not a completion
certificate or the final research decision. The sealed original Stage1 study is
retained separately, with all175 original manifest hashes verified unchanged.

| Evidence | Current state |
| --- | --- |
| Qwen3-4B-Base, Ouro1.4B and Ouro2.6B multiple-choice experiments | Complete under the extension's frozen protocol;18 raw files,31,250 readouts; independent CPU rebuild matched15 tables/8,000 rows and both selected configurations |
| Ouro1.4B math validation | Complete:100 questions ×9 conditions =900 records; strict/numeric token audit and recovery lineage passed |
| Ouro2.6B formal math validation | Stopped:207/900 records,23 complete questions ×9 conditions; two native access-violation failures retained and charged |
| Ouro2.6B CPU-affinity diagnostic | Natural native exit0,216 isolated diagnostic records on a declared24-question prefix; all9 new conditions passed independent token/text/scoring/trace checks. Diagnostic rows are not merged into formal data |
| Held-out math tests | Not started; test sample count and validation-selected math settings are not yet frozen |
| Actual stopping/latency | Producer and independent auditors implemented; actual GPU experiment not yet run |
| Final research decision and full delivery audit | Pending; no extension Go/No-Go verdict is claimed |
| Cumulative charged GPU wall time |31,268.171330696205 seconds =8.685603147415613 hours, including sealed Stage1 and failed/diagnostic GPU work; hard cap50 hours |
| Official author-patch parity | Unverified. The connected GitHub API again returned404 for L-z-Chen/Training-Free-Looped-Transformer on2026-10-26 |

The last authenticated server observation at UTC03:07:11 found no experiment
GPU/supervisor process. The formal2.6B file remained4,795,153 bytes, SHA256
`b2bb4dd03966cebaba23d32f7aac6981a5c58f16900841318e44da6e98d8efd9`.
The diagnostic archive's SHA256 is
`9f26a46357d66934dbf60d738ab55d508964a2c4307478e384daf1bf1f760c2b`.

`results/native26_affinity_independent_audit.json` records the independent216-row
diagnostic check. The complete diagnostic evidence is under
`logs/native26_affinity_context/`, with the nine newly generated rows separately
identified. The crash root cause and long-run stability remain unestablished.
Fresh affinity-gate code is present but has not yet authorized/launched a new
formal resume at this upload checkpoint. The earlier2.6B gate remains invalidated.

For a CPU-only rebuild of the completed MC subset:

```powershell
python -X utf8 scripts/rebuild_cpu.py --scope mc
```

This command passed locally, including16 protocol/auditor tests. Its receipt is
under `logs/cpu_rebuild/20261026T025556138961Z/`. The full default rebuild correctly
rejects the currently missing test/latency/research-decision inputs; it has not
passed. Synthetic auditor tests do not establish actual stopping or speedup.

Weights, Python environments, dataset parquet caches, SSH transfer requests and
credentials are excluded from the GitHub checkpoint. Lightweight prepared
requests, tokenizers, source pins, code, raw trajectories, tables, figures,
failure evidence and rebuild logs are retained. Source limitations and negative
results remain part of the record.

# 4090 E3 source snapshot

Read-only snapshot of the isolated site's source, source_next, and orders. MANIFEST.json hashes each copied file. Source code imports the pre-existing 4090 venv and pinned model paths; checkpoint and user credentials are excluded. Current L-only nested run remains PENDING and must be added to final results only after full audit.

Runtime: torch 2.5.1+cu124, transformers 4.55.0, 4090, BF16 SDPA, no cache. Entrypoints: writer4090_reference64.py; writer4090_treatment64.py and _bare/_cpu_audit; writer4090_reference1030.py and writer4090_attention1030.py with bare/math (validation equivalents end 299); e3_conditional_technical_v2.py and its bare/endpoint/math; e3_conditional_stagea64_v2.py with bare/math; e3_conditional_l1030.py with bare/math then e3_conditional_l299.py with bare/math. All require their receipts and immutable source/inputs; inspect each source and frozen orders before reproduction. Cross-hardware baseline equivalence is not assumed.

# Context-consistent recurrent update

All nine development conditions on256questions and five frozen confirmation conditions on512questions. Original25shot ARC Joint-Option, no Test/training/search. Read ASTRA_FINAL_DECISION.md and CONTEXT_UPDATE_RESULTS.md. All work by Sol; requested ASTRA filename retained.

CPU replay: Python3.10, NumPy1.26.4, `python code/reproduce_release.py --release .`. This assembles lossless shards and exactly recomputes strict effects, paired intervals, trajectories, attention associations and doses. It audits every input/candidate/tie/norm and exact-input-group separation. It does not rerun GPU forwards.

Raw BF16 attention observation and full-vocabulary technical tensors stay on server, identified by RAW_TENSOR_MANIFEST.json. GPU reruns require pinned external checkpoint and fresh root; adapt paths, run technical.py, then run_study.py and finish scripts. Original source/execution hashes and freezes remain preserved.

Strict utility requires unique native and modified top candidates and keeps the full cohort denominator. Random norms match actual structured Q doses per token; unmatched cases remain. Norm row scale for Random is the *structured reference* clipping scale, while the randomized scalar amplitude is in dose_matching.amplitude. P/R/Q total doses differ, limiting region-semantics attribution. Attention observations are descriptive associations, not mediation proof.

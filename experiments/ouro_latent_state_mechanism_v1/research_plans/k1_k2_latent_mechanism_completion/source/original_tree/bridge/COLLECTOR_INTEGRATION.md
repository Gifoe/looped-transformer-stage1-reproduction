# Collector integration (candidate; no launch)

CLI: python bridge/b2_freeze_science.py --root ROOT --readiness READY.json ; python bridge/b2_collect.py --root ROOT --kind reference|treatment --stage train_all|train64|train1030|validation299.

READY.json requires sources (relative path->SHA256; include all bridge producer/manual/audit/analysis/codec and imported dependencies), model_path, lock_path and receipts. Receipt keys: technical and each train64/train1030/validation299 × pending/audit/analysis/E2audit. Each receipt has path, sha256, exact status and optional bindings {JSONfield:relative_dependency_path}. Creator explicitly checks PASS technical and E2/audit, COMPLETE pending/analysis, audit pending_sha256 and analysis audit_sha256. Missing E2 receipt stops; external supervisor can wait for its creation. No exception is swallowed. Current exact PASS status must be reconciled with actual technical receipt before freeze; no invented receipt. Freeze uses exclusive output, original AB scientific order/source/input SHA receipts and complete three AB stages. b2_audit.py and b2_analyze.py must exist and be explicitly hashed before freeze.

Kernel source_identity remains AB origin; producer_identity is actual loaded B2 runtime plus its sciencefreeze SHA. All stored provenance paths are root-relative. Ref train_all has1094 original Train; cyclic donor IDs sorted within original split are frozen before T. Reference CPU audit must bind PENDING hash before treatments. Validation references wait for Train1030 coverage and analysis. Each resumed case verifies exact input/source/producer/external source headers plus CAS chunks. Full split reference PENDING IDs and case SHA are checked on donor lookup. New cases use actual tensor byte upper bound +8GiB reserve before save. Model identity must equal AB identity excluding only prior sourcefreeze SHA. No automatic stage extension or audit/analysis execution exists.

Only compilation checked for collector/freezer. Root review and CPU fault-injection/integration tests against actual readiness/analysis schema remain required; no claim of execution readiness.

## Integration correction

Unique sciencefreeze is B2_FULL_SOURCE_FREEZE.json. Trace JSON keys are normalized strings at production and comparison. Case metadata preserves kernel_status and adds execution_status COLLECTED_AWAITING_AUDIT; PENDING COMPLETE means collection only.

Technical READY receipt additionally requires technical_freeze_path, undefined_proof_audit_path and undefined_proof_audit_sha256. The independent proof receipt must have status PASS, technical_gate_sha256, case_metadata_sha256 mapping all3 IDs to metadataSHA, source_sha256. It certifies legitimate undefined math using raw witnesses/donor references, not reason strings. Creator separately verifies actual technicalstatus, all3 CAS records, sourcefreeze/source hashes, 25 endpoint expected coverage, available replayed versus undefined disjointness. READY requires final_order_path, queue_path, codec_path explicitly included in sources. READY itself and all source/dataheaders are hashed and checked again immediately before exclusive freeze creation.

CPU test_schema_cpu.py PASS proves trace JSON roundtrip and rejects forged pending/audit/analysis dependency binding and incorrect status even with refreshed outer file hash. This is not full collector runtime validation.

# Numerical audit

{
  "status": "PASS",
  "base_pair": "CW000",
  "official_zero_identity": [
    {
      "id": "CW000_L0_donor",
      "official_all49152_bothpasses_bitwise": true
    },
    {
      "id": "CW000_L0_donor_edit",
      "official_all49152_bothpasses_bitwise": true
    },
    {
      "id": "CW000_L0_donor_unrelated_source",
      "official_all49152_bothpasses_bitwise": true
    },
    {
      "id": "CW000_L0_original",
      "official_all49152_bothpasses_bitwise": true
    },
    {
      "id": "CW000_L0_original_edit",
      "official_all49152_bothpasses_bitwise": true
    },
    {
      "id": "CW000_L0_original_unrelated_source",
      "official_all49152_bothpasses_bitwise": true
    },
    {
      "id": "CW000_L1_donor",
      "official_all49152_bothpasses_bitwise": true
    },
    {
      "id": "CW000_L1_donor_edit",
      "official_all49152_bothpasses_bitwise": true
    },
    {
      "id": "CW000_L1_donor_unrelated_source",
      "official_all49152_bothpasses_bitwise": true
    },
    {
      "id": "CW000_L1_original",
      "official_all49152_bothpasses_bitwise": true
    },
    {
      "id": "CW000_L1_original_edit",
      "official_all49152_bothpasses_bitwise": true
    },
    {
      "id": "CW000_L1_original_unrelated_source",
      "official_all49152_bothpasses_bitwise": true
    }
  ],
  "source_future_mapping_and_options_bitwise": true,
  "same_variable_all_condition_null": true,
  "all_other_attention_rows_byte_equal_asserted": true,
  "counterfactual_K1_endpoint_unchanged": true,
  "witness_path": "/root/autodl-tmp/ouro-causal-edge-sol-20261117/raw/TECHNICAL_ATTENTION_WITNESSES.pt",
  "formal_pair_estimated_seconds": 8.983096561394632,
  "full96pair_estimated_seconds": 862.3772698938847,
  "actual_technical_wall_seconds": 9.988211608491838,
  "utc": "2026-11-16T18:56:10.111069+00:00"
}

{
  "status": "PASS",
  "candidates": 9,
  "max_absolute_reconstruction_error": 1.2212453270876722e-15,
  "source_sha256": "c1d5828fe4407808181d1859af389f5e8efbc8c2c3d5cb42552f3029f1356731",
  "scope": "CPUFP64 actualpostRoPEQ/K/V rowcounterfactual, fullcausalprefixnormalization and exactBF16writeback; noGPU/model",
  "utc": "2026-11-16T18:58:43.639571+00:00"
}

{
  "status": "PASS",
  "full49152_endpoints": 16128,
  "max_candidate_logprob_FP64_reference_error": 7.247561129020141e-07,
  "matched_old_development_units": 128,
  "scope": "Every native/counterfactual vector hash, candidate score/tie/prediction/logprob, symbolic cross label, source/target positions, unchangedK1 andsame-variablefullnull. Oldboundaryreference usedonlyaftercurrentnativefullhash parity; freshnotpooledwitholdheldout.",
  "utc": "2026-11-16T19:13:02.795890+00:00"
}

Original native SDPA is retained. Only the designated target row is corrected by native_out+(FP64 counter_row-FP64 base_row). Both reconstructions use identical precision, causal prefix, scale and normalization. Projection, norms, residuals and MLP execute naturally. Same-variable source tensors trigger exact null bypass; zero observer runs match unhooked full49152 K1/K2 on all12technical worlds. Every formal non-target attention row remains byte-identical at the patch site, and every K1 endpoint remains exact under K2 edits.

Actual BF16 doses, requested correction norms, source probability masses and complete-row probability changes are saved for every intervention. Shuffled directions match the actual structured BF16 row dose as closely as feasible; no clipping cap from the old CATC study is reused, and no unmatched cases are discarded. Backend and original checkpoint/source hashes remain frozen.


## Fresh wording zero identity

{
  "status": "PASS",
  "tests": [
    {
      "id": "NEW000_L0_original",
      "template": 0,
      "unhooked_observer_formal_full49152_bothpasses_bitwise": true
    },
    {
      "id": "NEW016_L0_original",
      "template": 1,
      "unhooked_observer_formal_full49152_bothpasses_bitwise": true
    },
    {
      "id": "NEW032_L0_original",
      "template": 2,
      "unhooked_observer_formal_full49152_bothpasses_bitwise": true
    }
  ],
  "numerical_audit_body_forwards": 6,
  "scope": "Firstfrozenbasepairperfreshwordingtemplate,selectedwithoutoutcomes; originalunhooked vsobservervsformal native outputs",
  "utc": "2026-11-16T19:16:01.321513+00:00"
}

The additional CPU archive comparison required aligning CPU/GPU tensor placement; the byte-equality requirement was unchanged. See FRESH_AUDIT_DEVICE_REPAIR.json.

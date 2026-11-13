# Primary fusion failure analysis and counterexamples

All gold-dependent classifications below are offline evaluation only; they do not change methods or select a new Test winner.

## joint5090

{
  "unchanged:K4_and_earlier_correct": 997,
  "unchanged:all_native_wrong": 87,
  "unchanged:earlier_recoverable_K4_error": 51,
  "rescue:earlier_recoverable_K4_error": 12,
  "damage:K4_only_correct": 10,
  "unchanged:K4_only_correct": 7,
  "damage:K4_and_earlier_correct": 8
}

Rescue examples (first5 by ID): ACTAAP_2014_5_8, Mercury_180828, Mercury_184975, Mercury_400084, Mercury_404435

Damage examples (first5 by ID): CSZ10358, Mercury_183820, Mercury_412202, Mercury_416407, Mercury_7139545

## fas_char5090

{
  "unchanged:K4_and_earlier_correct": 672,
  "unchanged:all_native_wrong": 381,
  "unchanged:earlier_recoverable_K4_error": 68,
  "rescue:earlier_recoverable_K4_error": 18,
  "damage:K4_and_earlier_correct": 16,
  "damage:K4_only_correct": 14,
  "unchanged:K4_only_correct": 3
}

Rescue examples (first5 by ID): ACTAAP_2014_5_8, AIMS_2009_4_5, MCAS_2004_5_7, MCAS_2004_9_15-v1, MCAS_2006_8_12

Damage examples (first5 by ID): MCAS_2009_5_6518, MDSA_2007_8_17, MDSA_2012_8_6, MEAP_2005_8_13, MEA_2016_5_4

Fusion changes decisions by mixing earlier, sometimes incorrect candidate competitions into K4. Native oracle headroom is not evidence that a label-free selector can identify those corrections. Full counts and every changed primary question are in E6A_PRIMARY_COUNTEREXAMPLES.json. Point estimates and intervals support failure to establish the declared primary gain; they do not imply that every possible fusion rule is impossible.

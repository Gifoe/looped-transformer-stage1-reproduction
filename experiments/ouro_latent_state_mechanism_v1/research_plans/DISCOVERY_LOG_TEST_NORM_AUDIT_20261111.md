# Discovery log addendum: test BF16 norm-control audit

Astra independently reviewed the complete ROUND_2 test norm arrays for 1,172 unique questions and 2,233,719 token positions, including source/coverage bindings and the actual readout-position ledger. This is a numerical control audit, not a new mechanism finding. The original repository and active PQR run remain unchanged.

The K1-direction/K2-norm and K2-direction/K1-norm controls have global C maximum absolute target-norm errors 0.107819 and 0.081558, respectively; every question's full-sequence maximum occurs at prefix token 0 or 1. These are not readout-token errors. Actual R maximum absolute errors are 0.0046496 and 0.0042615; maximum relative errors are 0.07144% and 0.08410%. For R, signed actual-minus-target counts are +/−/zero 560/612/0 and 567/605/0, respectively.

BF16 conversion therefore approximately preserves target norms; it is not an exact norm swap. Small measured norm error alone does not imply zero prediction effect. The arrays do not measure post-BF16 angular error, and do not establish that a pure direction change reconstructs a natural mechanism. No ROUND_3 outcomes were used.

Source SHA256: test coverage 1cdee583defb6fd580001b532e9fefd11de8f46b7943707ebdb9a30c3d4da4a7; full norm gzip e37bde984a27638ce436672e729c92bbfa83716fdb708e0e5b0fcaaa025898c4; exact-source recovery receipt c5e34b0414583a9be43acdae765d4b1ae404cee0add5dc7dde1783fd1bc103b7; actual R ledger 051f11ea05f984e98408aa69659deda24c12c40e09a8d60e758ca1aaec80785e; R ledger receipt 1591ea3f12dea80b02f1d0828156ebfdc5ed1aba42ea76fbbf89eb9e4d8e61b5.

The original gzip exporter wrote its output but raised NameError while constructing its receipt; the independent recovery pass compared all 1,172 exported arrays exactly to the coverage-hash-bound original metadata and recorded a separate receipt. The original failed exporter and output were retained.
